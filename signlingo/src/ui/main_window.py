import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QMenuBar, QMenu, QStatusBar, QLabel, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QFont

from .camera_widget import CameraWidget
from .translation_panel import TranslationPanel
from .language_selector import LanguageSelector


class _Signals(QObject):
    """Thread-safe signals for pipeline -> UI updates."""
    frame_update = pyqtSignal(object, str, float, str)
    translation_update = pyqtSignal(str, str, str, str)


class MainWindow(QMainWindow):
    """Main application window for SignLingo."""

    def __init__(self, pipeline=None):
        super().__init__()
        self.pipeline = pipeline
        self._signals = _Signals()
        self._auto_speak = True
        self._show_roman = True
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
        self._load_stylesheet()

    def _setup_ui(self):
        self.setWindowTitle("SignLingo — ASL to Indian Language Translator")
        self.setMinimumSize(1280, 720)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # LEFT: Camera (60%)
        self.camera_widget = CameraWidget()
        main_layout.addWidget(self.camera_widget, stretch=6)

        # RIGHT: Controls (40%)
        right_panel = QWidget()
        right_panel.setObjectName("panel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(12, 12, 12, 12)

        self.language_selector = LanguageSelector()
        right_layout.addWidget(self.language_selector)

        self.translation_panel = TranslationPanel()
        right_layout.addWidget(self.translation_panel, stretch=1)

        # Status
        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("glossLabel")
        right_layout.addWidget(self._status_label)

        main_layout.addWidget(right_panel, stretch=4)

        # Status bar
        self.statusBar().showMessage("SignLingo ready — select language and start signing")

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        new_session = QAction("New Session", self)
        new_session.triggered.connect(self._on_new_session)
        file_menu.addAction(new_session)

        save_transcript = QAction("Save Transcript", self)
        save_transcript.triggered.connect(self._on_save_transcript)
        file_menu.addAction(save_transcript)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About SignLingo", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

        hw_status = QAction("Hardware Status", self)
        hw_status.triggered.connect(self._on_hw_status)
        help_menu.addAction(hw_status)

    def _connect_signals(self):
        self._signals.frame_update.connect(self._on_frame_update)
        self._signals.translation_update.connect(self._on_translation_update)

        self.language_selector.language_changed.connect(self._on_language_changed)
        self.language_selector.show_roman_changed.connect(self._on_show_roman_changed)
        self.language_selector.auto_speak_changed.connect(self._on_auto_speak_changed)
        self.translation_panel.tts_requested.connect(self._on_tts_requested)

    def _load_stylesheet(self):
        qss_path = Path(__file__).parent / 'styles.qss'
        if qss_path.exists():
            with open(qss_path, 'r', encoding='utf-8') as f:
                self.setStyleSheet(f.read())

    # --- Pipeline callbacks (called from background thread) ---

    def on_ui_update(self, frame, label: str, confidence: float, gloss: str):
        """Called by pipeline from background thread."""
        self._signals.frame_update.emit(frame, label, confidence, gloss)

    def on_translation_result(self, native: str, roman: str, gloss: str, language: str):
        """Called by pipeline from background thread."""
        self._signals.translation_update.emit(native, roman, gloss, language)

    # --- Qt slot handlers (main thread) ---

    def _on_frame_update(self, frame, label: str, confidence: float, gloss: str):
        self.camera_widget.update_frame(frame, label, confidence)
        self.translation_panel.update_gloss(gloss)
        conf_pct = int(confidence * 100)
        self._status_label.setText(
            f"Sign: {label or '—'}  |  Confidence: {conf_pct}%  |  Gloss: {gloss or '—'}"
        )

    def _on_translation_update(self, native: str, roman: str, gloss: str, language: str):
        display_roman = roman if self._show_roman else ''
        self.translation_panel.update_translation(native, display_roman, gloss, 1.0, language)
        self.statusBar().showMessage(f"Translated: {gloss} → {native}")

    def _on_language_changed(self, language: str):
        if self.pipeline:
            self.pipeline.set_language(language)
        self.statusBar().showMessage(f"Language changed to: {language.title()}")

    def _on_show_roman_changed(self, show: bool):
        self._show_roman = show

    def _on_auto_speak_changed(self, auto: bool):
        self._auto_speak = auto
        if self.pipeline:
            self.pipeline.config.setdefault('tts', {})['enabled'] = auto

    def _on_tts_requested(self, native: str, language: str):
        if self.pipeline:
            self.pipeline.tts.speak(native, language)

    def _on_new_session(self):
        if self.pipeline:
            self.pipeline.composer.clear()
            self.pipeline._clear_fused_feature_buffer()
            self.pipeline.transcript.close()
            from ..output.transcript_logger import TranscriptLogger
            self.pipeline.transcript = TranscriptLogger(
                self.pipeline.config.get('app', {}).get('transcript_dir', 'transcripts')
            )
        self.statusBar().showMessage("New session started")

    def _on_save_transcript(self):
        self.statusBar().showMessage("Transcript auto-saved to transcripts/ folder")

    def _on_about(self):
        QMessageBox.about(
            self,
            "About SignLingo",
            "SignLingo v1.0\n\n"
            "Real-time ASL recognition with Indian language translation.\n\n"
            "Architecture: HSTFe (Hybrid Swin-ViT Temporal Fusion Encoder)\n"
            "Translation: pluggable CGME (default offline rule-based; optional Ollama)\n\n"
            "License: Apache 2.0"
        )

    def _on_hw_status(self):
        import torch
        cuda_info = "CUDA available" if torch.cuda.is_available() else "CPU only"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            cuda_info = f"GPU: {gpu_name} ({vram} MB VRAM)"

        trans = ""
        if self.pipeline and self.pipeline.cgme:
            b = self.pipeline.cgme.backend
            trans = f"{b.backend_name} ({'ready' if b.is_available() else 'unavailable'})"

        QMessageBox.information(
            self,
            "Hardware Status",
            f"GPU: {cuda_info}\n"
            f"Translation: {trans or '—'}\n"
            f"Mode: {'HSTFe' if self.pipeline and not self.pipeline._use_fallback else 'Fallback Recognizer'}"
        )

    def closeEvent(self, event):
        if self.pipeline:
            self.pipeline.stop()
        event.accept()
