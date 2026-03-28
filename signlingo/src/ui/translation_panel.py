from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar,
    QPushButton, QListWidget, QListWidgetItem, QSizePolicy
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, pyqtSignal


class TranslationPanel(QWidget):
    """Displays ASL gloss, native translation, Roman transliteration, and history."""

    tts_requested = pyqtSignal(str, str)  # (native_text, language)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_native = ''
        self._last_language = 'hindi'
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ASL Gloss label
        self._gloss_label = QLabel("ASL Gloss: —")
        self._gloss_label.setObjectName("glossLabel")
        self._gloss_label.setWordWrap(True)
        layout.addWidget(self._gloss_label)

        # Native script display
        self._native_label = QLabel("—")
        self._native_label.setObjectName("nativeLabel")
        self._native_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._native_label.setWordWrap(True)
        self._native_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        try:
            font = QFont("Noto Sans", 24)
        except Exception:
            font = QFont()
            font.setPointSize(24)
        self._native_label.setFont(font)
        layout.addWidget(self._native_label, stretch=3)

        # Roman transliteration
        self._roman_label = QLabel("")
        self._roman_label.setObjectName("romanLabel")
        self._roman_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._roman_label.setWordWrap(True)
        layout.addWidget(self._roman_label)

        # Confidence bar
        self._confidence_bar = QProgressBar()
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setValue(0)
        self._confidence_bar.setTextVisible(True)
        self._confidence_bar.setFormat("Confidence: %p%")
        layout.addWidget(self._confidence_bar)

        # TTS replay button
        self._tts_btn = QPushButton("🔊 Speak Again")
        self._tts_btn.clicked.connect(self._on_tts_clicked)
        layout.addWidget(self._tts_btn)

        # History list
        history_label = QLabel("Recent Translations:")
        history_label.setObjectName("historyHeader")
        layout.addWidget(history_label)

        self._history_list = QListWidget()
        self._history_list.setMaximumHeight(150)
        layout.addWidget(self._history_list)

    def update_translation(self, native: str, roman: str, gloss: str,
                           confidence: float, language: str = 'hindi'):
        """Update all translation display elements."""
        self._last_native = native
        self._last_language = language

        self._gloss_label.setText(f"ASL Gloss: {gloss}")
        self._native_label.setText(native)
        self._roman_label.setText(roman)
        self._confidence_bar.setValue(int(confidence * 100))

        # Update confidence bar color via dynamic property
        if confidence > 0.75:
            self._confidence_bar.setProperty("level", "high")
        elif confidence > 0.5:
            self._confidence_bar.setProperty("level", "medium")
        else:
            self._confidence_bar.setProperty("level", "low")
        self._confidence_bar.style().unpolish(self._confidence_bar)
        self._confidence_bar.style().polish(self._confidence_bar)

        # Add to history (max 5)
        item_text = f"{gloss} → {native}"
        self._history_list.insertItem(0, QListWidgetItem(item_text))
        while self._history_list.count() > 5:
            self._history_list.takeItem(self._history_list.count() - 1)

    def update_gloss(self, gloss: str):
        """Update only the current gloss (live, before translation)."""
        self._gloss_label.setText(f"ASL Gloss: {gloss}")

    def _on_tts_clicked(self):
        if self._last_native:
            self.tts_requested.emit(self._last_native, self._last_language)
