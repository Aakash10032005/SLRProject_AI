from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QRadioButton,
    QCheckBox, QButtonGroup, QLabel
)
from PyQt6.QtCore import pyqtSignal

LANGUAGES = [
    ('hindi', '🇮🇳 Hindi'),
    ('tamil', '🌺 Tamil'),
    ('telugu', '🌸 Telugu'),
    ('malayalam', '🌴 Malayalam'),
    ('marathi', '🏔️ Marathi'),
]


class LanguageSelector(QWidget):
    """Radio button language selector with transliteration and TTS options."""

    language_changed = pyqtSignal(str)
    show_roman_changed = pyqtSignal(bool)
    auto_speak_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        title = QLabel("Target Language")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self._button_group = QButtonGroup(self)
        self._radio_buttons = {}

        for lang_key, lang_label in LANGUAGES:
            rb = QRadioButton(lang_label)
            rb.setObjectName("langRadio")
            self._button_group.addButton(rb)
            self._radio_buttons[lang_key] = rb
            layout.addWidget(rb)
            rb.toggled.connect(lambda checked, k=lang_key: self._on_language_toggled(checked, k))

        # Default: Hindi
        self._radio_buttons['hindi'].setChecked(True)

        # Options
        self._show_roman = QCheckBox("Show Transliteration")
        self._show_roman.setChecked(True)
        self._show_roman.toggled.connect(self.show_roman_changed.emit)
        layout.addWidget(self._show_roman)

        self._auto_speak = QCheckBox("Auto-speak translation")
        self._auto_speak.setChecked(True)
        self._auto_speak.toggled.connect(self.auto_speak_changed.emit)
        layout.addWidget(self._auto_speak)

    def _on_language_toggled(self, checked: bool, lang_key: str):
        if checked:
            self.language_changed.emit(lang_key)

    def get_selected_language(self) -> str:
        for key, rb in self._radio_buttons.items():
            if rb.isChecked():
                return key
        return 'hindi'

    def is_show_roman(self) -> bool:
        return self._show_roman.isChecked()

    def is_auto_speak(self) -> bool:
        return self._auto_speak.isChecked()
