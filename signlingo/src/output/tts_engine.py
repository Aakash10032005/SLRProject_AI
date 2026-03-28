import subprocess
import threading
import logging
import shutil

logger = logging.getLogger(__name__)

LANGUAGE_VOICES = {
    'hindi': 'hi',
    'tamil': 'ta',
    'telugu': 'te',
    'malayalam': 'ml',
    'marathi': 'mr',
}


class TTSEngine:
    """
    Text-to-speech engine with espeak-ng primary and pyttsx3 fallback.
    Runs in daemon thread to avoid blocking main thread.
    """

    def __init__(self):
        self._current_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._espeak_available = shutil.which('espeak-ng') is not None
        self._pyttsx3_engine = None

        if not self._espeak_available:
            logger.warning("espeak-ng not found in PATH. Falling back to pyttsx3.")
            try:
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
            except Exception as e:
                logger.error(f"pyttsx3 init failed: {e}. TTS disabled.")

    def speak(self, text: str, language: str):
        """Speak text in the given language. Runs in a daemon thread."""
        thread = threading.Thread(
            target=self._speak_worker,
            args=(text, language),
            daemon=True
        )
        thread.start()

    def _speak_worker(self, text: str, language: str):
        self.stop_speaking()
        voice_code = LANGUAGE_VOICES.get(language.lower(), 'en')

        if self._espeak_available:
            try:
                with self._lock:
                    self._current_proc = subprocess.Popen(
                        ['espeak-ng', '-v', voice_code, '-s', '140', text],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                self._current_proc.wait()
            except Exception as e:
                logger.error(f"espeak-ng error: {e}")
        elif self._pyttsx3_engine:
            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
            except Exception as e:
                logger.error(f"pyttsx3 error: {e}")

    def stop_speaking(self):
        """Kill current TTS subprocess if running."""
        with self._lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                except Exception:
                    pass
            self._current_proc = None
