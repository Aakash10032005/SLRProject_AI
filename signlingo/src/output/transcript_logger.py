import os
import time
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class TranscriptLogger:
    """Logs signs and translations to a session transcript file."""

    def __init__(self, transcript_dir: str = 'transcripts'):
        Path(transcript_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._path = Path(transcript_dir) / f"session_{timestamp}.txt"
        self._file = open(self._path, 'w', encoding='utf-8')
        self._file.write(f"SignLingo Session — {datetime.now().isoformat()}\n")
        self._file.write("=" * 60 + "\n\n")
        self._file.flush()
        logger.info(f"Transcript logging to: {self._path}")

    def log_sign(self, label: str, confidence: float, timestamp: float):
        """Log a committed sign."""
        line = f"[{timestamp:.2f}] SIGN: {label} (conf={confidence:.2f})\n"
        self._file.write(line)
        self._file.flush()

    def log_translation(self, asl_gloss: str, native: str, roman: str, language: str):
        """Log a completed translation."""
        ts = datetime.now().strftime('%H:%M:%S')
        self._file.write(f"\n[{ts}] GLOSS: {asl_gloss}\n")
        self._file.write(f"  [{language.upper()}] {native}\n")
        if roman:
            self._file.write(f"  [Roman] {roman}\n")
        self._file.write("\n")
        self._file.flush()

    def close(self):
        """Flush and close the transcript file."""
        if self._file and not self._file.closed:
            self._file.write("\n--- Session ended ---\n")
            self._file.flush()
            self._file.close()
            logger.info(f"Transcript saved: {self._path}")
