class SentenceComposer:
    """Accumulates committed sign labels into an ASL gloss string."""

    def __init__(self):
        self._signs: list[str] = []

    def add_sign(self, label: str):
        """Append a committed sign label to the buffer."""
        if label:
            self._signs.append(label.upper())

    def get_gloss(self) -> str:
        """Return all buffered signs joined as ASL gloss."""
        return ' '.join(self._signs)

    def clear(self):
        """Reset the sign buffer."""
        self._signs.clear()

    def is_sentence_ready(self, pause_detected: bool) -> bool:
        """True when a sentence pause is detected and buffer has signs."""
        return pause_detected and len(self._signs) > 0
