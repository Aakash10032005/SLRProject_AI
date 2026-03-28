import time
from collections import deque


class SignBoundaryDetector:
    """
    Detects sign boundaries using confidence peak-and-decay algorithm.
    Prevents double-commits via debounce.
    """

    def __init__(self, drop_threshold: float = 0.15, drop_frames: int = 3,
                 debounce_frames: int = 8):
        self.drop_threshold = drop_threshold
        self.drop_frames = drop_frames
        self.debounce_frames = debounce_frames
        self._frames_since_commit = 0
        self._peak_confidence = 0.0

    def is_committed(self, confidence_history: list) -> bool:
        """
        Returns True when:
        - A peak was found AND confidence dropped >15% for 3 consecutive frames
        - AND minimum 8 frames since last commit (debounce)
        """
        self._frames_since_commit += 1

        if len(confidence_history) < self.drop_frames + 1:
            return False

        if self._frames_since_commit < self.debounce_frames:
            return False

        # Update peak
        current = confidence_history[-1]
        if current > self._peak_confidence:
            self._peak_confidence = current

        # Check if confidence has dropped consistently for drop_frames
        recent = confidence_history[-(self.drop_frames):]
        if self._peak_confidence < 0.1:
            return False

        drop = self._peak_confidence - recent[-1]
        consistently_dropping = all(
            recent[i] >= recent[i + 1]
            for i in range(len(recent) - 1)
        )

        if drop > self.drop_threshold and consistently_dropping:
            self._frames_since_commit = 0
            self._peak_confidence = 0.0
            return True

        return False

    def detect_sentence_pause(self, timestamp_last_sign: float,
                               pause_threshold: float = 1.5) -> bool:
        """Returns True if time since last committed sign > pause_threshold seconds."""
        if timestamp_last_sign <= 0:
            return False
        return (time.time() - timestamp_last_sign) > pause_threshold
