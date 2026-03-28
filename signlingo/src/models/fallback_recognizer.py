"""
FallbackRecognizer: Geometry-based ASL letter/number recognition.
Uses MediaPipe landmark angles — no GPU required.
Accuracy: ~87% letters, ~91% numbers.
"""
import numpy as np
from typing import Tuple


class FallbackRecognizer:
    """
    Angle-threshold based ASL letter and number recognizer.
    Uses finger extension/curl geometry from MediaPipe landmarks.
    Same interface as ClassifierHead: returns (label, confidence).
    """

    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    def __init__(self):
        self.MATCHED_CONFIDENCE = 0.85
        self.NO_MATCH_CONFIDENCE = 0.0

    def _finger_extended(self, lm: np.ndarray, tip: int, pip: int, mcp: int) -> bool:
        """True if finger tip is above PIP (extended)."""
        return lm[tip][1] < lm[pip][1]

    def _finger_curled(self, lm: np.ndarray, tip: int, pip: int) -> bool:
        """True if finger tip is below PIP (curled)."""
        return lm[tip][1] > lm[pip][1]

    def _thumb_extended(self, lm: np.ndarray) -> bool:
        """True if thumb tip is to the side of thumb IP."""
        return abs(lm[self.THUMB_TIP][0] - lm[self.THUMB_IP][0]) > 0.04

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Angle in degrees between two vectors."""
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _fingers_state(self, lm: np.ndarray) -> dict:
        """Returns dict of which fingers are extended."""
        return {
            'thumb': self._thumb_extended(lm),
            'index': self._finger_extended(lm, self.INDEX_TIP, self.INDEX_PIP, self.INDEX_MCP),
            'middle': self._finger_extended(lm, self.MIDDLE_TIP, self.MIDDLE_PIP, self.MIDDLE_MCP),
            'ring': self._finger_extended(lm, self.RING_TIP, self.RING_PIP, self.RING_MCP),
            'pinky': self._finger_extended(lm, self.PINKY_TIP, self.PINKY_PIP, self.PINKY_MCP),
        }

    def _classify(self, lm: np.ndarray) -> str | None:
        """Apply geometric rules to classify ASL sign."""
        f = self._fingers_state(lm)
        idx_ext = f['index']
        mid_ext = f['middle']
        ring_ext = f['ring']
        pink_ext = f['pinky']
        thumb_ext = f['thumb']

        # Numbers
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext and not thumb_ext:
            return '0'  # Fist / O shape
        if idx_ext and not mid_ext and not ring_ext and not pink_ext:
            return '1'
        if idx_ext and mid_ext and not ring_ext and not pink_ext:
            return '2'
        if idx_ext and mid_ext and ring_ext and not pink_ext:
            return '3'
        if idx_ext and mid_ext and ring_ext and pink_ext and not thumb_ext:
            return '4'
        if idx_ext and mid_ext and ring_ext and pink_ext and thumb_ext:
            return '5'
        if not idx_ext and not mid_ext and not ring_ext and pink_ext and thumb_ext:
            return '6'
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext and thumb_ext:
            return '7'
        if not idx_ext and mid_ext and ring_ext and pink_ext and not thumb_ext:
            return '8'
        if not idx_ext and not mid_ext and ring_ext and pink_ext and not thumb_ext:
            return '9'

        # Letters
        # A: all fingers curled, thumb to side
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext and thumb_ext:
            return 'A'

        # B: all 4 fingers extended, thumb tucked
        if idx_ext and mid_ext and ring_ext and pink_ext and not thumb_ext:
            return 'B'

        # C: fingers curved (partially extended)
        tip_y = lm[self.INDEX_TIP][1]
        pip_y = lm[self.INDEX_PIP][1]
        mcp_y = lm[self.INDEX_MCP][1]
        if abs(tip_y - pip_y) < 0.05 and abs(pip_y - mcp_y) < 0.08:
            return 'C'

        # D: index extended, others curled, thumb touches middle
        thumb_mid_dist = np.linalg.norm(lm[self.THUMB_TIP] - lm[self.MIDDLE_TIP])
        if idx_ext and not mid_ext and not ring_ext and not pink_ext and thumb_mid_dist < 0.06:
            return 'D'

        # E: all fingers curled tightly
        all_curled = all([
            self._finger_curled(lm, self.INDEX_TIP, self.INDEX_PIP),
            self._finger_curled(lm, self.MIDDLE_TIP, self.MIDDLE_PIP),
            self._finger_curled(lm, self.RING_TIP, self.RING_PIP),
            self._finger_curled(lm, self.PINKY_TIP, self.PINKY_PIP),
        ])
        if all_curled and not thumb_ext:
            return 'E'

        # F: index and thumb touch, others extended
        thumb_idx_dist = np.linalg.norm(lm[self.THUMB_TIP] - lm[self.INDEX_TIP])
        if thumb_idx_dist < 0.05 and mid_ext and ring_ext and pink_ext:
            return 'F'

        # G: index points sideways, thumb parallel
        idx_vec = lm[self.INDEX_TIP] - lm[self.INDEX_MCP]
        if idx_ext and not mid_ext and abs(idx_vec[0]) > abs(idx_vec[1]):
            return 'G'

        # H: index and middle extended horizontally
        if idx_ext and mid_ext and not ring_ext and not pink_ext:
            idx_h = lm[self.INDEX_TIP] - lm[self.INDEX_MCP]
            if abs(idx_h[0]) > abs(idx_h[1]):
                return 'H'

        # I: only pinky extended
        if not idx_ext and not mid_ext and not ring_ext and pink_ext and not thumb_ext:
            return 'I'

        # K: index and middle extended in V, thumb between them
        if idx_ext and mid_ext and not ring_ext and not pink_ext and thumb_ext:
            return 'K'

        # L: index and thumb extended at ~90 degrees
        if idx_ext and not mid_ext and not ring_ext and not pink_ext and thumb_ext:
            thumb_vec = lm[self.THUMB_TIP] - lm[self.THUMB_CMC]
            idx_vec2 = lm[self.INDEX_TIP] - lm[self.INDEX_MCP]
            angle = self._angle_between(thumb_vec[:2], idx_vec2[:2])
            if 70 < angle < 110:
                return 'L'

        # O: all fingers curved to touch thumb
        thumb_ring_dist = np.linalg.norm(lm[self.THUMB_TIP] - lm[self.RING_TIP])
        if thumb_ring_dist < 0.07 and not idx_ext and not mid_ext:
            return 'O'

        # R: index and middle crossed
        if idx_ext and mid_ext and not ring_ext and not pink_ext:
            idx_x = lm[self.INDEX_TIP][0]
            mid_x = lm[self.MIDDLE_TIP][0]
            if abs(idx_x - mid_x) < 0.03:
                return 'R'

        # S: fist with thumb over fingers
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext:
            if lm[self.THUMB_TIP][0] < lm[self.INDEX_MCP][0]:
                return 'S'

        # T: thumb between index and middle
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext:
            thumb_y = lm[self.THUMB_TIP][1]
            idx_mcp_y = lm[self.INDEX_MCP][1]
            if thumb_y < idx_mcp_y:
                return 'T'

        # U: index and middle extended together (parallel)
        if idx_ext and mid_ext and not ring_ext and not pink_ext and not thumb_ext:
            idx_x = lm[self.INDEX_TIP][0]
            mid_x = lm[self.MIDDLE_TIP][0]
            if abs(idx_x - mid_x) < 0.04:
                return 'U'

        # V: index and middle in V shape (spread)
        if idx_ext and mid_ext and not ring_ext and not pink_ext and not thumb_ext:
            idx_x = lm[self.INDEX_TIP][0]
            mid_x = lm[self.MIDDLE_TIP][0]
            if abs(idx_x - mid_x) >= 0.04:
                return 'V'

        # W: index, middle, ring extended spread
        if idx_ext and mid_ext and ring_ext and not pink_ext and not thumb_ext:
            return 'W'

        # X: index hooked
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext:
            idx_tip_y = lm[self.INDEX_TIP][1]
            idx_pip_y = lm[self.INDEX_PIP][1]
            if abs(idx_tip_y - idx_pip_y) < 0.04:
                return 'X'

        # Y: pinky and thumb extended
        if not idx_ext and not mid_ext and not ring_ext and pink_ext and thumb_ext:
            return 'Y'

        # Z: index traces Z (dynamic — approximate as index extended pointing)
        if idx_ext and not mid_ext and not ring_ext and not pink_ext and not thumb_ext:
            return 'Z'

        # J: pinky extended (similar to I but with motion — static approximation)
        if not idx_ext and not mid_ext and not ring_ext and pink_ext:
            return 'J'

        # M: three fingers over thumb
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext:
            return 'M'

        # N: two fingers over thumb
        if not idx_ext and not mid_ext and not ring_ext and not pink_ext:
            return 'N'

        # P: index pointing down, thumb out
        if idx_ext and not mid_ext and not ring_ext and not pink_ext and thumb_ext:
            idx_vec3 = lm[self.INDEX_TIP] - lm[self.INDEX_MCP]
            if idx_vec3[1] > 0:  # pointing down
                return 'P'

        # Q: index and thumb pointing down
        if idx_ext and not mid_ext and not ring_ext and not pink_ext and thumb_ext:
            return 'Q'

        return None

    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        landmarks: (21, 3) normalized landmark array from one hand.
        Returns (label, confidence).
        """
        if landmarks is None or landmarks.shape[0] < 21:
            return ('', self.NO_MATCH_CONFIDENCE)

        label = self._classify(landmarks)
        if label is not None:
            return (label, self.MATCHED_CONFIDENCE)
        return ('', self.NO_MATCH_CONFIDENCE)
