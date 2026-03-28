import cv2
import numpy as np


class OpticalFlowAnalyzer:
    """
    Lucas-Kanade sparse optical flow for hand motion complexity estimation.
    Tracks hand landmark points across frames.
    """

    LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    def compute_complexity(self, prev_landmarks: np.ndarray,
                           curr_landmarks: np.ndarray) -> float:
        """
        Compute normalized motion complexity from landmark displacement.
        Returns float in [0.0, 1.0].
        """
        if prev_landmarks is None or curr_landmarks is None:
            return 0.0
        if prev_landmarks.shape != curr_landmarks.shape:
            return 0.0

        # Compute displacement magnitudes between corresponding landmarks
        diff = curr_landmarks - prev_landmarks
        magnitudes = np.linalg.norm(diff.reshape(-1, 3)[:, :2], axis=1)
        mean_disp = float(np.mean(magnitudes))

        # Normalize to [0, 1] — clamp at 0.3 as max expected displacement
        return min(mean_disp / 0.3, 1.0)
