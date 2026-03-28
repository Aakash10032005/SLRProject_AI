import numpy as np


class LandmarkNormalizer:
    """Normalizes hand landmarks for skin-tone and scale robustness."""

    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Z-score normalize relative to wrist keypoint.
        Translates so wrist (index 0) is at origin, scales by palm size.
        """
        if landmarks.shape[0] == 0:
            return landmarks

        lm = landmarks.copy().reshape(-1, 3)
        wrist = lm[0].copy()

        # Translate to wrist origin
        lm -= wrist

        # Scale by palm size (distance from wrist to middle finger MCP = index 9)
        if len(lm) > 9:
            palm_size = np.linalg.norm(lm[9])
            if palm_size > 1e-6:
                lm /= palm_size

        return lm.reshape(landmarks.shape)

    def flatten_to_vector(self, landmarks_per_hand: list, max_hands: int = 2) -> np.ndarray:
        """
        Flatten normalized landmarks to a (126,) vector.
        Pads with zeros if fewer than max_hands detected.
        """
        result = np.zeros(max_hands * 63)
        for i, lm in enumerate(landmarks_per_hand[:max_hands]):
            normalized = self.normalize(lm)
            result[i * 63:(i + 1) * 63] = normalized.flatten()
        return result
