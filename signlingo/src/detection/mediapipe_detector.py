import mediapipe as mp
import numpy as np
import cv2
from dataclasses import dataclass, field


@dataclass
class DetectionResult:
    landmarks_flat: np.ndarray = field(default_factory=lambda: np.zeros(126))
    landmarks_per_hand: list = field(default_factory=list)
    handedness: list = field(default_factory=list)
    bounding_boxes: list = field(default_factory=list)
    num_hands: int = 0
    annotated_frame: np.ndarray = None


class HandDetector:
    """MediaPipe-based hand detector returning structured DetectionResult."""

    def __init__(self, max_num_hands=2, min_detection_confidence=0.7,
                 min_tracking_confidence=0.6, model_complexity=0):
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run hand detection on an RGB frame."""
        annotated = frame.copy()
        result = DetectionResult(annotated_frame=annotated)

        mp_result = self._hands.process(frame)
        if not mp_result.multi_hand_landmarks:
            return result

        h, w = frame.shape[:2]
        landmarks_per_hand = []
        bounding_boxes = []
        handedness_list = []

        for idx, hand_lms in enumerate(mp_result.multi_hand_landmarks):
            # Draw skeleton
            self._mp_draw.draw_landmarks(
                annotated, hand_lms, self._mp_hands.HAND_CONNECTIONS
            )

            # Extract (21, 3) landmark array
            lm_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark])
            landmarks_per_hand.append(lm_array)

            # Bounding box in pixel coords
            xs = lm_array[:, 0] * w
            ys = lm_array[:, 1] * h
            pad = 20
            x1, y1 = int(xs.min()) - pad, int(ys.min()) - pad
            x2, y2 = int(xs.max()) + pad, int(ys.max()) + pad
            bounding_boxes.append((x1, y1, x2, y2))

            # Handedness label
            if mp_result.multi_handedness and idx < len(mp_result.multi_handedness):
                label = mp_result.multi_handedness[idx].classification[0].label
                handedness_list.append(label)
            else:
                handedness_list.append("Unknown")

        # Flatten all landmarks to (126,) — 2 hands × 21 × 3
        flat = np.zeros(126)
        for i, lm in enumerate(landmarks_per_hand[:2]):
            flat[i * 63:(i + 1) * 63] = lm.flatten()

        result.landmarks_flat = flat
        result.landmarks_per_hand = landmarks_per_hand
        result.handedness = handedness_list
        result.bounding_boxes = bounding_boxes
        result.num_hands = len(landmarks_per_hand)
        result.annotated_frame = annotated
        return result

    def close(self):
        self._hands.close()
