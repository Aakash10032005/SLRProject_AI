import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.mediapipe_detector import HandDetector, DetectionResult


def test_no_hands_blank_frame():
    """HandDetector on blank frame should return num_hands=0 without crashing."""
    detector = HandDetector()
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank_frame)

    assert isinstance(result, DetectionResult)
    assert result.num_hands == 0
    assert result.landmarks_flat.shape == (126,)
    assert np.all(result.landmarks_flat == 0)
    assert result.bounding_boxes == []
    assert result.handedness == []
    detector.close()
    print("test_no_hands_blank_frame PASSED")


def test_detection_result_annotated_frame():
    """Annotated frame should be returned even with no hands."""
    detector = HandDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert result.annotated_frame is not None
    assert result.annotated_frame.shape == (480, 640, 3)
    detector.close()
    print("test_detection_result_annotated_frame PASSED")


if __name__ == '__main__':
    test_no_hands_blank_frame()
    test_detection_result_annotated_frame()
    print("All detection tests passed.")
