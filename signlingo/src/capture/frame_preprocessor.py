import cv2
import numpy as np
import torch
from torchvision import transforms


class FramePreprocessor:
    """Handles all frame preprocessing for MediaPipe and model inputs."""

    def __init__(self, device: torch.device):
        self.device = device
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self._to_tensor = transforms.ToTensor()
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def preprocess_for_mediapipe(self, frame: np.ndarray) -> np.ndarray:
        """Resize to 640x480 and convert BGR -> RGB."""
        resized = cv2.resize(frame, (640, 480))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def preprocess_for_swin(self, frame: np.ndarray, bbox: tuple) -> torch.Tensor:
        """Crop bbox, resize to 224x224, normalize. Returns (1,3,224,224)."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            crop = frame
        else:
            crop = frame[y1:y2, x1:x2]

        crop_rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = self._normalize(self._to_tensor(crop_rgb)).unsqueeze(0)
        return tensor.to(self.device)

    def preprocess_for_vit(self, frame: np.ndarray) -> torch.Tensor:
        """Resize full frame to 224x224, normalize. Returns (1,3,224,224)."""
        resized = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        tensor = self._normalize(self._to_tensor(resized)).unsqueeze(0)
        return tensor.to(self.device)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE equalization per channel for lighting robustness."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
