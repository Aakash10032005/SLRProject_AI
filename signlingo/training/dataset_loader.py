"""
Dataset loaders for SignLingo training.

Classes
-------
ASLAlphabetDataset   — Kaggle static image dataset  (~87k images, ~4 GB)
WLASLDataset         — Legacy eager loader (kept for compatibility)
WLASLLazyDataset     — Lazy MP4 loader (preferred, zero disk extraction)
CombinedDataset      — Merges alphabet + WLASL with correct label offsets
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

# Re-export lazy loader so callers can import from one place
from training.wlasl_dataset import WLASLLazyDataset, LITE_CLASSES  # noqa: F401


# ── ASL Alphabet (static images) ─────────────────────────────────────────────

class ASLAlphabetDataset(Dataset):
    """
    Kaggle ASL Alphabet dataset.
    Layout: <dataset_path>/<letter>/<image>.jpg
    Returns (hand_crop, full_frame, label) — same tensor for both streams.
    """

    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.label_to_idx: dict[str, int] = {}
        self._load_samples()

    def _load_samples(self):
        classes = sorted(d.name for d in self.dataset_path.iterdir() if d.is_dir())
        self.label_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls_name in classes:
            cls_dir = self.dataset_path / cls_name
            for img_file in cls_dir.glob('*.jpg'):
                self.samples.append((str(img_file), self.label_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        if self.transform:
            img = self.transform(img)
        return img, img, label


# ── Legacy WLASL eager loader (kept for compatibility) ────────────────────────

class WLASLDataset(Dataset):
    """
    Legacy WLASL loader — reads all frames eagerly.
    Prefer WLASLLazyDataset for production training to avoid 35–50 GB extraction.
    """

    def __init__(self, dataset_path: str, transform=None, frames_per_clip: int = 16):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples: list[tuple[str, int]] = []
        self.label_to_idx: dict[str, int] = {}
        self._load_samples()

    def _load_samples(self):
        video_dir = self.dataset_path / 'videos'
        if not video_dir.exists():
            raise FileNotFoundError(f"WLASL videos directory not found: {video_dir}")
        classes = sorted(d.name for d in video_dir.iterdir() if d.is_dir())
        self.label_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        for cls_name in classes:
            for mp4 in (video_dir / cls_name).glob('*.mp4'):
                self.samples.append((str(mp4), self.label_to_idx[cls_name]))

    def _extract_frames(self, video_path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return [np.zeros((224, 224, 3), dtype=np.uint8)]
        indices = np.linspace(0, len(frames) - 1, self.frames_per_clip, dtype=int)
        return [frames[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)
        frame = frames[len(frames) // 2]
        img = Image.fromarray(cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB))
        if self.transform:
            img = self.transform(img)
        return img, img, label


# ── Combined dataset ──────────────────────────────────────────────────────────

class CombinedDataset(Dataset):
    """
    Merges ASLAlphabetDataset + WLASLLazyDataset (or WLASLDataset) with
    correct label offsets so indices don't collide.

    Label layout:
        0–25   : A–Z  (alphabet, offset=0)
        26–35  : 0–9  (numbers, if present in alphabet dataset)
        36–535 : WLASL words (offset=36)
    """

    def __init__(
        self,
        alphabet_dataset: Optional[ASLAlphabetDataset] = None,
        wlasl_dataset: Optional[Dataset] = None,
        alphabet_offset: int = 0,
        wlasl_offset: int = 36,
    ):
        self.alphabet = alphabet_dataset
        self.wlasl = wlasl_dataset
        self.alphabet_offset = alphabet_offset
        self.wlasl_offset = wlasl_offset

        n_alpha = len(alphabet_dataset) if alphabet_dataset else 0
        n_wlasl = len(wlasl_dataset) if wlasl_dataset else 0
        self._len = n_alpha + n_wlasl

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        alpha_len = len(self.alphabet) if self.alphabet else 0
        if idx < alpha_len:
            hand, full, label = self.alphabet[idx]
            return hand, full, label + self.alphabet_offset
        wlasl_idx = idx - alpha_len
        hand, full, label = self.wlasl[wlasl_idx]
        return hand, full, label + self.wlasl_offset
