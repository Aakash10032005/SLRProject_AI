"""
WLASL Lazy-Loading Dataset
==========================
Reads MP4 clips on-the-fly via cv2 — zero frames extracted to disk.
Supports full WLASL (2000+ classes) and a "Lite" subset manifest.

Directory layout expected:
    <dataset_root>/
        videos/
            <class_name>/
                <video_id>.mp4
        WLASL_v0.3.json          (optional — used for metadata only)
        lite_manifest.json       (optional — subset class list)

Usage:
    ds = WLASLLazyDataset('data/wlasl', transform=get_train_transforms())
    ds_lite = WLASLLazyDataset('data/wlasl', manifest='data/wlasl/lite_manifest.json', ...)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Lite subset: 50 high-frequency signs + full alphabet ──────────────────────
LITE_CLASSES: List[str] = [
    # Alphabet (26)
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    # High-frequency words (24)
    'hello', 'thank-you', 'please', 'sorry', 'yes', 'no',
    'help', 'water', 'food', 'eat', 'drink', 'more',
    'stop', 'go', 'come', 'want', 'need', 'like',
    'good', 'bad', 'name', 'where', 'what', 'understand',
]


def _sample_frames(cap: cv2.VideoCapture, n_frames: int) -> List[np.ndarray]:
    """
    Lazily sample n_frames evenly spaced from an open VideoCapture.
    Returns list of BGR uint8 arrays. Never writes to disk.
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback: read all frames (some encoders don't report count)
        frames: List[np.ndarray] = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        if not frames:
            return [np.zeros((224, 224, 3), dtype=np.uint8)]
        total = len(frames)
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        return [frames[i] for i in indices]

    indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames if frames else [np.zeros((224, 224, 3), dtype=np.uint8)]


def _bgr_to_pil(frame: np.ndarray, size: int = 224) -> Image.Image:
    """Resize BGR frame to (size×size) and convert to PIL RGB."""
    resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


class WLASLLazyDataset(Dataset):
    """
    Lazy WLASL video dataset.

    Parameters
    ----------
    dataset_root : str | Path
        Root directory containing ``videos/<class>/<id>.mp4``.
    transform : callable, optional
        torchvision transform applied to each PIL frame.
    n_frames : int
        Number of frames to sample per clip (default 8).
        Kept low to stay within 4 GB VRAM budget during training.
    manifest : str | Path | None
        Path to a JSON file ``{"classes": ["hello", "yes", ...]}`` for
        subset filtering. If None, all classes are used.
    label_offset : int
        Added to every label index (for CombinedDataset alignment).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        transform=None,
        n_frames: int = 8,
        manifest: Optional[str | Path] = None,
        label_offset: int = 0,
    ):
        self.root = Path(dataset_root)
        self.transform = transform
        self.n_frames = n_frames
        self.label_offset = label_offset

        self.samples: List[Tuple[Path, int]] = []   # (video_path, label_idx)
        self.label_to_idx: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}

        allowed = self._load_manifest(manifest)
        self._index_videos(allowed)

        logger.info(
            "WLASLLazyDataset: %d clips | %d classes | root=%s",
            len(self.samples), len(self.label_to_idx), self.root,
        )

    # ── Manifest ──────────────────────────────────────────────────────────────

    def _load_manifest(self, manifest: Optional[str | Path]) -> Optional[set]:
        if manifest is None:
            return None
        p = Path(manifest)
        if not p.exists():
            logger.warning("Manifest not found: %s — using all classes", p)
            return None
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        classes = data.get('classes', [])
        logger.info("Manifest loaded: %d classes", len(classes))
        return {c.lower() for c in classes}

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_videos(self, allowed: Optional[set]):
        video_dir = self.root / 'videos'
        if not video_dir.exists():
            raise FileNotFoundError(
                f"WLASL videos directory not found: {video_dir}\n"
                "Expected layout: <root>/videos/<class_name>/<id>.mp4"
            )

        class_dirs = sorted(
            d for d in video_dir.iterdir()
            if d.is_dir() and (allowed is None or d.name.lower() in allowed)
        )

        if not class_dirs:
            raise RuntimeError(
                f"No matching class directories found in {video_dir}. "
                "Check your manifest or dataset layout."
            )

        for idx, cls_dir in enumerate(class_dirs):
            cls_name = cls_dir.name
            self.label_to_idx[cls_name] = idx
            self.idx_to_label[idx] = cls_name
            for mp4 in cls_dir.glob('*.mp4'):
                self.samples.append((mp4, idx))

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        cap = cv2.VideoCapture(str(video_path))
        try:
            frames = _sample_frames(cap, self.n_frames)
        finally:
            cap.release()

        # Representative frame: middle of the clip
        mid = frames[len(frames) // 2]
        img = _bgr_to_pil(mid)

        if self.transform:
            img = self.transform(img)

        return img, img, label + self.label_offset   # (hand_crop, full_frame, label)

    def get_clip(self, idx: int) -> List[Image.Image]:
        """Return all sampled frames as PIL images (for inspection/debugging)."""
        video_path, _ = self.samples[idx]
        cap = cv2.VideoCapture(str(video_path))
        try:
            frames = _sample_frames(cap, self.n_frames)
        finally:
            cap.release()
        return [_bgr_to_pil(f) for f in frames]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def class_distribution(self) -> dict[str, int]:
        """Returns {class_name: sample_count} for imbalance analysis."""
        dist: dict[str, int] = {}
        for _, label in self.samples:
            name = self.idx_to_label[label]
            dist[name] = dist.get(name, 0) + 1
        return dist
