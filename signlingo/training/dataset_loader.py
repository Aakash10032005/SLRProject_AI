import os
import cv2
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image


class WLASLDataset(Dataset):
    """
    WLASL dataset loader.
    Expects directory structure: dataset_path/videos/<class_name>/<video>.mp4
    Or JSON annotation file: dataset_path/WLASL_v0.3.json
    """

    def __init__(self, dataset_path: str, transform=None, frames_per_clip: int = 16):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.samples = []
        self.label_to_idx = {}
        self._load_samples()

    def _load_samples(self):
        video_dir = self.dataset_path / 'videos'
        if not video_dir.exists():
            raise FileNotFoundError(f"WLASL videos directory not found: {video_dir}")

        classes = sorted([d.name for d in video_dir.iterdir() if d.is_dir()])
        self.label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls_name in classes:
            cls_dir = video_dir / cls_name
            for video_file in cls_dir.glob('*.mp4'):
                self.samples.append((str(video_file), self.label_to_idx[cls_name]))

    def _extract_frames(self, video_path: str):
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

        # Sample evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, self.frames_per_clip, dtype=int)
        return [frames[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._extract_frames(video_path)

        # Use middle frame as representative
        frame = frames[len(frames) // 2]
        frame_rgb = cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        if self.transform:
            img = self.transform(img)

        return img, img, label  # (hand_crop, full_frame, label)


class ASLAlphabetDataset(Dataset):
    """
    Kaggle ASL Alphabet dataset loader.
    Expects: dataset_path/<letter>/<image>.jpg
    """

    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.samples = []
        self.label_to_idx = {}
        self._load_samples()

    def _load_samples(self):
        classes = sorted([d.name for d in self.dataset_path.iterdir() if d.is_dir()])
        self.label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls_name in classes:
            cls_dir = self.dataset_path / cls_name
            for img_file in cls_dir.glob('*.jpg'):
                self.samples.append((str(img_file), self.label_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB').resize((224, 224))

        if self.transform:
            img = self.transform(img)

        return img, img, label  # same image for both streams


class CombinedDataset(Dataset):
    """Merges WLASL and ASL Alphabet datasets with proper label offsets."""

    def __init__(self, alphabet_dataset: ASLAlphabetDataset,
                 wlasl_dataset: WLASLDataset = None,
                 alphabet_offset: int = 0, wlasl_offset: int = 36):
        self.alphabet = alphabet_dataset
        self.wlasl = wlasl_dataset
        self.alphabet_offset = alphabet_offset
        self.wlasl_offset = wlasl_offset

    def __len__(self):
        total = len(self.alphabet)
        if self.wlasl:
            total += len(self.wlasl)
        return total

    def __getitem__(self, idx):
        if idx < len(self.alphabet):
            hand, full, label = self.alphabet[idx]
            return hand, full, label + self.alphabet_offset
        else:
            hand, full, label = self.wlasl[idx - len(self.alphabet)]
            return hand, full, label + self.wlasl_offset
