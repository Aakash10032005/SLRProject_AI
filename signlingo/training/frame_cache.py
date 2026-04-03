"""
Optional JPEG Frame Cache
=========================
Pre-extracts frames from MP4s to JPEG (quality=85) for faster repeated reads.
Use ONLY when you have SSD space and want to speed up epoch 2+ I/O.

Rules:
- JPEG quality=85 (~60% smaller than PNG, negligible accuracy impact)
- Cache lives OUTSIDE the dataset directory (separate path)
- Augmentation is NEVER cached — always in-memory
- Run once, then pass cache_dir to WLASLLazyDataset

Usage:
    python training/frame_cache.py \\
        --root data/wlasl \\
        --cache_dir data/wlasl_cache \\
        --n_frames 8 \\
        --manifest data/wlasl/lite_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

JPEG_QUALITY = 85


def cache_dataset(
    root: Path,
    cache_dir: Path,
    n_frames: int = 8,
    manifest: Path | None = None,
):
    """Extract n_frames per clip to JPEG. Skips already-cached clips."""
    video_dir = root / 'videos'
    if not video_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {video_dir}")

    allowed = None
    if manifest and manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        allowed = {c.lower() for c in data.get('classes', [])}

    class_dirs = sorted(
        d for d in video_dir.iterdir()
        if d.is_dir() and (allowed is None or d.name.lower() in allowed)
    )

    total_clips = sum(len(list(d.glob('*.mp4'))) for d in class_dirs)
    logger.info(f"Caching {total_clips} clips from {len(class_dirs)} classes → {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached, skipped = 0, 0

    for cls_dir in tqdm(class_dirs, desc='Classes'):
        cls_cache = cache_dir / cls_dir.name
        cls_cache.mkdir(exist_ok=True)

        for mp4 in cls_dir.glob('*.mp4'):
            clip_cache = cls_cache / mp4.stem
            # Skip if already cached (check for at least 1 frame file)
            if clip_cache.exists() and any(clip_cache.glob('*.jpg')):
                skipped += 1
                continue

            clip_cache.mkdir(exist_ok=True)
            cap = cv2.VideoCapture(str(mp4))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                skipped += 1
                continue

            indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
            for frame_idx, pos in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
                ret, frame = cap.read()
                if not ret:
                    continue
                resized = cv2.resize(frame, (224, 224))
                out_path = clip_cache / f"{frame_idx:03d}.jpg"
                cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cap.release()
            cached += 1

    logger.info(f"Done. Cached: {cached} clips, Skipped (already cached): {skipped}")


def main():
    parser = argparse.ArgumentParser(description='Pre-cache WLASL frames as JPEG')
    parser.add_argument('--root', required=True, help='WLASL dataset root')
    parser.add_argument('--cache_dir', required=True, help='Output cache directory (on SSD)')
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument('--manifest', type=str, default=None)
    args = parser.parse_args()

    cache_dataset(
        root=Path(args.root),
        cache_dir=Path(args.cache_dir),
        n_frames=args.n_frames,
        manifest=Path(args.manifest) if args.manifest else None,
    )


if __name__ == '__main__':
    main()
