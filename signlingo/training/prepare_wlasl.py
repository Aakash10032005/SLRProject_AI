"""
WLASL Dataset Preparation Helper
=================================
Run this BEFORE downloading to:
  1. Verify disk space
  2. Generate the lite_manifest.json
  3. Validate expected directory layout
  4. Print a download checklist

Does NOT download anything. Call with --download only when ready.

Usage:
    python training/prepare_wlasl.py --root data/wlasl
    python training/prepare_wlasl.py --root data/wlasl --mode lite
    python training/prepare_wlasl.py --root data/wlasl --validate   # after download
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FULL_DATASET_GB = 8.0       # raw MP4 upper bound
LITE_DATASET_GB = 3.0       # lite subset upper bound
BUFFER_GB = 10.0            # working buffer (augmentation, checkpoints, logs)
CHECKPOINT_GB = 0.5         # hstfe_v1.pth

LITE_CLASSES = [
    # Alphabet
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    # High-frequency words
    'hello', 'thank-you', 'please', 'sorry', 'yes', 'no',
    'help', 'water', 'food', 'eat', 'drink', 'more',
    'stop', 'go', 'come', 'want', 'need', 'like',
    'good', 'bad', 'name', 'where', 'what', 'understand',
]

WLASL_DOWNLOAD_URLS = {
    'json': 'https://raw.githubusercontent.com/dxli94/WLASL/master/data/WLASL_v0.3.json',
    'videos': 'https://github.com/dxli94/WLASL  (follow repo instructions for video download)',
    'kaggle_asl': 'kaggle datasets download -d grassknoted/asl-alphabet',
}


# ── Disk check ────────────────────────────────────────────────────────────────

def check_disk_space(root: Path, required_gb: float) -> bool:
    drive = root.anchor or '.'
    usage = shutil.disk_usage(drive)
    free_gb = usage.free / (1024 ** 3)
    logger.info(f"Free disk space on {drive}: {free_gb:.1f} GB  (required: {required_gb:.1f} GB)")
    if free_gb < required_gb:
        logger.error(
            f"Insufficient disk space. Need {required_gb:.1f} GB, have {free_gb:.1f} GB."
        )
        return False
    return True


# ── Manifest generation ───────────────────────────────────────────────────────

def write_lite_manifest(root: Path) -> Path:
    manifest = {'classes': LITE_CLASSES, 'count': len(LITE_CLASSES)}
    out = root / 'lite_manifest.json'
    root.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Lite manifest written: {out}  ({len(LITE_CLASSES)} classes)")
    return out


# ── Post-download validation ──────────────────────────────────────────────────

def validate_layout(root: Path, manifest_path: Path | None = None) -> bool:
    video_dir = root / 'videos'
    if not video_dir.exists():
        logger.error(f"Missing: {video_dir}")
        logger.error("Expected layout: <root>/videos/<class_name>/<id>.mp4")
        return False

    class_dirs = [d for d in video_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.error(f"No class subdirectories found in {video_dir}")
        return False

    total_clips = sum(len(list(d.glob('*.mp4'))) for d in class_dirs)
    logger.info(f"Found {len(class_dirs)} classes, {total_clips} MP4 clips in {video_dir}")

    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        expected = set(c.lower() for c in manifest.get('classes', []))
        found = set(d.name.lower() for d in class_dirs)
        missing = expected - found
        if missing:
            logger.warning(f"{len(missing)} manifest classes not found on disk: {sorted(missing)[:10]}...")
        else:
            logger.info("All manifest classes present on disk.")

    return True


# ── Checklist printer ─────────────────────────────────────────────────────────

def print_checklist(mode: str, root: Path):
    required_gb = LITE_DATASET_GB if mode == 'lite' else FULL_DATASET_GB
    required_gb += BUFFER_GB + CHECKPOINT_GB

    print("\n" + "=" * 65)
    print(f"  WLASL Download Checklist  [{mode.upper()} mode]")
    print("=" * 65)
    print(f"\n  Target directory : {root.resolve()}")
    print(f"  Required space   : ~{required_gb:.0f} GB free")
    print(f"  Classes          : {len(LITE_CLASSES) if mode == 'lite' else '2000+'}")
    print(f"  Clips            : {'~1,000' if mode == 'lite' else '~21,083'}")
    print()
    print("  Steps:")
    print("  1. Ensure free disk space (see above)")
    print("  2. Clone WLASL repo:")
    print("       git clone https://github.com/dxli94/WLASL.git")
    print("  3. Follow WLASL repo instructions to request video access")
    print("       (videos are hosted separately — requires form submission)")
    print("  4. Place videos at:")
    print(f"       {root}/videos/<class_name>/<video_id>.mp4")
    print("  5. (Optional) Download ASL Alphabet from Kaggle:")
    print(f"       {WLASL_DOWNLOAD_URLS['kaggle_asl']}")
    print("  6. Run validation:")
    print(f"       python training/prepare_wlasl.py --root {root} --validate")
    print("  7. Run training:")
    if mode == 'lite':
        print(f"       python -m training.train_hstfe \\")
        print(f"         --dataset_path {root} \\")
        print(f"         --mode wlasl_lite \\")
        print(f"         --epochs 40 --batch_size 16")
    else:
        print(f"       python -m training.train_hstfe \\")
        print(f"         --dataset_path {root} \\")
        print(f"         --mode wlasl \\")
        print(f"         --epochs 40 --batch_size 8")
    print()
    print("  Storage rules (MANDATORY):")
    print("  - Do NOT extract frames to disk (lazy loading is active)")
    print("  - Do NOT store augmented images (in-memory only)")
    print("  - Checkpoints go to: models/weights/  (outside dataset dir)")
    print("  - Use JPEG quality=85 if any caching is needed")
    print("=" * 65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='WLASL dataset preparation (no download)')
    parser.add_argument('--root', type=str, default='data/wlasl',
                        help='Target dataset root directory')
    parser.add_argument('--mode', choices=['lite', 'full'], default='lite',
                        help='lite = 50 classes, full = all 2000+ classes')
    parser.add_argument('--validate', action='store_true',
                        help='Validate layout after download (run post-download)')
    args = parser.parse_args()

    root = Path(args.root)

    if args.validate:
        manifest = root / 'lite_manifest.json'
        ok = validate_layout(root, manifest if manifest.exists() else None)
        sys.exit(0 if ok else 1)

    # Pre-download planning
    required_gb = (LITE_DATASET_GB if args.mode == 'lite' else FULL_DATASET_GB) + BUFFER_GB
    space_ok = check_disk_space(root.parent if not root.exists() else root, required_gb)

    manifest_path = write_lite_manifest(root)

    print_checklist(args.mode, root)

    if not space_ok:
        logger.warning("Disk space check FAILED. Free up space before downloading.")
        sys.exit(1)

    logger.info("Planning complete. DO NOT download yet — confirm with user first.")
    logger.info(f"Lite manifest ready at: {manifest_path}")


if __name__ == '__main__':
    main()
