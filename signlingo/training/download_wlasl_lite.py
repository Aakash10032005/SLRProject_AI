"""
WLASL Lite Downloader
=====================
Downloads only the 50 lite-manifest classes from WLASL_v0.3.json.
Uses direct HTTP URLs only (no yt-dlp / YouTube dependency).
YouTube-sourced clips are skipped with a warning.

Output layout:
    <saveto>/videos/<gloss>/<video_id>.mp4

Usage (from signlingo/ directory):
    python training/download_wlasl_lite.py \\
        --json   data/WLASL_repo/start_kit/WLASL_v0.3.json \\
        --manifest data/wlasl/lite_manifest.json \\
        --saveto data/wlasl \\
        --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

REFERERS = {
    'aslpro': 'http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi',
    'aslbrick': 'http://aslbricks.org/',
}


def _is_youtube(url: str) -> bool:
    return 'youtube.com' in url or 'youtu.be' in url


def _referer_for(url: str) -> str:
    for key, ref in REFERERS.items():
        if key in url:
            return ref
    return ''


def download_one(gloss: str, video_id: str, url: str, saveto: Path) -> tuple[bool, str]:
    """Download a single video. Returns (success, reason)."""
    if _is_youtube(url):
        return False, 'youtube-skip'

    ext = '.swf' if 'aslpro' in url else '.mp4'
    out_dir = saveto / 'videos' / gloss
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}{ext}"

    if out_path.exists() and out_path.stat().st_size > 1024:
        return True, 'already-exists'

    headers = dict(HEADERS)
    ref = _referer_for(url)
    if ref:
        headers['Referer'] = ref

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) < 512:
            return False, f'too-small ({len(data)} bytes)'
        with open(out_path, 'wb') as f:
            f.write(data)
        time.sleep(random.uniform(0.3, 0.8))   # be polite
        return True, 'downloaded'
    except Exception as e:
        return False, str(e)


def run(json_path: Path, manifest_path: Path, saveto: Path, workers: int):
    # Load manifest
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    allowed = {c.lower() for c in manifest.get('classes', [])}
    logger.info("Lite manifest: %d classes", len(allowed))

    # Load WLASL JSON
    with open(json_path, encoding='utf-8') as f:
        wlasl = json.load(f)

    # Build task list
    tasks: list[tuple[str, str, str]] = []   # (gloss, video_id, url)
    for entry in wlasl:
        gloss = entry['gloss'].lower()
        if gloss not in allowed:
            continue
        for inst in entry['instances']:
            tasks.append((entry['gloss'], inst['video_id'], inst['url']))

    logger.info("Total clips to attempt: %d (YouTube will be skipped)", len(tasks))

    ok, skipped_yt, failed = 0, 0, 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, gloss, vid, url, saveto): (gloss, vid)
            for gloss, vid, url in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            gloss, vid = futures[fut]
            success, reason = fut.result()
            if reason == 'youtube-skip':
                skipped_yt += 1
            elif success:
                ok += 1
            else:
                failed += 1
                logger.warning("FAIL  %s/%s — %s", gloss, vid, reason)

            if i % 50 == 0:
                logger.info("Progress: %d/%d  ok=%d  yt_skip=%d  fail=%d",
                            i, len(tasks), ok, skipped_yt, failed)

    logger.info("Done. Downloaded=%d  YouTube-skipped=%d  Failed=%d", ok, skipped_yt, failed)

    # Summary per class
    video_dir = saveto / 'videos'
    logger.info("\nPer-class clip counts:")
    for cls in sorted(allowed):
        cls_dir = video_dir / cls
        if cls_dir.exists():
            n = len(list(cls_dir.glob('*.*')))
            logger.info("  %-20s %d clips", cls, n)
        else:
            logger.warning("  %-20s 0 clips (no directory)", cls)


def main():
    parser = argparse.ArgumentParser(description='Download WLASL lite subset (no YouTube)')
    parser.add_argument('--json',     default='data/WLASL_repo/start_kit/WLASL_v0.3.json')
    parser.add_argument('--manifest', default='data/wlasl/lite_manifest.json')
    parser.add_argument('--saveto',   default='data/wlasl')
    parser.add_argument('--workers',  type=int, default=4)
    args = parser.parse_args()

    run(
        json_path=Path(args.json),
        manifest_path=Path(args.manifest),
        saveto=Path(args.saveto),
        workers=args.workers,
    )


if __name__ == '__main__':
    main()
