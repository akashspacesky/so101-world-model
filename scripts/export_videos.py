#!/usr/bin/env python3
"""
Export LeRobot parquet episodes → MP4 videos + metadata CSV
for CogVideoX fine-tuning with the official diffusers training script.

Usage:
  python scripts/export_videos.py \
      --data_dir /workspace/data/raw \
      --output_dir /workspace/data/videos \
      --max_episodes 5000 \
      --fps 10
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import imageio
    import imageio.v3 as iio
except ImportError:
    raise ImportError("Run: pip install imageio[ffmpeg]")

from wm.data.dataset import (
    _decode_image_cell,
    _dataset_root,
    _find_image_columns,
    _find_action_column,
    _infer_task,
    _is_quality_repo,
    _JUNK_RE,
)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def export_episodes(
    data_dir: Path,
    output_dir: Path,
    max_episodes: int = 5000,
    fps: int = 10,
    min_frames: int = 50,
    target_size: tuple[int, int] = (480, 480),
    quality_filter: bool = True,
) -> None:
    output_dir = Path(output_dir)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_parquets = sorted(Path(data_dir).rglob("episode_*.parquet"))

    by_dataset: dict[Path, list[Path]] = defaultdict(list)
    for pf in all_parquets:
        by_dataset[_dataset_root(pf)].append(pf)

    if quality_filter:
        kept = {d: pfs for d, pfs in by_dataset.items() if _is_quality_repo(d.name)}
        print(f"Quality filter: kept {len(kept)}/{len(by_dataset)} datasets")
    else:
        kept = dict(by_dataset)

    parquet_files = sorted(pf for pfs in kept.values() for pf in pfs)[:max_episodes]
    print(f"Exporting up to {max_episodes} episodes from {len(parquet_files)} parquet files...")

    metadata = []
    exported = 0
    skipped = 0

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            if len(df) < min_frames:
                skipped += 1
                continue

            img_cols = _find_image_columns(df)
            if not img_cols:
                skipped += 1
                continue
            img_col = img_cols[0]

            sample = df[img_col].iloc[0]
            if sample is None:
                skipped += 1
                continue
            if isinstance(sample, dict):
                if "bytes" not in sample and "path" not in sample:
                    skipped += 1
                    continue
                if "path" in sample and sample["path"]:
                    if Path(sample["path"]).suffix.lower() in _VIDEO_EXTS:
                        skipped += 1
                        continue

            root = _dataset_root(pf)
            task = _infer_task(df, pf)

            # Decode frames
            frames = []
            for cell in df[img_col]:
                img = _decode_image_cell(cell, base_dir=root)
                if img is not None:
                    img = img.resize(target_size, Image.BICUBIC)
                    frames.append(np.array(img))

            if len(frames) < min_frames:
                skipped += 1
                continue

            # Write MP4
            video_name = f"{pf.parent.parent.name}__{pf.stem}.mp4"
            video_path = videos_dir / video_name
            iio.imwrite(
                str(video_path),
                np.stack(frames),
                fps=fps,
                codec="libx264",
                quality=7,
            )

            metadata.append({"video": str(video_path.relative_to(output_dir)), "caption": task})
            exported += 1

            if exported % 100 == 0:
                print(f"  {exported} exported, {skipped} skipped...")

        except Exception as e:
            skipped += 1
            continue

    # Write metadata CSV
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "caption"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nDone: {exported} videos exported, {skipped} skipped")
    print(f"Videos: {videos_dir}")
    print(f"Metadata: {csv_path}")
    print(f"\nNext step:")
    print(f"  python scripts/train_world_model.py --data_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("/workspace/data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("/workspace/data/videos"))
    parser.add_argument("--max_episodes", type=int, default=5000)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min_frames", type=int, default=50)
    parser.add_argument("--no_quality_filter", action="store_true")
    args = parser.parse_args()

    export_episodes(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
        fps=args.fps,
        min_frames=args.min_frames,
        quality_filter=not args.no_quality_filter,
    )


if __name__ == "__main__":
    main()
