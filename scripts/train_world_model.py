#!/usr/bin/env python3
"""
Fine-tune CogVideoX-2b on SO-101 videos using LoRA.

Requires exported videos from scripts/export_videos.py first.

Usage:
  python scripts/train_world_model.py \
      --data_dir /workspace/data/videos \
      --output_dir /workspace/checkpoints/wm \
      --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import urllib.request
from pathlib import Path


DIFFUSERS_TRAIN_SCRIPT = "https://raw.githubusercontent.com/huggingface/diffusers/v0.32.2/examples/cogvideo/train_cogvideox_lora.py"


def download_train_script(dest: Path) -> Path:
    script_path = dest / "train_cogvideox_lora.py"
    if script_path.exists():
        return script_path
    print("Downloading official diffusers CogVideoX training script...")
    urllib.request.urlretrieve(DIFFUSERS_TRAIN_SCRIPT, script_path)
    print(f"Saved to {script_path}")
    return script_path


def prepare_data_files(data_dir: Path) -> Path:
    """
    The v0.32.2 training script expects two line-separated text files
    in instance_data_root:
      - video.txt  : one video filename per line
      - text.txt   : one caption per line

    Reads metadata.csv produced by export_videos.py and writes these files.
    """
    videos_dir = data_dir / "videos"
    csv_path = data_dir / "metadata.csv"

    if csv_path.exists():
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
        video_names = [Path(r["video"]).name for r in rows]
        captions = [r["caption"] for r in rows]
    else:
        # Fallback: glob all mp4s, use a generic caption
        video_names = sorted(p.name for p in videos_dir.glob("*.mp4"))
        captions = ["SO-101 robot arm manipulation"] * len(video_names)

    (videos_dir / "video.txt").write_text("\n".join(video_names))
    (videos_dir / "text.txt").write_text("\n".join(captions))
    print(f"Prepared {len(video_names)} video/caption pairs in {videos_dir}")
    return videos_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("/workspace/data/videos"))
    parser.add_argument("--output_dir", type=Path, default=Path("/workspace/checkpoints/wm"))
    parser.add_argument("--model_id", default="THUDM/CogVideoX-2b")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    videos_dir = args.data_dir / "videos"
    if not videos_dir.exists():
        print(f"ERROR: videos dir not found at {videos_dir}")
        print("Run first: python scripts/export_videos.py --data_dir /workspace/data/raw --output_dir /workspace/data/videos")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_data_files(args.data_dir)
    script_path = download_train_script(args.output_dir)

    cmd = [
        sys.executable, str(script_path),
        "--pretrained_model_name_or_path", args.model_id,
        "--instance_data_root", str(videos_dir),
        "--video_column", "video.txt",
        "--caption_column", "text.txt",
        "--output_dir", str(args.output_dir),
        "--mixed_precision", args.mixed_precision,
        "--num_train_epochs", str(args.epochs),
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation),
        "--learning_rate", str(args.lr),
        "--rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_rank),
        "--checkpointing_steps", "500",
        "--gradient_checkpointing",
        "--enable_slicing",
        "--enable_tiling",
        "--seed", "42",
    ]

    print("Launching training...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
