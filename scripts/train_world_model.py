#!/usr/bin/env python3
"""
Fine-tune CogVideoX-2b on SO-101 videos using LoRA.

Requires exported videos from scripts/export_videos.py first.

Usage:
  python scripts/train_world_model.py \
      --data_dir /workspace/data/videos \
      --output_dir /workspace/checkpoints/wm \
      --epochs 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DIFFUSERS_TRAIN_SCRIPT = "https://raw.githubusercontent.com/huggingface/diffusers/v0.32.2/examples/cogvideo/train_cogvideox_lora.py"


def download_train_script(dest: Path) -> Path:
    script_path = dest / "train_cogvideox_lora.py"
    if script_path.exists():
        return script_path
    print("Downloading official diffusers CogVideoX training script...")
    import urllib.request
    urllib.request.urlretrieve(DIFFUSERS_TRAIN_SCRIPT, script_path)
    print(f"Saved to {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("/workspace/data/videos"),
                        help="Directory with videos/ subdir and metadata.csv from export_videos.py")
    parser.add_argument("--output_dir", type=Path, default=Path("/workspace/checkpoints/wm"))
    parser.add_argument("--model_id", default="THUDM/CogVideoX-2b")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
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

    # Download the official training script if needed
    script_path = download_train_script(args.output_dir)

    # Build the command
    cmd = [
        sys.executable, str(script_path),
        "--pretrained_model_name_or_path", args.model_id,
        "--instance_data_root", str(args.data_dir / "videos"),
        "--instance_prompt", "SO-101 robot arm manipulation task",
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
