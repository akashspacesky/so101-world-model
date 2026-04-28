#!/usr/bin/env python3
"""Fine-tune CogVideoX-2b on SO-101 data via LoRA. Runs on cloud (A100/H100)."""

import argparse
from pathlib import Path

from wm.training.train_wm import train_world_model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune world model on SO-101 video data")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/wm"))
    parser.add_argument("--model_id", default="THUDM/CogVideoX-2b")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--clip_len", type=int, default=13)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--wandb", default=None, metavar="PROJECT")
    parser.add_argument("--resume_from", type=Path, default=None)
    args = parser.parse_args()

    train_world_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        lr=args.lr,
        clip_len=args.clip_len,
        max_episodes=args.max_episodes,
        wandb_project=args.wandb,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
