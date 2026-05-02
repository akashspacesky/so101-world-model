#!/usr/bin/env python3
"""Train the Inverse Dynamics Model on downloaded SO-101 data."""

import argparse
from pathlib import Path

from wm.training.train_idm import train_idm


def main():
    parser = argparse.ArgumentParser(description="Train IDM on SO-101 frame pairs")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"), help="Downloaded SO-101 data root")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/idm"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit for quick runs")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--min_frames", type=int, default=50, help="Min frames per episode")
    parser.add_argument("--no_quality_filter", action="store_true", help="Disable junk repo filter")
    parser.add_argument("--wandb", default=None, metavar="PROJECT", help="WandB project name")
    args = parser.parse_args()

    train_idm(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        freeze_encoder=args.freeze_encoder,
        max_episodes=args.max_episodes,
        num_workers=args.num_workers,
        min_frames=args.min_frames,
        quality_filter=not args.no_quality_filter,
        wandb_project=args.wandb,
    )


if __name__ == "__main__":
    main()
