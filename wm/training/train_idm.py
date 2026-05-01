"""
IDM training loop.

Supervised training of the Inverse Dynamics Model on all SO-101 data.
Given (frame_t, frame_{t+1}) pairs, predict the action executed between them.

Designed to run on a single A100 (40GB). ~2-4 hours for full SO-101 dataset.
Also runs on M2 Air (MPS) for quick validation with small data slices.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from wm.data.dataset import IDMDataset
from wm.models.idm import InverseDynamicsModel


def train_idm(
    data_dir: Path,
    output_dir: Path,
    # Model
    action_dim: int = 6,
    feature_dim: int = 768,
    hidden_dim: int = 512,
    num_layers: int = 3,
    freeze_encoder: bool = True,
    # Training
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    val_split: float = 0.05,
    # Misc
    device: str = "cuda",
    num_workers: int = 4,
    log_every: int = 50,
    save_every: int = 5,
    max_episodes: int | None = None,
    wandb_project: str | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)

    # Logging
    logger = None
    if wandb_project:
        try:
            import wandb
            wandb.init(project=wandb_project, config=locals())
            logger = wandb
        except ImportError:
            print("wandb not available, logging to console only")

    print(f"Device: {device}")
    print(f"Loading dataset from {data_dir}")

    dataset = IDMDataset(data_dir, max_episodes=max_episodes)
    if len(dataset) == 0:
        raise RuntimeError(f"No frame pairs found in {data_dir}. Run download_so101_data.py first.")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device != "cpu",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device != "cpu",
    )

    model = InverseDynamicsModel(
        action_dim=action_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        freeze_encoder=freeze_encoder,
    ).to(device_obj)

    n_train = model.num_trainable_params()
    n_total = model.num_total_params()
    print(f"IDM params: {n_train:,} trainable / {n_total:,} total")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float("inf")
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            frame_t = batch["frame_t"].to(device_obj)
            frame_next = batch["frame_next"].to(device_obj)
            action_gt = batch["action"].to(device_obj)

            loss_dict = model.loss(frame_t, frame_next, action_gt)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if step % log_every == 0:
                log_data = {
                    "train/loss": loss.item(),
                    "train/mse": loss_dict["mse"].item(),
                    "train/huber": loss_dict["huber"].item(),
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if logger:
                    logger.log(log_data, step=step)
                print(
                    f"  step {step:6d} | loss {loss.item():.4f} "
                    f"| mse {loss_dict['mse'].item():.4f}"
                )

        scheduler.step()

        # Validation
        val_loss = _validate(model, val_loader, device_obj)
        elapsed = time.time() - t0
        avg_train = epoch_loss / len(train_loader)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train {avg_train:.4f} | val {val_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        if logger:
            logger.log({"val/loss": val_loss, "epoch": epoch}, step=step)

        # Save checkpoint
        if epoch % save_every == 0:
            _save_checkpoint(model, optimizer, epoch, val_loss, output_dir / f"epoch_{epoch:03d}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, val_loss, output_dir / "best.pt")
            print(f"  ✓ New best val loss: {val_loss:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


@torch.no_grad()
def _validate(model: InverseDynamicsModel, loader: DataLoader, device) -> float:
    model.eval()
    total = 0.0
    for batch in loader:
        frame_t = batch["frame_t"].to(device)
        frame_next = batch["frame_next"].to(device)
        action_gt = batch["action"].to(device)
        loss_dict = model.loss(frame_t, frame_next, action_gt)
        total += loss_dict["loss"].item()
    model.train()
    return total / max(1, len(loader))


def _save_checkpoint(model, optimizer, epoch, val_loss, path: Path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "model_config": {
            "action_dim": model.head[-1].out_features,
            "feature_dim": model.feature_dim,
        },
    }, path)
