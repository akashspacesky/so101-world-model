"""
PyTorch datasets for world model and IDM training.

IDMDataset:          (frame_t, frame_{t+1}) → action_t   — supervised IDM training
WorldModelDataset:   video_clip + text_annotation         — world model fine-tuning
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from wm.data.preprocessor import (
    build_frame_pairs,
    decode_image,
    make_dino_transform,
    make_wm_transform,
    normalize_action,
    extract_video_clip,
)


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _find_image_columns(df: pd.DataFrame) -> list[str]:
    """Find all image columns in a LeRobot parquet dataframe."""
    img_cols = []
    for col in df.columns:
        if "image" in col.lower() or "camera" in col.lower():
            img_cols.append(col)
    return img_cols


def _find_action_column(df: pd.DataFrame) -> str | None:
    for candidate in ["action", "actions", "joint_action", "motor_action"]:
        if candidate in df.columns:
            return candidate
    return None


def _load_episode(parquet_path: Path) -> dict:
    """
    Load a single LeRobot episode from a parquet file.

    Returns dict with:
        frames: list[PIL.Image]  — from first image column found
        actions: np.ndarray [T, 6]
        task: str
    """
    df = _load_parquet(parquet_path)

    # Find image column (use first camera found)
    img_cols = _find_image_columns(df)
    if not img_cols:
        raise ValueError(f"No image column found in {parquet_path}")
    img_col = img_cols[0]

    # Decode images
    frames = []
    for raw in df[img_col]:
        if raw is None:
            continue
        # LeRobot stores images as dicts with 'bytes' key or raw bytes
        if isinstance(raw, dict) and "bytes" in raw:
            raw = raw["bytes"]
        frames.append(decode_image(raw))

    # Decode actions
    action_col = _find_action_column(df)
    if action_col and action_col in df.columns:
        actions_raw = df[action_col].tolist()
        # Handle list-of-lists or list-of-arrays
        actions = np.array([
            a if isinstance(a, (list, np.ndarray)) else [a]
            for a in actions_raw
        ], dtype=np.float32)
    else:
        actions = np.zeros((len(frames), 6), dtype=np.float32)

    # Task description — look in metadata or derive from path
    task = ""
    for col in ["task", "language_instruction", "task_description", "instruction"]:
        if col in df.columns and df[col].notna().any():
            task = str(df[col].dropna().iloc[0])
            break
    if not task:
        # Derive from dataset name embedded in path
        task = parquet_path.parent.parent.name.replace("__", "/").replace("_", " ")

    return {"frames": frames, "actions": actions, "task": task}


class IDMDataset(Dataset):
    """
    Inverse Dynamics Model training dataset.

    Each sample: (frame_t_tensor, frame_{t+1}_tensor, action_t_normalized)
    where action_t is what the robot executed to go from frame_t to frame_{t+1}.
    """

    def __init__(
        self,
        data_dir: Path,
        transform: Callable | None = None,
        max_episodes: int | None = None,
        cache_dir: Path | None = None,
    ):
        self.transform = transform or make_dino_transform()
        self.pairs: list[tuple[Image.Image, Image.Image, np.ndarray]] = []

        parquet_files = sorted(Path(data_dir).rglob("episode_*.parquet"))
        if max_episodes:
            parquet_files = parquet_files[:max_episodes]

        print(f"IDMDataset: loading {len(parquet_files)} episodes from {data_dir}")
        for pf in parquet_files:
            try:
                ep = _load_episode(pf)
                pairs = build_frame_pairs(ep["frames"], ep["actions"])
                self.pairs.extend(pairs)
            except Exception as e:
                print(f"  WARNING: skipping {pf}: {e}")

        print(f"IDMDataset: {len(self.pairs)} frame pairs loaded")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame_t, frame_next, action = self.pairs[idx]
        return {
            "frame_t": self.transform(frame_t),
            "frame_next": self.transform(frame_next),
            "action": normalize_action(action),
        }


class WorldModelDataset(Dataset):
    """
    World model fine-tuning dataset.

    Each sample: (video_clip_tensor [T, C, H, W], text_prompt)
    CogVideoX fine-tuning expects normalized frames in [-1, 1].
    """

    def __init__(
        self,
        data_dir: Path,
        clip_len: int = 13,
        stride: int = 2,
        transform: Callable | None = None,
        max_episodes: int | None = None,
    ):
        self.clip_len = clip_len
        self.stride = stride
        self.transform = transform or make_wm_transform()
        self.episodes: list[dict] = []

        parquet_files = sorted(Path(data_dir).rglob("episode_*.parquet"))
        if max_episodes:
            parquet_files = parquet_files[:max_episodes]

        print(f"WorldModelDataset: loading {len(parquet_files)} episodes")
        for pf in parquet_files:
            try:
                ep = _load_episode(pf)
                if len(ep["frames"]) >= clip_len:
                    self.episodes.append(ep)
            except Exception as e:
                print(f"  WARNING: skipping {pf}: {e}")

        print(f"WorldModelDataset: {len(self.episodes)} valid episodes loaded")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        ep = self.episodes[idx]
        clip_frames = extract_video_clip(ep["frames"], self.clip_len, self.stride)
        video = torch.stack([self.transform(f) for f in clip_frames])  # [T, C, H, W]
        return {
            "video": video,
            "text": ep["task"],
            "first_frame": self.transform(clip_frames[0]),
        }


class EpisodeBuffer:
    """
    In-memory buffer of recent deployment episodes for the data flywheel.
    Stores raw frames + actions from live robot sessions.
    """

    def __init__(self, max_episodes: int = 1000):
        self.episodes: list[dict] = []
        self.max_episodes = max_episodes

    def add(self, frames: list[Image.Image], actions: np.ndarray, task: str, success: bool):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
        self.episodes.append({
            "frames": frames,
            "actions": actions,
            "task": task,
            "success": success,
        })

    def successful_episodes(self) -> list[dict]:
        return [e for e in self.episodes if e["success"]]

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, ep in enumerate(self.episodes):
            ep_dir = output_dir / f"episode_{i:06d}"
            ep_dir.mkdir(exist_ok=True)
            # Save frames as JPEG
            for j, frame in enumerate(ep["frames"]):
                frame.save(ep_dir / f"frame_{j:04d}.jpg")
            np.save(ep_dir / "actions.npy", ep["actions"])
            (ep_dir / "meta.json").write_text(json.dumps({
                "task": ep["task"],
                "success": ep["success"],
                "num_frames": len(ep["frames"]),
            }))
