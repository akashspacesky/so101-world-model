"""
PyTorch datasets for world model and IDM training.

IDMDataset:          (frame_t, frame_{t+1}) → action_t   — supervised IDM training
WorldModelDataset:   video_clip + text_annotation         — world model fine-tuning

Supports all LeRobot dataset formats:
  v1/v2: images as bytes embedded in parquet
  v3:    images as individual JPEG/PNG files on disk, path referenced in parquet

Design: fully lazy — images loaded from disk in __getitem__, not at init.
"""

from __future__ import annotations

import functools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from wm.data.preprocessor import (
    decode_image,
    make_dino_transform,
    make_wm_transform,
    normalize_action,
    extract_video_clip,
)

ACTION_DIM = 6  # SO-101 / SO-100 are both 6 DoF

# Repo names matching these patterns are junk (test uploads, personal experiments)
_JUNK_RE = re.compile(
    r'(?:^|[-_])test\d*$'
    r'|(?:^|[-_])trial\d*$'
    r'|(?:^|[-_])debug\d*$'
    r'|(?:^|[-_])tmp\d*$'
    r'|(?:^|[-_])temp\d*$'
    r'|(?:^|[-_])demo\d*$'
    r'|(?:^|[-_])exp\d*$',
    re.IGNORECASE,
)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _is_quality_repo(dataset_dir_name: str) -> bool:
    """
    Filter based on the local directory name (encodes repo_id as user__dataset).
    Returns False for obvious junk repos (test uploads, personal experiments).
    """
    parts = dataset_dir_name.split("__", 1)
    name = parts[-1] if len(parts) > 1 else dataset_dir_name
    return not _JUNK_RE.search(name)


def _find_image_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if any(kw in c.lower() for kw in ("image", "camera", "video", "rgb", "pixel", "frame", "obs"))
        and "action" not in c.lower()
    ]


def _find_action_column(df: pd.DataFrame) -> str | None:
    for candidate in ["action", "actions", "joint_action", "motor_action"]:
        if candidate in df.columns:
            return candidate
    return None


def _extract_action(row_value) -> np.ndarray | None:
    """Extract a 6-DoF action from a parquet cell, handling variable shapes."""
    if row_value is None:
        return None
    if isinstance(row_value, (list, np.ndarray)):
        arr = np.array(row_value, dtype=np.float32).flatten()
    elif isinstance(row_value, (int, float)):
        arr = np.array([row_value], dtype=np.float32)
    else:
        return None
    if len(arr) == 0:
        return None
    if len(arr) >= ACTION_DIM:
        return arr[:ACTION_DIM]
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[:len(arr)] = arr
    return out


def _decode_image_cell(cell, base_dir: Path | None = None) -> Image.Image | None:
    if cell is None:
        return None
    try:
        if isinstance(cell, dict):
            if "bytes" in cell and cell["bytes"] is not None:
                return decode_image(cell["bytes"])
            if "path" in cell and cell["path"] and base_dir is not None:
                img_path = base_dir / cell["path"]
                if img_path.exists():
                    return Image.open(img_path).convert("RGB")
            return None
        if isinstance(cell, (bytes, bytearray)):
            return decode_image(cell)
        if isinstance(cell, np.ndarray):
            return decode_image(cell)
        if isinstance(cell, str) and base_dir is not None:
            img_path = base_dir / cell
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
    except Exception:
        return None
    return None


def _dataset_root(parquet_path: Path) -> Path:
    return parquet_path.parent.parent


# ---------------------------------------------------------------------------
# Lazy index
# ---------------------------------------------------------------------------

class _EpisodeIndex:
    """
    Scans all parquet files and builds a flat index of valid frame pairs.

    Two-stage quality filter:
      1. Repo name filter (no I/O) — drops test/debug/junk repos instantly
      2. Episode filter — drops episodes with fewer than min_frames frames
    """

    def __init__(
        self,
        data_dir: Path,
        max_episodes: int | None = None,
        min_frames: int = 50,
        quality_filter: bool = True,
    ):
        self.entries: list[tuple[Path, Path, int, int, str, str | None]] = []

        all_parquets = sorted(Path(data_dir).rglob("episode_*.parquet"))

        # Group by dataset dir so we can apply repo-level filter once per repo
        by_dataset: dict[Path, list[Path]] = defaultdict(list)
        for pf in all_parquets:
            by_dataset[_dataset_root(pf)].append(pf)

        if quality_filter:
            kept = {d: pfs for d, pfs in by_dataset.items() if _is_quality_repo(d.name)}
            n_junk = len(by_dataset) - len(kept)
        else:
            kept = dict(by_dataset)
            n_junk = 0

        parquet_files = sorted(pf for pfs in kept.values() for pf in pfs)
        if max_episodes:
            parquet_files = parquet_files[:max_episodes]

        print(f"Indexing {len(parquet_files)} episodes from {len(kept)} datasets")
        if quality_filter:
            print(f"  ({n_junk} junk repos filtered by name)")

        skipped = 0
        for pf in parquet_files:
            try:
                n = self._index_file(pf, min_frames=min_frames)
                if n == 0:
                    skipped += 1
            except Exception:
                skipped += 1

        print(f"  {len(self.entries):,} frame pairs indexed")
        print(f"  {skipped} episodes skipped (short / bad format / video-only)")

    def _index_file(self, pf: Path, min_frames: int = 50) -> int:
        df = pd.read_parquet(pf)
        if len(df) < min_frames:
            return 0
        img_cols = _find_image_columns(df)
        if not img_cols:
            return 0
        img_col = img_cols[0]

        sample = df[img_col].iloc[0]
        if sample is None:
            return 0

        if isinstance(sample, dict):
            if "bytes" not in sample and "path" not in sample:
                return 0
            if "path" in sample and sample["path"] and Path(sample["path"]).suffix.lower() in _VIDEO_EXTS:
                return 0

        root = _dataset_root(pf)
        action_col = _find_action_column(df)
        count = 0
        for i in range(len(df) - 1):
            self.entries.append((pf, root, i, i + 1, img_col, action_col))
            count += 1
        return count


# ---------------------------------------------------------------------------
# Parquet LRU cache
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _cached_parquet(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(path_str)


# ---------------------------------------------------------------------------
# IDMDataset
# ---------------------------------------------------------------------------

class IDMDataset(Dataset):
    """
    Lazy IDM dataset. Indexes all parquet files at init, loads images on demand.
    Supports v1/v2 (bytes in parquet) and v3 (images on disk) LeRobot formats.
    """

    def __init__(
        self,
        data_dir: Path,
        transform: Callable | None = None,
        max_episodes: int | None = None,
        min_frames: int = 50,
        quality_filter: bool = True,
    ):
        self.transform = transform or make_dino_transform()
        self.index = _EpisodeIndex(
            data_dir,
            max_episodes=max_episodes,
            min_frames=min_frames,
            quality_filter=quality_filter,
        )

    def __len__(self) -> int:
        return len(self.index.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pf, root, row_i, row_j, img_col, action_col = self.index.entries[idx]
        df = _cached_parquet(str(pf))

        frame_t = _decode_image_cell(df[img_col].iloc[row_i], base_dir=root)
        frame_next = _decode_image_cell(df[img_col].iloc[row_j], base_dir=root)

        blank = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if frame_t is None:
            frame_t = blank
        if frame_next is None:
            frame_next = blank

        action = None
        if action_col:
            action = _extract_action(df[action_col].iloc[row_i])
        if action is None:
            action = np.zeros(ACTION_DIM, dtype=np.float32)

        return {
            "frame_t": self.transform(frame_t),
            "frame_next": self.transform(frame_next),
            "action": normalize_action(action),
        }


# ---------------------------------------------------------------------------
# WorldModelDataset
# ---------------------------------------------------------------------------

class WorldModelDataset(Dataset):
    """Lazy world model fine-tuning dataset. Supports v1/v2/v3 LeRobot formats."""

    def __init__(
        self,
        data_dir: Path,
        clip_len: int = 13,
        stride: int = 2,
        transform: Callable | None = None,
        max_episodes: int | None = None,
        min_frames: int = 50,
        quality_filter: bool = True,
    ):
        self.clip_len = clip_len
        self.stride = stride
        self.transform = transform or make_wm_transform()
        self.valid_files: list[tuple[Path, Path, str, str]] = []

        all_parquets = sorted(Path(data_dir).rglob("episode_*.parquet"))

        by_dataset: dict[Path, list[Path]] = defaultdict(list)
        for pf in all_parquets:
            by_dataset[_dataset_root(pf)].append(pf)

        if quality_filter:
            kept = {d: pfs for d, pfs in by_dataset.items() if _is_quality_repo(d.name)}
        else:
            kept = dict(by_dataset)

        parquet_files = sorted(pf for pfs in kept.values() for pf in pfs)
        if max_episodes:
            parquet_files = parquet_files[:max_episodes]

        required_frames = max(clip_len * stride, min_frames)
        print(f"WorldModelDataset: scanning {len(parquet_files)} episodes...")
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                img_cols = _find_image_columns(df)
                if not img_cols or len(df) < required_frames:
                    continue
                task = _infer_task(df, pf)
                self.valid_files.append((pf, _dataset_root(pf), img_cols[0], task))
            except Exception:
                continue
        print(f"WorldModelDataset: {len(self.valid_files)} valid episodes")

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> dict:
        pf, root, img_col, task = self.valid_files[idx]
        df = _cached_parquet(str(pf))
        frames = [
            img for cell in df[img_col]
            if (img := _decode_image_cell(cell, base_dir=root)) is not None
        ]
        clip_frames = extract_video_clip(frames, self.clip_len, self.stride)
        video = torch.stack([self.transform(f) for f in clip_frames])
        return {"video": video, "text": task, "first_frame": self.transform(clip_frames[0])}


def _infer_task(df: pd.DataFrame, path: Path) -> str:
    for col in ["task", "language_instruction", "task_description", "instruction"]:
        if col in df.columns and df[col].notna().any():
            return str(df[col].dropna().iloc[0])
    return path.parent.parent.name.replace("__", "/").replace("_", " ")


# ---------------------------------------------------------------------------
# EpisodeBuffer (flywheel)
# ---------------------------------------------------------------------------

class EpisodeBuffer:
    """In-memory buffer of recent deployment episodes for the data flywheel."""

    def __init__(self, max_episodes: int = 1000):
        self.episodes: list[dict] = []
        self.max_episodes = max_episodes

    def add(self, frames: list[Image.Image], actions: np.ndarray, task: str, success: bool):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
        self.episodes.append({"frames": frames, "actions": actions, "task": task, "success": success})

    def successful_episodes(self) -> list[dict]:
        return [e for e in self.episodes if e["success"]]

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, ep in enumerate(self.episodes):
            ep_dir = output_dir / f"episode_{i:06d}"
            ep_dir.mkdir(exist_ok=True)
            for j, frame in enumerate(ep["frames"]):
                frame.save(ep_dir / f"frame_{j:04d}.jpg")
            np.save(ep_dir / "actions.npy", ep["actions"])
            (ep_dir / "meta.json").write_text(json.dumps({
                "task": ep["task"], "success": ep["success"],
                "num_frames": len(ep["frames"]),
            }))
