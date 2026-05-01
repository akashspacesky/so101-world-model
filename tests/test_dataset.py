"""
Tests for the lazy IDMDataset and parquet parsing utilities.
No network access, no GPU — uses synthetic in-memory parquet files.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from wm.data.dataset import (
    _decode_image_cell,
    _extract_action,
    _find_action_column,
    _find_image_columns,
    ACTION_DIM,
    IDMDataset,
    EpisodeBuffer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _write_fake_parquet(path: Path, num_frames: int = 10, action_dim: int = 6):
    """Write a synthetic LeRobot-style parquet file with image bytes + actions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(num_frames):
        rows.append({
            "observation.images.top": {"bytes": _make_image_bytes()},
            "action": list(np.random.randn(action_dim).astype(np.float32)),
            "task": "pick up the block",
        })
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_fake_dataset(data_dir: Path, num_episodes: int = 3, num_frames: int = 10, action_dim: int = 6):
    """Write a fake dataset with multiple episode parquet files."""
    for i in range(num_episodes):
        _write_fake_parquet(
            data_dir / f"fake_ds__task1" / "data" / f"episode_{i:06d}.parquet",
            num_frames=num_frames,
            action_dim=action_dim,
        )


# ---------------------------------------------------------------------------
# Unit tests for parsing helpers
# ---------------------------------------------------------------------------

def test_find_image_columns():
    df = pd.DataFrame({"observation.images.top": [], "action": [], "task": []})
    cols = _find_image_columns(df)
    assert "observation.images.top" in cols


def test_find_action_column():
    df = pd.DataFrame({"action": [1, 2], "task": ["a", "b"]})
    assert _find_action_column(df) == "action"

    df2 = pd.DataFrame({"joint_action": [1, 2]})
    assert _find_action_column(df2) == "joint_action"

    df3 = pd.DataFrame({"foo": [1, 2]})
    assert _find_action_column(df3) is None


def test_decode_image_cell_bytes():
    raw = _make_image_bytes()
    img = _decode_image_cell(raw)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_decode_image_cell_dict_with_bytes():
    raw = _make_image_bytes()
    img = _decode_image_cell({"bytes": raw, "path": "frame_000.jpg"})
    assert isinstance(img, Image.Image)


def test_decode_image_cell_dict_path_only():
    """v2 video-format: path-only dict should return None (skipped)."""
    result = _decode_image_cell({"path": "videos/episode_000.mp4"})
    assert result is None


def test_decode_image_cell_none():
    assert _decode_image_cell(None) is None


def test_extract_action_6dof():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    out = _extract_action(arr.tolist())
    assert out is not None
    assert out.shape == (ACTION_DIM,)
    np.testing.assert_array_almost_equal(out, arr)


def test_extract_action_truncates_7dof():
    arr = list(range(7))
    out = _extract_action(arr)
    assert out is not None
    assert out.shape == (ACTION_DIM,)
    assert list(out) == list(range(6))


def test_extract_action_pads_short():
    arr = [1.0, 2.0, 3.0]
    out = _extract_action(arr)
    assert out is not None
    assert out.shape == (ACTION_DIM,)
    assert out[0] == 1.0
    assert out[3] == 0.0  # padded


def test_extract_action_none():
    assert _extract_action(None) is None


# ---------------------------------------------------------------------------
# IDMDataset integration tests
# ---------------------------------------------------------------------------

def test_idm_dataset_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        _write_fake_dataset(data_dir, num_episodes=3, num_frames=10)

        ds = IDMDataset(data_dir)
        # 3 episodes * (10 frames - 1) = 27 pairs
        assert len(ds) == 27

        sample = ds[0]
        assert "frame_t" in sample
        assert "frame_next" in sample
        assert "action" in sample
        assert sample["frame_t"].shape == (3, 224, 224)
        assert sample["frame_next"].shape == (3, 224, 224)
        assert sample["action"].shape == (ACTION_DIM,)


def test_idm_dataset_action_normalized():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        _write_fake_dataset(data_dir, num_episodes=2, num_frames=8)
        ds = IDMDataset(data_dir)
        for i in range(min(10, len(ds))):
            action = ds[i]["action"]
            # Normalized actions from real data won't necessarily be in [-1,1]
            # but should be finite
            assert action.isfinite().all()


def test_idm_dataset_v3_disk_images():
    """v3 format: path-only cells resolved relative to dataset root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        ds_root = data_dir / "v3_ds"
        ep_path = ds_root / "data" / "episode_000000.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)

        # Write actual JPEG files to disk
        rows = []
        for i in range(5):
            img_rel = f"episode_000000/cam_0/{i:06d}.jpg"
            img_abs = ds_root / img_rel
            img_abs.parent.mkdir(parents=True, exist_ok=True)
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_abs)
            rows.append({
                "observation.images.top": {"path": img_rel},
                "action": [float(i)] * 6,
            })
        pd.DataFrame(rows).to_parquet(ep_path, index=False)

        ds = IDMDataset(data_dir)
        assert len(ds) == 4  # 5 frames → 4 pairs
        sample = ds[0]
        assert sample["frame_t"].shape == (3, 224, 224)


def test_idm_dataset_skips_unknown_format():
    """Cells with neither bytes nor path should be skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        ep_path = data_dir / "bad_ds" / "data" / "episode_000000.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"observation.images.top": {"unknown_key": "???"}, "action": [0.0]*6}
                for _ in range(5)]
        pd.DataFrame(rows).to_parquet(ep_path, index=False)

        ds = IDMDataset(data_dir)
        assert len(ds) == 0


def test_idm_dataset_handles_missing_action_column():
    """Episodes without an action column should use zero actions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        ep_path = data_dir / "no_action" / "data" / "episode_000000.parquet"
        ep_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"observation.images.top": {"bytes": _make_image_bytes()}} for _ in range(5)]
        pd.DataFrame(rows).to_parquet(ep_path, index=False)

        ds = IDMDataset(data_dir)
        assert len(ds) == 4
        assert ds[0]["action"].shape == (ACTION_DIM,)


def test_idm_dataset_max_episodes():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        _write_fake_dataset(data_dir, num_episodes=5, num_frames=6)

        ds_full = IDMDataset(data_dir)
        ds_limited = IDMDataset(data_dir, max_episodes=2)
        assert len(ds_limited) < len(ds_full)
        assert len(ds_limited) == 2 * 5  # 2 episodes * 5 pairs each


def test_episode_buffer():
    buf = EpisodeBuffer(max_episodes=3)
    for i in range(4):
        frames = [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))] * 5
        actions = np.zeros((5, 6), dtype=np.float32)
        buf.add(frames, actions, task=f"task_{i}", success=(i % 2 == 0))

    assert len(buf.episodes) == 3  # max_episodes=3, oldest evicted
    assert len(buf.successful_episodes()) == 1  # only episode 2 (i=2) survives eviction
