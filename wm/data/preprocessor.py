"""
Frame and action preprocessing for SO-101 world model training.

LeRobot v2 datasets store images as raw bytes in parquet columns.
This module handles decoding, resizing, and normalization to
formats expected by DINOv2 and CogVideoX.
"""

from __future__ import annotations

import io
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# CogVideoX expects 480×720 or 480×480 frames
WORLD_MODEL_SIZE = (480, 480)
# DINOv2 ViT-B/14 expects 224×224 or 518×518 (using 224 for speed)
DINO_SIZE = (224, 224)

# ImageNet stats for DINOv2
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)

# SO-101 action normalization: joint positions in degrees, roughly [-180, 180]
# We normalize to [-1, 1] for training
ACTION_MIN = -180.0
ACTION_MAX = 180.0


def decode_image(raw: bytes | np.ndarray | Image.Image) -> Image.Image:
    """Decode image from bytes, numpy array, or PIL Image."""
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, np.ndarray):
        return Image.fromarray(raw).convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise TypeError(f"Cannot decode image from {type(raw)}")


def make_dino_transform() -> T.Compose:
    return T.Compose([
        T.Resize(DINO_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(DINO_SIZE),
        T.ToTensor(),
        T.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])


def make_wm_transform() -> T.Compose:
    """Transform for world model (CogVideoX): resize to 480×480, normalize to [-1, 1]."""
    return T.Compose([
        T.Resize(WORLD_MODEL_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(WORLD_MODEL_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def normalize_action(action: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Normalize SO-101 joint positions to [-1, 1]."""
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(action).float()
    return (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) * 2.0 - 1.0


def denormalize_action(action: torch.Tensor) -> torch.Tensor:
    """Convert normalized actions back to degrees."""
    return (action + 1.0) / 2.0 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN


def frames_to_tensor(
    frames: Sequence[Image.Image],
    transform: T.Compose,
) -> torch.Tensor:
    """Stack a list of PIL images into a (T, C, H, W) tensor."""
    tensors = [transform(f) for f in frames]
    return torch.stack(tensors)


def extract_video_clip(
    frames: list[Image.Image],
    clip_len: int = 13,
    stride: int = 1,
    start: int | None = None,
) -> list[Image.Image]:
    """Extract a fixed-length clip from a frame list, with optional striding."""
    total = len(frames)
    if total < clip_len:
        # Pad by repeating last frame
        frames = frames + [frames[-1]] * (clip_len - total)
        return frames[:clip_len]

    if start is None:
        max_start = total - (clip_len - 1) * stride - 1
        start = np.random.randint(0, max(1, max_start))

    indices = [min(start + i * stride, total - 1) for i in range(clip_len)]
    return [frames[i] for i in indices]


def build_frame_pairs(
    frames: list[Image.Image],
    actions: np.ndarray,
) -> list[tuple[Image.Image, Image.Image, np.ndarray]]:
    """
    Build (frame_t, frame_{t+1}, action_t) triplets for IDM training.
    action_t is what the robot did to go from frame_t to frame_{t+1}.
    """
    pairs = []
    for i in range(len(frames) - 1):
        if i < len(actions):
            pairs.append((frames[i], frames[i + 1], actions[i]))
    return pairs
