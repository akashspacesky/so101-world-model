"""
DINOv2-B frame encoder.

Encodes single frames into 768-dimensional feature vectors.
Used by the IDM to represent (frame_t, frame_{t+1}) pairs.
Kept frozen during IDM training for stability; can be fine-tuned optionally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


DINO_MODEL_ID = "facebook/dinov2-base"  # 86M params, 768-dim output


class FrameEncoder(nn.Module):
    """
    DINOv2-B visual encoder that maps a single frame to a feature vector.

    Input:  (B, 3, 224, 224) — normalized with ImageNet stats
    Output: (B, 768)         — [CLS] token from the last layer
    """

    def __init__(
        self,
        model_id: str = DINO_MODEL_ID,
        freeze: bool = True,
        proj_dim: int | None = None,
    ):
        super().__init__()
        self.dino = AutoModel.from_pretrained(model_id)

        if freeze:
            for p in self.dino.parameters():
                p.requires_grad_(False)

        self.hidden_dim = self.dino.config.hidden_size  # 768 for ViT-B

        # Optional projection to a smaller dimension
        self.proj = None
        if proj_dim is not None and proj_dim != self.hidden_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_dim, proj_dim),
                nn.LayerNorm(proj_dim),
            )
            self.output_dim = proj_dim
        else:
            self.output_dim = self.hidden_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) — ImageNet-normalized frames
        Returns:
            features: (B, output_dim)
        """
        outputs = self.dino(pixel_values=pixel_values)
        # [CLS] token is the first token of last_hidden_state
        cls_features = outputs.last_hidden_state[:, 0]  # (B, 768)

        if self.proj is not None:
            cls_features = self.proj(cls_features)

        return cls_features

    def encode_pair(
        self,
        frame_t: torch.Tensor,
        frame_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a pair of frames. Returns (z_t, z_next), each (B, output_dim)."""
        # Single forward for efficiency when frames are from same video
        combined = torch.cat([frame_t, frame_next], dim=0)  # (2B, C, H, W)
        features = self.forward(combined)
        B = frame_t.shape[0]
        return features[:B], features[B:]
