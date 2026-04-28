"""
Inverse Dynamics Model (IDM) for SO-101.

Given two consecutive frames (frame_t, frame_{t+1}), predicts the action
the robot executed to transition between them.

This is the crown jewel of the system — trained on all available SO-101
data, it can extract executable actions from any plausible future video.

Architecture:
    DINOv2-B encoder → cross-attention between z_t and z_{t+1}
    → MLP head → 6-DoF SO-101 joint action
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.models.video_encoder import FrameEncoder


class CrossAttentionFusion(nn.Module):
    """
    Fuses features from two frames via bidirectional cross-attention.

    z_t attends to z_{t+1} and vice versa, then concatenated.
    This gives the IDM context about *what changed* between frames.
    """

    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_t_to_next = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_next_to_t = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_t = nn.LayerNorm(feature_dim)
        self.norm_next = nn.LayerNorm(feature_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t:    (B, D) — features from frame t
            z_next: (B, D) — features from frame t+1
        Returns:
            fused: (B, 2*D) — concatenated attended features
        """
        # Add sequence dim for MHA: (B, 1, D)
        z_t_s = z_t.unsqueeze(1)
        z_next_s = z_next.unsqueeze(1)

        # z_t queries z_{t+1}: what information in next frame explains the change?
        attended_t, _ = self.attn_t_to_next(z_t_s, z_next_s, z_next_s)
        attended_t = self.norm_t(z_t_s + attended_t).squeeze(1)

        # z_{t+1} queries z_t: grounding in where we started
        attended_next, _ = self.attn_next_to_t(z_next_s, z_t_s, z_t_s)
        attended_next = self.norm_next(z_next_s + attended_next).squeeze(1)

        return torch.cat([attended_t, attended_next], dim=-1)  # (B, 2*D)


class InverseDynamicsModel(nn.Module):
    """
    Predicts SO-101 actions from consecutive frame pairs.

    Input:  (frame_t, frame_{t+1}) — each (B, 3, 224, 224)
    Output: action (B, action_dim) — normalized SO-101 joint positions in [-1, 1]
    """

    def __init__(
        self,
        action_dim: int = 6,
        feature_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        encoder_model_id: str = "facebook/dinov2-base",
    ):
        super().__init__()

        self.encoder = FrameEncoder(
            model_id=encoder_model_id,
            freeze=freeze_encoder,
        )
        self.feature_dim = self.encoder.output_dim

        self.fusion = CrossAttentionFusion(
            self.feature_dim, num_heads=num_heads, dropout=dropout
        )

        fused_dim = self.feature_dim * 2
        layers: list[nn.Module] = []
        in_dim = fused_dim
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_dim))
        self.head = nn.Sequential(*layers)

    def forward(
        self,
        frame_t: torch.Tensor,
        frame_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frame_t:    (B, 3, 224, 224)
            frame_next: (B, 3, 224, 224)
        Returns:
            action: (B, action_dim) in [-1, 1] (tanh not applied — use with MSE loss)
        """
        z_t, z_next = self.encoder.encode_pair(frame_t, frame_next)
        fused = self.fusion(z_t, z_next)
        action = self.head(fused)
        return action

    def predict(
        self,
        frame_t: torch.Tensor,
        frame_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference-time prediction with tanh clamping to ensure [-1, 1] range.
        Returns denormalized degree actions via preprocessor.denormalize_action().
        """
        with torch.no_grad():
            raw = self.forward(frame_t, frame_next)
            return torch.tanh(raw)

    def loss(
        self,
        frame_t: torch.Tensor,
        frame_next: torch.Tensor,
        action_gt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss.

        Uses MSE as primary loss plus Huber for robustness to outlier actions.
        """
        pred = self.forward(frame_t, frame_next)
        mse = F.mse_loss(pred, action_gt)
        huber = F.huber_loss(pred, action_gt, delta=0.5)
        total = 0.5 * mse + 0.5 * huber
        return {"loss": total, "mse": mse, "huber": huber}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
