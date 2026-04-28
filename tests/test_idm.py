"""
IDM unit tests — run without GPU or internet (DINOv2 loaded from HF cache if available,
otherwise tests are skipped to avoid network calls in CI).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from wm.data.preprocessor import (
    denormalize_action,
    make_dino_transform,
    normalize_action,
)


@pytest.fixture
def random_frames():
    """Two random 224×224 PIL Images."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr), Image.fromarray(arr + 5)


@pytest.fixture
def dino_transform():
    return make_dino_transform()


def test_normalize_action_roundtrip():
    action = np.array([0.0, 45.0, -90.0, 180.0, -180.0, 12.5])
    normalized = normalize_action(action)
    recovered = denormalize_action(normalized)
    assert torch.allclose(recovered, torch.tensor(action, dtype=torch.float32), atol=1e-4)


def test_normalize_action_range():
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    n = normalize_action(action)
    assert n.min() >= -1.0 - 1e-6
    assert n.max() <= 1.0 + 1e-6


def test_dino_transform_output_shape(random_frames, dino_transform):
    frame_t, frame_next = random_frames
    t = dino_transform(frame_t)
    assert t.shape == (3, 224, 224), f"Expected (3, 224, 224), got {t.shape}"


def test_frame_encoder_shapes():
    """FrameEncoder output shape without downloading DINOv2 — uses mock."""
    from unittest.mock import MagicMock, patch
    import torch

    with patch("wm.models.video_encoder.AutoModel") as MockAutoModel:
        mock_dino = MagicMock()
        mock_dino.config.hidden_size = 768
        # Simulate forward: returns object with last_hidden_state (B, 197, 768)
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_dino.return_value = mock_output
        MockAutoModel.from_pretrained.return_value = mock_dino

        from wm.models.video_encoder import FrameEncoder
        encoder = FrameEncoder(freeze=False)

        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"


def test_idm_shapes():
    """IDM forward pass shape test with mocked DINOv2."""
    from unittest.mock import MagicMock, patch
    import torch

    with patch("wm.models.video_encoder.AutoModel") as MockAutoModel:
        mock_dino = MagicMock()
        mock_dino.config.hidden_size = 768
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(4, 197, 768)
        mock_dino.return_value = mock_output
        MockAutoModel.from_pretrained.return_value = mock_dino

        from wm.models.idm import InverseDynamicsModel
        idm = InverseDynamicsModel(action_dim=6, freeze_encoder=False)

        B = 2
        frame_t = torch.randn(B, 3, 224, 224)
        frame_next = torch.randn(B, 3, 224, 224)
        action = idm.forward(frame_t, frame_next)
        assert action.shape == (B, 6), f"Expected ({B}, 6), got {action.shape}"


def test_idm_loss_finite():
    """IDM loss should be finite and positive."""
    from unittest.mock import MagicMock, patch
    import torch

    with patch("wm.models.video_encoder.AutoModel") as MockAutoModel:
        mock_dino = MagicMock()
        mock_dino.config.hidden_size = 768
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(4, 197, 768)
        mock_dino.return_value = mock_output
        MockAutoModel.from_pretrained.return_value = mock_dino

        from wm.models.idm import InverseDynamicsModel
        idm = InverseDynamicsModel(action_dim=6, freeze_encoder=False)

        B = 2
        frame_t = torch.randn(B, 3, 224, 224)
        frame_next = torch.randn(B, 3, 224, 224)
        action_gt = torch.randn(B, 6)

        loss_dict = idm.loss(frame_t, frame_next, action_gt)
        assert torch.isfinite(loss_dict["loss"]), "Loss is not finite"
        assert loss_dict["loss"].item() > 0, "Loss should be positive"


def test_idm_predict_bounded():
    """IDM.predict() output should be in [-1, 1] after tanh."""
    from unittest.mock import MagicMock, patch
    import torch

    with patch("wm.models.video_encoder.AutoModel") as MockAutoModel:
        mock_dino = MagicMock()
        mock_dino.config.hidden_size = 768

        # side_effect so the mock returns the right batch size regardless of input
        def _dino_forward(pixel_values=None, **kwargs):
            B = pixel_values.shape[0]
            out = MagicMock()
            out.last_hidden_state = torch.randn(B, 197, 768)
            return out

        mock_dino.side_effect = _dino_forward
        MockAutoModel.from_pretrained.return_value = mock_dino

        from wm.models.idm import InverseDynamicsModel
        idm = InverseDynamicsModel(action_dim=6, freeze_encoder=False)
        idm.eval()

        frame_t = torch.randn(1, 3, 224, 224)
        frame_next = torch.randn(1, 3, 224, 224)
        pred = idm.predict(frame_t, frame_next)
        assert pred.min() >= -1.0 - 1e-6
        assert pred.max() <= 1.0 + 1e-6
