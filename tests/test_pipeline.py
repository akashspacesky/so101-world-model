"""
Full pipeline integration tests — mock world model, mock DINOv2.
No network access or hardware required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import MagicMock, patch


def make_random_frame(size: tuple[int, int] = (480, 480)) -> Image.Image:
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_mock_idm(action_dim: int = 6):
    with patch("wm.models.video_encoder.AutoModel") as MockAutoModel:
        mock_dino = MagicMock()
        mock_dino.config.hidden_size = 768
        mock_output = MagicMock()
        # encode_pair calls forward with 2B batch → need 2*B CLS tokens
        mock_dino.return_value = mock_output
        MockAutoModel.from_pretrained.return_value = mock_dino

        from wm.models.idm import InverseDynamicsModel

        # Patch _load inside encode_pair to return fixed tensors
        idm = InverseDynamicsModel(action_dim=action_dim, freeze_encoder=False)

        # Replace encoder forward with a simple linear mock
        def mock_encoder_forward(pixel_values):
            B = pixel_values.shape[0]
            mock_out = MagicMock()
            mock_out.last_hidden_state = torch.randn(B, 197, 768)
            return mock_out

        idm.encoder.dino = MagicMock(side_effect=mock_encoder_forward)
        idm.encoder.dino.config = MagicMock()
        idm.encoder.dino.config.hidden_size = 768

        return idm


def test_mock_world_model_output():
    from wm.models.world_model import MockWorldModel
    wm = MockWorldModel()
    frame = make_random_frame()
    frames = wm.generate(frame, "pick up the block", num_frames=5)
    assert len(frames) == 5
    assert all(isinstance(f, Image.Image) for f in frames)


def test_mock_world_model_batch():
    from wm.models.world_model import MockWorldModel
    wm = MockWorldModel()
    frame = make_random_frame()
    candidates = wm.generate_batch(frame, "pick up the block", num_candidates=3, num_frames=4)
    assert len(candidates) == 3
    assert all(len(c) == 4 for c in candidates)


def test_pipeline_plan_no_clip():
    """Pipeline plan() with MockWorldModel and mocked IDM, no CLIP scorer."""
    from wm.models.world_model import MockWorldModel
    from wm.models.pipeline import SO101Pipeline

    with patch("wm.models.video_encoder.AutoModel") as MockAutoModel:
        mock_dino_inst = MagicMock()
        mock_dino_inst.config.hidden_size = 768

        def dino_forward(pixel_values):
            B = pixel_values.shape[0]
            out = MagicMock()
            out.last_hidden_state = torch.randn(B, 197, 768)
            return out

        mock_dino_inst.side_effect = dino_forward
        MockAutoModel.from_pretrained.return_value = mock_dino_inst

        from wm.models.idm import InverseDynamicsModel
        idm = InverseDynamicsModel(action_dim=6, freeze_encoder=False)
        idm.encoder.dino = mock_dino_inst

        wm = MockWorldModel()
        pipeline = SO101Pipeline(
            world_model=wm,
            idm=idm,
            device="cpu",
            num_candidates=2,
            use_clip_scoring=False,
        )

        frame = make_random_frame()
        result = pipeline.plan(frame, "pick up the block", num_frames=4)

        assert result.actions.shape[1] == 6, "Actions should be 6-DoF"
        assert len(result.video_frames) == 4
        assert result.num_candidates == 2


def test_preprocessor_build_frame_pairs():
    from wm.data.preprocessor import build_frame_pairs
    frames = [make_random_frame() for _ in range(5)]
    actions = np.random.randn(5, 6).astype(np.float32)
    pairs = build_frame_pairs(frames, actions)
    assert len(pairs) == 4  # T-1 pairs
    for ft, fnext, act in pairs:
        assert isinstance(ft, Image.Image)
        assert isinstance(fnext, Image.Image)
        assert act.shape == (6,)


def test_episode_buffer():
    from wm.data.dataset import EpisodeBuffer
    buf = EpisodeBuffer(max_episodes=5)
    for i in range(3):
        frames = [make_random_frame() for _ in range(4)]
        actions = np.zeros((4, 6), dtype=np.float32)
        buf.add(frames, actions, task=f"task_{i}", success=(i % 2 == 0))

    assert len(buf.episodes) == 3
    assert len(buf.successful_episodes()) == 2  # episodes 0 and 2


def test_flywheel_record_submit_no_upload():
    """Flywheel records and serializes without uploading."""
    import tempfile
    from wm.data.flywheel import DataFlywheel

    with tempfile.TemporaryDirectory() as tmpdir:
        flywheel = DataFlywheel(
            hf_repo="test/repo",
            local_cache=tmpdir,
            auto_upload=False,
        )
        flywheel.start_episode(task="pick up the block")

        for _ in range(3):
            frame = make_random_frame()
            action = np.zeros(6, dtype=np.float32)
            flywheel.record_step(frame, action)

        result = flywheel.submit_episode(success=True, force_upload=False)
        assert result["success"] is True
        assert result["num_frames"] == 3
        assert result["uploaded"] is False
        assert "local_path" in result
