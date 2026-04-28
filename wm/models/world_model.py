"""
CogVideoX-2b I2V world model wrapper.

Wraps HuggingFace CogVideoX Image-to-Video pipeline for SO-101.
Given a starting frame and text instruction, generates a plausible
future video of the task being completed.

Fine-tuning is done via LoRA (see wm/training/train_wm.py).
Inference runs on cloud (A100/H100) or any machine with enough VRAM.

CogVideoX-2b-I2V:
  - 2B parameters, bfloat16
  - Generates 49 frames at 8fps (~6s video) or 13 frames at 8fps
  - Input: first frame (PIL Image) + text prompt
  - Output: list of PIL Images
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

# Lazy imports — only needed at runtime, not at import time for tests
def _load_cogvideox():
    from diffusers import CogVideoXImageToVideoPipeline
    return CogVideoXImageToVideoPipeline


COGVIDEOX_MODEL_ID = "THUDM/CogVideoX-2b"


class SO101WorldModel:
    """
    CogVideoX-2b I2V fine-tuned for SO-101 manipulation.

    Usage:
        wm = SO101WorldModel.from_pretrained()
        frames = wm.generate(current_frame, "pick up the red block")
    """

    def __init__(self, pipe, device: str | torch.device = "cuda"):
        self.pipe = pipe
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = COGVIDEOX_MODEL_ID,
        lora_path: str | Path | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_slicing: bool = True,
        enable_tiling: bool = True,
    ) -> "SO101WorldModel":
        Pipeline = _load_cogvideox()
        pipe = Pipeline.from_pretrained(model_id, torch_dtype=dtype)

        if lora_path is not None:
            lora_path = Path(lora_path)
            if lora_path.exists():
                pipe.load_lora_weights(str(lora_path))
                print(f"Loaded LoRA weights from {lora_path}")

        if enable_slicing:
            pipe.vae.enable_slicing()
        if enable_tiling:
            pipe.vae.enable_tiling()

        pipe = pipe.to(device)
        return cls(pipe, device=device)

    def generate(
        self,
        current_frame: Image.Image,
        text_instruction: str,
        num_frames: int = 13,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: int | None = None,
    ) -> list[Image.Image]:
        """
        Generate a future video given the current frame and instruction.

        Args:
            current_frame:      Starting frame from SO-101 camera
            text_instruction:   Task description, e.g. "pick up the red block"
            num_frames:         Number of frames to generate (13 = ~1.6s at 8fps)
            num_inference_steps: Diffusion steps (more = better quality, slower)
            guidance_scale:     Classifier-free guidance scale
            seed:               RNG seed for reproducibility

        Returns:
            List of PIL Images — the generated future video frames
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # CogVideoX expects images resized to 480×480
        frame_480 = current_frame.resize((480, 480), Image.LANCZOS)

        output = self.pipe(
            image=frame_480,
            prompt=text_instruction,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        return output.frames[0]  # list of PIL Images

    def generate_batch(
        self,
        current_frame: Image.Image,
        text_instruction: str,
        num_candidates: int = 3,
        **kwargs,
    ) -> list[list[Image.Image]]:
        """
        Generate multiple candidate videos for test-time compute scaling.
        Each candidate uses a different random seed.
        """
        candidates = []
        for i in range(num_candidates):
            frames = self.generate(
                current_frame,
                text_instruction,
                seed=i * 42,
                **kwargs,
            )
            candidates.append(frames)
        return candidates


class MockWorldModel:
    """
    Mock world model for testing without CogVideoX installed.
    Returns random PIL Images of the correct size.
    """

    def generate(
        self,
        current_frame: Image.Image,
        text_instruction: str,
        num_frames: int = 13,
        **kwargs,
    ) -> list[Image.Image]:
        import numpy as np
        return [
            Image.fromarray(np.random.randint(0, 255, (480, 480, 3), dtype=np.uint8))
            for _ in range(num_frames)
        ]

    def generate_batch(
        self,
        current_frame: Image.Image,
        text_instruction: str,
        num_candidates: int = 3,
        **kwargs,
    ) -> list[list[Image.Image]]:
        return [self.generate(current_frame, text_instruction, **kwargs) for _ in range(num_candidates)]
