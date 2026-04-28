"""
SO101Pipeline: full inference pipeline.

1. World model generates N candidate future videos from (current_frame, text)
2. CLIP scores each candidate's final frame against the goal text
3. Best candidate selected
4. IDM extracts actions from consecutive frame pairs
5. Actions returned for robot execution

This implements 1X's test-time compute scaling:
  - N=1: 30% success on hard tasks
  - N=3-5: 45%+ success (roughly)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from wm.data.preprocessor import denormalize_action, make_dino_transform
from wm.models.idm import InverseDynamicsModel
from wm.models.world_model import SO101WorldModel, MockWorldModel


@dataclass
class PlanResult:
    """Result of a single planning call."""
    actions: np.ndarray          # (T, 6) denormalized SO-101 joint positions in degrees
    video_frames: list[Image.Image]  # The selected candidate video
    goal_score: float            # CLIP similarity score of final frame vs goal
    num_candidates: int          # How many candidates were generated


class CLIPGoalScorer:
    """
    Scores a video candidate against the goal text using CLIP.
    Higher score = final frame is more consistent with the goal description.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def score(self, frame: Image.Image, text: str) -> float:
        """Returns cosine similarity between frame and text in CLIP embedding space."""
        inputs = self.processor(
            text=[text],
            images=[frame],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image  # (1, 1)
        return logits.item()

    def score_candidates(
        self,
        candidates: list[list[Image.Image]],
        goal_text: str,
    ) -> list[float]:
        """Score all candidates by their final frame."""
        return [self.score(c[-1], goal_text) for c in candidates]


class SO101Pipeline:
    """
    Full inference pipeline: text + frame → robot actions.

    Args:
        world_model:    Video generation model (SO101WorldModel or MockWorldModel)
        idm:            Inverse dynamics model (InverseDynamicsModel)
        device:         Torch device for IDM inference
        num_candidates: Number of candidate videos to generate per step
        use_clip_scoring: Score candidates with CLIP (requires CLIP model)
    """

    def __init__(
        self,
        world_model: SO101WorldModel | MockWorldModel,
        idm: InverseDynamicsModel,
        device: str | torch.device = "cuda",
        num_candidates: int = 3,
        use_clip_scoring: bool = True,
    ):
        self.world_model = world_model
        self.idm = idm.to(device)
        self.idm.eval()
        self.device = device
        self.num_candidates = num_candidates
        self._dino_transform = make_dino_transform()

        self.scorer: CLIPGoalScorer | None = None
        if use_clip_scoring:
            try:
                self.scorer = CLIPGoalScorer(device=str(device))
            except Exception as e:
                print(f"WARNING: CLIP scorer unavailable ({e}), using random selection")

    @classmethod
    def from_checkpoints(
        cls,
        idm_ckpt: str | Path,
        wm_lora: str | Path | None = None,
        device: str = "cuda",
        num_candidates: int = 3,
        mock_wm: bool = False,
    ) -> "SO101Pipeline":
        """Load pipeline from saved checkpoints."""
        # Load IDM
        idm = InverseDynamicsModel()
        state = torch.load(idm_ckpt, map_location=device)
        idm.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded IDM from {idm_ckpt}")

        # Load world model
        if mock_wm:
            wm = MockWorldModel()
        else:
            wm = SO101WorldModel.from_pretrained(lora_path=wm_lora, device=device)

        return cls(wm, idm, device=device, num_candidates=num_candidates)

    def plan(
        self,
        current_frame: Image.Image,
        text_instruction: str,
        num_frames: int = 13,
        wm_inference_steps: int = 50,
    ) -> PlanResult:
        """
        Full planning step: generate video candidates → score → extract actions.

        Args:
            current_frame:      Current RGB frame from SO-101 camera
            text_instruction:   Task to execute, e.g. "pick up the cup"
            num_frames:         Video length (13 ≈ 1.6s at 8fps)
            wm_inference_steps: Diffusion steps for world model

        Returns:
            PlanResult with actions array and selected video
        """
        # Step 1: generate N candidate future videos
        candidates = self.world_model.generate_batch(
            current_frame,
            text_instruction,
            num_candidates=self.num_candidates,
            num_frames=num_frames,
            num_inference_steps=wm_inference_steps,
        )

        # Step 2: select best candidate
        if self.scorer is not None and len(candidates) > 1:
            scores = self.scorer.score_candidates(candidates, text_instruction)
            best_idx = int(np.argmax(scores))
            goal_score = scores[best_idx]
        else:
            best_idx = 0
            goal_score = 0.0

        best_video = candidates[best_idx]

        # Step 3: extract actions from consecutive frame pairs via IDM
        actions = self._extract_actions(best_video)

        return PlanResult(
            actions=actions,
            video_frames=best_video,
            goal_score=goal_score,
            num_candidates=len(candidates),
        )

    def _extract_actions(self, frames: list[Image.Image]) -> np.ndarray:
        """Run IDM on consecutive frame pairs to extract action sequence."""
        if len(frames) < 2:
            return np.zeros((1, 6), dtype=np.float32)

        frame_tensors = torch.stack([
            self._dino_transform(f) for f in frames
        ]).to(self.device)  # (T, 3, 224, 224)

        actions = []
        with torch.no_grad():
            for t in range(len(frame_tensors) - 1):
                ft = frame_tensors[t].unsqueeze(0)
                ft1 = frame_tensors[t + 1].unsqueeze(0)
                action_norm = self.idm.predict(ft, ft1)  # (1, 6) in [-1, 1]
                action_deg = denormalize_action(action_norm)
                actions.append(action_deg.squeeze(0).cpu().numpy())

        return np.stack(actions)  # (T-1, 6)

    def execute_on_robot(
        self,
        current_frame: Image.Image,
        text_instruction: str,
        robot_env,
        replan_every: int = 6,
        max_steps: int = 50,
    ) -> dict:
        """
        Full closed-loop execution on SO-101.

        Replans every `replan_every` steps using the latest camera frame.
        """
        total_reward = 0.0
        obs = {"frame": current_frame}
        done = False
        step = 0
        episode_frames = [current_frame]
        episode_actions = []

        while not done and step < max_steps:
            result = self.plan(obs["frame"], text_instruction)

            for action in result.actions:
                if done or step >= max_steps:
                    break
                next_obs, reward, done, info = robot_env.step(action)
                total_reward += reward
                obs = next_obs
                episode_frames.append(obs.get("frame", current_frame))
                episode_actions.append(action)
                step += 1

                if step % replan_every == 0 and not done:
                    break  # replan with new frame

        return {
            "total_reward": total_reward,
            "steps": step,
            "success": done and total_reward > 0,
            "frames": episode_frames,
            "actions": np.stack(episode_actions) if episode_actions else np.zeros((0, 6)),
        }
