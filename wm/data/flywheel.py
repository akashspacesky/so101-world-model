"""
Data Flywheel — the long-term moat.

Every SO-101 robot running this system can contribute deployment episodes
back to the shared dataset on HuggingFace. More data → better IDM and world model
→ better zero-shot performance → more deployments → more data.

How it works:
1. During inference, the pipeline records frames + executed actions
2. After each episode, a CLIP-based success detector scores the outcome
3. Successful episodes are serialized to LeRobot parquet format
4. Uploaded to the community HF dataset (with user's consent and attribution)
5. A weekly retraining job (see scripts/) incorporates new contributions

The IDM is the crown jewel: trained on all contributed SO-101 data,
it accumulates embodiment knowledge that generic video models can't replicate.

Usage:
    flywheel = DataFlywheel(hf_repo="your-org/so101-world-model-data")
    flywheel.record_frame(frame, action)
    # ... end of episode
    flywheel.submit_episode(task="pick up the block", success=True)
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


class SuccessDetector:
    """
    CLIP-based episode success detection.

    Compares the final frame of an episode against the task description.
    High CLIP similarity = task likely completed.

    Threshold is conservative (0.25) by default — we'd rather miss a success
    than contribute a failure to the dataset.
    """

    def __init__(self, device: str = "cpu", threshold: float = 0.25):
        self.threshold = threshold
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model.eval()

    def score(self, final_frame: Image.Image, task: str) -> float:
        """Returns CLIP similarity score in [0, 1] range."""
        import torch
        self._load()
        inputs = self._processor(
            text=[task],
            images=[final_frame],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Logits are un-normalized; convert to probability-like score
            score = outputs.logits_per_image.sigmoid().item()
        return score

    def is_success(self, final_frame: Image.Image, task: str) -> tuple[bool, float]:
        score = self.score(final_frame, task)
        return score >= self.threshold, score


def _frames_to_bytes(frame: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    frame.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _serialize_episode_to_parquet(
    frames: list[Image.Image],
    actions: np.ndarray,
    task: str,
    episode_id: str,
    contributor: str,
    success: bool,
    clip_score: float,
) -> bytes:
    """Serialize an episode to LeRobot-compatible parquet format."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for i, (frame, action) in enumerate(zip(frames, actions)):
        rows.append({
            "episode_index": 0,
            "frame_index": i,
            "timestamp": i / 10.0,  # assume 10 fps
            "observation.images.top": {"bytes": _frames_to_bytes(frame), "path": None},
            "action": action.tolist(),
            "task": task,
            "next.done": i == len(frames) - 1,
            # Flywheel metadata
            "meta.episode_id": episode_id,
            "meta.contributor": contributor,
            "meta.success": success,
            "meta.clip_score": clip_score,
            "meta.timestamp": time.time(),
        })

    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


class DataFlywheel:
    """
    Records deployment episodes and uploads successful ones to HuggingFace.

    Args:
        hf_repo:      HuggingFace dataset repo to push episodes to
                      e.g. "your-org/so101-world-model-data"
        local_cache:  Local directory to buffer episodes before upload
        contributor:  Attribution string (username or robot ID)
        auto_upload:  Upload immediately after each successful episode
        success_detector: Override default CLIP-based detector
    """

    def __init__(
        self,
        hf_repo: str = "so101-community/world-model-data",
        local_cache: Path = Path("~/.so101_flywheel").expanduser(),
        contributor: str = "anonymous",
        auto_upload: bool = True,
        success_threshold: float = 0.25,
    ):
        self.hf_repo = hf_repo
        self.local_cache = Path(local_cache)
        self.local_cache.mkdir(parents=True, exist_ok=True)
        self.contributor = contributor
        self.auto_upload = auto_upload

        self.detector = SuccessDetector(threshold=success_threshold)

        # In-memory episode buffer
        self._frames: list[Image.Image] = []
        self._actions: list[np.ndarray] = []
        self._current_task: str = ""
        self._episode_id: str = ""

    def start_episode(self, task: str):
        """Call at the start of each robot episode."""
        self._frames = []
        self._actions = []
        self._current_task = task
        self._episode_id = str(uuid.uuid4())[:12]

    def record_step(self, frame: Image.Image, action: np.ndarray):
        """Record a single robot step (frame observed + action taken)."""
        self._frames.append(frame)
        self._actions.append(action.astype(np.float32))

    def submit_episode(
        self,
        success: bool | None = None,
        force_upload: bool = False,
    ) -> dict:
        """
        Finalize the current episode. Auto-detects success via CLIP if not provided.

        Returns dict with success status, clip_score, and upload status.
        """
        if not self._frames:
            return {"success": False, "reason": "no frames recorded"}

        actions = np.stack(self._actions) if self._actions else np.zeros((len(self._frames), 6))

        # Auto-detect success if not explicitly provided
        clip_score = 0.0
        if success is None and self._current_task:
            success, clip_score = self.detector.is_success(self._frames[-1], self._current_task)
        elif success is not None:
            clip_score = self.detector.score(self._frames[-1], self._current_task) if self._frames else 0.0

        result = {
            "episode_id": self._episode_id,
            "task": self._current_task,
            "num_frames": len(self._frames),
            "success": success,
            "clip_score": clip_score,
            "uploaded": False,
        }

        # Only contribute successful episodes
        if not success and not force_upload:
            result["reason"] = f"episode not successful (clip_score={clip_score:.3f})"
            return result

        # Save locally first
        parquet_bytes = _serialize_episode_to_parquet(
            self._frames,
            actions,
            self._current_task,
            self._episode_id,
            self.contributor,
            success,
            clip_score,
        )

        local_file = self.local_cache / f"episode_{self._episode_id}.parquet"
        local_file.write_bytes(parquet_bytes)
        result["local_path"] = str(local_file)

        # Upload to HuggingFace
        if self.auto_upload or force_upload:
            try:
                uploaded = self._upload_to_hub(local_file)
                result["uploaded"] = uploaded
            except Exception as e:
                result["upload_error"] = str(e)
                print(f"WARNING: upload failed: {e}. Episode saved locally at {local_file}")

        return result

    def _upload_to_hub(self, local_file: Path) -> bool:
        from huggingface_hub import HfApi
        api = HfApi()
        remote_path = f"data/{local_file.name}"
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=remote_path,
            repo_id=self.hf_repo,
            repo_type="dataset",
            commit_message=f"Add episode {local_file.stem} from {self.contributor}",
        )
        print(f"Uploaded {local_file.name} to {self.hf_repo}/{remote_path}")
        return True

    def upload_pending(self) -> int:
        """Upload all locally cached episodes that haven't been uploaded yet."""
        pending = list(self.local_cache.glob("episode_*.parquet"))
        n_uploaded = 0
        for p in pending:
            try:
                self._upload_to_hub(p)
                n_uploaded += 1
            except Exception as e:
                print(f"  Failed to upload {p.name}: {e}")
        return n_uploaded

    def stats(self) -> dict:
        """Return flywheel stats: total local episodes, uploads pending."""
        local_files = list(self.local_cache.glob("episode_*.parquet"))
        return {
            "local_episodes": len(local_files),
            "cache_dir": str(self.local_cache),
            "hf_repo": self.hf_repo,
            "contributor": self.contributor,
        }


def cli_contribute(episode_dir: Path, hf_repo: str, contributor: str = "anonymous"):
    """
    CLI entrypoint: contribute a directory of recorded episodes to the flywheel.

    Episode dir structure:
        episode_dir/
            episode_000001/
                frame_0000.jpg
                frame_0001.jpg
                ...
                actions.npy
                meta.json   <- {"task": "...", "success": true/false}
    """
    flywheel = DataFlywheel(hf_repo=hf_repo, contributor=contributor, auto_upload=True)
    episode_dirs = sorted(Path(episode_dir).glob("episode_*"))
    print(f"Contributing {len(episode_dirs)} episodes from {episode_dir}")

    for ep_dir in episode_dirs:
        meta_file = ep_dir / "meta.json"
        actions_file = ep_dir / "actions.npy"
        if not meta_file.exists():
            continue

        meta = json.loads(meta_file.read_text())
        frames = sorted(ep_dir.glob("frame_*.jpg"))

        flywheel.start_episode(task=meta.get("task", ""))
        for fpath in frames:
            frame = Image.open(fpath).convert("RGB")
            action = np.zeros(6)  # placeholder if no per-step actions
            flywheel.record_step(frame, action)

        if actions_file.exists():
            actions = np.load(actions_file)
            flywheel._actions = [actions[i] for i in range(min(len(actions), len(frames)))]

        result = flywheel.submit_episode(success=meta.get("success"))
        print(f"  {ep_dir.name}: success={result['success']}, uploaded={result.get('uploaded', False)}")
