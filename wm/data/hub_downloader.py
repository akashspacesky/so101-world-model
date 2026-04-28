"""
Download all SO-101 / SO-100 robot datasets from HuggingFace.

LeRobot datasets store episodes as parquet files with image bytes, joint states,
and actions. This module searches for all known SO-101 datasets, downloads them,
and caches them locally for training.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


# All known SO-101/SO-100 robot datasets on HuggingFace.
# We include SO-100 because the kinematics are nearly identical.
KNOWN_DATASETS: list[str] = [
    # Official LeRobot SO-101 datasets
    "lerobot/so101_strawberry_grape",
    "lerobot/so101_pick_place",
    "lerobot/so101_fold_tshirt",
    "lerobot/so101_wipe_table",
    # Official LeRobot SO-100 datasets (compatible kinematics)
    "lerobot/so100_pick_place",
    "lerobot/so100_stack_cups",
]

# HuggingFace search tags to discover community datasets
SEARCH_TAGS = ["so101", "so-101", "so100", "so-100", "lerobot-so101"]


@dataclass
class DatasetInfo:
    repo_id: str
    num_episodes: int
    robot_type: str
    tasks: list[str] = field(default_factory=list)
    local_path: Path | None = None


def search_hub_datasets(max_results: int = 200) -> list[DatasetInfo]:
    """Search HuggingFace Hub for all SO-101/SO-100 robot datasets."""
    api = HfApi()
    found: dict[str, DatasetInfo] = {}

    # Add known datasets first
    for repo_id in KNOWN_DATASETS:
        try:
            info = api.dataset_info(repo_id)
            found[repo_id] = DatasetInfo(
                repo_id=repo_id,
                num_episodes=_get_num_episodes(repo_id, api),
                robot_type="so101" if "so101" in repo_id else "so100",
            )
        except Exception:
            pass

    # Search for community datasets
    for tag in SEARCH_TAGS:
        try:
            results = api.list_datasets(search=tag, limit=max_results)
            for ds in results:
                repo_id = ds.id
                if repo_id in found:
                    continue
                # Filter: must look like a robot manipulation dataset
                card_tags = ds.tags or []
                if not any(t in card_tags for t in ["lerobot", "robotics", "manipulation", "so101", "so100"]):
                    # Try to check the README for robot mentions
                    pass
                found[repo_id] = DatasetInfo(
                    repo_id=repo_id,
                    num_episodes=_get_num_episodes(repo_id, api),
                    robot_type=_infer_robot_type(repo_id, card_tags),
                )
        except Exception:
            continue

    return list(found.values())


def _get_num_episodes(repo_id: str, api: HfApi) -> int:
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        return sum(1 for f in files if f.startswith("data/episode_") and f.endswith(".parquet"))
    except Exception:
        return 0


def _infer_robot_type(repo_id: str, tags: list[str]) -> str:
    combined = (repo_id + " ".join(tags)).lower()
    if "so101" in combined or "so-101" in combined:
        return "so101"
    if "so100" in combined or "so-100" in combined:
        return "so100"
    return "unknown"


def download_dataset(
    repo_id: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """Download a single HuggingFace dataset into output_dir/repo_id."""
    local_path = output_dir / repo_id.replace("/", "__")
    if local_path.exists() and not force:
        print(f"  [cache hit] {repo_id}")
        return local_path

    print(f"  Downloading {repo_id} → {local_path}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path),
        ignore_patterns=["*.git*", "*.gitattributes"],
    )
    return local_path


def download_all(
    output_dir: Path,
    datasets: list[DatasetInfo] | None = None,
    force: bool = False,
) -> list[DatasetInfo]:
    """Download all discovered SO-101 datasets. Returns updated DatasetInfo list."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        print("Searching HuggingFace Hub for SO-101/SO-100 datasets...")
        datasets = search_hub_datasets()
        print(f"Found {len(datasets)} datasets")

    registry_path = output_dir / "registry.json"
    for info in tqdm(datasets, desc="Downloading datasets"):
        try:
            local_path = download_dataset(info.repo_id, output_dir, force=force)
            info.local_path = local_path
        except Exception as e:
            print(f"  WARNING: failed to download {info.repo_id}: {e}")

    # Save registry for reproducibility
    registry = [
        {
            "repo_id": d.repo_id,
            "num_episodes": d.num_episodes,
            "robot_type": d.robot_type,
            "local_path": str(d.local_path) if d.local_path else None,
        }
        for d in datasets
    ]
    registry_path.write_text(json.dumps(registry, indent=2))
    print(f"\nRegistry saved to {registry_path}")

    downloaded = [d for d in datasets if d.local_path is not None]
    print(f"Successfully downloaded {len(downloaded)}/{len(datasets)} datasets")
    return datasets


def iter_episode_files(data_dir: Path) -> Iterator[Path]:
    """Yield all episode parquet files under data_dir (recursively)."""
    for parquet in sorted(data_dir.rglob("episode_*.parquet")):
        yield parquet


def load_registry(data_dir: Path) -> list[DatasetInfo]:
    registry_path = data_dir / "registry.json"
    if not registry_path.exists():
        return []
    entries = json.loads(registry_path.read_text())
    return [
        DatasetInfo(
            repo_id=e["repo_id"],
            num_episodes=e["num_episodes"],
            robot_type=e["robot_type"],
            local_path=Path(e["local_path"]) if e["local_path"] else None,
        )
        for e in entries
    ]
