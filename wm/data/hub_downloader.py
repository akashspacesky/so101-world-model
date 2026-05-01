"""
Download SO-101 / SO-100 robot datasets from HuggingFace.

Search strategy:
  1. Enumerate all datasets tagged "lerobot" + keyword search for so101/so100
  2. Filter by repo name / tags — no per-repo API calls (fast)
  3. Download everything that matches; quality filtering happens in the dataloader

Run `python scripts/download_so101_data.py --scan` to preview without downloading.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm


SO101_KEYWORDS = ["so101", "so-101"]
SO100_KEYWORDS = ["so100", "so-100"]
ALL_KEYWORDS = SO101_KEYWORDS + SO100_KEYWORDS


@dataclass
class DatasetInfo:
    repo_id: str
    robot_type: str
    local_path: Path | None = None


def _api_call_with_retry(fn, retries: int = 4, **kwargs) -> list:
    for attempt in range(retries):
        try:
            return list(fn(**kwargs))
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                wait = 5 * (2 ** attempt)
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Warning: {e}")
                return []
    return []


def search_hub_datasets(verbose: bool = True) -> list[DatasetInfo]:
    """
    Fast scan: one API call per search term, no per-repo lookups.
    Returns all SO-101/SO-100 datasets found on HuggingFace.
    """
    api = HfApi()
    found: dict[str, DatasetInfo] = {}

    if verbose:
        print("Scanning HuggingFace Hub for SO-101/SO-100 datasets...")

    # All lerobot-tagged datasets (catches the community explosion)
    lerobot_ds = _api_call_with_retry(api.list_datasets, filter="lerobot", limit=5000)
    if verbose:
        print(f"  {len(lerobot_ds)} datasets tagged 'lerobot'")

    # Direct keyword search for so101/so100 (catches untagged datasets)
    keyword_ds: list = []
    for kw in ALL_KEYWORDS:
        keyword_ds += _api_call_with_retry(api.list_datasets, search=kw, limit=2000)
    if verbose:
        print(f"  {len(keyword_ds)} results from keyword search")

    for ds in lerobot_ds + keyword_ds:
        repo_id = ds.id
        if repo_id in found:
            continue
        tags = ds.tags or []
        combined = (repo_id + " " + " ".join(tags)).lower()
        if not any(kw in combined for kw in ALL_KEYWORDS):
            continue
        found[repo_id] = DatasetInfo(
            repo_id=repo_id,
            robot_type=_infer_robot_type(repo_id, tags),
        )

    return sorted(found.values(), key=lambda d: d.repo_id)


def _infer_robot_type(repo_id: str, tags: list[str]) -> str:
    combined = (repo_id + " " + " ".join(tags)).lower()
    if any(kw in combined for kw in SO101_KEYWORDS):
        return "so101"
    if any(kw in combined for kw in SO100_KEYWORDS):
        return "so100"
    return "unknown"


def print_scan_report(datasets: list[DatasetInfo]) -> None:
    so101 = [d for d in datasets if d.robot_type == "so101"]
    so100 = [d for d in datasets if d.robot_type == "so100"]

    print(f"\n{'='*60}")
    print(f"SCAN RESULTS  (no per-repo API calls — size unknown until download)")
    print(f"{'='*60}")
    print(f"  Total datasets found: {len(datasets):>5}")
    print(f"  SO-101:               {len(so101):>5}")
    print(f"  SO-100:               {len(so100):>5}")
    print(f"{'='*60}")
    print(f"\nSample datasets:")
    for d in datasets[:30]:
        print(f"  {d.repo_id:<60}  [{d.robot_type}]")
    if len(datasets) > 30:
        print(f"  ... and {len(datasets) - 30} more")


def download_dataset(repo_id: str, output_dir: Path, force: bool = False) -> Path:
    local_path = output_dir / repo_id.replace("/", "__")
    if local_path.exists() and not force:
        return local_path
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_path),
        ignore_patterns=[
            "*.git*", "*.gitattributes",
            # Video files — can't decode without ffmpeg/av, skip entirely
            "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
        ],
    )
    return local_path


def download_all(
    output_dir: Path,
    datasets: list[DatasetInfo] | None = None,
    force: bool = False,
) -> list[DatasetInfo]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = search_hub_datasets()
        print_scan_report(datasets)

    print(f"\nDownloading {len(datasets)} datasets → {output_dir}")

    for info in tqdm(datasets, desc="Downloading"):
        try:
            info.local_path = download_dataset(info.repo_id, output_dir, force=force)
        except Exception as e:
            print(f"  WARNING: {info.repo_id}: {e}")

    registry = [
        {"repo_id": d.repo_id, "robot_type": d.robot_type,
         "local_path": str(d.local_path) if d.local_path else None}
        for d in datasets
    ]
    registry_path = output_dir / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2))

    downloaded = [d for d in datasets if d.local_path]
    print(f"Downloaded {len(downloaded)}/{len(datasets)} datasets")
    print(f"Registry → {registry_path}")
    return datasets


def iter_episode_files(data_dir: Path) -> Iterator[Path]:
    yield from sorted(data_dir.rglob("episode_*.parquet"))


def load_registry(data_dir: Path) -> list[DatasetInfo]:
    registry_path = data_dir / "registry.json"
    if not registry_path.exists():
        return []
    return [
        DatasetInfo(
            repo_id=e["repo_id"],
            robot_type=e["robot_type"],
            local_path=Path(e["local_path"]) if e["local_path"] else None,
        )
        for e in json.loads(registry_path.read_text())
    ]
