"""
Download SO-101 / SO-100 robot datasets from HuggingFace.

Strategy:
  1. Scan: one API call per search term to find all matching repos
  2. Per repo: one API call (list_repo_files) to get the file list
  3. Download: parallel direct GET requests — no per-file HEAD/ETag checks

snapshot_download() HEAD-checks every file against local ETags before
downloading. For v3 datasets with 10k individual frame files that means
10k HEAD requests → 429s. Direct GETs to HF's CDN bypass this entirely.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterator

import requests
from huggingface_hub import HfApi, hf_hub_url
from tqdm import tqdm


SO101_KEYWORDS = ["so101", "so-101"]
SO100_KEYWORDS = ["so100", "so-100"]
ALL_KEYWORDS = SO101_KEYWORDS + SO100_KEYWORDS

_SKIP = {".gitattributes", ".gitignore", ".git"}


@dataclass
class DatasetInfo:
    repo_id: str
    robot_type: str
    local_path: Path | None = None


# ---------------------------------------------------------------------------
# Hub scanning
# ---------------------------------------------------------------------------

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
    api = HfApi()
    found: dict[str, DatasetInfo] = {}

    if verbose:
        print("Scanning HuggingFace Hub for SO-101/SO-100 datasets...")

    lerobot_ds = _api_call_with_retry(api.list_datasets, filter="lerobot", limit=5000)
    if verbose:
        print(f"  {len(lerobot_ds)} datasets tagged 'lerobot'")

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
    print(f"SCAN RESULTS  (size unknown until download)")
    print(f"{'='*60}")
    print(f"  Total: {len(datasets):>5}   SO-101: {len(so101):>5}   SO-100: {len(so100):>5}")
    print(f"{'='*60}")
    for d in datasets[:30]:
        print(f"  {d.repo_id:<60}  [{d.robot_type}]")
    if len(datasets) > 30:
        print(f"  ... and {len(datasets) - 30} more")


# ---------------------------------------------------------------------------
# Direct GET download (no HEAD / ETag check)
# ---------------------------------------------------------------------------

def _download_file(url: str, local_path: Path, token: str | None, retries: int = 6) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    wait = 15 * (2 ** attempt)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                tmp = local_path.with_suffix(local_path.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
                tmp.rename(local_path)
                return
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def download_dataset(
    repo_id: str,
    output_dir: Path,
    force: bool = False,
    file_workers: int = 8,
) -> Path:
    """
    Download one dataset.
    - 1 API call to list all files
    - Parallel direct GET downloads, no per-file HEAD requests
    """
    local_path = output_dir / repo_id.replace("/", "__")
    if local_path.exists() and not force:
        return local_path

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Single API call — replaces N HEAD requests
    try:
        all_files = list(api.list_repo_files(repo_id, repo_type="dataset"))
    except Exception as e:
        raise RuntimeError(f"Cannot list {repo_id}: {e}")

    files = [
        f for f in all_files
        if Path(f).name not in _SKIP and not f.startswith(".git")
    ]

    def _fetch(file_path: str) -> None:
        url = hf_hub_url(repo_id=repo_id, filename=file_path, repo_type="dataset")
        _download_file(url, local_path / file_path, token)

    with ThreadPoolExecutor(max_workers=file_workers) as pool:
        futures = [pool.submit(_fetch, f) for f in files]
        for future in as_completed(futures):
            future.result()

    return local_path


# ---------------------------------------------------------------------------
# Bulk download across all datasets
# ---------------------------------------------------------------------------

def download_all(
    output_dir: Path,
    datasets: list[DatasetInfo] | None = None,
    force: bool = False,
    workers: int = 4,
    file_workers: int = 8,
) -> list[DatasetInfo]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if datasets is None:
        datasets = search_hub_datasets()
        print_scan_report(datasets)

    print(f"\nDownloading {len(datasets)} datasets → {output_dir}")
    print(f"  {workers} dataset workers × {file_workers} file workers = {workers*file_workers} concurrent GETs")

    lock = Lock()
    pbar = tqdm(total=len(datasets), desc="Datasets")

    def _download_one(info: DatasetInfo) -> DatasetInfo:
        try:
            info.local_path = download_dataset(
                info.repo_id, output_dir, force=force, file_workers=file_workers
            )
        except Exception as e:
            print(f"\n  WARNING: {info.repo_id}: {e}")
        with lock:
            pbar.update(1)
        return info

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, info): info for info in datasets}
        for future in as_completed(futures):
            future.result()

    pbar.close()

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def iter_episode_files(data_dir: Path) -> Iterator[Path]:
    yield from sorted(Path(data_dir).rglob("episode_*.parquet"))


def load_registry(data_dir: Path) -> list[DatasetInfo]:
    registry_path = Path(data_dir) / "registry.json"
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
