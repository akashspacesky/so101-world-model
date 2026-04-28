#!/usr/bin/env python3
"""Download all SO-101/SO-100 datasets from HuggingFace Hub."""

import argparse
from pathlib import Path

from wm.data.hub_downloader import download_all, search_hub_datasets


def main():
    parser = argparse.ArgumentParser(description="Download all SO-101 HF datasets")
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    parser.add_argument("--list_only", action="store_true", help="Just list datasets, don't download")
    args = parser.parse_args()

    print("Searching HuggingFace Hub for SO-101/SO-100 datasets...")
    datasets = search_hub_datasets()

    print(f"\nFound {len(datasets)} datasets:")
    total_episodes = 0
    for ds in datasets:
        print(f"  {ds.repo_id:<50} {ds.num_episodes:>5} episodes  [{ds.robot_type}]")
        total_episodes += ds.num_episodes
    print(f"\nTotal episodes: {total_episodes}")

    if args.list_only:
        return

    print(f"\nDownloading to {args.output_dir}")
    download_all(args.output_dir, datasets=datasets, force=args.force)


if __name__ == "__main__":
    main()
