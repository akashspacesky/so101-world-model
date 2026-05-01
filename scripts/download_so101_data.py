#!/usr/bin/env python3
"""
Download all SO-101/SO-100 datasets from HuggingFace Hub.

Usage:
  # Fast scan — see how many datasets exist (no download):
  python scripts/download_so101_data.py --scan

  # Download everything:
  python scripts/download_so101_data.py --output_dir /workspace/data/raw
"""

import argparse
from pathlib import Path

from wm.data.hub_downloader import download_all, print_scan_report, search_hub_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--scan", action="store_true", help="Preview without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download cached datasets")
    args = parser.parse_args()

    datasets = search_hub_datasets()
    print_scan_report(datasets)

    if args.scan:
        return

    download_all(args.output_dir, datasets=datasets, force=args.force)


if __name__ == "__main__":
    main()
