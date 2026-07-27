#!/usr/bin/env python3
"""Orchestrate MIND download → Postgres ingest → ranking feature parquet.

Keeps step-0 data setup as a real script under scripts/ (not one-off shell).

Examples:
  python scripts/setup_mind_data.py
  python scripts/setup_mind_data.py --size small --split train --feature-limit 50000
  python scripts/setup_mind_data.py --skip-download --skip-ingest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_mind import run as download_mind



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MIND, ingest into Postgres, and build ranking features."
    )
    parser.add_argument("--size", choices=["small", "large"], default="small")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "mind",
        help="Root directory for MIND raw + extracted files",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "data" / "ranking_features.parquet",
        help="Where to write ranking feature parquet",
    )
    parser.add_argument(
        "--feature-limit",
        type=int,
        default=50_000,
        help="Max labeled impression rows for the ranking dataset",
    )
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    return parser.parse_args()


def mind_split_dir(data_dir: Path, size: str, split: str) -> Path:
    folder = f"MIND{size}_{split}"
    return data_dir / split / folder


def run_module(module: str, args: list[str]) -> None:
    command = [sys.executable, "-m", module, *args]
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()

    if not args.skip_download:
        print(f"Downloading MIND-{args.size} into {args.data_dir}")
        paths = download_mind(
            size=args.size,
            data_dir=args.data_dir,
            force=args.force_download,
        )
        split_dir = paths[args.split]
    else:
        split_dir = mind_split_dir(args.data_dir, args.size, args.split)

    news_path = split_dir / "news.tsv"
    behaviors_path = split_dir / "behaviors.tsv"
    if not news_path.exists() or not behaviors_path.exists():
        msg = (
            f"Missing MIND files under {split_dir}. "
            "Run without --skip-download, or pass --data-dir to the extracted dataset."
        )
        raise SystemExit(msg)

    if not args.skip_ingest:
        print(f"Ingesting {args.split} split from {split_dir}")
        run_module(
            "pipelines.ingest_mind",
            [
                "--news-path",
                str(news_path),
                "--behaviors-path",
                str(behaviors_path),
                "--batch-size",
                str(args.batch_size),
            ],
        )

    if not args.skip_features:
        print(f"Building ranking features -> {args.output_path}")
        run_module(
            "pipelines.build_ranking_dataset",
            [
                "--output-path",
                str(args.output_path),
                "--limit",
                str(args.feature_limit),
            ],
        )

    print("MIND data setup complete.")
    print(f"  news:      {news_path}")
    print(f"  behaviors: {behaviors_path}")
    if not args.skip_features:
        print(f"  features:  {args.output_path}")


if __name__ == "__main__":
    main()
