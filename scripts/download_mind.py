#!/usr/bin/env python3
"""Download and extract the MIND dataset (small or large)."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

BASE_URL = "https://mind201910small.blob.core.windows.net/release"
DATASET_FILES = {
    "small": ("MINDsmall_train.zip", "MINDsmall_dev.zip"),
    "large": ("MINDlarge_train.zip", "MINDlarge_dev.zip"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract the MIND dataset.")
    parser.add_argument(
        "--size",
        choices=sorted(DATASET_FILES),
        default="small",
        help="MIND variant to download (default: small)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mind"),
        help="Root directory for MIND raw + extracted files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if files already exist",
    )
    return parser.parse_args()


def dataset_url(filename: str) -> str:
    return f"{BASE_URL}/{filename}"


def download_file(url: str, destination: Path, force: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"Skipping download (exists): {destination}")
        return destination

    print(f"Downloading {url}")
    print(f"  -> {destination}")
    try:
        urlretrieve(url, destination)  # noqa: S310 - fixed Microsoft Azure blob URLs
    except URLError as exc:
        if destination.exists():
            destination.unlink()
        msg = f"Failed to download {url}: {exc}"
        raise SystemExit(msg) from exc
    return destination


def extract_zip(zip_path: Path, extract_dir: Path, force: bool = False) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / "news.tsv"
    if marker.exists() and not force:
        print(f"Skipping extract (exists): {extract_dir}")
        return extract_dir

    if force and extract_dir.exists():
        for child in extract_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    if not marker.exists():
        msg = f"Expected news.tsv after extracting {zip_path} into {extract_dir}"
        raise SystemExit(msg)
    return extract_dir


def split_name(filename: str) -> str:
    # MINDsmall_train.zip -> train, MINDlarge_dev.zip -> dev
    stem = filename.removesuffix(".zip")
    return stem.rsplit("_", 1)[-1]


def extract_dir_for(data_dir: Path, size: str, filename: str) -> Path:
    split = split_name(filename)
    folder_name = filename.removesuffix(".zip")
    return data_dir / split / folder_name


def run(size: str, data_dir: Path, force: bool = False) -> dict[str, Path]:
    raw_dir = data_dir / "raw"
    paths: dict[str, Path] = {}

    for filename in DATASET_FILES[size]:
        zip_path = download_file(dataset_url(filename), raw_dir / filename, force=force)
        target = extract_dir_for(data_dir, size, filename)
        extract_zip(zip_path, target, force=force)
        paths[split_name(filename)] = target

    return paths


def main() -> None:
    args = parse_args()
    paths = run(size=args.size, data_dir=args.data_dir, force=args.force)
    print("MIND dataset ready:")
    for split, path in sorted(paths.items()):
        print(f"  {split}: {path}")
        print(f"    news:      {path / 'news.tsv'}")
        print(f"    behaviors: {path / 'behaviors.tsv'}")


if __name__ == "__main__":
    main()
    sys.exit(0)
