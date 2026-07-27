"""Unit tests for MIND download helpers (no network)."""

from pathlib import Path

from scripts.download_mind import (
    dataset_url,
    extract_dir_for,
    extract_zip,
    split_name,
)


def test_dataset_url() -> None:
    assert dataset_url("MINDsmall_train.zip").endswith("/MINDsmall_train.zip")


def test_split_name() -> None:
    assert split_name("MINDsmall_train.zip") == "train"
    assert split_name("MINDlarge_dev.zip") == "dev"


def test_extract_dir_for() -> None:
    path = extract_dir_for(Path("data/mind"), "small", "MINDsmall_train.zip")
    assert path == Path("data/mind/train/MINDsmall_train")


def test_extract_zip_skips_when_news_exists(tmp_path: Path) -> None:
    zip_path = tmp_path / "dummy.zip"
    zip_path.write_text("not a real zip")
    extract_dir = tmp_path / "out"
    extract_dir.mkdir()
    (extract_dir / "news.tsv").write_text("n1\n")

    result = extract_zip(zip_path, extract_dir, force=False)
    assert result == extract_dir
