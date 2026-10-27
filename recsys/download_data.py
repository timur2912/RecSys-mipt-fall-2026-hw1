"""Download and extract the Avito ML Cup 2025 dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "sharthz23/avito-cup-2025-recsys"
)
DEFAULT_OUTPUT_DIR = Path("data/raw/avito")
ARCHIVE_NAME = "avito_ml_cup_2025.zip"
DATA_FILES = ("clickstream.pq", "cat_features.pq", "events.pq")
CHUNK_SIZE = 1024 * 1024


def download_file(url: str, destination: Path, force: bool = False) -> None:
    """Download a file atomically while reporting progress."""
    if destination.exists() and not force:
        print(f"Archive already exists: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary_path.open(
            "wb"
        ) as output:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = downloaded / total_size * 100
                    print(
                        f"\rDownloading: {progress:6.2f}% "
                        f"({downloaded / 1024**3:.2f} GiB)",
                        end="",
                        flush=True,
                    )

        temporary_path.replace(destination)
        print(f"\nDownloaded archive: {destination}")
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def extract_files(
    archive: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    """Extract the files required by the experiment."""
    with zipfile.ZipFile(archive) as source_archive:
        for member_name in DATA_FILES:
            destination = output_dir / member_name
            if destination.exists() and not force:
                print(f"Dataset file already exists: {destination}")
                continue

            temporary_path = destination.with_suffix(destination.suffix + ".part")
            try:
                with (
                    source_archive.open(member_name) as source,
                    temporary_path.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                temporary_path.replace(destination)
                print(f"Extracted dataset file: {destination}")
            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run the download and extraction workflow."""
    args = parse_args()
    archive_path = args.output_dir / ARCHIVE_NAME

    try:
        download_file(DATASET_URL, archive_path, force=args.force)
        extract_files(archive_path, args.output_dir, force=args.force)
    except (OSError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"Failed to prepare the dataset: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
