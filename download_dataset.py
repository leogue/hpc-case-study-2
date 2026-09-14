#!/usr/bin/env python3
"""Download and extract the SCANIA Component X dataset from Kaggle."""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "mariembenkamel/scania-component-x-dataset"
)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data"
EXPECTED_FILES = {
    "test_labels.csv",
    "test_operational_readouts.csv",
    "test_specifications.csv",
    "train_operational_readouts.csv",
    "train_specifications.csv",
    "train_tte.csv",
    "validation_labels.csv",
    "validation_operational_readouts.csv",
    "validation_specifications.csv",
}


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": "scania-component-x-downloader/1.0"}
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        credentials = base64.b64encode(f"{username}:{key}".encode()).decode()
        headers["Authorization"] = f"Basic {credentials}"
    return headers


def download(archive: Path) -> None:
    request = urllib.request.Request(DATASET_URL, headers=request_headers())
    print(f"Téléchargement depuis Kaggle vers {archive.parent}…")

    with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                status = f"{format_size(downloaded)} / {format_size(total)} ({percent:.1f} %)"
            else:
                status = format_size(downloaded)
            print(f"\r{status}", end="", flush=True)
    print()


def archive_prefix(archive: zipfile.ZipFile) -> str | None:
    """Return the common top-level directory, if every file has one."""
    file_parts = [
        PurePosixPath(info.filename).parts
        for info in archive.infolist()
        if not info.is_dir()
    ]
    if file_parts and all(len(parts) > 1 for parts in file_parts):
        first = file_parts[0][0]
        if all(parts[0] == first for parts in file_parts):
            return first
    return None


def extract(archive_path: Path, output_dir: Path) -> None:
    print(f"Extraction dans {output_dir}…")
    output_root = output_dir.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise zipfile.BadZipFile(f"fichier corrompu dans l'archive : {bad_file}")

        prefix = archive_prefix(archive)
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if prefix and parts and parts[0] == prefix:
                parts = parts[1:]
            if not parts:
                continue

            target = output_dir.joinpath(*parts).resolve()
            if not target.is_relative_to(output_root):
                raise zipfile.BadZipFile(f"chemin non sûr dans l'archive : {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)


def missing_expected_files(output_dir: Path) -> set[str]:
    return {name for name in EXPECTED_FILES if not (output_dir / name).is_file()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge et extrait le dataset SCANIA Component X sans cache externe."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"dossier de destination (défaut : {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="retélécharger même si tous les CSV sont déjà présents",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="conserver l'archive ZIP dans le dossier de destination",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output.expanduser().resolve()

    if not args.force and output_dir.is_dir() and not missing_expected_files(output_dir):
        print(f"Le dataset est déjà présent dans {output_dir}")
        print("Utilisez --force pour le télécharger à nouveau.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="scania-component-x-", suffix=".zip", dir=output_dir
    )
    os.close(descriptor)
    archive_path = Path(temporary_name)

    try:
        download(archive_path)
        extract(archive_path, output_dir)
        missing = missing_expected_files(output_dir)
        if missing:
            names = ", ".join(sorted(missing))
            raise RuntimeError(f"fichiers attendus absents après extraction : {names}")

        if args.keep_archive:
            kept_archive = output_dir / "scania-component-x-dataset.zip"
            archive_path.replace(kept_archive)
            print(f"Archive conservée : {kept_archive}")
        print(f"Dataset disponible dans : {output_dir}")
        return 0
    except (OSError, urllib.error.URLError, zipfile.BadZipFile, RuntimeError) as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
