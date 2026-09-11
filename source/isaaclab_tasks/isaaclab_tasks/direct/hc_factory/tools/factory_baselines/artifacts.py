"""Verified file archives for reusing existing experiment directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def archive_files(directory: Path, names: list[str], archive_name: str) -> Path:
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    if Path(archive_name).name != archive_name or archive_name in names:
        raise ValueError("Archive must be a distinct filename in the existing directory")
    if any(Path(name).name != name for name in names) or len(set(names)) != len(names):
        raise ValueError("Archive inputs must be unique direct-child filenames")
    paths = [directory / name for name in names if (directory / name).exists()]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Only ordinary files may be archived")
    if not paths:
        raise ValueError("No files to archive")
    hashes = {}
    archive_path = directory / archive_name
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
            archive.write(path, path.name)
        archive.writestr("archive_manifest.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        for path in paths:
            with archive.open(path.name) as stream:
                saved = hashlib.file_digest(stream, "sha256").hexdigest()
            with path.open("rb") as stream:
                current = hashlib.file_digest(stream, "sha256").hexdigest()
            if saved != hashes[path.name] or current != saved:
                raise RuntimeError(f"Archive verification failed; originals retained: {path}")
    for path in paths:
        path.unlink()
    return archive_path
