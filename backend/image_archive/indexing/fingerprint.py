"""Hashing and signature helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def file_sha256(path: Union[str, Path]) -> str:
    """Compute a SHA256 hash for a file."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stat_signature(path: Union[str, Path]) -> str:
    """Return a cheap change detector based on file size and mtime."""

    stat = Path(path).stat()
    return f"{stat.st_size}:{int(stat.st_mtime_ns)}"
