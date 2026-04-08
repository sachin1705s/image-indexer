"""File discovery helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

from image_archive.constants import SUPPORTED_EXTENSIONS


def is_supported_image(path: Path) -> bool:
    """Return True when the path extension is supported."""

    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _normalize_fragment(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def should_exclude_path(
    path: Path,
    *,
    excluded_dir_names: Optional[set[str]] = None,
    excluded_path_fragments: Optional[list[str]] = None,
) -> bool:
    """Return True when a file or directory path matches exclusion rules."""

    dir_names = excluded_dir_names or set()
    fragments = excluded_path_fragments or []
    parts = {part.lower() for part in path.parts}
    if parts.intersection(dir_names):
        return True
    normalized = _normalize_fragment(str(path))
    return any(fragment in normalized for fragment in fragments)


def discover_image_files(
    paths: Iterable[Union[str, Path]],
    *,
    excluded_dir_names: Optional[Iterable[str]] = None,
    excluded_path_fragments: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Recursively discover supported image files."""

    excluded_dirs = {_normalize_fragment(item) for item in (excluded_dir_names or [])}
    excluded_fragments = [_normalize_fragment(item) for item in (excluded_path_fragments or [])]
    discovered: list[Path] = []
    for root in paths:
        root_path = Path(root).expanduser().resolve()
        if root_path.is_file() and is_supported_image(root_path) and not should_exclude_path(
            root_path,
            excluded_dir_names=excluded_dirs,
            excluded_path_fragments=excluded_fragments,
        ):
            discovered.append(root_path)
            continue
        if not root_path.exists():
            continue
        for current_root, dirs, files in os.walk(root_path):
            current_path = Path(current_root).resolve()
            dirs[:] = [
                item
                for item in dirs
                if not should_exclude_path(
                    current_path / item,
                    excluded_dir_names=excluded_dirs,
                    excluded_path_fragments=excluded_fragments,
                )
            ]
            for file_name in files:
                candidate = (current_path / file_name).resolve()
                if not is_supported_image(candidate):
                    continue
                if should_exclude_path(
                    candidate,
                    excluded_dir_names=excluded_dirs,
                    excluded_path_fragments=excluded_fragments,
                ):
                    continue
                discovered.append(candidate)
    return sorted(dict.fromkeys(discovered))
