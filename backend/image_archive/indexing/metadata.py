"""Image metadata extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from PIL import Image


def _timestamp_to_iso(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


@dataclass
class ImageMetadata:
    """Normalized metadata for a file."""

    file_type: str
    file_size: int
    width: int
    height: int
    created_at: Optional[str]
    modified_at: Optional[str]
    folder: str


def extract_image_metadata(path: Union[str, Path]) -> ImageMetadata:
    """Read dimensions and filesystem timestamps."""

    file_path = Path(path).expanduser().resolve()
    stat = file_path.stat()
    with Image.open(file_path) as image:
        width, height = image.size

    created_raw = getattr(stat, "st_birthtime", None)
    modified_raw = getattr(stat, "st_mtime", None)
    return ImageMetadata(
        file_type=file_path.suffix.lower().lstrip("."),
        file_size=int(stat.st_size),
        width=int(width),
        height=int(height),
        created_at=_timestamp_to_iso(created_raw),
        modified_at=_timestamp_to_iso(modified_raw),
        folder=str(file_path.parent),
    )
