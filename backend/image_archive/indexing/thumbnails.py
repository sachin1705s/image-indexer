"""Thumbnail generation and caching."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image, ImageOps


def generate_thumbnail(
    source_path: Union[str, Path],
    thumbnail_dir: Union[str, Path],
    cache_key: str,
    *,
    size: tuple[int, int] = (512, 512),
) -> str:
    """Generate or reuse a cached JPEG thumbnail."""

    source = Path(source_path).expanduser().resolve()
    target_dir = Path(thumbnail_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{cache_key[:24]}.jpg"
    if target.exists():
        return str(target)

    with Image.open(source) as image:
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        prepared.thumbnail(size)
        prepared.save(target, format="JPEG", quality=85, optimize=True)
    return str(target)
