"""Configuration loading and persistence."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from image_archive.constants import (
    APP_NAME,
    DEFAULT_CONFIG_FILENAME,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_PIPELINE_VERSION,
)


def default_data_dir() -> Path:
    """Return the per-user app data directory for an installed CLI."""

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")) / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / APP_NAME


def default_config_dir() -> Path:
    """Return the per-user config directory for an installed CLI."""

    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")) / APP_NAME


def default_config_path() -> Path:
    """Return the default config file path."""

    return default_config_dir() / DEFAULT_CONFIG_FILENAME


@dataclass
class AppConfig:
    """User-editable application configuration."""

    app_data_dir: str = field(default_factory=lambda: str(default_data_dir()))
    indexed_paths: list[str] = field(default_factory=list)
    thumbnail_dir: str = field(default_factory=lambda: str(default_data_dir() / "thumbnails"))
    sqlite_path: str = field(default_factory=lambda: str(default_data_dir() / "archive.db"))
    faiss_index_path: str = field(default_factory=lambda: str(default_data_dir() / "index.faiss"))
    upload_dir: str = field(default_factory=lambda: str(default_data_dir() / "uploads"))
    embedding_model_name: str = "openai/clip-vit-base-patch32"
    enrichment_backend: str = "clip"
    enrichment_model: str = "openai/clip-vit-base-patch32"
    enrichment_mode: str = "fast"
    enrichment_version: str = "clip-zero-shot-v1"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: int = 120
    fast_image_max_size: int = 448
    document_image_max_size: int = 1024
    excluded_dir_names: list[str] = field(
        default_factory=lambda: [
            ".git",
            "__pycache__",
            "node_modules",
            "photo library.photoslibrary",
            ".photoslibrary",
            "derivatives",
            "rendered",
            "proxies",
            "thumbnails",
            "previews",
            "frames",
            "caches",
            "cache",
        ]
    )
    excluded_path_fragments: list[str] = field(
        default_factory=lambda: [
            ".photoslibrary/resources/derivatives",
            ".photoslibrary/resources/rendered",
            ".photoslibrary/resources/proxies",
            "/video frames/",
            "/video thumbnails/",
        ]
    )
    device: str = "auto"
    batch_size: int = 8
    num_workers: int = 4
    host: str = "127.0.0.1"
    port: int = 8000
    text_search_min_score: float = 0.24
    low_information_max_score: float = 0.30
    pipeline_version: str = DEFAULT_PIPELINE_VERSION
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION

    def with_indexed_path(self, path: str) -> "AppConfig":
        """Return a config with the path registered once."""

        resolved = str(Path(path).expanduser().resolve())
        items = list(dict.fromkeys([*self.indexed_paths, resolved]))
        updated = asdict(self)
        updated["indexed_paths"] = items
        return AppConfig(**updated)


class ConfigManager:
    """Load, resolve, and persist config files."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path).expanduser().resolve() if path else default_config_path()

    def default(self) -> AppConfig:
        return AppConfig()

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> AppConfig:
        if not self.exists():
            return self.default()

        with self.path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        valid_keys = {item.name for item in fields(AppConfig)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return AppConfig(**filtered)

    def save(self, config: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(asdict(config), handle, sort_keys=False)

    def resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return (self.path.parent / path).resolve()

    def ensure_directories(self, config: AppConfig) -> dict[str, Path]:
        paths = {
            "app_data_dir": self.resolve_path(config.app_data_dir),
            "thumbnail_dir": self.resolve_path(config.thumbnail_dir),
            "upload_dir": self.resolve_path(config.upload_dir),
            "sqlite_path": self.resolve_path(config.sqlite_path),
            "faiss_index_path": self.resolve_path(config.faiss_index_path),
        }

        paths["app_data_dir"].mkdir(parents=True, exist_ok=True)
        paths["thumbnail_dir"].mkdir(parents=True, exist_ok=True)
        paths["upload_dir"].mkdir(parents=True, exist_ok=True)
        paths["sqlite_path"].parent.mkdir(parents=True, exist_ok=True)
        paths["faiss_index_path"].parent.mkdir(parents=True, exist_ok=True)
        return paths

    def summary(self, config: AppConfig) -> dict[str, Any]:
        return {
            "config_path": str(self.path),
            "sqlite_path": str(self.resolve_path(config.sqlite_path)),
            "faiss_index_path": str(self.resolve_path(config.faiss_index_path)),
            "thumbnail_dir": str(self.resolve_path(config.thumbnail_dir)),
            "indexed_paths": [str(Path(item).expanduser().resolve()) for item in config.indexed_paths],
        }
