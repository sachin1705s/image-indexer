"""High-level service helpers used by CLI and API layers."""

from __future__ import annotations

from pathlib import Path

from image_archive.config import AppConfig, ConfigManager
from image_archive.db import Database
from image_archive.faiss_store import FaissIndexStore
from image_archive.models.factory import build_embedding_model, build_enrichment_model


def initialize_app_state(config_manager: ConfigManager, config: AppConfig) -> None:
    """Create app directories, DB schema, and FAISS files."""

    paths = config_manager.ensure_directories(config)
    Database(paths["sqlite_path"]).initialize()
    FaissIndexStore(paths["faiss_index_path"], config.embedding_dimension).save()


def build_runtime(
    config_manager: ConfigManager,
    config: AppConfig,
) -> tuple[Database, FaissIndexStore]:
    """Build common stateful runtime services."""

    paths = config_manager.ensure_directories(config)
    database = Database(paths["sqlite_path"])
    database.initialize()
    vector_store = FaissIndexStore(paths["faiss_index_path"], config.embedding_dimension)
    return database, vector_store


def normalize_index_paths(paths: list[str]) -> list[str]:
    """Normalize and deduplicate target paths."""

    normalized = [str(Path(item).expanduser().resolve()) for item in paths]
    return list(dict.fromkeys(normalized))


def build_models(config: AppConfig):
    """Instantiate enrichment and embedding backends."""

    embedding = build_embedding_model(config)
    return embedding, build_enrichment_model(config, embedding)


def build_embedding_backend(config: AppConfig):
    """Instantiate only the embedding backend."""

    return build_embedding_model(config)


def build_enrichment_backend(config: AppConfig):
    """Instantiate only the enrichment backend."""

    return build_enrichment_model(config)
