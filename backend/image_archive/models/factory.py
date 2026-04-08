"""Model factory helpers."""

from __future__ import annotations

from typing import Optional

from image_archive.config import AppConfig
from image_archive.models.base import EnrichmentModel, EmbeddingModel
from image_archive.models.local import ClipEmbeddingModel, ClipZeroShotEnrichmentModel
from image_archive.models.ollama import OllamaGemma4E2BEnrichmentModel


def build_embedding_model(config: AppConfig) -> EmbeddingModel:
    """Instantiate the configured embedding model."""

    return ClipEmbeddingModel(
        model_name=config.embedding_model_name,
        device=config.device,
        dimension=config.embedding_dimension,
    )


def build_enrichment_model(config: AppConfig, embedding_model: Optional[EmbeddingModel] = None) -> EnrichmentModel:
    """Instantiate the configured structured enrichment backend."""

    if config.enrichment_backend == "clip":
        if embedding_model is not None and isinstance(embedding_model, ClipEmbeddingModel):
            clip_model = embedding_model
        else:
            clip_model = ClipEmbeddingModel(
                model_name=config.embedding_model_name,
                device=config.device,
                dimension=config.embedding_dimension,
            )
        return ClipZeroShotEnrichmentModel(clip_model, enrichment_version=config.enrichment_version)
    if config.enrichment_backend == "ollama":
        return OllamaGemma4E2BEnrichmentModel(config)
    raise ValueError(f"Unsupported enrichment backend: {config.enrichment_backend}")
