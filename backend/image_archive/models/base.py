"""Abstract model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from PIL import Image

from image_archive.types import EnrichmentResult


class EmbeddingModel(ABC):
    """Interface for image and text embedding backends."""

    model_name: str
    dimension: int

    @abstractmethod
    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return one embedding per image."""

    @abstractmethod
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return one embedding per input text."""


class CaptionModel(ABC):
    """Interface for local caption generators."""

    model_name: str

    @abstractmethod
    def caption_images(self, images: Sequence[Image.Image]) -> list[str]:
        """Return one caption per image."""


class EnrichmentModel(ABC):
    """Interface for structured image enrichment backends."""

    model_name: str
    enrichment_version: str

    @abstractmethod
    def enrich_images(self, images: Sequence[Image.Image], *, mode: str = "fast") -> list[EnrichmentResult]:
        """Return one enrichment payload per image."""
