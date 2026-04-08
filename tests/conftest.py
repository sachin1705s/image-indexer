from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from PIL import Image

from image_archive.config import AppConfig, ConfigManager
from image_archive.db import Database
from image_archive.types import EnrichmentResult


class InMemoryVectorStore:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.vectors: dict[int, np.ndarray] = {}

    def add_or_update(self, item_id: int, vector: np.ndarray) -> None:
        value = np.asarray(vector, dtype=np.float32)
        value = value / np.clip(np.linalg.norm(value), a_min=1e-12, a_max=None)
        self.vectors[item_id] = value

    def search(self, query_vector: np.ndarray, *, top_k: int, exclude_ids: Optional[list[int]] = None):
        query = np.asarray(query_vector, dtype=np.float32)
        query = query / np.clip(np.linalg.norm(query), a_min=1e-12, a_max=None)
        exclude = set(exclude_ids or [])
        pairs = []
        for item_id, vector in self.vectors.items():
            if item_id in exclude:
                continue
            pairs.append((item_id, float(np.dot(query, vector))))
        pairs.sort(key=lambda item: item[1], reverse=True)
        return pairs[:top_k]

    def save(self) -> None:
        return None


class FakeEmbeddingModel:
    model_name = "fake-clip"
    dimension = 4

    def encode_images(self, images):
        vectors = []
        for image in images:
            array = np.asarray(image.resize((1, 1))).astype(np.float32)[0, 0]
            red, green, blue = array[:3]
            vector = np.array([red + 1, green + 1, blue + 1, red + green + blue + 1], dtype=np.float32)
            vector = vector / np.clip(np.linalg.norm(vector), a_min=1e-12, a_max=None)
            vectors.append(vector)
        return np.stack(vectors)

    def encode_texts(self, texts):
        vectors = []
        for text in texts:
            value = text.lower()
            vector = np.array(
                [
                    1.0 if "red" in value or "warm" in value else 0.2,
                    1.0 if "green" in value or "forest" in value else 0.2,
                    1.0 if "blue" in value or "cool" in value else 0.2,
                    1.0 if "portrait" in value or "editorial" in value else 0.2,
                ],
                dtype=np.float32,
            )
            vector = vector / np.clip(np.linalg.norm(vector), a_min=1e-12, a_max=None)
            vectors.append(vector)
        return np.stack(vectors)


@dataclass
class FakeEnrichmentModel:
    model_name: str = "fake-clip-zero-shot"
    enrichment_version: str = "fake-v1"
    calls: list[str] = field(default_factory=list)

    def enrich_images(self, images, *, mode: str = "fast"):
        self.calls.append(mode)
        results = []
        for image in images:
            color = np.asarray(image.resize((1, 1))).astype(np.uint8)[0, 0]
            red, green, blue = color[:3]
            if blue >= red and blue >= green:
                results.append(
                    EnrichmentResult(
                        content_type="screenshot" if mode == "fast" else "document",
                        auto_tags=["blue", "layout", "screen"],
                        style_labels=["minimal", "cool"],
                        object_labels=["window", "panel"],
                        has_person=False,
                        has_face=False,
                        person_count_bucket="0",
                        face_count_bucket="0",
                        is_document=(mode == "document"),
                        is_screenshot=True,
                        text_heavy=True,
                        short_summary="blue screen-like composition",
                    )
                )
            elif red >= green and red >= blue:
                results.append(
                    EnrichmentResult(
                        content_type="photo",
                        auto_tags=["red", "portrait", "warm"],
                        style_labels=["editorial", "warm-light"],
                        object_labels=["person"],
                        has_person=True,
                        has_face=True,
                        person_count_bucket="1",
                        face_count_bucket="1",
                        short_summary="warm red portrait",
                    )
                )
            else:
                results.append(
                    EnrichmentResult(
                        content_type="photo",
                        auto_tags=["green", "botanical"],
                        style_labels=["natural", "textured"],
                        object_labels=["plant"],
                        has_person=False,
                        has_face=False,
                        person_count_bucket="0",
                        face_count_bucket="0",
                        short_summary="green botanical scene",
                    )
                )
        return results


@dataclass
class TestState:
    root: Path
    config_path: Path
    config_manager: ConfigManager
    config: AppConfig
    database: Database
    vector_store: InMemoryVectorStore
    embedding_model: FakeEmbeddingModel
    enrichment_model: FakeEnrichmentModel


@pytest.fixture()
def test_state(tmp_path: Path) -> TestState:
    config_path = tmp_path / "config.yaml"
    config_manager = ConfigManager(config_path)
    config = AppConfig(
        app_data_dir=".image-archive",
        thumbnail_dir=".image-archive/thumbnails",
        sqlite_path=".image-archive/archive.db",
        faiss_index_path=".image-archive/index.faiss",
        upload_dir=".image-archive/uploads",
        batch_size=2,
        embedding_dimension=4,
        indexed_paths=[],
        enrichment_version="fake-v1",
        enrichment_backend="clip",
        enrichment_model="fake-clip",
        enrichment_mode="fast",
    )
    config_manager.save(config)
    resolved = config_manager.ensure_directories(config)
    database = Database(resolved["sqlite_path"])
    database.initialize()
    return TestState(
        root=tmp_path,
        config_path=config_path,
        config_manager=config_manager,
        config=config,
        database=database,
        vector_store=InMemoryVectorStore(dimension=4),
        embedding_model=FakeEmbeddingModel(),
        enrichment_model=FakeEnrichmentModel(),
    )


@pytest.fixture()
def image_factory(tmp_path: Path):
    def _create(name: str, color: tuple[int, int, int]) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), color=color).save(path)
        return path

    return _create
