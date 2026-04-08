"""Search services for text and image similarity retrieval."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

from image_archive.config import AppConfig
from image_archive.db import Database
from image_archive.faiss_store import VectorStore
from image_archive.models.base import EmbeddingModel


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromisoformat(f"{value}T00:00:00")
        except ValueError:
            return None


def _split_csv(value: Any) -> list[str]:
    if not value:
        return []
    return [item for item in str(value).split(",") if item]


def _score_search_text(search_text: Optional[str], query_terms: list[str]) -> float:
    if not search_text:
        return 0.0
    text = search_text.lower()
    matches = sum(1 for term in query_terms if term in text)
    return matches * 0.08


LOW_INFORMATION_QUERY_TERMS = {
    "blank",
    "black",
    "white",
    "minimal",
    "minimalist",
    "slide",
    "slides",
    "screenshot",
    "screenshots",
    "screen",
    "ui",
    "document",
    "workspace",
    "background",
    "gradient",
}


@lru_cache(maxsize=4096)
def _is_low_information_image(image_path: str) -> bool:
    """Detect mostly-empty generated slides without needing a reindex."""

    path = Path(image_path)
    if not path.exists():
        return False
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB").resize((64, 64))
        array = np.asarray(image, dtype=np.float32)
    except Exception:
        return False

    gray = array.mean(axis=2)
    contrast = float(gray.std())
    edge_energy = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    return contrast < 8.0 and edge_energy < 1.5


class SearchService:
    """Combined metadata + vector search behavior."""

    def __init__(
        self,
        *,
        config: AppConfig,
        database: Database,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.config = config
        self.database = database
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def _matches_filters(
        self,
        row: dict[str, Any],
        *,
        folder: Optional[str],
        date_from: Optional[str],
        date_to: Optional[str],
        content_type: Optional[str],
    ) -> bool:
        if folder and not str(row["file_path"]).startswith(folder):
            return False
        if content_type and row.get("content_type") != content_type:
            return False

        modified_at = _parse_date(row.get("modified_at"))
        lower = _parse_date(date_from)
        upper = _parse_date(date_to)
        if lower and modified_at and modified_at < lower:
            return False
        if upper and modified_at and modified_at > upper:
            return False
        return True

    def _serialize_asset(self, row: dict[str, Any], score: float = 0.0) -> dict[str, Any]:
        asset_id = int(row["id"])
        return {
            "id": asset_id,
            "file_path": row["file_path"],
            "file_hash": row.get("file_hash"),
            "file_type": row.get("file_type"),
            "width": row.get("width"),
            "height": row.get("height"),
            "created_at": row.get("created_at"),
            "modified_at": row.get("modified_at"),
            "indexed_at": row.get("indexed_at"),
            "folder": row.get("folder"),
            "thumbnail_path": row.get("thumbnail_path"),
            "summary": row.get("short_summary") or row.get("caption"),
            "content_type": row.get("content_type") or "other",
            "auto_tags": _split_csv(row.get("auto_tags")),
            "style_labels": _split_csv(row.get("style_labels")),
            "object_labels": _split_csv(row.get("object_labels")),
            "has_person": bool(row.get("has_person")),
            "has_face": bool(row.get("has_face")),
            "person_count_bucket": row.get("person_count_bucket") or "0",
            "face_count_bucket": row.get("face_count_bucket") or "0",
            "is_document": bool(row.get("is_document")),
            "is_screenshot": bool(row.get("is_screenshot")),
            "text_heavy": bool(row.get("text_heavy")),
            "thumbnail_url": f"/assets/{asset_id}/thumbnail",
            "image_url": f"/assets/{asset_id}/image",
            "score": round(float(score), 6),
        }

    def text_search(
        self,
        query: str,
        *,
        folder: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        content_type: Optional[str] = None,
        top_k: int = 24,
    ) -> list[dict[str, Any]]:
        text_vector = self.embedding_model.encode_texts([query])[0]
        query_terms = [term.lower() for term in query.split() if len(term) > 2]

        vector_hits = self.vector_store.search(text_vector, top_k=max(top_k * 8, top_k))
        merged_scores: dict[int, float] = {asset_id: score for asset_id, score in vector_hits}

        keyword_rows = self.database.list_search_matches(
            query_terms,
            folder_prefix=folder,
            date_from=date_from,
            date_to=date_to,
            content_type=content_type,
            limit=max(top_k * 4, top_k),
        )
        for row in keyword_rows:
            merged_scores[int(row["id"])] = merged_scores.get(int(row["id"]), 0.0) + _score_search_text(
                row["search_text"], query_terms
            )

        ordered_ids = sorted(merged_scores, key=lambda item: merged_scores[item], reverse=True)
        rows = [dict(item) for item in self.database.get_assets_by_ids(ordered_ids)]

        results: list[dict[str, Any]] = []
        for row in rows:
            if not self._matches_filters(
                row,
                folder=folder,
                date_from=date_from,
                date_to=date_to,
                content_type=content_type,
            ):
                continue
            asset_id = int(row["id"])
            score = merged_scores.get(asset_id, 0.0) + _score_search_text(row.get("search_text"), query_terms)
            if score < self.config.text_search_min_score:
                continue
            if (
                score < self.config.low_information_max_score
                and not any(term in LOW_INFORMATION_QUERY_TERMS for term in query_terms)
                and _is_low_information_image(row.get("thumbnail_path") or row.get("file_path") or "")
            ):
                continue
            results.append(self._serialize_asset(row, score=score))
            if len(results) >= top_k:
                break
        return results

    def asset_similar_search(self, asset_id: int, *, top_k: int = 24) -> list[dict[str, Any]]:
        embedding_row = self.database.get_embedding_row(asset_id)
        if not embedding_row or not embedding_row["vector_blob"]:
            raise ValueError(f"No stored embedding found for asset {asset_id}")

        vector = np.frombuffer(embedding_row["vector_blob"], dtype=np.float32)
        hits = self.vector_store.search(vector, top_k=top_k, exclude_ids=[asset_id])
        rows = [dict(item) for item in self.database.get_assets_by_ids([hit[0] for hit in hits])]
        score_lookup = {asset: score for asset, score in hits}
        return [self._serialize_asset(row, score=score_lookup.get(int(row["id"]), 0.0)) for row in rows]

    def uploaded_image_search(self, image_bytes: bytes, *, top_k: int = 24) -> list[dict[str, Any]]:
        with Image.open(BytesIO(image_bytes)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        vector = self.embedding_model.encode_images([image])[0]
        hits = self.vector_store.search(vector, top_k=top_k)
        rows = [dict(item) for item in self.database.get_assets_by_ids([hit[0] for hit in hits])]
        score_lookup = {asset: score for asset, score in hits}
        return [self._serialize_asset(row, score=score_lookup.get(int(row["id"]), 0.0)) for row in rows]

    def asset_detail(self, asset_id: int) -> dict[str, Any]:
        row = self.database.get_asset(asset_id)
        if row is None:
            raise KeyError(f"Unknown asset id {asset_id}")
        return self._serialize_asset(dict(row))

    def folders(self) -> list[str]:
        return self.config.indexed_paths

    def stats(self) -> dict[str, Any]:
        details = self.database.stats()
        return {
            "asset_count": details["asset_count"],
            "index_dimension": self.vector_store.dimension,
            "indexed_paths": self.config.indexed_paths,
            "last_run": details["last_run"],
        }
