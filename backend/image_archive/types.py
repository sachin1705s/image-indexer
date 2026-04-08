"""Shared types for enrichment, timing, and search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

ALLOWED_CONTENT_TYPES = {
    "photo",
    "document",
    "screenshot",
    "ui",
    "illustration",
    "scan",
    "other",
}
COUNT_BUCKETS = {"0", "1", "2-3", "4+"}


def _clean_label(value: str) -> str:
    cleaned = value.strip().lower().replace("_", "-")
    return "-".join(part for part in cleaned.split() if part)


def normalize_labels(values: list[str], *, limit: int = 8) -> list[str]:
    """Normalize short labels for tags and styles."""

    output: list[str] = []
    for item in values:
        label = _clean_label(str(item))
        if not label or label in output:
            continue
        output.append(label)
        if len(output) >= limit:
            break
    return output


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, str):
        if "," in value:
            return [item.strip() for item in value.split(",")]
        return [value]
    return [str(value)]


def normalize_content_type(value: str) -> str:
    """Clamp content types to the supported taxonomy."""

    normalized = _clean_label(value)
    return normalized if normalized in ALLOWED_CONTENT_TYPES else "other"


def normalize_count_bucket(value: Any) -> str:
    """Normalize person and face count buckets."""

    raw = str(value).strip().lower()
    if raw in COUNT_BUCKETS:
        return raw
    if raw in {"none", "zero"}:
        return "0"
    if raw in {"one", "single"}:
        return "1"
    if raw in {"2", "3", "2-3", "few"}:
        return "2-3"
    if raw in {"4", "4+", "many"}:
        return "4+"
    return "0"


def _prefer_content_type(primary: str, secondary: str) -> str:
    if primary == "screenshot" and secondary in {"document", "ui", "other"}:
        return primary
    if primary == "ui" and secondary == "document":
        return primary
    return secondary or primary


def coerce_bool(value: Any) -> bool:
    """Best-effort boolean coercion for model output."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class EnrichmentResult:
    """Structured metadata emitted by the enrichment model."""

    content_type: str = "other"
    auto_tags: list[str] = field(default_factory=list)
    style_labels: list[str] = field(default_factory=list)
    object_labels: list[str] = field(default_factory=list)
    has_person: bool = False
    has_face: bool = False
    person_count_bucket: str = "0"
    face_count_bucket: str = "0"
    is_document: bool = False
    is_screenshot: bool = False
    text_heavy: bool = False
    short_summary: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EnrichmentResult":
        """Normalize arbitrary model JSON into the expected structure."""

        object_labels = _coerce_list(payload.get("object_labels") or payload.get("objects"))
        style_labels = _coerce_list(payload.get("style_labels") or payload.get("styles"))
        auto_tags = _coerce_list(payload.get("auto_tags") or payload.get("tags"))

        return cls(
            content_type=normalize_content_type(str(payload.get("content_type", "other"))),
            auto_tags=normalize_labels(auto_tags),
            style_labels=normalize_labels(style_labels),
            object_labels=normalize_labels(object_labels),
            has_person=coerce_bool(payload.get("has_person")),
            has_face=coerce_bool(payload.get("has_face")),
            person_count_bucket=normalize_count_bucket(payload.get("person_count_bucket", "0")),
            face_count_bucket=normalize_count_bucket(payload.get("face_count_bucket", "0")),
            is_document=coerce_bool(payload.get("is_document")),
            is_screenshot=coerce_bool(payload.get("is_screenshot")),
            text_heavy=coerce_bool(payload.get("text_heavy")),
            short_summary=str(payload.get("short_summary", "")).strip()[:240],
            raw_response=payload,
        )

    def merge(self, other: "EnrichmentResult") -> "EnrichmentResult":
        """Merge a higher-detail pass over an initial result."""

        return EnrichmentResult(
            content_type=_prefer_content_type(self.content_type, other.content_type),
            auto_tags=normalize_labels(self.auto_tags + other.auto_tags, limit=12),
            style_labels=normalize_labels(self.style_labels + other.style_labels, limit=10),
            object_labels=normalize_labels(self.object_labels + other.object_labels, limit=12),
            has_person=self.has_person or other.has_person,
            has_face=self.has_face or other.has_face,
            person_count_bucket=other.person_count_bucket or self.person_count_bucket,
            face_count_bucket=other.face_count_bucket or self.face_count_bucket,
            is_document=self.is_document or other.is_document,
            is_screenshot=self.is_screenshot or other.is_screenshot,
            text_heavy=self.text_heavy or other.text_heavy,
            short_summary=other.short_summary or self.short_summary,
            raw_response={**self.raw_response, **other.raw_response},
        )

    def requires_document_pass(self) -> bool:
        """Return True when a higher-detail pass is likely worthwhile."""

        return (
            self.content_type in {"document", "screenshot", "scan", "ui"}
            or self.is_document
            or self.is_screenshot
            or self.text_heavy
        )

    def search_text(self) -> str:
        """Build denormalized searchable text from structured fields."""

        parts = [
            self.content_type,
            *self.auto_tags,
            *self.style_labels,
            *self.object_labels,
            self.short_summary,
        ]
        if self.has_person:
            parts.append("person")
        if self.has_face:
            parts.append("face")
        if self.is_document:
            parts.append("document")
        if self.is_screenshot:
            parts.append("screenshot")
        if self.text_heavy:
            parts.append("text-heavy")
        return " ".join(part for part in parts if part).strip()

    def to_record(self) -> dict[str, Any]:
        """Serialize to database-friendly primitives."""

        return {
            "content_type": self.content_type,
            "auto_tags": ",".join(self.auto_tags),
            "style_labels": ",".join(self.style_labels),
            "object_labels": ",".join(self.object_labels),
            "has_person": int(self.has_person),
            "has_face": int(self.has_face),
            "person_count_bucket": self.person_count_bucket,
            "face_count_bucket": self.face_count_bucket,
            "is_document": int(self.is_document),
            "is_screenshot": int(self.is_screenshot),
            "text_heavy": int(self.text_heavy),
            "short_summary": self.short_summary,
            "search_text": self.search_text(),
            "raw_json": self.raw_response,
        }


@dataclass
class PipelineTimings:
    """Per-stage runtime summary for an indexing run."""

    scan_seconds: float = 0.0
    prep_seconds: float = 0.0
    embedding_seconds: float = 0.0
    enrichment_seconds: float = 0.0
    persist_seconds: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "scan_seconds": round(self.scan_seconds, 3),
            "prep_seconds": round(self.prep_seconds, 3),
            "embedding_seconds": round(self.embedding_seconds, 3),
            "enrichment_seconds": round(self.enrichment_seconds, 3),
            "persist_seconds": round(self.persist_seconds, 3),
        }

    @property
    def total_seconds(self) -> float:
        return (
            self.scan_seconds
            + self.prep_seconds
            + self.embedding_seconds
            + self.enrichment_seconds
            + self.persist_seconds
        )

    def throughput_per_minute(self, indexed_count: int) -> float:
        if indexed_count <= 0 or self.total_seconds <= 0:
            return 0.0
        return round(indexed_count / (self.total_seconds / 60.0), 2)


Vector = np.ndarray
