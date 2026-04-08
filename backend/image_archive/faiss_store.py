"""FAISS vector store wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np


class VectorStore(Protocol):
    """Protocol used by the search and indexing services."""

    dimension: int

    def add_or_update(self, item_id: int, vector: np.ndarray) -> None:
        ...

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        exclude_ids: Optional[list[int]] = None,
    ) -> list[tuple[int, float]]:
        ...

    def save(self) -> None:
        ...


class FaissIndexStore:
    """Persistent FAISS index using asset ids as vector ids."""

    def __init__(self, path: Union[str, Path], dimension: int) -> None:
        self.path = Path(path).expanduser().resolve()
        self.dimension = dimension
        self.meta_path = self.path.with_suffix(".json")
        self._faiss = self._import_faiss()
        self.index = self._load_or_create()

    def _import_faiss(self):  # type: ignore[no-untyped-def]
        try:
            import faiss  # pylint: disable=import-outside-toplevel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "faiss-cpu is required for vector search. Install project dependencies first."
            ) from exc
        return faiss

    def _create_index(self):  # type: ignore[no-untyped-def]
        return self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.dimension))

    def _load_or_create(self):  # type: ignore[no-untyped-def]
        if self.path.exists():
            index = self._faiss.read_index(str(self.path))
            stored_dim = self._read_dimension()
            if stored_dim and stored_dim != self.dimension:
                raise ValueError(
                    f"FAISS index dimension mismatch: expected {self.dimension}, found {stored_dim}"
                )
            return index

        self.path.parent.mkdir(parents=True, exist_ok=True)
        index = self._create_index()
        self._write_meta()
        self._faiss.write_index(index, str(self.path))
        return index

    def _read_dimension(self) -> Optional[int]:
        if not self.meta_path.exists():
            return None
        data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return int(data.get("dimension", 0)) or None

    def _write_meta(self) -> None:
        payload = {"dimension": self.dimension}
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(value)
        return value

    def add_or_update(self, item_id: int, vector: np.ndarray) -> None:
        normalized = self._normalize(vector)
        self.remove(item_id)
        self.index.add_with_ids(normalized, np.asarray([item_id], dtype=np.int64))

    def remove(self, item_id: int) -> None:
        if getattr(self.index, "ntotal", 0) == 0:
            return
        self.index.remove_ids(np.asarray([item_id], dtype=np.int64))

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        exclude_ids: Optional[list[int]] = None,
    ) -> list[tuple[int, float]]:
        if getattr(self.index, "ntotal", 0) == 0:
            return []

        exclude = set(exclude_ids or [])
        search_k = min(max(top_k + len(exclude), top_k), int(self.index.ntotal))
        scores, ids = self.index.search(self._normalize(query_vector), search_k)
        results: list[tuple[int, float]] = []
        for asset_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if asset_id < 0 or asset_id in exclude:
                continue
            results.append((int(asset_id), float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_meta()
        self._faiss.write_index(self.index, str(self.path))
