"""Incremental indexing pipeline."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from image_archive.config import AppConfig, ConfigManager
from image_archive.db import Database, utc_now_iso
from image_archive.faiss_store import VectorStore
from image_archive.indexing.discovery import discover_image_files
from image_archive.indexing.fingerprint import file_sha256, stat_signature
from image_archive.indexing.metadata import ImageMetadata, extract_image_metadata
from image_archive.indexing.thumbnails import generate_thumbnail
from image_archive.models.base import EmbeddingModel, EnrichmentModel
from image_archive.types import EnrichmentResult, PipelineTimings


@dataclass
class PipelineStats:
    """Summary counters for an indexing run."""

    scanned: int = 0
    planned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    timings: PipelineTimings = field(default_factory=PipelineTimings)

    @property
    def avg_ms_per_asset(self) -> float:
        if self.indexed <= 0:
            return 0.0
        return round((self.timings.total_seconds / self.indexed) * 1000.0, 2)


@dataclass
class PendingAsset:
    """A discovered asset with its current DB state and work flags."""

    path: Path
    existing: Optional[dict]
    signature: str
    needs_hash: bool
    needs_metadata: bool
    needs_thumbnail: bool
    needs_embedding: bool
    needs_enrichment: bool

    @property
    def needs_processing(self) -> bool:
        return any(
            [
                self.needs_hash,
                self.needs_metadata,
                self.needs_thumbnail,
                self.needs_embedding,
                self.needs_enrichment,
            ]
        )


@dataclass
class PreparedAsset:
    """In-memory prepared data for persistence."""

    pending: PendingAsset
    image: Optional[Image.Image]
    metadata: Optional[ImageMetadata]
    file_hash: Optional[str]
    thumbnail_path: Optional[str]
    embedding: Optional[np.ndarray]
    enrichment: Optional[EnrichmentResult]


def _row_to_dict(row: Optional[object]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


class IndexingPipeline:
    """Incremental, resumable image indexing pipeline."""

    def __init__(
        self,
        *,
        config: AppConfig,
        config_manager: ConfigManager,
        database: Database,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        enrichment_model: EnrichmentModel,
    ) -> None:
        self.config = config
        self.config_manager = config_manager
        self.database = database
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.enrichment_model = enrichment_model
        resolved = config_manager.ensure_directories(config)
        self.thumbnail_dir = resolved["thumbnail_dir"]

    def run(self, paths: list[str]) -> PipelineStats:
        """Index one or more folders."""
        return self.run_with_options(paths)

    def run_with_options(self, paths: list[str], *, limit: Optional[int] = None) -> PipelineStats:
        """Index one or more folders with optional discovery limits."""

        target_paths = [str(Path(item).expanduser().resolve()) for item in paths]
        run_id = self.database.start_index_run(target_paths)
        stats = PipelineStats()
        status = "completed"
        try:
            scan_started = time.perf_counter()
            pending_assets = self._plan(target_paths, stats, limit=limit)
            stats.timings.scan_seconds += time.perf_counter() - scan_started
            stats.planned = len(pending_assets)
            tqdm.write(
                "Discovered "
                f"{stats.scanned} supported images. "
                f"To process: {stats.planned}. "
                f"Already current: {stats.skipped}."
            )
            if pending_assets:
                self._process_pending(pending_assets, stats)
                self.vector_store.save()
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except Exception:
            status = "failed"
            raise
        finally:
            self.database.finish_index_run(
                run_id,
                status=status,
                num_scanned=stats.scanned,
                num_indexed=stats.indexed,
                num_skipped=stats.skipped,
                num_failed=stats.failed,
                timings=stats.timings.to_dict(),
                avg_ms_per_asset=stats.avg_ms_per_asset,
                throughput_per_min=stats.timings.throughput_per_minute(stats.indexed),
            )
        return stats

    def _plan(self, paths: list[str], stats: PipelineStats, *, limit: Optional[int] = None) -> list[PendingAsset]:
        candidates = discover_image_files(
            paths,
            excluded_dir_names=self.config.excluded_dir_names,
            excluded_path_fragments=self.config.excluded_path_fragments,
        )
        if limit is not None and limit >= 0:
            candidates = candidates[:limit]
        pending: list[PendingAsset] = []

        for file_path in tqdm(candidates, desc="Scanning", unit="file"):
            stats.scanned += 1
            signature = stat_signature(file_path)
            existing = _row_to_dict(self.database.get_asset_by_path(str(file_path)))
            changed = existing is None or existing.get("stat_signature") != signature

            thumbnail_exists = bool(existing and existing.get("thumbnail_path") and Path(existing["thumbnail_path"]).exists())
            has_enrichment = bool(existing and existing.get("enrichment_model") and existing.get("search_text"))
            enrichment_stale = bool(existing) and (
                existing.get("enrichment_model") != self.enrichment_model.model_name
                or existing.get("enrichment_version") != self.enrichment_model.enrichment_version
                or existing.get("enrichment_mode") != self.config.enrichment_mode
            )
            embedding_stale = bool(existing) and existing.get("embedding_model") != self.config.embedding_model_name
            needs_metadata = changed or not (
                existing and existing.get("width") and existing.get("height") and existing.get("modified_at")
            )
            needs_hash = changed or not (existing and existing.get("file_hash"))
            needs_thumbnail = changed or not thumbnail_exists
            needs_embedding = changed or not (existing and existing.get("vector_blob")) or embedding_stale
            needs_enrichment = changed or not has_enrichment or enrichment_stale
            stale_pipeline = bool(existing) and existing.get("pipeline_version") != self.config.pipeline_version
            if stale_pipeline:
                needs_metadata = True
                needs_thumbnail = True
                needs_embedding = True
                needs_enrichment = True

            item = PendingAsset(
                path=file_path,
                existing=existing,
                signature=signature,
                needs_hash=needs_hash,
                needs_metadata=needs_metadata,
                needs_thumbnail=needs_thumbnail,
                needs_embedding=needs_embedding,
                needs_enrichment=needs_enrichment,
            )
            if item.needs_processing:
                pending.append(item)
            else:
                stats.skipped += 1
        return pending

    def _process_pending(self, pending_assets: list[PendingAsset], stats: PipelineStats) -> None:
        batch_size = max(int(self.config.batch_size), 1)
        overall = tqdm(total=len(pending_assets), desc="Indexing", unit="image")
        try:
            for start in range(0, len(pending_assets), batch_size):
                batch = pending_assets[start : start + batch_size]
                prep_started = time.perf_counter()
                prepared = self._prepare_batch(batch, stats)
                stats.timings.prep_seconds += time.perf_counter() - prep_started
                if not prepared:
                    continue

                embedding_targets = [item for item in prepared if item.pending.needs_embedding and item.image is not None]
                if embedding_targets:
                    embed_started = time.perf_counter()
                    try:
                        vectors = self.embedding_model.encode_images([item.image for item in embedding_targets])
                        for item, vector in zip(embedding_targets, vectors):
                            item.embedding = vector
                    except Exception as exc:  # pragma: no cover - depends on model backend
                        for item in embedding_targets:
                            stats.failed += 1
                            self.database.log_error(str(item.pending.path), "embedding", str(exc))
                    stats.timings.embedding_seconds += time.perf_counter() - embed_started

                enrichment_targets = [item for item in prepared if item.pending.needs_enrichment and item.image is not None]
                if enrichment_targets:
                    enrich_started = time.perf_counter()
                    self._enrich_batch(enrichment_targets, stats)
                    stats.timings.enrichment_seconds += time.perf_counter() - enrich_started

                persist_started = time.perf_counter()
                for item in tqdm(prepared, desc="Persisting", unit="asset", leave=False, position=1):
                    self._persist_prepared(item)
                    stats.indexed += 1
                    overall.update(1)
                    overall.set_postfix(indexed=f"{stats.indexed}/{stats.planned}", failed=stats.failed)
                stats.timings.persist_seconds += time.perf_counter() - persist_started

                for item in prepared:
                    if item.image is not None:
                        item.image.close()
        finally:
            overall.close()

    def _prepare_item(self, item: PendingAsset) -> PreparedAsset:
        existing = item.existing or {}
        needs_image = item.needs_embedding or item.needs_enrichment or item.needs_thumbnail
        image = None
        if needs_image:
            with Image.open(item.path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")

        metadata = extract_image_metadata(item.path) if item.needs_metadata else None
        file_hash = file_sha256(item.path) if item.needs_hash else existing.get("file_hash")
        thumbnail_path = (
            generate_thumbnail(item.path, self.thumbnail_dir, file_hash or existing.get("file_hash") or item.signature)
            if item.needs_thumbnail
            else existing.get("thumbnail_path")
        )
        return PreparedAsset(
            pending=item,
            image=image,
            metadata=metadata,
            file_hash=file_hash,
            thumbnail_path=thumbnail_path,
            embedding=None,
            enrichment=None,
        )

    def _prepare_batch(self, batch: list[PendingAsset], stats: PipelineStats) -> list[PreparedAsset]:
        prepared_items: list[PreparedAsset] = []
        with ThreadPoolExecutor(max_workers=max(1, int(self.config.num_workers))) as executor:
            futures = {executor.submit(self._prepare_item, item): item for item in batch}
            for future in tqdm(futures, desc="Processing", unit="asset", leave=False, position=1):
                item = futures[future]
                try:
                    prepared_items.append(future.result())
                except Exception as exc:  # pragma: no cover - exercised indirectly
                    stats.failed += 1
                    self.database.log_error(str(item.path), "prepare", str(exc))
        return prepared_items

    def _enrich_batch(self, items: list[PreparedAsset], stats: PipelineStats) -> None:
        try:
            fast_results = self.enrichment_model.enrich_images([item.image for item in items if item.image is not None], mode="fast")
        except Exception as exc:  # pragma: no cover - depends on Ollama/runtime
            for item in items:
                stats.failed += 1
                self.database.log_error(str(item.pending.path), "enrichment-fast", str(exc))
            return

        for item, result in zip(items, fast_results):
            item.enrichment = result

        if self.config.enrichment_mode != "balanced":
            return

        detailed_items = [item for item in items if item.enrichment and item.enrichment.requires_document_pass()]
        if not detailed_items:
            return

        try:
            detailed_results = self.enrichment_model.enrich_images(
                [item.image for item in detailed_items if item.image is not None],
                mode="document",
            )
            for item, result in zip(detailed_items, detailed_results):
                assert item.enrichment is not None
                item.enrichment = item.enrichment.merge(result)
        except Exception as exc:  # pragma: no cover - depends on Ollama/runtime
            for item in detailed_items:
                stats.failed += 1
                self.database.log_error(str(item.pending.path), "enrichment-document", str(exc))

    def _persist_prepared(self, item: PreparedAsset) -> None:
        existing = item.pending.existing or {}
        metadata = item.metadata
        summary = item.enrichment.short_summary if item.enrichment else existing.get("short_summary") or existing.get("caption")
        payload = {
            "file_path": str(item.pending.path),
            "file_hash": item.file_hash or existing.get("file_hash"),
            "file_type": metadata.file_type if metadata else existing.get("file_type"),
            "file_size": metadata.file_size if metadata else existing.get("file_size"),
            "width": metadata.width if metadata else existing.get("width"),
            "height": metadata.height if metadata else existing.get("height"),
            "created_at": metadata.created_at if metadata else existing.get("created_at"),
            "modified_at": metadata.modified_at if metadata else existing.get("modified_at"),
            "indexed_at": utc_now_iso(),
            "folder": metadata.folder if metadata else existing.get("folder"),
            "thumbnail_path": item.thumbnail_path or existing.get("thumbnail_path"),
            "caption": summary,
            "caption_model": self.enrichment_model.model_name if summary else existing.get("caption_model"),
            "stat_signature": item.pending.signature,
            "pipeline_version": self.config.pipeline_version,
            "is_deleted": 0,
        }

        asset_id = self.database.upsert_asset(payload)
        if item.pending.needs_embedding and item.embedding is not None:
            vector = np.asarray(item.embedding, dtype=np.float32)
            self.vector_store.add_or_update(asset_id, vector)
            self.database.upsert_embedding(
                asset_id,
                faiss_row_id=asset_id,
                embedding_model=self.config.embedding_model_name,
                embedding_dim=int(vector.shape[-1]),
                vector_blob=vector.astype(np.float32).tobytes(),
            )

        if item.pending.needs_enrichment and item.enrichment is not None:
            record = item.enrichment.to_record()
            self.database.upsert_enrichment(
                asset_id,
                {
                    **record,
                    "enrichment_model": self.enrichment_model.model_name,
                    "enrichment_version": self.enrichment_model.enrichment_version,
                    "enrichment_mode": self.config.enrichment_mode,
                    "enriched_at": utc_now_iso(),
                },
            )
