"""SQLite schema and repository helpers."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Union


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


ASSET_SELECT = """
SELECT
    assets.*,
    embeddings.embedding_model,
    embeddings.embedding_dim,
    embeddings.vector_blob,
    enrichment.content_type,
    enrichment.auto_tags,
    enrichment.style_labels,
    enrichment.object_labels,
    enrichment.has_person,
    enrichment.has_face,
    enrichment.person_count_bucket,
    enrichment.face_count_bucket,
    enrichment.is_document,
    enrichment.is_screenshot,
    enrichment.text_heavy,
    enrichment.short_summary,
    enrichment.search_text,
    enrichment.enrichment_model,
    enrichment.enrichment_version,
    enrichment.enrichment_mode,
    enrichment.enriched_at,
    enrichment.raw_json
FROM assets
LEFT JOIN embeddings ON embeddings.asset_id = assets.id
LEFT JOIN enrichment ON enrichment.asset_id = assets.id
"""


class Database:
    """Small repository wrapper around sqlite3."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path).expanduser().resolve()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    created_at TEXT,
                    modified_at TEXT,
                    indexed_at TEXT,
                    folder TEXT,
                    thumbnail_path TEXT,
                    caption TEXT,
                    caption_model TEXT,
                    stat_signature TEXT,
                    pipeline_version TEXT,
                    is_deleted INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    asset_id INTEGER PRIMARY KEY,
                    faiss_row_id INTEGER NOT NULL UNIQUE,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    vector_blob BLOB,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS enrichment (
                    asset_id INTEGER PRIMARY KEY,
                    content_type TEXT,
                    auto_tags TEXT,
                    style_labels TEXT,
                    object_labels TEXT,
                    has_person INTEGER NOT NULL DEFAULT 0,
                    has_face INTEGER NOT NULL DEFAULT 0,
                    person_count_bucket TEXT,
                    face_count_bucket TEXT,
                    is_document INTEGER NOT NULL DEFAULT 0,
                    is_screenshot INTEGER NOT NULL DEFAULT 0,
                    text_heavy INTEGER NOT NULL DEFAULT 0,
                    short_summary TEXT,
                    search_text TEXT,
                    enrichment_model TEXT NOT NULL,
                    enrichment_version TEXT NOT NULL,
                    enrichment_mode TEXT NOT NULL,
                    enriched_at TEXT NOT NULL,
                    raw_json TEXT,
                    FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS indexing_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    num_scanned INTEGER NOT NULL DEFAULT 0,
                    num_indexed INTEGER NOT NULL DEFAULT 0,
                    num_skipped INTEGER NOT NULL DEFAULT 0,
                    num_failed INTEGER NOT NULL DEFAULT 0,
                    target_paths TEXT
                );

                CREATE TABLE IF NOT EXISTS errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_path TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    error_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_assets_folder ON assets(folder);
                CREATE INDEX IF NOT EXISTS idx_assets_modified_at ON assets(modified_at);
                CREATE INDEX IF NOT EXISTS idx_runs_started_at ON indexing_runs(started_at);
                CREATE INDEX IF NOT EXISTS idx_enrichment_content_type ON enrichment(content_type);
                """
            )
            self._ensure_column(conn, "indexing_runs", "timings_json", "TEXT")
            self._ensure_column(conn, "indexing_runs", "avg_ms_per_asset", "REAL")
            self._ensure_column(conn, "indexing_runs", "throughput_per_min", "REAL")

    def start_index_run(self, target_paths: list[str]) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO indexing_runs (started_at, status, target_paths)
                VALUES (?, ?, ?)
                """,
                (utc_now_iso(), "running", "\n".join(target_paths)),
            )
            return int(cursor.lastrowid)

    def finish_index_run(
        self,
        run_id: int,
        *,
        status: str,
        num_scanned: int,
        num_indexed: int,
        num_skipped: int,
        num_failed: int,
        timings: Optional[dict[str, Any]] = None,
        avg_ms_per_asset: float = 0.0,
        throughput_per_min: float = 0.0,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE indexing_runs
                SET finished_at = ?, status = ?, num_scanned = ?, num_indexed = ?, num_skipped = ?, num_failed = ?,
                    timings_json = ?, avg_ms_per_asset = ?, throughput_per_min = ?
                WHERE id = ?
                """,
                (
                    utc_now_iso(),
                    status,
                    num_scanned,
                    num_indexed,
                    num_skipped,
                    num_failed,
                    json.dumps(timings or {}),
                    avg_ms_per_asset,
                    throughput_per_min,
                    run_id,
                ),
            )

    def log_error(self, asset_path: str, stage: str, error_text: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO errors (asset_path, stage, error_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (asset_path, stage, error_text, utc_now_iso()),
            )

    def get_asset_by_path(self, file_path: str) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                f"""
                {ASSET_SELECT}
                WHERE assets.file_path = ?
                """,
                (file_path,),
            ).fetchone()

    def get_asset(self, asset_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                f"""
                {ASSET_SELECT}
                WHERE assets.id = ?
                """,
                (asset_id,),
            ).fetchone()

    def get_assets_by_ids(self, asset_ids: list[int]) -> list[sqlite3.Row]:
        if not asset_ids:
            return []
        placeholders = ",".join(["?"] * len(asset_ids))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                {ASSET_SELECT}
                WHERE assets.id IN ({placeholders}) AND assets.is_deleted = 0
                """,
                tuple(asset_ids),
            ).fetchall()
        lookup = {int(row["id"]): row for row in rows}
        return [lookup[item] for item in asset_ids if item in lookup]

    def get_embedding_row(self, asset_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM embeddings WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()

    def get_enrichment_row(self, asset_id: int) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM enrichment WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()

    def upsert_asset(self, payload: dict[str, Any]) -> int:
        file_path = payload["file_path"]
        existing = self.get_asset_by_path(file_path)
        if existing:
            asset_id = int(existing["id"])
            updates = {**dict(existing), **payload}
            with self.connect() as conn:
                conn.execute(
                    """
                    UPDATE assets
                    SET file_hash = ?, file_type = ?, file_size = ?, width = ?, height = ?,
                        created_at = ?, modified_at = ?, indexed_at = ?, folder = ?,
                        thumbnail_path = ?, caption = ?, caption_model = ?, stat_signature = ?,
                        pipeline_version = ?, is_deleted = ?
                    WHERE id = ?
                    """,
                    (
                        updates.get("file_hash"),
                        updates.get("file_type"),
                        updates.get("file_size"),
                        updates.get("width"),
                        updates.get("height"),
                        updates.get("created_at"),
                        updates.get("modified_at"),
                        updates.get("indexed_at"),
                        updates.get("folder"),
                        updates.get("thumbnail_path"),
                        updates.get("caption"),
                        updates.get("caption_model"),
                        updates.get("stat_signature"),
                        updates.get("pipeline_version"),
                        int(updates.get("is_deleted", 0)),
                        asset_id,
                    ),
                )
            return asset_id

        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO assets (
                    file_path, file_hash, file_type, file_size, width, height,
                    created_at, modified_at, indexed_at, folder, thumbnail_path,
                    caption, caption_model, stat_signature, pipeline_version, is_deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload.get("file_path"),
                    payload.get("file_hash"),
                    payload.get("file_type"),
                    payload.get("file_size"),
                    payload.get("width"),
                    payload.get("height"),
                    payload.get("created_at"),
                    payload.get("modified_at"),
                    payload.get("indexed_at"),
                    payload.get("folder"),
                    payload.get("thumbnail_path"),
                    payload.get("caption"),
                    payload.get("caption_model"),
                    payload.get("stat_signature"),
                    payload.get("pipeline_version"),
                    int(payload.get("is_deleted", 0)),
                ),
            )
            return int(cursor.lastrowid)

    def upsert_embedding(
        self,
        asset_id: int,
        *,
        faiss_row_id: int,
        embedding_model: str,
        embedding_dim: int,
        vector_blob: bytes,
    ) -> None:
        existing = self.get_embedding_row(asset_id)
        with self.connect() as conn:
            if existing:
                conn.execute(
                    """
                    UPDATE embeddings
                    SET faiss_row_id = ?, embedding_model = ?, embedding_dim = ?, vector_blob = ?, updated_at = ?
                    WHERE asset_id = ?
                    """,
                    (faiss_row_id, embedding_model, embedding_dim, vector_blob, utc_now_iso(), asset_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO embeddings (asset_id, faiss_row_id, embedding_model, embedding_dim, vector_blob, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (asset_id, faiss_row_id, embedding_model, embedding_dim, vector_blob, utc_now_iso()),
                )

    def upsert_enrichment(self, asset_id: int, payload: dict[str, Any]) -> None:
        existing = self.get_enrichment_row(asset_id)
        serialized = {
            **payload,
            "raw_json": json.dumps(payload.get("raw_json", {})),
        }
        with self.connect() as conn:
            if existing:
                conn.execute(
                    """
                    UPDATE enrichment
                    SET content_type = ?, auto_tags = ?, style_labels = ?, object_labels = ?,
                        has_person = ?, has_face = ?, person_count_bucket = ?, face_count_bucket = ?,
                        is_document = ?, is_screenshot = ?, text_heavy = ?, short_summary = ?,
                        search_text = ?, enrichment_model = ?, enrichment_version = ?,
                        enrichment_mode = ?, enriched_at = ?, raw_json = ?
                    WHERE asset_id = ?
                    """,
                    (
                        serialized.get("content_type"),
                        serialized.get("auto_tags"),
                        serialized.get("style_labels"),
                        serialized.get("object_labels"),
                        int(serialized.get("has_person", 0)),
                        int(serialized.get("has_face", 0)),
                        serialized.get("person_count_bucket"),
                        serialized.get("face_count_bucket"),
                        int(serialized.get("is_document", 0)),
                        int(serialized.get("is_screenshot", 0)),
                        int(serialized.get("text_heavy", 0)),
                        serialized.get("short_summary"),
                        serialized.get("search_text"),
                        serialized.get("enrichment_model"),
                        serialized.get("enrichment_version"),
                        serialized.get("enrichment_mode"),
                        serialized.get("enriched_at"),
                        serialized.get("raw_json"),
                        asset_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO enrichment (
                        asset_id, content_type, auto_tags, style_labels, object_labels,
                        has_person, has_face, person_count_bucket, face_count_bucket,
                        is_document, is_screenshot, text_heavy, short_summary, search_text,
                        enrichment_model, enrichment_version, enrichment_mode, enriched_at, raw_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset_id,
                        serialized.get("content_type"),
                        serialized.get("auto_tags"),
                        serialized.get("style_labels"),
                        serialized.get("object_labels"),
                        int(serialized.get("has_person", 0)),
                        int(serialized.get("has_face", 0)),
                        serialized.get("person_count_bucket"),
                        serialized.get("face_count_bucket"),
                        int(serialized.get("is_document", 0)),
                        int(serialized.get("is_screenshot", 0)),
                        int(serialized.get("text_heavy", 0)),
                        serialized.get("short_summary"),
                        serialized.get("search_text"),
                        serialized.get("enrichment_model"),
                        serialized.get("enrichment_version"),
                        serialized.get("enrichment_mode"),
                        serialized.get("enriched_at"),
                        serialized.get("raw_json"),
                    ),
                )

    def count_assets(self) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM assets WHERE is_deleted = 0").fetchone()
            return int(row["count"])

    def last_run(self) -> Optional[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM indexing_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()

    def stats(self) -> dict[str, Any]:
        last_run = self.last_run()
        payload = dict(last_run) if last_run else None
        if payload and payload.get("timings_json"):
            payload["timings"] = json.loads(payload["timings_json"])
        return {
            "asset_count": self.count_assets(),
            "last_run": payload,
        }

    def list_search_matches(
        self,
        terms: list[str],
        *,
        folder_prefix: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        content_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        clauses = ["assets.is_deleted = 0"]
        params: list[Any] = []

        if folder_prefix:
            clauses.append("assets.file_path LIKE ?")
            params.append(f"{folder_prefix}%")
        if date_from:
            clauses.append("(assets.modified_at IS NULL OR assets.modified_at >= ?)")
            params.append(date_from)
        if date_to:
            clauses.append("(assets.modified_at IS NULL OR assets.modified_at <= ?)")
            params.append(date_to)
        if content_type:
            clauses.append("enrichment.content_type = ?")
            params.append(content_type)

        if terms:
            term_clauses = []
            for term in terms:
                term_clauses.append("LOWER(COALESCE(enrichment.search_text, assets.caption, '')) LIKE ?")
                params.append(f"%{term.lower()}%")
            clauses.append(f"({' OR '.join(term_clauses)})")

        params.append(limit)
        where = " AND ".join(clauses)
        with self.connect() as conn:
            return conn.execute(
                f"""
                {ASSET_SELECT}
                WHERE {where}
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
