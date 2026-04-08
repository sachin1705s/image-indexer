"""FastAPI application factory."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from image_archive.config import AppConfig
from image_archive.db import Database
from image_archive.faiss_store import VectorStore
from image_archive.models.base import EmbeddingModel
from image_archive.search.service import SearchService


def frontend_directory() -> Path:
    """Return the packaged frontend directory, with a repo fallback for editable dev."""

    packaged = Path(str(files("image_archive").joinpath("frontend")))
    if (packaged / "index.html").exists():
        return packaged
    return Path(__file__).resolve().parents[3] / "frontend"


def create_app(
    *,
    config: AppConfig,
    database: Database,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
) -> FastAPI:
    """Build the FastAPI app with injected dependencies."""

    app = FastAPI(title="Image Archive Search", version="0.1.0")
    search_service = SearchService(
        config=config,
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    frontend_dir = frontend_directory()
    app.mount("/ui", StaticFiles(directory=frontend_dir), name="ui")

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.get("/styles.css")
    def styles() -> FileResponse:
        return FileResponse(frontend_dir / "styles.css")

    @app.get("/app.js")
    def script() -> FileResponse:
        return FileResponse(frontend_dir / "app.js")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search/text")
    def search_text(payload: dict) -> dict:
        query = str(payload.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail="query is required")
        results = search_service.text_search(
            query,
            folder=payload.get("folder"),
            date_from=payload.get("date_from"),
            date_to=payload.get("date_to"),
            content_type=payload.get("content_type"),
            top_k=int(payload.get("top_k", 24)),
        )
        return {"results": results}

    @app.post("/search/similar")
    async def search_similar(
        asset_id: Optional[int] = Form(default=None),
        top_k: int = Form(default=24),
        file: Optional[UploadFile] = File(default=None),
    ) -> dict:
        if asset_id is None and file is None:
            raise HTTPException(status_code=400, detail="asset_id or file is required")

        try:
            if asset_id is not None:
                results = search_service.asset_similar_search(asset_id, top_k=top_k)
            else:
                assert file is not None
                results = search_service.uploaded_image_search(await file.read(), top_k=top_k)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"results": results}

    @app.get("/assets/{asset_id}")
    def asset_detail(asset_id: int) -> dict:
        try:
            return search_service.asset_detail(asset_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/assets/{asset_id}/similar")
    def asset_similar(asset_id: int, top_k: int = 24) -> dict:
        try:
            results = search_service.asset_similar_search(asset_id, top_k=top_k)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"results": results}

    @app.get("/assets/{asset_id}/image")
    def asset_image(asset_id: int) -> FileResponse:
        row = database.get_asset(asset_id)
        if row is None:
            raise HTTPException(status_code=404, detail="asset not found")
        image_path = Path(row["file_path"])
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="original file missing")
        return FileResponse(image_path)

    @app.get("/assets/{asset_id}/thumbnail")
    def asset_thumbnail(asset_id: int) -> FileResponse:
        row = database.get_asset(asset_id)
        if row is None:
            raise HTTPException(status_code=404, detail="asset not found")
        thumbnail_path = Path(row["thumbnail_path"])
        if not thumbnail_path.exists():
            raise HTTPException(status_code=404, detail="thumbnail missing")
        return FileResponse(thumbnail_path)

    @app.get("/folders")
    def folders() -> dict:
        return {"folders": search_service.folders()}

    @app.get("/content-types")
    def content_types() -> dict:
        return {
            "content_types": [
                "photo",
                "document",
                "screenshot",
                "ui",
                "illustration",
                "scan",
                "other",
            ]
        }

    @app.get("/stats")
    def stats() -> dict:
        return search_service.stats()

    return app
