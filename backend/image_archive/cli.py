"""Typer CLI entrypoint."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
import uvicorn

from image_archive.runtime_env import prepare_runtime_environment

prepare_runtime_environment()

from image_archive.api.app import create_app
from image_archive.config import AppConfig, ConfigManager
from image_archive.indexing.pipeline import IndexingPipeline
from image_archive.services.library import (
    build_embedding_backend,
    build_enrichment_backend,
    build_models,
    build_runtime,
    initialize_app_state,
    normalize_index_paths,
)
from image_archive.services.folder_picker import pick_folders

app = typer.Typer(help="Local-first creative image archive search")


def _server_url(host: str, port: int) -> str:
    """Return the browser URL a local user should open."""

    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{browser_host}:{port}"


def _terminal_hyperlink(url: str, label: str) -> str:
    """Return a clickable terminal hyperlink when the terminal supports OSC 8."""

    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def _print_server_instructions(host: str, port: int) -> None:
    """Print a clean, copy/click-friendly link before Uvicorn logs start."""

    url = _server_url(host, port)
    link = _terminal_hyperlink(url, "Open Image Archive Search")
    typer.echo("")
    typer.secho("Image Archive Search is live", bold=True, fg=typer.colors.GREEN)
    typer.echo(f"Open: {link}")
    typer.echo(f"URL:  {url}")
    typer.echo("Tip: keep this terminal open. Press Ctrl+C here when you want to stop the app.")
    typer.echo("")


def load_config_manager(config_path: Optional[str]) -> tuple[ConfigManager, AppConfig]:
    """Load the config manager and config."""

    manager = ConfigManager(config_path)
    config = manager.load()
    return manager, config


@app.command()
def init(config_path: Optional[str] = typer.Option(None, help="Path to the config file")) -> None:
    """Create app directories, database, and vector index."""

    manager, config = load_config_manager(config_path)
    manager.save(config)
    initialize_app_state(manager, config)
    summary = manager.summary(config)
    typer.echo(f"Config: {summary['config_path']}")
    typer.echo(f"SQLite: {summary['sqlite_path']}")
    typer.echo(f"FAISS: {summary['faiss_index_path']}")


@app.command()
def index(
    path: Optional[str] = typer.Argument(None, help="Folder of images to index"),
    config_path: Optional[str] = typer.Option(None, help="Path to the config file"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Index only the first N discovered images."),
) -> None:
    """Index one local image folder."""

    manager, config = load_config_manager(config_path)
    target_paths = [path] if path else pick_folders()
    if not target_paths:
        raise typer.Abort()
    for target in target_paths:
        config = config.with_indexed_path(target)
    manager.save(config)

    database, vector_store = build_runtime(manager, config)
    embedding_model, enrichment_model = build_models(config)
    pipeline = IndexingPipeline(
        config=config,
        config_manager=manager,
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
        enrichment_model=enrichment_model,
    )
    stats = pipeline.run_with_options(normalize_index_paths(target_paths), limit=limit)
    typer.echo(
        f"Scanned={stats.scanned} Indexed={stats.indexed} Skipped={stats.skipped} Failed={stats.failed}"
    )
    typer.echo(f"Timing={stats.timings.to_dict()} Throughput/min={stats.timings.throughput_per_minute(stats.indexed)}")


@app.command()
def reindex(config_path: Optional[str] = typer.Option(None, help="Path to the config file")) -> None:
    """Re-run indexing for all configured folders."""

    manager, config = load_config_manager(config_path)
    if not config.indexed_paths:
        raise typer.BadParameter("No indexed_paths configured. Run `image-archive index <path>` first.")

    targets = normalize_index_paths(config.indexed_paths)
    database, vector_store = build_runtime(manager, config)
    embedding_model, enrichment_model = build_models(config)
    pipeline = IndexingPipeline(
        config=config,
        config_manager=manager,
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
        enrichment_model=enrichment_model,
    )
    stats = pipeline.run(targets)
    typer.echo(
        f"Scanned={stats.scanned} Indexed={stats.indexed} Skipped={stats.skipped} Failed={stats.failed}"
    )
    typer.echo(f"Timing={stats.timings.to_dict()} Throughput/min={stats.timings.throughput_per_minute(stats.indexed)}")


@app.command()
def status(config_path: Optional[str] = typer.Option(None, help="Path to the config file")) -> None:
    """Show archive status."""

    manager, config = load_config_manager(config_path)
    database, _vector_store = build_runtime(manager, config)
    summary = manager.summary(config)
    stats_payload = database.stats()
    last_run = stats_payload["last_run"]
    typer.echo(f"Indexed assets: {stats_payload['asset_count']}")
    typer.echo(f"DB path: {summary['sqlite_path']}")
    typer.echo(f"Index path: {summary['faiss_index_path']}")
    typer.echo(f"Indexed folders: {len(config.indexed_paths)}")
    typer.echo(f"Last index run: {last_run['finished_at'] if last_run else 'never'}")
    if last_run and last_run.get("timings"):
        typer.echo(f"Last timings: {last_run['timings']}")
        typer.echo(f"Throughput/min: {last_run.get('throughput_per_min') or 0}")


@app.command()
def serve(
    config_path: Optional[str] = typer.Option(None, help="Path to the config file"),
    host: Optional[str] = typer.Option(None, help="Host override"),
    port: Optional[int] = typer.Option(None, help="Port override"),
) -> None:
    """Start the local API server and serve the web UI."""

    manager, config = load_config_manager(config_path)
    database, vector_store = build_runtime(manager, config)
    embedding_model = build_embedding_backend(config)
    fastapi_app = create_app(
        config=config,
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
    )
    server_host = host or config.host
    server_port = port or config.port
    _print_server_instructions(server_host, server_port)
    uvicorn.run(
        fastapi_app,
        host=server_host,
        port=server_port,
        log_level="info",
    )


@app.command()
def run(
    config_path: Optional[str] = typer.Option(None, help="Path to the config file"),
    root: str = typer.Option("~", help="Initial folder for the terminal navigator"),
    start_server: Optional[bool] = typer.Option(None, "--start-server/--no-start-server", help="Start the local server after indexing."),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Index only the first N discovered images."),
) -> None:
    """Guided workflow: init, pick folders, index, and optionally serve."""

    manager, config = load_config_manager(config_path)
    manager.save(config)
    initialize_app_state(manager, config)

    selected = pick_folders(Path(root).expanduser())
    if not selected:
        raise typer.Abort()

    for target in selected:
        config = config.with_indexed_path(target)
    manager.save(config)

    database, vector_store = build_runtime(manager, config)
    embedding_model = build_embedding_backend(config)
    enrichment_model = build_enrichment_backend(config)
    pipeline = IndexingPipeline(
        config=config,
        config_manager=manager,
        database=database,
        vector_store=vector_store,
        embedding_model=embedding_model,
        enrichment_model=enrichment_model,
    )
    stats = pipeline.run_with_options(normalize_index_paths(selected), limit=limit)
    typer.echo(
        f"Scanned={stats.scanned} Indexed={stats.indexed} Skipped={stats.skipped} Failed={stats.failed}"
    )
    typer.echo(f"Timing={stats.timings.to_dict()} Throughput/min={stats.timings.throughput_per_minute(stats.indexed)}")

    should_serve = start_server if start_server is not None else typer.confirm("Start the local web app now?", default=True)
    if should_serve:
        fastapi_app = create_app(
            config=config,
            database=database,
            vector_store=vector_store,
            embedding_model=embedding_model,
        )
        _print_server_instructions(config.host, config.port)
        uvicorn.run(
            fastapi_app,
            host=config.host,
            port=config.port,
            log_level="info",
        )


@app.command()
def reset(
    config_path: Optional[str] = typer.Option(None, help="Path to the config file"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    delete_config: bool = typer.Option(False, "--delete-config", help="Also delete the config file."),
) -> None:
    """Delete the local index database, vector index, thumbnails, and uploads."""

    manager, config = load_config_manager(config_path)
    paths = manager.summary(config)
    data_dir = manager.resolve_path(config.app_data_dir)
    summary = manager.summary(config)

    typer.echo(f"Archive data: {data_dir}")
    typer.echo(f"Config file: {summary['config_path']}")
    if not yes and not typer.confirm("Delete indexed data? Original images will not be touched.", default=False):
        raise typer.Abort()

    targets = [
        manager.resolve_path(config.sqlite_path),
        manager.resolve_path(config.faiss_index_path),
        Path(paths["faiss_index_path"]).with_suffix(".json"),
        manager.resolve_path(config.thumbnail_dir),
        manager.resolve_path(config.upload_dir),
    ]
    for sqlite_sidecar in [
        manager.resolve_path(config.sqlite_path).with_suffix(".db-shm"),
        manager.resolve_path(config.sqlite_path).with_suffix(".db-wal"),
    ]:
        targets.append(sqlite_sidecar)

    deleted_any = False
    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        deleted_any = True

    typer.echo("Deleted indexed data." if deleted_any else "No indexed data files found.")

    if delete_config and manager.path.exists():
        manager.path.unlink()
        typer.echo("Deleted config file.")


def main() -> None:
    """Console script entrypoint."""

    app()


if __name__ == "__main__":
    main()
