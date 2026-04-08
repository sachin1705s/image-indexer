from typer.testing import CliRunner

from image_archive.cli import app


class DummyPipeline:
    def __init__(self, **_kwargs) -> None:
        self.kwargs = _kwargs

    def run_with_options(self, paths, *, limit=None):
        class _Stats:
            scanned = min(len(paths), limit) if limit else len(paths)
            indexed = min(len(paths), limit) if limit else len(paths)
            skipped = 0
            failed = 0

            class _Timings:
                def to_dict(self):
                    return {"scan_seconds": 0.1}

                def throughput_per_minute(self, _indexed):
                    return 100.0

            timings = _Timings()

        return _Stats()


def test_guided_run_saves_selected_folders(monkeypatch, test_state) -> None:
    runner = CliRunner()
    selected_folder = test_state.root / "library"
    selected_folder.mkdir(exist_ok=True)

    monkeypatch.setattr("image_archive.cli.pick_folders", lambda *_args, **_kwargs: [str(selected_folder)])
    monkeypatch.setattr("image_archive.cli.build_runtime", lambda _manager, _config: (test_state.database, test_state.vector_store))
    monkeypatch.setattr("image_archive.cli.build_embedding_backend", lambda _config: test_state.embedding_model)
    monkeypatch.setattr("image_archive.cli.build_enrichment_backend", lambda _config: test_state.enrichment_model)
    monkeypatch.setattr("image_archive.cli.IndexingPipeline", DummyPipeline)
    monkeypatch.setattr("image_archive.cli.initialize_app_state", lambda *_args, **_kwargs: None)

    result = runner.invoke(
        app,
        ["run", "--config-path", str(test_state.config_path), "--root", str(test_state.root), "--no-start-server"],
    )

    assert result.exit_code == 0
    refreshed = test_state.config_manager.load()
    assert str(selected_folder.resolve()) in refreshed.indexed_paths


def test_index_command_passes_limit(monkeypatch, test_state) -> None:
    runner = CliRunner()
    selected_folder = test_state.root / "library"
    selected_folder.mkdir(exist_ok=True)
    captured = {}

    class CapturePipeline(DummyPipeline):
        def run_with_options(self, paths, *, limit=None):
            captured["paths"] = paths
            captured["limit"] = limit
            return super().run_with_options(paths, limit=limit)

    monkeypatch.setattr("image_archive.cli.build_runtime", lambda _manager, _config: (test_state.database, test_state.vector_store))
    monkeypatch.setattr("image_archive.cli.build_models", lambda _config: (test_state.embedding_model, test_state.enrichment_model))
    monkeypatch.setattr("image_archive.cli.IndexingPipeline", CapturePipeline)

    result = runner.invoke(
        app,
        ["index", str(selected_folder), "--config-path", str(test_state.config_path), "--limit", "100"],
    )

    assert result.exit_code == 0
    assert captured["limit"] == 100


def test_serve_prints_clickable_local_url(monkeypatch, test_state) -> None:
    runner = CliRunner()
    captured = {}

    monkeypatch.setattr("image_archive.cli.build_runtime", lambda _manager, _config: (test_state.database, test_state.vector_store))
    monkeypatch.setattr("image_archive.cli.build_embedding_backend", lambda _config: test_state.embedding_model)
    monkeypatch.setattr("image_archive.cli.create_app", lambda **_kwargs: object())

    def fake_run(_app, *, host, port, log_level):
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr("image_archive.cli.uvicorn.run", fake_run)

    result = runner.invoke(
        app,
        ["serve", "--config-path", str(test_state.config_path), "--host", "0.0.0.0", "--port", "9999"],
    )

    assert result.exit_code == 0
    assert "Image Archive Search is live" in result.output
    assert "Open Image Archive Search" in result.output
    assert "URL:  http://127.0.0.1:9999" in result.output
    assert captured == {"host": "0.0.0.0", "port": 9999, "log_level": "info"}


def test_reset_deletes_archive_data_with_explicit_confirmation(test_state) -> None:
    runner = CliRunner()
    archive_dir = test_state.config_manager.resolve_path(test_state.config.app_data_dir)
    db_path = test_state.config_manager.resolve_path(test_state.config.sqlite_path)
    thumbnail_dir = test_state.config_manager.resolve_path(test_state.config.thumbnail_dir)
    thumbnail_marker = thumbnail_dir / "marker.jpg"
    thumbnail_marker.write_text("indexed data", encoding="utf-8")

    result = runner.invoke(
        app,
        ["reset", "--config-path", str(test_state.config_path), "--yes"],
    )

    assert result.exit_code == 0
    assert "Deleted indexed data." in result.output
    assert archive_dir.exists()
    assert not db_path.exists()
    assert not thumbnail_dir.exists()
    assert test_state.config_path.exists()
