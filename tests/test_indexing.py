from pathlib import Path

from image_archive.indexing.pipeline import IndexingPipeline


def test_indexing_reports_overall_progress(test_state, image_factory, monkeypatch) -> None:
    library = test_state.root / "library"
    library.mkdir()
    for index in range(3):
        image_factory(f"library/image_{index}.jpg", (index * 40, 10, 10))

    class FakeTqdm:
        instances = []
        writes = []

        def __init__(self, iterable=None, total=None, desc=None, unit=None, leave=True, position=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.unit = unit
            self.leave = leave
            self.position = position
            self.updates = []
            self.postfixes = []
            FakeTqdm.instances.append(self)

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else [])

        def update(self, amount=1):
            self.updates.append(amount)

        def set_postfix(self, **kwargs):
            self.postfixes.append(kwargs)

        def close(self):
            return None

        @staticmethod
        def write(message):
            FakeTqdm.writes.append(message)

    monkeypatch.setattr("image_archive.indexing.pipeline.tqdm", FakeTqdm)

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    stats = pipeline.run([str(library)])

    progress = next(instance for instance in FakeTqdm.instances if instance.desc == "Indexing")
    assert progress.total == 3
    assert sum(progress.updates) == 3
    assert stats.planned == 3
    assert any("To process: 3" in message for message in FakeTqdm.writes)


def test_indexing_writes_assets_and_skips_unchanged(test_state, image_factory) -> None:
    library = test_state.root / "library"
    library.mkdir()
    first = image_factory("library/red.jpg", (255, 0, 0))
    second = image_factory("library/green.png", (0, 255, 0))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )

    first_run = pipeline.run([str(library)])
    second_run = pipeline.run([str(library)])

    assert first_run.scanned == 2
    assert first_run.indexed == 2
    assert first_run.skipped == 0
    assert second_run.scanned == 2
    assert second_run.indexed == 0
    assert second_run.skipped == 2
    assert test_state.database.count_assets() == 2
    assert first.exists()
    assert second.exists()


def test_database_persists_generated_fields(test_state, image_factory) -> None:
    library = test_state.root / "library"
    library.mkdir()
    path = image_factory("library/blue.png", (0, 0, 255))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    pipeline.run([str(library)])

    asset = test_state.database.get_asset_by_path(str(path))
    assert asset is not None
    assert asset["file_hash"]
    assert asset["width"] == 64
    assert asset["height"] == 64
    assert asset["thumbnail_path"]
    assert asset["search_text"]
    assert asset["content_type"]


def test_reindex_reprocesses_when_enrichment_version_changes(test_state, image_factory) -> None:
    library = test_state.root / "library"
    library.mkdir()
    image_factory("library/red.jpg", (255, 0, 0))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    first_run = pipeline.run([str(library)])

    test_state.config.enrichment_version = "fake-v2"
    test_state.enrichment_model.enrichment_version = "fake-v2"
    second_run = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    ).run([str(library)])

    assert first_run.indexed == 1
    assert second_run.indexed == 1


def test_fast_clip_enrichment_does_not_use_document_second_pass(test_state, image_factory) -> None:
    library = test_state.root / "library"
    library.mkdir()
    image_factory("library/blue.png", (0, 0, 255))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    pipeline.run([str(library)])

    assert "fast" in test_state.enrichment_model.calls
    assert "document" not in test_state.enrichment_model.calls


def test_indexing_limit_only_processes_first_n_images(test_state, image_factory) -> None:
    library = test_state.root / "library"
    library.mkdir()
    for index in range(5):
        image_factory(f"library/image_{index}.jpg", (index * 20, 0, 0))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    stats = pipeline.run_with_options([str(library)], limit=3)

    assert stats.scanned == 3
    assert stats.indexed == 3
    assert test_state.database.count_assets() == 3
