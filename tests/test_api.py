from fastapi.testclient import TestClient
import numpy as np

from image_archive.api.app import create_app
from image_archive.indexing.pipeline import IndexingPipeline


def seed_assets(test_state, image_factory):
    library = test_state.root / "library"
    library.mkdir(exist_ok=True)
    image_factory("library/red.jpg", (255, 0, 0))
    image_factory("library/blue.jpg", (0, 0, 255))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    pipeline.run([str(library)])


def test_text_search_endpoint_returns_ranked_results(test_state, image_factory) -> None:
    seed_assets(test_state, image_factory)
    app = create_app(
        config=test_state.config,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
    )
    client = TestClient(app)

    response = client.post("/search/text", json={"query": "warm portrait", "top_k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    assert "portrait" in payload["results"][0]["auto_tags"]


def test_similar_search_endpoint_uses_asset_embedding(test_state, image_factory) -> None:
    seed_assets(test_state, image_factory)
    app = create_app(
        config=test_state.config,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
    )
    client = TestClient(app)

    red_asset = test_state.database.list_search_matches(["red"], limit=1)[0]
    response = client.post(
        "/search/similar",
        files={},
        data={"asset_id": str(red_asset["id"]), "top_k": "5"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    assert payload["results"][0]["id"] != red_asset["id"]


def test_text_search_can_filter_by_content_type(test_state, image_factory) -> None:
    seed_assets(test_state, image_factory)
    app = create_app(
        config=test_state.config,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
    )
    client = TestClient(app)

    response = client.post("/search/text", json={"query": "screen", "content_type": "screenshot", "top_k": 5})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"]
    assert all(item["content_type"] == "screenshot" for item in payload["results"])


def test_text_search_filters_weak_low_information_results(test_state, image_factory) -> None:
    library = test_state.root / "library"
    library.mkdir(exist_ok=True)
    image_factory("library/blank-slide.png", (180, 180, 180))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    pipeline.run([str(library)])

    row = dict(test_state.database.get_asset_by_path(str(library / "blank-slide.png")))
    test_state.vector_store.vectors[int(row["id"])] = np.array([0.275, 0.0, 0.0, 0.0], dtype=np.float32)
    test_state.config.low_information_max_score = 0.30
    app = create_app(
        config=test_state.config,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
    )
    client = TestClient(app)

    response = client.post("/search/text", json={"query": "warm", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_text_search_keeps_low_information_results_when_query_mentions_them(test_state, image_factory) -> None:
    library = test_state.root / "library"
    library.mkdir(exist_ok=True)
    image_factory("library/blank-slide.png", (180, 180, 180))

    pipeline = IndexingPipeline(
        config=test_state.config,
        config_manager=test_state.config_manager,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
        enrichment_model=test_state.enrichment_model,
    )
    pipeline.run([str(library)])
    row = dict(test_state.database.get_asset_by_path(str(library / "blank-slide.png")))
    test_state.vector_store.vectors[int(row["id"])] = np.array([0.275, 0.0, 0.0, 0.0], dtype=np.float32)
    test_state.config.low_information_max_score = 0.30
    app = create_app(
        config=test_state.config,
        database=test_state.database,
        vector_store=test_state.vector_store,
        embedding_model=test_state.embedding_model,
    )
    client = TestClient(app)

    response = client.post("/search/text", json={"query": "warm slide", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["results"]
