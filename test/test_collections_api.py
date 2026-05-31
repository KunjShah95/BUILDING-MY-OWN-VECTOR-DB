"""API tests for collections and text ingest (requires PostgreSQL)."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from database.schema import Base

TEST_DATABASE_URL = "postgresql://user:password@localhost:5432/vector_db_test"
engine = create_engine(TEST_DATABASE_URL)
_DB_AVAILABLE = False
try:
    with engine.connect():
        _DB_AVAILABLE = True
except SQLAlchemyError:
    _DB_AVAILABLE = False
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_FAKE_DIM = 8
_FAKE_VECTOR = [0.1] * _FAKE_DIM


def setup_test_db():
    if not _DB_AVAILABLE:
        pytest.skip("PostgreSQL test database unavailable; skipping collection tests.")
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def client(db_session):
    from api.main import app
    from config.database import get_db

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def db_session():
    setup_test_db()
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def text_collection(client):
    payload = {
        "name": "Test Docs",
        "collection_id": "test-docs-phase2",
        "modality": "text",
        "dimension": _FAKE_DIM,
        "embedding_model": "fake-model",
    }
    response = client.post("/collections", json=payload)
    assert response.status_code == 201
    return payload["collection_id"]


@patch("services.embedding_service.embed_text", return_value=_FAKE_VECTOR)
def test_text_ingest_and_search(mock_embed, client, text_collection):
    ingest = client.post(
        f"/collections/{text_collection}/ingest/text",
        json={"text": "The quick brown fox", "metadata": {"source": "unit-test"}},
    )
    assert ingest.status_code == 201
    data = ingest.json()
    assert data["success"] is True
    assert mock_embed.called

    search = client.post(
        f"/collections/{text_collection}/search",
        json={"query": "brown fox", "k": 3, "method": "brute"},
    )
    assert search.status_code == 200
    search_data = search.json()
    assert search_data["success"] is True
    assert search_data.get("query_text") == "brown fox"
    assert search_data["total_results"] >= 1


@patch("services.embedding_service.embed_text", return_value=_FAKE_VECTOR)
def test_search_with_metadata_filter(mock_embed, client, text_collection):
    client.post(
        f"/collections/{text_collection}/ingest/text",
        json={"text": "alpha doc", "metadata": {"tier": "gold"}},
    )
    client.post(
        f"/collections/{text_collection}/ingest/text",
        json={"text": "beta doc", "metadata": {"tier": "silver"}},
    )

    search = client.post(
        f"/collections/{text_collection}/search",
        json={"query": "doc", "k": 5, "filters": {"tier": "gold"}},
    )
    assert search.status_code == 200
    results = search.json()["results"]
    assert all(r.get("metadata", {}).get("tier") == "gold" for r in results)


def test_global_search_filters(client):
    client.post(
        "/vectors",
        json={
            "vector": [1.0, 0.0, 0.0],
            "vector_id": "filter_a",
            "metadata": {"category": "x"},
        },
    )
    client.post(
        "/vectors",
        json={
            "vector": [0.0, 1.0, 0.0],
            "vector_id": "filter_b",
            "metadata": {"category": "y"},
        },
    )

    response = client.post(
        "/search",
        json={
            "query_vector": [1.0, 0.0, 0.0],
            "k": 5,
            "method": "brute",
            "filters": {"category": "x"},
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["vector_id"] == "filter_a"
