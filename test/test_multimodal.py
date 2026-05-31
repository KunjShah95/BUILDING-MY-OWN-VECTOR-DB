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

_DIM = 3


def _mock_embed_text(text: str, model_name=None):
    """Deterministic tiny embeddings for CI without loading transformers."""
    t = text.lower()
    if "python" in t:
        return [1.0, 0.1, 0.0]
    if "javascript" in t or " js " in f" {t} ":
        return [0.1, 1.0, 0.0]
    if "database" in t or "sql" in t:
        return [0.0, 0.1, 1.0]
    return [0.33, 0.33, 0.34]


def setup_test_db():
    if not _DB_AVAILABLE:
        pytest.skip("PostgreSQL test database unavailable; skipping multimodal tests.")
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        pytest.skip(f"Could not create test database tables: {exc}")


@pytest.fixture
def db_session():
    setup_test_db()
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


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
def mock_embed():
    with patch("services.multimodal_service.embed_text", side_effect=_mock_embed_text):
        yield


@pytest.fixture
def collection_id(client):
    payload = {
        "name": "Test Docs",
        "collection_id": "test-multimodal-docs",
        "dimension": _DIM,
        "modality": "text",
    }
    response = client.post("/collections", json=payload)
    assert response.status_code == 201
    return payload["collection_id"]


def test_create_collection(client):
    response = client.post(
        "/collections",
        json={
            "name": "Standalone",
            "collection_id": "standalone-col",
            "dimension": _DIM,
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["success"] is True
    assert data["collection"]["collection_id"] == "standalone-col"


@patch("services.multimodal_service.embed_text", side_effect=_mock_embed_text)
def test_ingest_and_search_text(mock_embed, client, collection_id):
    ingest_python = client.post(
        f"/collections/{collection_id}/ingest/text",
        json={
            "text": "Python is a popular programming language for data science.",
            "vector_id": "doc-python",
        },
    )
    assert ingest_python.status_code == 201

    ingest_js = client.post(
        f"/collections/{collection_id}/ingest/text",
        json={
            "text": "JavaScript powers interactive web development in browsers.",
            "vector_id": "doc-javascript",
        },
    )
    assert ingest_js.status_code == 201

    search = client.post(
        f"/collections/{collection_id}/search/text",
        json={"query": "python programming", "k": 2, "method": "brute"},
    )
    assert search.status_code == 200
    body = search.json()
    assert body["success"] is True
    assert body["total_results"] >= 1
    top_id = body["results"][0]["vector_id"]
    assert top_id == "doc-python"


@patch("services.multimodal_service.embed_text", side_effect=_mock_embed_text)
def test_list_and_delete_collection(mock_embed, client, collection_id):
    listing = client.get("/collections")
    assert listing.status_code == 200
    ids = [c["collection_id"] for c in listing.json()["collections"]]
    assert collection_id in ids

    got = client.get(f"/collections/{collection_id}")
    assert got.status_code == 200

    deleted = client.delete(f"/collections/{collection_id}")
    assert deleted.status_code == 200

    missing = client.get(f"/collections/{collection_id}")
    assert missing.status_code == 404
