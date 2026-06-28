import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from database.schema import Base
from services.media_store import save_media

TEST_DATABASE_URL = "postgresql://user:password@localhost:5432/vector_db_test"
engine = create_engine(TEST_DATABASE_URL)
_DB_AVAILABLE = False
try:
    with engine.connect():
        _DB_AVAILABLE = True
except SQLAlchemyError:
    _DB_AVAILABLE = False

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_IMAGE_DIM = 4
_AUDIO_DIM = 4


def _mock_embed_image(source, model_name=None):
    return [1.0, 0.0, 0.0, 0.0]


def _mock_embed_audio(source, model_name=None):
    return [0.0, 1.0, 0.0, 0.0]


def setup_test_db():
    if not _DB_AVAILABLE:
        pytest.skip("PostgreSQL test database unavailable; skipping media tests.")
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


def test_media_store_writes_under_collection(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_PATH", str(tmp_path / "media_storage"))
    from config.settings import get_settings

    get_settings.cache_clear()

    uri = save_media("photos", "cat.jpg", b"fake-image-bytes")
    assert "photos/" in uri and uri.endswith(".jpg")
    full = tmp_path / "media_storage" / "photos"
    assert full.exists()
    assert any(p.suffix == ".jpg" for p in full.iterdir())


@patch("services.multimodal_service.embed_image", side_effect=_mock_embed_image)
def test_ingest_image_endpoint(mock_embed, client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_PATH", str(tmp_path / "media_storage"))
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()

    col = client.post(
        "/collections",
        json={
            "name": "Photos",
            "collection_id": "test-photos",
            "modality": "image",
            "dimension": _IMAGE_DIM,
        },
    )
    assert col.status_code == 201

    files = {"file": ("sample.png", b"\x89PNG", "image/png")}
    data = {"metadata": json.dumps({"tag": "sample"}), "vector_id": "img-1"}
    resp = client.post("/collections/test-photos/ingest/image", files=files, data=data)
    assert resp.status_code == 201
    body = resp.json()
    assert body["success"] is True
    assert body["vector_id"] == "img-1"
    meta = body.get("vector", {}).get("meta_data") or body.get("vector", {}).get("metadata")
    if meta:
        assert meta.get("content_uri")


@patch("services.multimodal_service.embed_image", side_effect=_mock_embed_image)
def test_search_image_endpoint(mock_embed, client, tmp_path, monkeypatch):
    monkeypatch.setenv("MEDIA_STORAGE_PATH", str(tmp_path / "media_storage"))
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()

    client.post(
        "/collections",
        json={
            "name": "Photos2",
            "collection_id": "test-photos-search",
            "modality": "image",
            "dimension": _IMAGE_DIM,
        },
    )
    client.post(
        "/collections/test-photos-search/ingest/image",
        files={"file": ("a.png", b"png-a", "image/png")},
        data={"vector_id": "a"},
    )

    search = client.post(
        "/collections/test-photos-search/search/image",
        files={"file": ("query.png", b"png-q", "image/png")},
        data={"k": "2", "method": "brute"},
    )
    assert search.status_code == 200
    assert search.json()["success"] is True


@patch("services.multimodal_service.embed_audio", side_effect=_mock_embed_audio)
def test_ingest_audio_wrong_modality(mock_embed, client):
    client.post(
        "/collections",
        json={
            "name": "Text only",
            "collection_id": "text-only-audio",
            "modality": "text",
            "dimension": 3,
        },
    )
    resp = client.post(
        "/collections/text-only-audio/ingest/audio",
        files={"file": ("x.wav", b"RIFF", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "modality" in str(resp.json()).lower() or "allow" in str(resp.json()).lower()


def test_openapi_lists_media_routes():
    """OpenAPI structure check — no database required."""
    from api.main import app

    with TestClient(app) as spec_client:
        spec = spec_client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    assert "/collections/{collection_id}/ingest/image" in paths
    assert "/collections/{collection_id}/ingest/audio" in paths
    assert "/collections/{collection_id}/search/image" in paths
    assert "/collections/{collection_id}/search/audio" in paths
    assert "/media" in paths


@patch("services.multimodal_service.embed_clip_text", return_value=[1.0, 0.0, 0.0, 0.0])
@patch("services.multimodal_service.embed_image", return_value=[1.0, 0.0, 0.0, 0.0])
def test_text_search_on_clip_image_collection(mock_img, mock_clip, client, tmp_path, monkeypatch):
    """Text-to-image search uses CLIP text encoder on image collections."""
    monkeypatch.setenv("MEDIA_STORAGE_PATH", str(tmp_path / "media_storage"))
    from config import settings as settings_module

    settings_module.get_settings.cache_clear()

    client.post(
        "/collections",
        json={
            "name": "CLIP Photos",
            "collection_id": "clip-photos",
            "modality": "image",
            "embedding_model": "clip-ViT-B-32",
            "dimension": 4,
        },
    )
    client.post(
        "/collections/clip-photos/ingest/image",
        files={"file": ("a.png", b"png-a", "image/png")},
        data={"vector_id": "img-a"},
    )

    search = client.post(
        "/collections/clip-photos/search/text",
        json={"query": "a red shoe", "k": 1, "method": "brute"},
    )
    assert search.status_code == 200
    assert search.json()["success"] is True
    mock_clip.assert_called()
