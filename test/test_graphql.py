"""Tests for Phase 18 GraphQL API (schema + resolver logic, no live DB/embed)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ── Schema introspection ──────────────────────────────────────────────────────

def test_schema_has_query_and_mutation():
    from api.graphql_schema import schema
    sdl = schema.as_str()
    assert "type Query" in sdl
    assert "type Mutation" in sdl


def test_schema_search_field():
    from api.graphql_schema import schema
    sdl = schema.as_str()
    assert "search(" in sdl
    assert "SearchResult" in sdl


def test_schema_collections_field():
    from api.graphql_schema import schema
    assert "collections" in schema.as_str()


def test_schema_web_search_field():
    from api.graphql_schema import schema
    assert "webSearch(" in schema.as_str()


def test_schema_insert_vector_mutation():
    from api.graphql_schema import schema
    assert "insertVector(" in schema.as_str()


def test_schema_delete_vector_mutation():
    from api.graphql_schema import schema
    assert "deleteVector(" in schema.as_str()


def test_schema_crawl_url_mutation():
    from api.graphql_schema import schema
    assert "crawlUrl(" in schema.as_str()


# ── Type helpers ──────────────────────────────────────────────────────────────

def test_search_result_from_dict():
    from api.graphql_schema import SearchResult
    r = SearchResult.from_dict({"vector_id": "v1", "score": 0.95, "metadata": {"k": "v"}})
    assert r.vector_id == "v1"
    assert r.score == 0.95
    assert json.loads(r.metadata) == {"k": "v"}


def test_search_result_no_metadata():
    from api.graphql_schema import SearchResult
    r = SearchResult.from_dict({"vector_id": "v2", "score": 0.5})
    assert r.metadata is None


def test_web_result_from_dict():
    from api.graphql_schema import WebResult
    r = WebResult.from_dict({
        "url": "https://example.com",
        "title": "Example",
        "snippet": "A test snippet",
        "score": 0.88,
    })
    assert r.url == "https://example.com"
    assert r.score == 0.88


def test_search_result_uses_distance_fallback():
    from api.graphql_schema import SearchResult
    r = SearchResult.from_dict({"id": "v3", "distance": 0.3})
    assert r.vector_id == "v3"
    assert r.score == 0.3


# ── Query execution (mocked backends) ────────────────────────────────────────

def _exec(query: str, variables: dict = None):
    """Execute a GraphQL query against the schema and return (data, errors)."""
    from api.graphql_schema import schema
    result = schema.execute_sync(query, variable_values=variables)
    return result.data, result.errors


def test_collections_query_empty(tmp_path, monkeypatch):
    """collections() reads index dirs — empty tmp dir → empty list."""
    import api.graphql_schema as gs
    monkeypatch.setattr(gs, "_INDEX_ROOT_FOR_COLLECTIONS", str(tmp_path))
    data, errors = _exec("{ collections { collectionId vectorCount hasSparse } }")
    assert errors is None
    assert data["collections"] == []


def test_collections_query_with_dir(tmp_path, monkeypatch):
    import api.graphql_schema as gs
    col = tmp_path / "mycol"
    col.mkdir()
    (col / "sparse.json").write_text('{"doc_count": 42}')
    monkeypatch.setattr(gs, "_INDEX_ROOT_FOR_COLLECTIONS", str(tmp_path))
    data, errors = _exec("{ collections { collectionId vectorCount hasSparse } }")
    assert errors is None
    cols = data["collections"]
    assert len(cols) == 1
    assert cols[0]["collectionId"] == "mycol"
    assert cols[0]["hasSparse"] is True
    assert cols[0]["vectorCount"] == 42


def test_index_stats_query(monkeypatch):
    import api.graphql_schema as gs
    import api.routers.web_search as ws
    monkeypatch.setattr(ws, "_search_cache", {})
    monkeypatch.setattr(gs, "_INDEX_ROOT_FOR_COLLECTIONS", "/nonexistent_xyz")
    data, errors = _exec(
        '{ indexStats(collectionId: "nonexistent") { collectionId totalDocs indexTypes cacheEntries } }'
    )
    assert errors is None
    assert data["indexStats"]["collectionId"] == "nonexistent"
    assert data["indexStats"]["cacheEntries"] == 0


def test_search_query_accepts_args():
    """Schema accepts search args — structural check."""
    from api.graphql_schema import schema
    assert "topK" in schema.as_str()


def test_search_result_list_type():
    from api.graphql_schema import schema
    sdl = schema.as_str()
    assert "[SearchResult!]!" in sdl


def test_mutation_result_has_vector_id():
    from api.graphql_schema import schema
    sdl = schema.as_str()
    assert "vectorId" in sdl


def test_crawl_job_type():
    from api.graphql_schema import schema
    sdl = schema.as_str()
    assert "CrawlJob" in sdl
    assert "jobId" in sdl


def test_web_search_response_type():
    from api.graphql_schema import schema
    sdl = schema.as_str()
    assert "WebSearchResponse" in sdl
    assert "route" in sdl


# ── Mutation direct tests ─────────────────────────────────────────────────────

def test_insert_vector_mutation_success(monkeypatch):
    import api.graphql_schema as gs
    mock_vs = MagicMock()
    mock_vs.store_vector.return_value = {"success": True, "message": "ok"}
    monkeypatch.setattr(gs, "_get_db", lambda: MagicMock())

    from api.graphql_schema import Mutation
    m = Mutation()
    with patch("services.vector_service.VectorService", return_value=mock_vs):
        result = m.insert_vector(
            collection_id="col1",
            vector_id="v42",
            vector=[0.1, 0.2, 0.3],
            metadata=None,
        )
    assert result.success is True
    assert result.vector_id == "v42"


def test_insert_vector_mutation_with_metadata(monkeypatch):
    import api.graphql_schema as gs
    mock_vs = MagicMock()
    mock_vs.store_vector.return_value = {"success": True, "message": "ok"}
    monkeypatch.setattr(gs, "_get_db", lambda: MagicMock())

    from api.graphql_schema import Mutation
    m = Mutation()
    with patch("services.vector_service.VectorService", return_value=mock_vs):
        result = m.insert_vector(
            collection_id="col1",
            vector_id="v43",
            vector=[0.5],
            metadata='{"label": "cat"}',
        )
    assert result.success is True
    call_kwargs = mock_vs.store_vector.call_args
    meta_passed = (
        call_kwargs.kwargs.get("metadata")
        or (call_kwargs.args[3] if len(call_kwargs.args) > 3 else None)
    )
    assert meta_passed == {"label": "cat"}


def test_delete_vector_mutation_success(monkeypatch):
    import api.graphql_schema as gs
    mock_vs = MagicMock()
    mock_vs.delete_vector.return_value = {"success": True, "message": "deleted"}
    monkeypatch.setattr(gs, "_get_db", lambda: MagicMock())

    from api.graphql_schema import Mutation
    m = Mutation()
    with patch("services.vector_service.VectorService", return_value=mock_vs):
        result = m.delete_vector(vector_id="v1")
    assert result.success is True
    assert result.vector_id == "v1"


def test_delete_vector_mutation_not_found(monkeypatch):
    import api.graphql_schema as gs
    mock_vs = MagicMock()
    mock_vs.delete_vector.return_value = {"success": False, "message": "not found"}
    monkeypatch.setattr(gs, "_get_db", lambda: MagicMock())

    from api.graphql_schema import Mutation
    m = Mutation()
    with patch("services.vector_service.VectorService", return_value=mock_vs):
        result = m.delete_vector(vector_id="ghost")
    assert result.success is False


def test_insert_vector_mutation_exception(monkeypatch):
    import api.graphql_schema as gs
    mock_vs = MagicMock()
    mock_vs.store_vector.side_effect = RuntimeError("db down")
    monkeypatch.setattr(gs, "_get_db", lambda: MagicMock())

    from api.graphql_schema import Mutation
    m = Mutation()
    with patch("services.vector_service.VectorService", return_value=mock_vs):
        result = m.insert_vector("col", "v1", [1.0], None)
    assert result.success is False
    assert "db down" in result.message
