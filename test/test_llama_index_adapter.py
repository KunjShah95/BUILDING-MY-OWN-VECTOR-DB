"""Tests for LlamaIndex VectorStore adapter (Phase 3)."""

import pytest
from unittest.mock import MagicMock, patch

from utils.llama_index_adapter import LlamaIndexVectorStore


class TestLlamaIndexInit:
    def test_init(self):
        store = LlamaIndexVectorStore(collection_id="docs", api_url="http://localhost:8000", api_key="key")
        assert store.collection_id == "docs"
        assert store.api_url == "http://localhost:8000"
        assert store.api_key == "key"

    def test_trailing_slash_stripped(self):
        store = LlamaIndexVectorStore(collection_id="d", api_url="http://h:8000/")
        assert store.api_url == "http://h:8000"

    def test_client_property(self):
        store = LlamaIndexVectorStore(collection_id="d")
        assert store.client is store


class TestLlamaIndexHeaders:
    def test_headers_no_key(self):
        store = LlamaIndexVectorStore(collection_id="d")
        h = store._headers()
        assert "Content-Type" in h
        assert "X-API-Key" not in h

    def test_headers_with_key(self):
        store = LlamaIndexVectorStore(collection_id="d", api_key="sk-test")
        h = store._headers()
        assert h["X-API-Key"] == "sk-test"


class TestLlamaIndexAdd:
    def test_add_success(self):
        store = LlamaIndexVectorStore(collection_id="d", api_url="http://h:8000")
        mock_node = MagicMock()
        mock_node.embedding = [0.1, 0.2, 0.3]
        mock_node.text = "hello world"
        mock_node.node_id = "node_1"
        mock_node.metadata = {"source": "test"}

        with patch.object(store, "_request") as mock_req:
            mock_req.return_value = {"success": True, "vector_ids": ["llama_node_1"]}
            ids = store.add([mock_node])
            assert ids == ["llama_node_1"]
            mock_req.assert_called_once()

    def test_add_failure(self):
        store = LlamaIndexVectorStore(collection_id="d", api_url="http://h:8000")
        with patch.object(store, "_request") as mock_req:
            mock_req.return_value = {"success": False, "message": "error"}
            ids = store.add([MagicMock(embedding=[0.1], text="x", node_id="n1", metadata={})])
            assert ids == []


class TestLlamaIndexDelete:
    def test_delete(self):
        store = LlamaIndexVectorStore(collection_id="d", api_url="http://h:8000")
        with patch.object(store, "_request") as mock_req:
            store.delete("doc_123")
            mock_req.assert_called_once_with("DELETE", "/vectors/llama_doc_123")


class TestLlamaIndexQuery:
    def test_query_with_results(self):
        store = LlamaIndexVectorStore(collection_id="d", api_url="http://h:8000")
        mock_query = MagicMock()
        mock_query.query_embedding = [0.1, 0.2, 0.3]
        mock_query.similarity_top_k = 3

        with patch.object(store, "_request") as mock_req:
            mock_req.return_value = {
                "results": [
                    {"vector_id": "v1", "distance": 0.1, "metadata": {"text": "doc1"}},
                    {"vector_id": "v2", "distance": 0.2, "metadata": {"text": "doc2"}},
                ]
            }
            result = store.query(mock_query)
            assert hasattr(result, "nodes")

    def test_query_no_langchain(self):
        store = LlamaIndexVectorStore(collection_id="d")
        with patch.dict("sys.modules", {"llama_index.core.vector_stores.types": None}):
            result = store.query(MagicMock())
            # Should handle gracefully (fallback object or error)
            assert result is not None


class TestLlamaIndexPersist:
    def test_persist_is_noop(self):
        store = LlamaIndexVectorStore(collection_id="d")
        store.persist("/some/path")  # Should not raise
