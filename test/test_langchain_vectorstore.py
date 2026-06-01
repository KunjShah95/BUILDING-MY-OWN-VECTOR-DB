"""Tests for the LangChain VectorStore integration.

Uses mocks so no running server is required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import pytest
from langchain_core.documents import Document

from vector_db_client.langchain_vectorstore import VectorDBVectorStore


def _mock_response(json_data: dict) -> MagicMock:
    """Build a fake ``httpx.Response`` so ``raise_for_status`` can parse it."""
    resp = MagicMock(spec=httpx.Response)
    resp.is_success = True
    resp.status_code = 200
    resp.json.return_value = json_data
    return resp


# ---- Fake embedding ---------------------------------------------------------


class FakeEmbeddings:
    """Simulates LangChain Embeddings — returns deterministic vectors."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] * 4 for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text))] * 4


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Patch the underlying VectorDBClient so no real HTTP calls happen."""
    with patch(
        "vector_db_client.langchain_vectorstore.VectorDBClient"
    ) as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance

        # Wire sub-APIs
        instance.vectors = MagicMock()
        instance.collections = MagicMock()
        instance.post = MagicMock()

        yield instance


@pytest.fixture
def store_global(mock_client):
    """VectorStore without a collection_id (uses global endpoints)."""
    return VectorDBVectorStore(
        embedding=FakeEmbeddings(),
        base_url="http://test:8000",
    )


@pytest.fixture
def store_collection(mock_client):
    """VectorStore scoped to a collection."""
    return VectorDBVectorStore(
        embedding=FakeEmbeddings(),
        base_url="http://test:8000",
        collection_id="my_docs",
    )


# ---- Initialisation ---------------------------------------------------------


class TestInit:
    def test_creates_client(self):
        with patch(
            "vector_db_client.langchain_vectorstore.VectorDBClient"
        ) as mock_cls:
            VectorDBVectorStore(
                embedding=FakeEmbeddings(),
                base_url="http://custom:9000",
                timeout=30.0,
            )
        mock_cls.assert_called_once_with(
            base_url="http://custom:9000",
            timeout=30.0,
        )

    def test_creates_collection_store(self, mock_client):
        store = VectorDBVectorStore(
            embedding=FakeEmbeddings(),
            base_url="http://test:8000",
            collection_id="my_docs",
        )
        assert store.collection_id == "my_docs"


# ---- add_texts --------------------------------------------------------------


class TestAddTexts:
    def test_adds_single_text_global(self, store_global, mock_client):
        mock_client.vectors.create.return_value = {"vector_id": "v1"}
        ids = store_global.add_texts(["hello"])
        assert ids == ["v1"]
        mock_client.vectors.create.assert_called_once()
        call_kwargs = mock_client.vectors.create.call_args[1]
        # vector should be based on embedding: len("hello")=5 → 5.0 repeated
        assert call_kwargs["vector"] == [5.0, 5.0, 5.0, 5.0]
        assert call_kwargs["metadata"]["text"] == "hello"

    def test_adds_multiple_texts_global(self, store_global, mock_client):
        def side_effect(**kwargs):
            # Return sequential IDs
            nonlocal call_count
            call_count += 1
            return {"vector_id": f"v{call_count}"}

        call_count = 0
        mock_client.vectors.create.side_effect = side_effect

        ids = store_global.add_texts(["a", "ab", "abc"])
        assert ids == ["v1", "v2", "v3"]

    def test_adds_with_metadatas_global(self, store_global, mock_client):
        mock_client.vectors.create.return_value = {"vector_id": "v1"}
        ids = store_global.add_texts(
            ["hello"],
            metadatas=[{"source": "test"}],
        )
        assert ids == ["v1"]
        call_kwargs = mock_client.vectors.create.call_args[1]
        assert call_kwargs["metadata"]["source"] == "test"
        assert call_kwargs["metadata"]["text"] == "hello"

    def test_adds_text_collection(self, store_collection, mock_client):
        mock_client.post.return_value = _mock_response({"vector_id": "v1"})
        ids = store_collection.add_texts(["hello"])
        assert ids == ["v1"]
        mock_client.post.assert_called_once()
        call_path = mock_client.post.call_args[0][0]
        assert "/collections/my_docs/ingest/text" in call_path

    def test_empty_texts_returns_empty(self, store_global, mock_client):
        ids = store_global.add_texts([])
        assert ids == []


# ---- similarity_search ------------------------------------------------------


class TestSimilaritySearch:
    SEARCH_RESULT = {
        "results": [
            {
                "vector_id": "v1",
                "distance": 0.1,
                "metadata": {"text": "hello world", "source": "doc1"},
            },
            {
                "vector_id": "v2",
                "distance": 0.3,
                "metadata": {"text": "foo bar", "source": "doc2"},
            },
        ],
        "total_results": 2,
        "success": True,
    }

    def test_search_global(self, store_global, mock_client):
        mock_client.vectors.search.return_value = self.SEARCH_RESULT
        docs = store_global.similarity_search("hello", k=5)

        assert len(docs) == 2
        assert docs[0].page_content == "hello world"
        assert docs[0].metadata["source"] == "doc1"
        assert docs[0].metadata["vector_id"] == "v1"
        assert docs[0].metadata["distance"] == 0.1

        # Verify search was called with the right params
        mock_client.vectors.search.assert_called_once()
        call_kwargs = mock_client.vectors.search.call_args[1]
        assert call_kwargs["k"] == 5
        assert call_kwargs["method"] == "hnsw"
        # Query vector from FakeEmbeddings: len("hello")=5
        assert call_kwargs["query_vector"] == [5.0, 5.0, 5.0, 5.0]

    def test_search_collection(self, store_collection, mock_client):
        mock_client.post.return_value = _mock_response(self.SEARCH_RESULT)
        docs = store_collection.similarity_search("hello", k=3)

        assert len(docs) == 2
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/collections/my_docs/search/text" in call_args[0][0]
        assert call_args[1]["json"]["k"] == 3

    def test_search_with_filters(self, store_global, mock_client):
        mock_client.vectors.search.return_value = self.SEARCH_RESULT
        store_global.similarity_search(
            "hello",
            k=5,
            filters={"source": "doc1"},
        )
        call_kwargs = mock_client.vectors.search.call_args[1]
        assert call_kwargs["filters"] == {"source": "doc1"}

    def test_search_empty_results(self, store_global, mock_client):
        mock_client.vectors.search.return_value = {"results": []}
        docs = store_global.similarity_search("hello")
        assert docs == []


# ---- to_documents helpers ---------------------------------------------------


class TestToDocuments:
    def test_extracts_text_from_metadata(self, store_global):
        results = [
            {
                "vector_id": "v1",
                "distance": 0.1,
                "metadata": {"text": "hello"},
            },
        ]
        docs = store_global._to_documents(results)
        assert docs[0].page_content == "hello"

    def test_extracts_text_from_content_key(self, store_global):
        results = [
            {
                "vector_id": "v2",
                "distance": 0.2,
                "metadata": {"content": "content text"},
            },
        ]
        docs = store_global._to_documents(results)
        assert docs[0].page_content == "content text"

    def test_falls_back_to_vector_id(self, store_global):
        results = [
            {
                "vector_id": "v3",
                "distance": 0.3,
                "metadata": {},
            },
        ]
        docs = store_global._to_documents(results)
        assert docs[0].page_content == "v3"

    def test_metadata_is_copied_not_mutated(self, store_global):
        results = [
            {
                "vector_id": "v1",
                "distance": 0.1,
                "metadata": {"text": "hello", "source": "doc"},
            },
        ]
        docs = store_global._to_documents(results)
        assert "text" not in docs[0].metadata  # text was popped
        assert docs[0].metadata["source"] == "doc"
        assert docs[0].metadata["vector_id"] == "v1"


# ---- from_texts / from_documents -------------------------------------------


class TestFactoryMethods:
    def test_from_texts(self, mock_client):
        mock_client.vectors.create.return_value = {"vector_id": "v1"}
        store = VectorDBVectorStore.from_texts(
            ["hello", "world"],
            embedding=FakeEmbeddings(),
            base_url="http://test:8000",
        )
        assert isinstance(store, VectorDBVectorStore)
        assert mock_client.vectors.create.call_count == 2

    def test_from_texts_with_collection(self, mock_client):
        mock_client.post.return_value = _mock_response({"vector_id": "v1"})
        store = VectorDBVectorStore.from_texts(
            ["hello"],
            embedding=FakeEmbeddings(),
            base_url="http://test:8000",
            collection_id="my_docs",
        )
        assert store.collection_id == "my_docs"
        mock_client.post.assert_called_once()

    def test_from_documents(self, mock_client):
        mock_client.vectors.create.return_value = {"vector_id": "v1"}
        docs = [
            Document(page_content="hello", metadata={"source": "doc1"}),
            Document(page_content="world", metadata={"source": "doc2"}),
        ]
        store = VectorDBVectorStore.from_documents(
            docs,
            embedding=FakeEmbeddings(),
            base_url="http://test:8000",
        )
        assert isinstance(store, VectorDBVectorStore)
        assert mock_client.vectors.create.call_count == 2


# ---- delete -----------------------------------------------------------------


class TestDelete:
    def test_delete_single(self, store_global, mock_client):
        result = store_global.delete(ids=["v1"])
        assert result is True
        mock_client.vectors.delete.assert_called_once_with("v1")

    def test_delete_multiple(self, store_global, mock_client):
        result = store_global.delete(ids=["v1", "v2"])
        assert result is True
        assert mock_client.vectors.delete.call_count == 2

    def test_delete_empty(self, store_global, mock_client):
        result = store_global.delete(ids=[])
        assert result is False
        mock_client.vectors.delete.assert_not_called()

    def test_delete_handles_error_gracefully(self, store_global, mock_client):
        mock_client.vectors.delete.side_effect = Exception("API error")
        result = store_global.delete(ids=["v_bad"])
        assert result is False  # No deletes succeeded


# ---- as_retriever ----------------------------------------------------------


class TestAsRetriever:
    def test_as_retriever_returns_langchain_retriever(self, store_global, mock_client):
        mock_client.vectors.search.return_value = {
            "results": [
                {
                    "vector_id": "v1",
                    "distance": 0.1,
                    "metadata": {"text": "hello"},
                },
            ],
        }
        retriever = store_global.as_retriever()
        docs = retriever.invoke("hello")
        assert len(docs) == 1
        assert docs[0].page_content == "hello"


# ---- import guard -----------------------------------------------------------


class TestImportGuard:
    def test_vector_db_vector_store_is_importable(self):
        from vector_db_client import VectorDBVectorStore

        assert VectorDBVectorStore is not None
