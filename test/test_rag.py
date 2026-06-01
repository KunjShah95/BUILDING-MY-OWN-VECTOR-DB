"""Tests for RAG pipeline."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest


class TestTextChunker:
    def test_chunk_recursive(self):
        from utils.text_chunker import chunk_text_recursive
        text = "This is a test. " * 500
        chunks = chunk_text_recursive(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_recursive_empty(self):
        from utils.text_chunker import chunk_text_recursive
        assert chunk_text_recursive("") == []

    def test_chunk_by_sentences(self):
        from utils.text_chunker import chunk_by_sentences
        text = "First sentence. Second sentence. Third sentence. " * 10
        chunks = chunk_by_sentences(text, max_sentences=3)
        assert len(chunks) > 5

    def test_chunk_tokens_fallback(self):
        from utils.text_chunker import chunk_tokens
        text = "word " * 1000
        chunks = chunk_tokens(text, chunk_size=100, overlap=10)
        assert len(chunks) >= 1


class TestPDFProcessor:
    def test_extract_text_function_exists(self):
        from utils.pdf_processor import extract_text_from_pdf
        assert callable(extract_text_from_pdf)

    def test_extract_text_pages_exists(self):
        from utils.pdf_processor import extract_text_pages
        assert callable(extract_text_pages)


class TestRAGService:
    @patch("services.rag_service.embed_texts")
    def test_ingest_pdf(self, mock_embed_texts):
        from services.rag_service import RAGService
        mock_embed_texts.return_value = [[0.1] * 384]
        service = RAGService(db_session=None)
        service._get_collection_service = MagicMock()
        service._get_collection_service().get_collection.return_value = {
            "success": True, "collection": {"collection_id": "c1", "modality": "text", "dimension": 384}
        }
        service._get_vector_service = MagicMock()
        service._get_vector_service().create_vector.return_value = {"success": True, "vector_id": "v1"}
        result = service.ingest_pdf(collection_id="c1", pdf_path="/nonexistent/test.pdf")
        assert "success" in result

    @patch("services.rag_service.embed_text")
    @patch("services.rag_service.openai_chat_completion")
    def test_query(self, mock_llm, mock_embed):
        from services.rag_service import RAGService
        mock_embed.return_value = [0.1] * 384
        mock_llm.return_value = "This is a test answer."
        service = RAGService(db_session=None)
        service._search_vectors = MagicMock(return_value={
            "success": True,
            "results": [{"vector_id": "v1", "distance": 0.1, "metadata": {"text": "context text", "source": "doc.pdf"}}]
        })
        result = service.query(collection_id="c1", query="test question")
        assert result["success"]
        assert result["answer"] == "This is a test answer."

    @patch("services.rag_service.embed_text")
    def test_query_no_results(self, mock_embed):
        from services.rag_service import RAGService
        mock_embed.return_value = [0.1] * 384
        service = RAGService(db_session=None)
        service._search_vectors = MagicMock(return_value={"success": True, "results": []})
        result = service.query(collection_id="c1", query="test")
        assert result["success"]
        assert result["answer"] == "No relevant documents found."

    def test_openai_fn_no_key(self):
        from services.rag_service import openai_chat_completion
        result = openai_chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            api_key="",
        )
        assert isinstance(result, str)
