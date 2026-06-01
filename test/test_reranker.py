"""Tests for cross-encoder re-ranking service."""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from services.reranker_service import RerankerService, _extract_text


# ---- Text Extraction Tests ------------------------------------------------


class TestExtractText:
    def test_extract_from_metadata_text(self):
        result = {"vector_id": "v1", "metadata": {"text": "hello world"}}
        assert _extract_text(result) == "hello world"

    def test_extract_from_metadata_content(self):
        result = {"vector_id": "v2", "metadata": {"content": "some content"}}
        assert _extract_text(result) == "some content"

    def test_extract_from_metadata_chunk_text(self):
        result = {"vector_id": "v3", "metadata": {"chunk_text": "chunk content"}}
        assert _extract_text(result) == "chunk content"

    def test_extract_from_top_level_text(self):
        result = {"vector_id": "v4", "text": "top level text"}
        assert _extract_text(result) == "top level text"

    def test_extract_from_top_level_content(self):
        result = {"vector_id": "v5", "content": "top level content"}
        assert _extract_text(result) == "top level content"

    def test_extract_falls_back_to_vector_id(self):
        result = {"vector_id": "v6"}
        assert _extract_text(result) == "v6"

    def test_extract_uses_meta_data_alias(self):
        result = {"vector_id": "v7", "meta_data": {"text": "from meta_data"}}
        assert _extract_text(result) == "from meta_data"

    def test_extract_skips_empty_text(self):
        result = {"vector_id": "v8", "metadata": {"text": "", "content": ""}}
        assert _extract_text(result) == "v8"


# ---- RerankerService Tests (mocked model) ---------------------------------


class FakeCrossEncoder:
    """Simulates a cross-encoder that scores (query, text) pairs."""

    def predict(self, pairs):
        # Return scores proportional to how much text matches "important"
        scores = []
        for query, text in pairs:
            if "important" in text.lower():
                scores.append(0.95)
            elif "relevant" in text.lower():
                scores.append(0.85)
            else:
                scores.append(0.1)
        return scores


class TestRerankerServiceLocal:
    @pytest.fixture
    def reranker(self):
        with patch(
            "services.reranker_service.RerankerService._load_model",
            return_value=True,
        ):
            rs = RerankerService()
            rs._model = FakeCrossEncoder()
            rs._model_loaded = True
            yield rs

    def make_result(self, vid, text, distance=0.5):
        return {
            "vector_id": vid,
            "distance": distance,
            "metadata": {"text": text},
        }

    def test_rerank_returns_sorted_by_relevance(self, reranker):
        results = [
            self.make_result("v1", "unrelated document", 0.1),
            self.make_result("v2", "important document about AI", 0.3),
            self.make_result("v3", "another important result", 0.5),
        ]
        reranked = reranker.rerank("AI query", results, top_k=3)
        assert len(reranked) == 3
        # Important docs should rank higher than unrelated
        assert reranked[0]["vector_id"] in ("v2", "v3")
        assert reranked[-1]["vector_id"] == "v1"

    def test_rerank_top_k_limits_results(self, reranker):
        results = [self.make_result(f"v{i}", f"text {i}") for i in range(10)]
        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3

    def test_rerank_adds_rerank_score(self, reranker):
        results = [self.make_result("v1", "important text")]
        reranked = reranker.rerank("query", results, top_k=1)
        assert "rerank_score" in reranked[0]
        assert isinstance(reranked[0]["rerank_score"], float)

    def test_rerank_empty_results(self, reranker):
        assert reranker.rerank("query", []) == []

    def test_rerank_no_text_in_results(self, reranker):
        results = [{"vector_id": "v1"}, {"vector_id": "v2"}]
        reranked = reranker.rerank("query", results, top_k=5)
        assert len(reranked) == 2  # returns original ordering

    def test_rerank_can_omit_score(self, reranker):
        results = [self.make_result("v1", "important text")]
        reranked = reranker.rerank("query", results, top_k=1, keep_rerank_score=False)
        assert "rerank_score" not in reranked[0]

    def test_rerank_some_results_missing_text(self, reranker):
        results = [
            self.make_result("v1", "important document"),
            {"vector_id": "v2"},  # no text
            self.make_result("v3", "also important"),
        ]
        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3
        # v2 has no text so it should be at the bottom
        assert reranked[-1]["vector_id"] == "v2"


class TestRerankerServiceUnavailable:
    def test_rerank_fallback_on_model_unavailable(self):
        """When model is unavailable, should return original ordering."""
        with patch(
            "services.reranker_service.RerankerService._load_model",
            return_value=False,
        ):
            rs = RerankerService()
            results = [
                {"vector_id": "v1", "distance": 0.1, "metadata": {"text": "a"}},
                {"vector_id": "v2", "distance": 0.2, "metadata": {"text": "b"}},
            ]
            reranked = rs.rerank("query", results, top_k=2)
            # Should return original ordering (already sorted by distance)
            assert len(reranked) == 2
            assert reranked[0]["vector_id"] == "v1"


class TestRerankerServiceCohere:
    def test_cohere_fallback(self):
        """Test Cohere API fallback when local model unavailable but key is set."""
        with patch(
            "services.reranker_service.RerankerService._load_model",
            return_value=False,
        ):
            with patch(
                "services.reranker_service.RerankerService._rerank_cohere"
            ) as mock_cohere:
                mock_cohere.return_value = [
                    {"vector_id": "v2", "rerank_score": 0.95},
                    {"vector_id": "v1", "rerank_score": 0.3},
                ]
                rs = RerankerService(cohere_api_key="test-key")
                results = [
                    {"vector_id": "v1", "metadata": {"text": "unrelated"}},
                    {"vector_id": "v2", "metadata": {"text": "important"}},
                ]
                reranked = rs.rerank("query", results, top_k=2)
                assert reranked[0]["vector_id"] == "v2"
                mock_cohere.assert_called_once()


class TestBatchRerank:
    def test_batch_rerank(self):
        with patch(
            "services.reranker_service.RerankerService._load_model",
            return_value=True,
        ):
            rs = RerankerService()
            rs._model = FakeCrossEncoder()
            rs._model_loaded = True
            queries = ["q1", "q2"]
            results_batch = [
                [
                    {"vector_id": "v1", "metadata": {"text": "important"}},
                    {"vector_id": "v2", "metadata": {"text": "junk"}},
                ],
                [
                    {"vector_id": "v3", "metadata": {"text": "relevant"}},
                    {"vector_id": "v4", "metadata": {"text": "noise"}},
                ],
            ]
            reranked_batch = rs.batch_rerank(queries, results_batch, top_k=2)
            assert len(reranked_batch) == 2
            assert reranked_batch[0][0]["vector_id"] == "v1"
            assert reranked_batch[1][0]["vector_id"] == "v3"


class TestIsAvailable:
    def test_is_available_true(self):
        with patch(
            "services.reranker_service.RerankerService._load_model",
            return_value=True,
        ):
            rs = RerankerService()
            assert rs.is_available is True

    def test_is_available_false(self):
        with patch(
            "services.reranker_service.RerankerService._load_model",
            return_value=False,
        ):
            rs = RerankerService()
            assert rs.is_available is False
