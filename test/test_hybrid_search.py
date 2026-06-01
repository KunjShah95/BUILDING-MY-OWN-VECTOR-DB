"""Tests for BM25 index and hybrid search engine."""

import json
import math
import pytest
import numpy as np

from utils.bm25_index import BM25Index
from utils.hybrid_search import reciprocal_rank_fusion, HybridSearchEngine


# ---- BM25 Index Tests -----------------------------------------------------


class TestBM25Index:
    def test_add_and_search(self):
        index = BM25Index()
        index.add_document("cat cat cat dog", "doc1")       # cat appears 3x
        index.add_document("the dog chased the cat", "doc2")  # cat appears 1x
        index.add_document("the bird flew over the mat", "doc3")  # cat 0x

        results = index.search("cat", k=5)
        assert len(results) == 2  # doc3 has zero matches
        assert all(isinstance(doc_id, str) and isinstance(score, float) for doc_id, score in results)
        # doc1 has highest TF of "cat" (3x) -> highest BM25 score
        assert results[0][0] == "doc1"

    def test_search_empty_index(self):
        index = BM25Index()
        results = index.search("anything", k=5)
        assert results == []

    def test_search_unknown_term(self):
        index = BM25Index()
        index.add_document("hello world", "doc1")
        results = index.search("zzzzz", k=5)
        assert results == []

    def test_add_and_remove(self):
        index = BM25Index()
        index.add_document("the cat sat on the mat", "doc1")
        index.add_document("the dog chased the cat", "doc2")
        assert len(index) == 2

        removed = index.remove_document("doc1")
        assert removed is True
        assert len(index) == 1
        assert "doc1" not in index.doc_ids

        removed = index.remove_document("nonexistent")
        assert removed is False

    def test_replace_document(self):
        index = BM25Index()
        index.add_document("cat cat cat", "doc1")
        results = index.search("cat", k=5)
        assert results[0][1] > 0

        index.add_document("dog dog dog", "doc1")  # replace
        results = index.search("cat", k=5)
        assert len(results) == 0
        results = index.search("dog", k=5)
        assert len(results) == 1

    def test_empty_text(self):
        index = BM25Index()
        index.add_document("", "doc1")
        assert len(index) == 0
        assert index.search("anything", k=5) == []

    def test_save_and_load(self, tmp_path):
        index = BM25Index(k1=1.2, b=0.5)
        index.add_document("the cat sat", "doc1")
        index.add_document("the dog ran", "doc2")

        path = str(tmp_path / "bm25.json")
        index.save(path)

        loaded = BM25Index.load(path)
        assert loaded.k1 == 1.2
        assert loaded.b == 0.5
        assert len(loaded) == 2
        assert loaded.doc_texts == index.doc_texts

        results = loaded.search("cat", k=5)
        assert len(results) == 1

    def test_clear(self):
        index = BM25Index()
        index.add_document("hello world", "doc1")
        index.clear()
        assert len(index) == 0
        assert index.search("hello", k=5) == []

    def test_batch_add(self):
        index = BM25Index()
        texts = ["the cat sat", "the dog ran", "the bird flew"]
        doc_ids = ["doc1", "doc2", "doc3"]
        index.add_documents(texts, doc_ids)
        assert len(index) == 3
        results = index.search("cat", k=5)
        assert len(results) == 1

    def test_different_query_term_counts(self):
        """Multi-occurrence query terms should boost scores."""
        index = BM25Index()
        index.add_document("cat cat cat", "doc1")
        index.add_document("cat dog", "doc2")

        results = index.search("cat cat", k=5)
        assert results[0][0] == "doc1"  # doc1 has higher cat frequency


# ---- RRF Fusion Tests -----------------------------------------------------


class TestReciprocalRankFusion:
    def test_fuse_both_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_fuse_dense_only(self):
        dense = [
            {"vector_id": "a", "distance": 0.1},
            {"vector_id": "b", "distance": 0.2},
        ]
        fused = reciprocal_rank_fusion(dense, [], top_n=2)
        assert len(fused) == 2
        assert fused[0]["vector_id"] == "a"

    def test_fuse_sparse_only(self):
        sparse = [
            {"vector_id": "b", "distance": 0.3},
            {"vector_id": "a", "distance": 0.4},
        ]
        fused = reciprocal_rank_fusion([], sparse, top_n=2)
        assert len(fused) == 2

    def test_fuse_combined(self):
        dense = [
            {"vector_id": "a", "distance": 0.1, "metadata": {"source": "dense"}},
            {"vector_id": "b", "distance": 0.2},
            {"vector_id": "c", "distance": 0.3},
        ]
        sparse = [
            {"vector_id": "b", "distance": 0.15},
            {"vector_id": "d", "distance": 0.25},
        ]
        fused = reciprocal_rank_fusion(dense, sparse, top_n=4)
        assert len(fused) == 4

        # "b" appears in both lists -> highest RRF score
        assert fused[0]["vector_id"] == "b"
        assert "rrf_score" in fused[0]
        assert "sources" in fused[0]
        assert "dense" in fused[0]["sources"]
        assert "sparse" in fused[0]["sources"]

    def test_rrf_score_order(self):
        """Documents appearing in both ranked lists should rank higher."""
        dense = [{"vector_id": f"d{i}"} for i in range(10)]
        sparse = [{"vector_id": "d0", "distance": 0.1}, {"vector_id": "d999"}]
        fused = reciprocal_rank_fusion(dense, sparse, top_n=10)
        # d0 should be first (appears in both lists)
        assert fused[0]["vector_id"] == "d0"

    def test_respects_top_n(self):
        dense = [{"vector_id": f"d{i}"} for i in range(50)]
        fused = reciprocal_rank_fusion(dense, [], top_n=10)
        assert len(fused) == 10

    def test_missing_vector_id(self):
        dense = [{"id": "a", "distance": 0.1}]
        fused = reciprocal_rank_fusion(dense, [], top_n=5)
        assert len(fused) == 1
        assert fused[0]["vector_id"] == "a"


# ---- HybridSearchEngine Tests ----------------------------------------------


class TestHybridSearchEngine:
    def test_engine_basic(self):
        engine = HybridSearchEngine(bm25_index=None)

        def dense_fn(**kwargs):
            return {"results": [{"vector_id": "a", "distance": 0.1}]}

        result = engine.search(
            dense_search_fn=dense_fn,
            query_vector=[0.1, 0.2],
            query_text="",
            k=5,
        )
        assert result["success"] is True
        assert len(result["results"]) == 1

    def test_engine_with_bm25(self):
        bm25 = BM25Index()
        bm25.add_document("cat sat on mat", "a")
        bm25.add_document("dog chased cat", "b")
        engine = HybridSearchEngine(bm25_index=bm25)

        def dense_fn(**kwargs):
            return {"results": [{"vector_id": "b", "distance": 0.1}]}

        result = engine.search(
            dense_search_fn=dense_fn,
            query_vector=[0.1, 0.2],
            query_text="cat",
            k=5,
        )
        assert result["success"] is True
        assert result["sparse_count"] > 0
        assert len(result["results"]) >= 2

    def test_engine_dense_k_multiplier(self):
        bm25 = BM25Index()
        engine = HybridSearchEngine(bm25_index=bm25)

        call_count = 0

        def dense_fn(**kwargs):
            nonlocal call_count
            call_count += 1
            assert kwargs.get("k") == 20  # 10 * 2
            return {"results": [{"vector_id": "a", "distance": 0.1}]}

        engine.search(dense_fn, [0.1], "", k=10, dense_k_multiplier=2)
        assert call_count == 1
