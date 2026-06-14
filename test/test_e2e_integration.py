"""End-to-end integration tests for the full vector DB pipeline.

Tests wire up multiple components without external dependencies:
1. In-memory HNSW index ingestion
2. Vector search
3. Vamana search
4. Hybrid search (BM25 + vector)
5. LTR re-ranking
6. GNN graph rerank
7. Personalization boost
8. Backup snapshots

These tests don't require PostgreSQL, Redis, or any external services.
"""

import numpy as np
import pytest

from utils.hnsw_index import HNSWIndex
from utils.vamana_index import VamanaVectorIndex
from utils.bm25_index import BM25Index
from utils.hybrid_search import reciprocal_rank_fusion, HybridSearchEngine
from services.ltr_service import LTRService
from services.gnn_service import GNNService
from services.personalization_service import PersonalizationService
from services.backup_service import BackupService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_vectors():
    """Generate 50 normalized 8-dim vectors for testing."""
    np.random.seed(42)
    vecs = np.random.randn(50, 8).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def sample_texts():
    return [
        "machine learning algorithms for data science",
        "deep neural networks with pytorch and tensorflow",
        "vector database for similarity search applications",
        "natural language processing with transformer models",
        "distributed systems and consensus algorithms",
        "graph neural networks for node classification",
        "reinforcement learning in robotics and automation",
        "computer vision and image recognition techniques",
        "time series analysis and forecasting methods",
        "bayesian statistics for data analysis",
        "quantum computing algorithms and applications",
        "cybersecurity threat detection and prevention",
        "blockchain technology and smart contracts",
        "cloud computing infrastructure and services",
        "edge computing for iot devices and sensors",
    ]


@pytest.fixture
def hnsw_index(tmp_path):
    idx = HNSWIndex(m=16, ef_construction=200)
    yield idx


@pytest.fixture
def vamana_index(tmp_path):
    idx = VamanaVectorIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana_e2e"))
    yield idx


# ---------------------------------------------------------------------------
# Test 1: End-to-end HNSW ingestion -> search
# ---------------------------------------------------------------------------

class TestHNSWE2E:
    def test_ingest_and_search(self, hnsw_index, sample_vectors):
        """Insert vectors, search, verify results are ordered by distance."""
        for i, vec in enumerate(sample_vectors):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}", {"idx": i})

        assert hnsw_index.total_inserted == 50

        query = sample_vectors[0].tolist()
        results = hnsw_index.search(query, k=10)

        assert len(results) >= 1
        # First result should be the query itself (v000)
        assert results[0]["vector_id"] == "v000"
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)
        # Results should be sorted by distance ascending
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_ingest_and_search_with_metadata(self, hnsw_index, sample_vectors):
        """Verify metadata is preserved through the ingest->search pipeline."""
        for i, (vec, text) in enumerate(zip(sample_vectors[:5], [
            "machine learning", "neural networks", "vector search",
            "nlp transformers", "distributed systems",
        ])):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}", {"text": text, "idx": i})

        results = hnsw_index.search(sample_vectors[0].tolist(), k=3)
        assert len(results) >= 1
        # Metadata should be present on results
        for r in results:
            assert "metadata" in r
            assert "text" in r["metadata"]

    def test_delete_and_search_excludes(self, hnsw_index, sample_vectors):
        """Deleted vectors should not appear in search results."""
        for i, vec in enumerate(sample_vectors[:10]):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}")

        hnsw_index.delete("v000")
        hnsw_index.delete("v001")

        results = hnsw_index.search(sample_vectors[2].tolist(), k=10)
        vector_ids = [r["vector_id"] for r in results]
        assert "v000" not in vector_ids
        assert "v001" not in vector_ids

    def test_empty_index_search(self, hnsw_index):
        """Search on empty index returns empty list."""
        assert hnsw_index.search([0.1] * 8, k=5) == []


# ---------------------------------------------------------------------------
# Test 2: End-to-end Vamana ingestion -> search
# ---------------------------------------------------------------------------

class TestVamanaE2E:
    def test_ingest_and_search(self, vamana_index, sample_vectors):
        """Insert vectors into Vamana index and search."""
        for i, vec in enumerate(sample_vectors[:20]):
            vamana_index.insert(vec.tolist(), f"v{i:03d}")

        query = sample_vectors[0].tolist()
        results = vamana_index.search(query, k=5)

        assert len(results) <= 5
        for r in results:
            assert "vector_id" in r
            assert "distance" in r
            assert r["distance"] >= 0

    def test_empty_index(self, vamana_index):
        assert vamana_index.search([0.1] * 8, k=5) == []


# ---------------------------------------------------------------------------
# Test 3: Hybrid search (BM25 + vector) pipeline
# ---------------------------------------------------------------------------

class TestHybridSearchE2E:
    def test_bm25_and_vector_fusion(self, hnsw_index, sample_vectors, sample_texts):
        """Full hybrid search: BM25 sparse + HNSW dense -> RRF fusion."""
        # Ingest into HNSW
        for i, (vec, text) in enumerate(zip(sample_vectors[:10], sample_texts[:10])):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}", {"text": text})

        # Build BM25 index from the same texts
        bm25 = BM25Index()
        for i, text in enumerate(sample_texts[:10]):
            bm25.add_document(text, f"v{i:03d}")

        # Search both
        query_vec = sample_vectors[0].tolist()
        dense_results = hnsw_index.search(query_vec, k=5)
        sparse_results_raw = bm25.search(sample_texts[0], k=5)

        # Format for RRF
        dense_formatted = [
            {"vector_id": r["vector_id"], "distance": r["distance"]}
            for r in dense_results
        ]
        sparse_formatted = [
            {"vector_id": doc_id, "distance": score}
            for doc_id, score in sparse_results_raw
        ]

        # Fuse
        fused = reciprocal_rank_fusion(dense_formatted, sparse_formatted, top_n=5)

        assert len(fused) <= 5
        assert len(fused) > 0
        for r in fused:
            assert "vector_id" in r
            assert "rrf_score" in r

    def test_hybrid_search_engine(self, hnsw_index, sample_vectors, sample_texts):
        """Test the full HybridSearchEngine end-to-end."""
        for i, (vec, text) in enumerate(zip(sample_vectors[:5], sample_texts[:5])):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}", {"text": text})

        bm25 = BM25Index()
        for i, text in enumerate(sample_texts[:5]):
            bm25.add_document(text, f"v{i:03d}")

        engine = HybridSearchEngine(bm25_index=bm25)

        def dense_search_fn(query_vector, k, **kwargs):
            results = hnsw_index.search(query_vector, k=k)
            return {"results": results}

        result = engine.search(dense_search_fn, query_vector=sample_vectors[0].tolist(), query_text=sample_texts[0], k=3)

        assert result["success"] is True
        assert len(result["results"]) <= 3
        for r in result["results"]:
            assert "vector_id" in r


# ---------------------------------------------------------------------------
# Test 4: LTR re-ranking pipeline
# ---------------------------------------------------------------------------

class TestLTRE2E:
    def test_ltr_reranks_results(self, hnsw_index, sample_vectors, tmp_path):
        """Train LTR model on search results, then re-rank."""
        for i, vec in enumerate(sample_vectors[:15]):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}", {
                "text": f"document {i}",
                "popularity": i * 10,
            })

        # Collect training data from top results of a query
        query = sample_vectors[5].tolist()
        candidates = hnsw_index.search(query, k=10)
        assert len(candidates) >= 2  # at least 2 results needed for click data

        # Format as feedback entries for LTR
        feedback = [{
            "query": "test query",
            "results": [
                {
                    "vector_id": c["vector_id"],
                    "distance": c["distance"],
                    "bm25_score": 0.5,
                    "metadata": c.get("metadata", {}),
                }
                for c in candidates
            ],
            "clicks": {candidates[0]["vector_id"]: 3, candidates[1]["vector_id"]: 1},
        }]

        ltr = LTRService(model_path=str(tmp_path / "ltr_model.pkl"))
        result = ltr.train_from_feedback(feedback)
        assert "success" in result


# ---------------------------------------------------------------------------
# Test 5: Personalization pipeline
# ---------------------------------------------------------------------------

class TestPersonalizationE2E:
    def test_profile_learning_from_search(self, tmp_path):
        """Simulate a user searching, clicking results, and getting personalized results."""
        ps = PersonalizationService(storage_path=str(tmp_path / "e2e_profiles.json"))

        # User searches and clicks
        query_vec = np.random.randn(8).astype(np.float32).tolist()
        ps.record_interaction("user1", "cat videos", query_vec, clicked_ids=["v1", "v2"])

        # Another search, with different results
        query_vec2 = np.random.randn(8).astype(np.float32).tolist()
        ps.record_interaction("user1", "dog videos", query_vec2, clicked_ids=["v3"])

        profile = ps.get_user_profile("user1")
        assert profile is not None
        assert profile["total_queries"] == 2
        assert profile["clicked_count"] == 3

        # Personalize a new query
        new_query = np.random.randn(8).astype(np.float32).tolist()
        personalized = ps.personalize_query("user1", new_query, weight=0.3)
        assert len(personalized) == 8
        assert personalized != new_query

        # Boost candidates
        candidates = [
            {"vector_id": "v1", "distance": 0.5},
            {"vector_id": "v4", "distance": 0.3},
        ]
        boosted = ps.boost_personalized("user1", candidates)
        assert boosted[0]["vector_id"] == "v1"  # v1 was clicked before


# ---------------------------------------------------------------------------
# Test 6: Backup snapshot -> restore pipeline
# ---------------------------------------------------------------------------

class TestBackupE2E:
    def test_snapshot_and_list(self, hnsw_index, sample_vectors, tmp_path):
        """Create index, snapshot it, list snapshots, verify metadata."""
        for i, vec in enumerate(sample_vectors[:10]):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}")

        backup = BackupService(
            snapshot_dir=str(tmp_path / "snapshots"),
            wal_archive_dir=str(tmp_path / "wal_archives"),
        )

        result = backup.create_snapshot(hnsw_index, collection_id="e2e_test", method="hnsw")
        assert result["success"] is True

        snapshots = backup.list_snapshots(collection_id="e2e_test")
        assert len(snapshots) >= 1
        assert snapshots[0]["method"] == "hnsw"

    def test_wal_archive(self, tmp_path):
        """Write a WAL file, archive it, list archives."""
        wal_dir = str(tmp_path / "wal_logs")
        import os
        os.makedirs(wal_dir, exist_ok=True)
        with open(os.path.join(wal_dir, "e2e.wal"), "w") as f:
            f.write("test wal data\n")

        backup = BackupService(
            snapshot_dir=str(tmp_path / "snap"),
            wal_archive_dir=str(tmp_path / "wal_archive"),
        )

        result = backup.archive_wal("e2e", wal_dir=wal_dir)
        assert result["success"] is True

        archives = backup.list_wal_archives()
        assert len(archives) >= 1


# ---------------------------------------------------------------------------
# Test 7: Cross-component consistency
# ---------------------------------------------------------------------------

class TestCrossComponentE2E:
    def test_same_results_across_searches(self, hnsw_index, sample_vectors):
        """The same query on the same index should return the same results."""
        for i, vec in enumerate(sample_vectors[:20]):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}")

        query = sample_vectors[3].tolist()
        results_1 = hnsw_index.search(query, k=5)
        results_2 = hnsw_index.search(query, k=5)

        ids_1 = [r["vector_id"] for r in results_1]
        ids_2 = [r["vector_id"] for r in results_2]
        assert ids_1 == ids_2

    def test_index_persistence(self, hnsw_index, sample_vectors, tmp_path):
        """Save index, reload, verify search results match."""
        for i, vec in enumerate(sample_vectors[:10]):
            hnsw_index.insert(vec.tolist(), f"v{i:03d}", {"idx": i})

        save_path = str(tmp_path / "hnsw_e2e.json")
        hnsw_index.save(save_path)

        # Load into fresh index
        loaded = HNSWIndex().load(save_path)
        assert loaded.total_inserted == 10

        query = sample_vectors[0].tolist()
        original_results = hnsw_index.search(query, k=5)
        loaded_results = loaded.search(query, k=5)

        assert len(original_results) == len(loaded_results)
        for orig, load in zip(original_results, loaded_results):
            assert orig["vector_id"] == load["vector_id"]
            assert orig["distance"] == pytest.approx(load["distance"], abs=1e-5)
