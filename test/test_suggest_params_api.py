"""Tests for the /api/indexer/suggest-params REST endpoint."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from examples.vector_indexer_api import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """A minimal FastAPI app with just the indexer router mounted."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def gaussian_vectors():
    """200 random 32D Gaussian vectors — sufficient for a data-driven analysis."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 32)).astype(np.float32).tolist()


@pytest.fixture
def clustered_vectors():
    """150 vectors, 16D, with 3 natural clusters — structured data."""
    rng = np.random.default_rng(42)
    centroids = np.array([[3.0] * 16, [-2.0] * 16, [0.5] * 16], dtype=np.float32)
    vecs = []
    for c in centroids:
        cluster = c + rng.standard_normal((50, 16)).astype(np.float32) * 0.3
        vecs.append(cluster)
    return np.vstack(vecs).tolist()


# ---------------------------------------------------------------------------
# /api/indexer/suggest-params — data-driven mode
# ---------------------------------------------------------------------------

class TestSuggestParamsDataDriven:
    def test_returns_success_with_recommendation(self, client, gaussian_vectors):
        """POST with vectors returns a data-driven recommendation."""
        response = client.post("/api/indexer/suggest-params", json={
            "vectors": gaussian_vectors,
            "recall_target": 0.95,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["mode"] == "data-driven"
        assert body["vectors_analysed"] == 200
        assert body["vector_dimension"] == 32

    def test_includes_dataset_stats(self, client, gaussian_vectors):
        """Response should include dataset statistics summary."""
        response = client.post("/api/indexer/suggest-params", json={
            "vectors": gaussian_vectors,
        })
        body = response.json()
        stats = body["dataset_stats"]
        assert stats["num_vectors"] == 200
        assert stats["dimension"] == 32
        assert isinstance(stats["intrinsic_dim_pca"], float)
        assert isinstance(stats["is_high_dimensional"], bool)
        assert isinstance(stats["is_normalized"], bool)
        assert isinstance(stats["mean_variance"], float)
        assert "cluster_separation_ratio" in stats

    def test_includes_recommendation_with_full_metadata(self, client, gaussian_vectors):
        """Recommendation should include HNSW params, expected recall, confidence, reasoning."""
        response = client.post("/api/indexer/suggest-params", json={
            "vectors": gaussian_vectors,
        })
        body = response.json()
        rec = body["recommendation"]
        hnsw = rec["hnsw"]

        assert hnsw["m"] > 0
        assert hnsw["m0"] == hnsw["m"] * 2
        assert hnsw["ef_construction"] > 0
        assert hnsw["ef_search"] > 0
        assert isinstance(rec["expected_recall"], float)
        assert rec["expected_recall"] > 0.0
        assert rec["confidence"] in ("high", "medium", "low")
        assert isinstance(rec["reasoning"], list)
        assert len(rec["reasoning"]) >= 1

    def test_higher_recall_target_gives_higher_ef(self, client, gaussian_vectors):
        """Increasing recall_target should increase ef_search and ef_construction."""
        res_low = client.post("/api/indexer/suggest-params", json={
            "vectors": gaussian_vectors,
            "recall_target": 0.90,
        })
        res_high = client.post("/api/indexer/suggest-params", json={
            "vectors": gaussian_vectors,
            "recall_target": 0.99,
        })

        low = res_low.json()["recommendation"]["hnsw"]
        high = res_high.json()["recommendation"]["hnsw"]

        assert high["ef_search"] >= low["ef_search"]
        assert high["ef_construction"] >= low["ef_construction"]

    def test_clustered_data_shows_separation_in_stats(self, client, clustered_vectors):
        """Well-separated clusters should be reflected in dataset statistics."""
        response = client.post("/api/indexer/suggest-params", json={
            "vectors": clustered_vectors,
        })
        body = response.json()
        stats = body["dataset_stats"]
        rec = body["recommendation"]

        # Clusters should be detected in the summary
        assert stats["cluster_separation_ratio"] > 1.5
        assert stats["estimated_cluster_count"] >= 1
        # Dataset has structure → confidence should not be "low"
        assert rec["confidence"] in ("high", "medium")

    def test_tiny_dataset_does_not_crash(self, client):
        """A tiny dataset (5 vectors) should still produce a valid response."""
        tiny = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        response = client.post("/api/indexer/suggest-params", json={
            "vectors": tiny,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["mode"] == "data-driven"
        assert body["vectors_analysed"] == 5
        # Even tiny datasets get a recommendation
        assert body["recommendation"]["hnsw"]["m"] > 0


# ---------------------------------------------------------------------------
# /api/indexer/suggest-params — size-based fallback mode
# ---------------------------------------------------------------------------

class TestSuggestParamsSizeBased:
    def test_omitting_vectors_returns_size_based(self, client):
        """Without vectors, the endpoint should return size-based heuristics."""
        response = client.post("/api/indexer/suggest-params", json={
            "num_vectors": 50000,
            "vector_dim": 128,
            "recall_target": 0.95,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["mode"] == "size-based"
        assert body["num_vectors"] == 50000
        assert "note" in body

    def test_size_based_returns_hnsw_and_ivf_params(self, client):
        """Size-based mode should return both HNSW and IVF params."""
        response = client.post("/api/indexer/suggest-params", json={
            "num_vectors": 100000,
        })
        body = response.json()
        rec = body["recommendation"]
        assert "hnsw" in rec
        assert "ivf" in rec
        assert rec["hnsw"]["m"] > 0
        assert rec["ivf"]["n_clusters"] > 0

    def test_size_based_defaults_to_10000_vectors(self, client):
        """When no num_vectors is given, should default to 10000."""
        response = client.post("/api/indexer/suggest-params", json={})
        body = response.json()
        assert body["num_vectors"] == 10000
        assert body["mode"] == "size-based"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestSuggestParamsErrors:
    def test_extreme_recall_target_clamped_by_recommender(self, client, gaussian_vectors):
        """Extreme recall targets should be clamped by the recommender, not crash."""
        response = client.post("/api/indexer/suggest-params", json={
            "vectors": gaussian_vectors,
            "recall_target": 1.5,  # Above 0.999 internal cap
        })
        # Pydantic allows 1.5 (le=2.0), then HNSWParameterRecommender clamps
        assert response.status_code == 200
        body = response.json()
        # The recommender's internal cap is 0.999
        assert body["recommendation"]["expected_recall"] <= 1.0
        # Extreme target should produce high ef values
        assert body["recommendation"]["hnsw"]["ef_search"] >= 100

    def test_empty_vectors_list_falls_back_to_size_mode(self, client):
        """An empty vectors list should fall back to size-based mode."""
        response = client.post("/api/indexer/suggest-params", json={
            "vectors": [],
            "num_vectors": 5000,
        })
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "size-based"
        assert body["num_vectors"] == 5000
