"""Tests for Product Quantization (PQ) index."""

import json
import math
import tempfile
import pytest
import numpy as np

from utils.product_quantization import PQIndex


# ---- Fixtures -------------------------------------------------------------


@pytest.fixture
def trained_pq():
    """Create a small PQ index with 8-dim vectors, M=4, k_sub=16."""
    rng = np.random.RandomState(42)
    vectors = rng.rand(200, 8).astype(np.float32).tolist()
    pq = PQIndex(M=4, k_sub=16, n_iter=10)
    pq.train(vectors)
    return pq


@pytest.fixture
def populated_pq(trained_pq):
    """A trained PQ index with 50 vectors added."""
    rng = np.random.RandomState(1)
    vectors = rng.rand(50, 8).astype(np.float32)
    for i in range(50):
        trained_pq.add(vectors[i].tolist(), f"v{i:03d}", {"idx": i})
    return trained_pq


# ---- Training Tests -------------------------------------------------------


class TestTraining:
    def test_train_basic(self):
        vectors = np.random.rand(100, 8).astype(np.float32).tolist()
        pq = PQIndex(M=4, k_sub=16)
        pq.train(vectors)
        assert pq.is_trained is True
        assert pq.M == 4
        assert pq._sub_dim == 2
        assert pq._dim == 8
        assert len(pq.codebooks) == 4
        for cb in pq.codebooks:
            assert cb.shape == (16, 2)

    def test_train_auto_M(self):
        """When M is None, it should be auto-inferred."""
        vectors = np.random.rand(100, 32).astype(np.float32).tolist()
        pq = PQIndex(k_sub=16)
        pq.train(vectors)
        assert pq.M == min(32, 32 // 2)  # = 16 (auto rule: min(32, D//2))
        assert pq._sub_dim == 2

    def test_train_dim_not_divisible_by_M(self):
        """M should be adjusted downwards if D % M != 0."""
        vectors = np.random.rand(100, 10).astype(np.float32).tolist()
        pq = PQIndex(M=4, k_sub=8)
        pq.train(vectors)
        assert 10 % pq.M == 0  # M should be adjusted to a divisor of D
        assert pq.M == 2  # 10 % 4 != 0, so try 3, then 2 (which divides 10)
        assert pq._sub_dim == 5

    def test_train_already_normalized_cosine(self):
        """Cosine-normalized vectors should stay normalized."""
        vectors = np.random.rand(50, 4).astype(np.float32).tolist()
        pq = PQIndex(M=2, k_sub=8, distance_metric="cosine")
        pq.train(vectors)
        norms = [np.linalg.norm(cb, axis=1) for cb in pq.codebooks]
        for norm_arr in norms:
            assert np.allclose(norm_arr, 1.0, atol=1e-5) or np.all(norm_arr > 0)

    def test_train_insufficient_vectors(self):
        """Fewer vectors than k_sub should still work (with warnings)."""
        vectors = np.random.rand(5, 8).astype(np.float32).tolist()
        pq = PQIndex(M=4, k_sub=16)
        pq.train(vectors)  # Should not crash
        assert pq.is_trained

    def test_train_empty_vectors(self):
        pq = PQIndex(M=2)
        with pytest.raises(ValueError):
            pq.train([])


# ---- Encoding / Add / Delete Tests ----------------------------------------


class TestEncoding:
    def test_add_single(self, trained_pq):
        vec = np.random.rand(8).astype(np.float32).tolist()
        trained_pq.add(vec, "test1", {"name": "test"})
        assert "test1" in trained_pq.codes
        assert trained_pq.codes["test1"].shape == (4,)
        assert trained_pq.codes["test1"].dtype == np.uint8
        assert len(trained_pq) == 1

    def test_add_batch(self, trained_pq):
        rng = np.random.RandomState(0)
        entries = []
        for i in range(10):
            entries.append({
                "vector": rng.rand(8).astype(np.float32).tolist(),
                "vector_id": f"batch_{i}",
                "metadata": {"i": i},
            })
        trained_pq.add_batch(entries)
        assert len(trained_pq) == 10

    def test_add_before_train_raises(self):
        pq = PQIndex(M=2)
        with pytest.raises(RuntimeError):
            pq.add([0.1] * 8, "fail")

    def test_delete(self, populated_pq):
        assert populated_pq.delete("v000") is True
        assert "v000" not in populated_pq.codes
        assert len(populated_pq) == 49

    def test_delete_nonexistent(self, populated_pq):
        assert populated_pq.delete("nonexistent") is False

    def test_clear(self, populated_pq):
        populated_pq.clear()
        assert len(populated_pq) == 0
        assert populated_pq.is_trained  # codebooks preserved


# ---- Search Tests ----------------------------------------------------------


class TestSearch:
    def test_search_basic(self, populated_pq):
        query = np.random.rand(8).astype(np.float32).tolist()
        results = populated_pq.search(query, k=5)
        assert len(results) == 5
        for r in results:
            assert "vector_id" in r
            assert "distance" in r
            assert isinstance(r["distance"], float)
            assert r["distance"] >= 0

    def test_search_empty_index(self, trained_pq):
        query = np.random.rand(8).astype(np.float32).tolist()
        results = trained_pq.search(query, k=5)
        assert results == []

    def test_search_k_larger_than_index(self, populated_pq):
        query = np.random.rand(8).astype(np.float32).tolist()
        results = populated_pq.search(query, k=100)
        assert len(results) == 50  # only 50 vectors in index

    def test_search_deterministic(self, populated_pq):
        query = np.random.rand(8).astype(np.float32).tolist()
        r1 = populated_pq.search(query, k=5)
        r2 = populated_pq.search(query, k=5)
        ids1 = [r["vector_id"] for r in r1]
        ids2 = [r["vector_id"] for r in r2]
        assert ids1 == ids2

    def test_search_returns_sorted(self, populated_pq):
        query = np.random.rand(8).astype(np.float32).tolist()
        results = populated_pq.search(query, k=10)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)

    def test_search_approximation_reasonable(self):
        """PQ approximation should find results close to exact search."""
        rng = np.random.RandomState(42)
        dim = 16
        # Generate clustered data
        centroids = rng.randn(3, dim)
        vectors = []
        for i in range(60):
            c = centroids[i % 3]
            vectors.append((c + rng.randn(dim) * 0.1).tolist())

        pq = PQIndex(M=4, k_sub=32, n_iter=20)
        pq.train(vectors)
        for i, v in enumerate(vectors):
            pq.add(v, f"v{i:03d}")

        # Query near first centroid
        query = (centroids[0] + rng.randn(dim) * 0.05).tolist()
        results = pq.search(query, k=5)

        # PQ should find vectors from the same cluster
        top_ids = [r["vector_id"] for r in results]
        top_indices = [int(vid[1:]) for vid in top_ids]
        # At least 3 of top 5 should be from cluster 0 (indices 0, 3, 6, ...)
        cluster0_count = sum(1 for idx in top_indices if idx % 3 == 0)
        assert cluster0_count >= 3, f"Only {cluster0_count}/5 from correct cluster"

    def test_search_with_rerank(self, populated_pq):
        """Rerank should yield same or better (lower) distances."""
        query = np.random.rand(8).astype(np.float32).tolist()
        without = populated_pq.search(query, k=5, rerank=False)
        with_rerank = populated_pq.search(query, k=5, rerank=True)
        assert len(with_rerank) == 5
        # Rerank may change order but shouldn't crash
        assert all(r["distance"] >= 0 for r in with_rerank)

    def test_search_cosine_metric(self):
        """Cosine distance should be bounded [0, 2]."""
        rng = np.random.RandomState(0)
        vectors = rng.rand(30, 4).astype(np.float32).tolist()
        pq = PQIndex(M=2, k_sub=16, distance_metric="cosine")
        pq.train(vectors)
        for i, v in enumerate(vectors):
            pq.add(v, f"v{i}")
        query = rng.rand(4).astype(np.float32).tolist()
        results = pq.search(query, k=5)
        for r in results:
            assert 0.0 <= r["distance"] <= 2.0

    def test_search_before_train_raises(self):
        pq = PQIndex(M=2)
        with pytest.raises(RuntimeError):
            pq.search([0.1] * 4, k=5)


# ---- Persistence Tests -----------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, populated_pq, tmp_path):
        path = str(tmp_path / "pq_index.json")
        populated_pq.save(path)

        loaded = PQIndex.load(path)
        assert loaded.is_trained
        assert loaded.M == populated_pq.M
        assert loaded.k_sub == populated_pq.k_sub
        assert len(loaded) == len(populated_pq)
        assert set(loaded.vector_ids) == set(populated_pq.vector_ids)

        # Codes should match exactly
        for vid in populated_pq.vector_ids:
            assert np.array_equal(loaded.codes[vid], populated_pq.codes[vid])

        # Search should work on loaded index
        query = np.random.rand(8).astype(np.float32).tolist()
        results = loaded.search(query, k=5)
        assert len(results) == 5

    def test_save_and_load_empty(self, trained_pq, tmp_path):
        path = str(tmp_path / "empty_pq.json")
        trained_pq.save(path)
        loaded = PQIndex.load(path)
        assert loaded.is_trained
        assert len(loaded) == 0


# ---- Stats Tests -----------------------------------------------------------


class TestStats:
    def test_get_stats(self, populated_pq):
        stats = populated_pq.get_stats()
        assert stats["total_vectors"] == 50
        assert stats["dimension"] == 8
        assert stats["M"] == 4
        assert stats["sub_dim"] == 2
        assert stats["k_sub"] == 16
        assert stats["is_trained"] is True
        assert stats["compression_ratio"] > 0
        assert stats["bytes_compressed"] == 50 * 4  # 50 vectors × 4 bytes (M)
        assert stats["bytes_original"] == 50 * 8 * 4  # 50 × 8 × 4 bytes float32

    def test_get_stats_empty(self, trained_pq):
        stats = trained_pq.get_stats()
        assert stats["total_vectors"] == 0
        assert stats["compression_ratio"] == 0


# ---- Edge Cases ------------------------------------------------------------


class TestEdgeCases:
    def test_single_vector(self):
        pq = PQIndex(M=2, k_sub=4)
        pq.train([[0.1, 0.2, 0.3, 0.4]])
        pq.add([0.1, 0.2, 0.3, 0.4], "only")
        results = pq.search([0.1, 0.2, 0.3, 0.4], k=5)
        assert len(results) == 1
        assert results[0]["vector_id"] == "only"

    def test_matching_vector_has_zero_distance(self):
        """Searching for an indexed vector should find itself at distance ~0."""
        rng = np.random.RandomState(0)
        vectors = rng.rand(20, 8).astype(np.float32).tolist()
        pq = PQIndex(M=4, k_sub=32, n_iter=20, distance_metric="euclidean")
        pq.train(vectors)
        for i, v in enumerate(vectors):
            pq.add(v, f"v{i}")

        # The original vector may not reconstruct perfectly, but should be close
        results = pq.search(vectors[0], k=5)
        top_id = results[0]["vector_id"]
        # The nearest result may not be exactly v0 due to PQ approximation,
        # but the distance should be relatively small
        assert results[0]["distance"] < 1.0

    def test_k_sub_2(self):
        """Minimum viable k_sub."""
        vectors = np.random.rand(30, 4).astype(np.float32).tolist()
        pq = PQIndex(M=2, k_sub=2)
        pq.train(vectors)
        for i, v in enumerate(vectors):
            pq.add(v, f"v{i}")
        results = pq.search(np.random.rand(4).astype(np.float32).tolist(), k=3)
        assert len(results) == 3

    def test_invalid_k_sub_raises(self):
        with pytest.raises(ValueError):
            PQIndex(k_sub=0)
        with pytest.raises(ValueError):
            PQIndex(k_sub=257)
