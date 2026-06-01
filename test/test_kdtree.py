import numpy as np
import pytest
from utils.kdtree_index import KDTreeIndex, KDTreeNode


class TestKDTreeBuild:
    def test_build_and_search(self):
        index = KDTreeIndex(distance_metric="cosine")
        vectors = np.random.randn(100, 128).astype(np.float64)
        ids = [f"v{i}" for i in range(100)]
        index.build(vectors, ids)
        query = np.random.randn(128).astype(np.float64)
        results = index.search(query, k=5)
        assert len(results) == 5
        for r in results:
            assert "id" in r
            assert "distance" in r

    def test_search_empty(self):
        index = KDTreeIndex()
        results = index.search(np.random.randn(128).tolist(), k=5)
        assert results == []

    def test_build_no_vectors(self):
        index = KDTreeIndex()
        index.build(np.array([]).reshape(0, 0), [])
        assert index.size == 0
        assert index.root is None

    def test_build_single_vector(self):
        index = KDTreeIndex()
        index.build(np.array([[1.0, 2.0, 3.0]]), ["v0"])
        assert index.size == 1
        results = index.search([1.0, 2.0, 3.0], k=1)
        assert len(results) == 1
        assert results[0]["id"] == "v0"


class TestKDTreeSearch:
    def test_search_exact_match(self):
        index = KDTreeIndex(distance_metric="euclidean")
        vectors = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0]])
        index.build(vectors, ["v0", "v1", "v2", "v3"])
        results = index.search([1.0, 1.0], k=2)
        assert results[0]["id"] == "v1"
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-10)

    def test_search_k_larger_than_size(self):
        index = KDTreeIndex()
        vectors = np.random.randn(5, 4)
        index.build(vectors, [f"v{i}" for i in range(5)])
        results = index.search(np.random.randn(4).tolist(), k=100)
        assert len(results) == 5

    def test_search_results_ordered(self):
        index = KDTreeIndex(distance_metric="euclidean")
        vectors = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [10.0, 10.0]])
        index.build(vectors, ["v0", "v1", "v2", "v3"])
        results = index.search([2.0, 2.0], k=4)
        distances = [r["distance"] for r in results]
        assert all(distances[i] <= distances[i + 1] for i in range(len(distances) - 1))


class TestKDTreeInsert:
    def test_insert_after_build(self):
        index = KDTreeIndex()
        index.build(np.random.randn(50, 64).astype(np.float64), [f"v{i}" for i in range(50)])
        index.insert(np.random.randn(64).tolist(), "v50", None)
        assert index.size == 51

    def test_insert_and_search(self):
        index = KDTreeIndex(distance_metric="euclidean")
        index.build(np.array([[0.0, 0.0], [10.0, 10.0]]), ["v0", "v1"])
        index.insert([5.0, 5.0], "v2", None)
        results = index.search([4.9, 5.1], k=1)
        assert results[0]["id"] == "v2"


class TestKDTreeDelete:
    def test_delete(self):
        index = KDTreeIndex()
        vectors = np.random.randn(10, 4)
        index.build(vectors, [f"v{i}" for i in range(10)])
        assert index.size == 10
        index.delete("v0")
        assert index.size == 9
        results = index.search(np.random.randn(4).tolist(), k=10)
        ids = [r["id"] for r in results]
        assert "v0" not in ids

    def test_delete_not_found(self):
        index = KDTreeIndex()
        index.build(np.array([[1.0, 2.0]]), ["v0"])
        result = index.delete("nonexistent")
        assert result is False
        assert index.size == 1

    def test_delete_last_vector(self):
        index = KDTreeIndex()
        index.build(np.array([[1.0, 2.0]]), ["v0"])
        assert index.size == 1
        index.delete("v0")
        assert index.size == 0
        assert index.root is None

    def test_delete_batch(self):
        index = KDTreeIndex()
        vectors = np.random.randn(10, 4)
        index.build(vectors, [f"v{i}" for i in range(10)])
        count = index.delete_batch(["v0", "v2", "v4"])
        assert count == 3
        assert index.size == 7
        results = index.search(np.random.randn(4).tolist(), k=10)
        ids = [r["id"] for r in results]
        assert "v0" not in ids
        assert "v2" not in ids
        assert "v4" not in ids

    def test_delete_batch_empty(self):
        index = KDTreeIndex()
        index.build(np.array([[1.0, 2.0], [3.0, 4.0]]), ["v0", "v1"])
        count = index.delete_batch([])
        assert count == 0
        assert index.size == 2

    def test_delete_and_search_exact(self):
        index = KDTreeIndex(distance_metric="euclidean")
        vectors = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [10.0, 10.0]])
        index.build(vectors, ["v0", "v1", "v2", "v3"])
        index.delete("v3")
        index.delete("v0")
        results = index.search([1.0, 1.0], k=2)
        assert results[0]["id"] == "v1"
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-10)


class TestKDTreeSaveLoad:
    def test_save_load(self, tmp_path):
        index = KDTreeIndex()
        vectors = np.random.randn(20, 32).astype(np.float64)
        ids = [f"v{i}" for i in range(20)]
        index.build(vectors, ids)
        path = str(tmp_path / "kdtree.json")
        index.save(path)
        loaded = KDTreeIndex.load(path)
        assert loaded.size == 20
        assert loaded.distance_metric == "cosine"

    def test_save_load_search(self, tmp_path):
        index = KDTreeIndex(distance_metric="euclidean")
        index.build(np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]]), ["a", "b", "c"])
        path = str(tmp_path / "kdtree.json")
        index.save(path)
        loaded = KDTreeIndex.load(path)
        results = loaded.search([1.1, 1.1], k=1)
        assert results[0]["id"] == "b"


class TestKDTreeDistance:
    def test_cosine_distance(self):
        index = KDTreeIndex(distance_metric="cosine")
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        index.build(vectors, ["v0", "v1"])
        results = index.search([0.0, 1.0], k=2)
        assert results[0]["id"] == "v1"

    def test_euclidean_distance(self):
        index = KDTreeIndex(distance_metric="euclidean")
        vectors = np.array([[1.0, 0.0], [5.0, 5.0]])
        index.build(vectors, ["v0", "v1"])
        results = index.search([1.1, 0.0], k=1)
        assert results[0]["id"] == "v0"

    def test_is_trained(self):
        index = KDTreeIndex()
        assert not index.is_trained
        index.build(np.array([[1.0, 2.0]]), ["v0"])
        assert index.is_trained


class TestKDTreeMetadata:
    def test_build_with_metadata(self):
        index = KDTreeIndex()
        vectors = np.random.randn(5, 3)
        ids = [f"v{i}" for i in range(5)]
        meta = [{"label": f"item_{i}"} for i in range(5)]
        index.build(vectors, ids, meta)
        results = index.search(np.random.randn(3).tolist(), k=1)
        assert "metadata" in results[0]
        assert "label" in results[0]["metadata"]

    def test_dim_property(self):
        index = KDTreeIndex()
        assert index.dim is None
        index.build(np.random.randn(10, 64), [f"v{i}" for i in range(10)])
        assert index.dim == 64
