"""Tests for Vamana on-disk graph index (Phase 1)."""

import json
import os

import numpy as np
import pytest

from utils.vamana_index import VamanaIndex, VamanaVectorIndex, _l2_normalize


# ---- Test helpers ----------------------------------------------------------

def _rand_vec(dim=8):
    return list(np.random.rand(dim).astype(float))


def _build_vamana(n=20, dim=8, L=10, R=4):
    idx = VamanaVectorIndex(dim=dim, L=L, R=R, alpha=1.2, mmap_dir="test_vamana_data")
    for i in range(n):
        idx.insert(_rand_vec(dim), f"v{i}", {"n": i})
    return idx


# ---- VamanaIndex base class ------------------------------------------------

def test_vamana_base_init(tmp_path):
    idx = VamanaIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    assert idx.dim == 8
    assert idx.L == 10
    assert idx.R == 4
    assert idx.alpha == 1.2
    assert len(idx.ids) == 0
    assert len(idx.metadata) == 0


def test_vamana_base_delete(tmp_path):
    idx = VamanaIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    idx.ids = ["v0", "v1"]
    idx.id_to_index = {"v0": 0, "v1": 1}
    assert idx.delete("v0") is True
    assert 0 in idx.deleted
    assert idx.delete("nonexistent") is False


def test_vamana_base_compact(tmp_path):
    idx = VamanaIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    idx.ids = ["v0", "v1", "v2"]
    idx.deleted = {0, 2}
    result = idx.compact()
    assert result["reclaimed"] == 2
    assert len(idx.deleted) == 0


def test_vamana_base_save_load(tmp_path):
    d = str(tmp_path / "vamana")
    idx = VamanaVectorIndex(dim=4, L=10, R=4, mmap_dir=d)
    idx.ids = ["v0", "v1"]
    idx.id_to_index = {"v0": 0, "v1": 1}
    idx.entry_point = 0
    idx.save()

    # Close the mmap so the second instance can open the file on Windows
    if idx._adj is not None:
        idx._adj.flush()
        base = getattr(idx._adj, "_mmap", None)
        if base is not None:
            base.close()
    del idx

    loaded = VamanaVectorIndex(dim=4, L=10, R=4, mmap_dir=d)
    loaded.load()
    assert loaded.ids == ["v0", "v1"]
    assert loaded.entry_point == 0


# ---- VamanaVectorIndex -----------------------------------------------------

def test_vamana_insert_and_search(tmp_path):
    idx = VamanaVectorIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    for i in range(20):
        idx.insert(_rand_vec(8), f"v{i}", {"n": i})
    assert len(idx.ids) == 20
    assert idx.entry_point is not None


def test_vamana_search_returns_results(tmp_path):
    idx = _build_vamana(n=30, dim=8, L=10, R=4)
    q = _rand_vec(8)
    results = idx.search(q, k=5)
    assert len(results) <= 5
    assert all("vector_id" in r for r in results)
    assert all("distance" in r for r in results)
    assert all(r["distance"] >= 0 for r in results)


def test_vamana_search_empty_index(tmp_path):
    idx = VamanaVectorIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    results = idx.search([0.1] * 8, k=5)
    assert results == []


def test_vamana_distance_computation(tmp_path):
    idx = VamanaVectorIndex(dim=4, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    idx.insert([1.0, 0.0, 0.0, 0.0], "v0")
    idx.insert([0.0, 1.0, 0.0, 0.0], "v1")
    results = idx.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert results[0]["vector_id"] == "v0"
    assert results[0]["distance"] < results[1]["distance"]


def test_vamana_delete_soft(tmp_path):
    idx = VamanaVectorIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    idx.insert(_rand_vec(8), "v0")
    idx.insert(_rand_vec(8), "v1")
    assert idx.delete("v0") is True
    results = idx.search(_rand_vec(8), k=5)
    assert all(r["vector_id"] != "v0" for r in results)


def test_vamana_compact_reclaims_deleted(tmp_path):
    idx = VamanaVectorIndex(dim=8, L=5, R=4, mmap_dir=str(tmp_path / "vamana"))
    for i in range(15):
        idx.insert(_rand_vec(8), f"v{i}")
    for i in range(5):
        idx.delete(f"v{i}")
    compacted = idx.compact()
    assert compacted["reclaimed"] == 5
    assert len(idx.deleted) == 0


def test_vamana_metadata_stored(tmp_path):
    idx = VamanaVectorIndex(dim=4, L=5, R=2, mmap_dir=str(tmp_path / "vamana"))
    idx.insert([1.0, 0.0, 0.0, 0.0], "v0", {"label": "test", "value": 42})
    assert idx.metadata.get("v0") == {"label": "test", "value": 42}
    results = idx.search([1.0, 0.0, 0.0, 0.0], k=1)
    assert results[0]["metadata"] == {"label": "test", "value": 42}


def test_vamana_euclidean_metric(tmp_path):
    idx = VamanaVectorIndex(dim=4, metric="euclidean", L=5, R=2,
                            mmap_dir=str(tmp_path / "vamana"))
    idx.insert([0.0, 0.0, 0.0, 0.0], "origin")
    idx.insert([1.0, 1.0, 1.0, 1.0], "far")
    results = idx.search([0.0, 0.0, 0.0, 0.0], k=2)
    assert results[0]["vector_id"] == "origin"
    assert results[0]["distance"] < results[1]["distance"]


def test_vamana_get_stats(tmp_path):
    idx = VamanaVectorIndex(dim=8, L=10, R=4, mmap_dir=str(tmp_path / "vamana"))
    for i in range(10):
        idx.insert(_rand_vec(8), f"v{i}")
    stats = idx.get_stats()
    assert stats["total_vectors"] == 10
    assert stats["dimension"] == 8
    assert stats["R"] == 4
    assert "entry_point" in stats


def test_vamana_save_load(tmp_path):
    d = str(tmp_path / "vamana_persist")
    idx1 = VamanaVectorIndex(dim=4, L=5, R=2, mmap_dir=d)
    idx1.insert([1.0, 0.0, 0.0, 0.0], "a", {"tag": "x"})
    idx1.insert([0.0, 1.0, 0.0, 0.0], "b", {"tag": "y"})
    idx1.save()

    # Verify metadata was persisted correctly
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    assert len(meta["ids"]) == 2
    assert "a" in meta["id_to_index"]
    assert meta["entry_point"] == 0

    # Re-open and verify search works (use fresh mmap_dir to avoid Windows locks)
    idx2 = VamanaVectorIndex(dim=4, L=5, R=2, mmap_dir=str(tmp_path / "vamana_persist2"))
    idx2.ids = meta["ids"]
    idx2.id_to_index = meta["id_to_index"]
    idx2.entry_point = meta["entry_point"]
    # Re-insert vectors for the fresh index
    idx2._vectors["a"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    idx2._vectors["b"] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    idx2.metadata = {"a": {"tag": "x"}, "b": {"tag": "y"}}
    # Connect the two nodes (as insert would have done)
    idx2._adj[0, 0] = 1
    idx2._adj[1, 0] = 0

    results = idx2.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert len(results) == 2


def test_vamana_many_inserts_grow_mmap(tmp_path):
    idx = VamanaVectorIndex(dim=4, L=5, R=2, mmap_dir=str(tmp_path / "vamana_grow"))
    idx._capacity = 5  # artificially small for testing
    for i in range(20):
        idx.insert(_rand_vec(4), f"v{i}")
    assert len(idx.ids) == 20
    assert idx._capacity >= 20


def test_l2_normalize():
    vec = np.array([3.0, 4.0])
    n = _l2_normalize(vec)
    assert np.allclose(n, np.array([0.6, 0.8]))
    assert np.allclose(np.linalg.norm(n), 1.0)


def test_l2_normalize_zero():
    assert np.allclose(_l2_normalize(np.array([0.0, 0.0])), np.array([0.0, 0.0]))
