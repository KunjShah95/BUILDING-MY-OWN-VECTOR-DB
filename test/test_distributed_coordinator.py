"""Tests for the distributed query coordinator."""

import pytest

from services.distributed_coordinator import (
    DistributedCoordinator, CallableShard, IndexShard,
)


def _shard_fn(prefix, base_dist=0.1):
    def fn(qv, k, **kw):
        return [
            {"vector_id": f"{prefix}{i}", "distance": base_dist * (i + 1), "metadata": {}}
            for i in range(k)
        ]
    return fn


def test_empty_shards():
    coord = DistributedCoordinator([])
    res = coord.search([0.1], k=3)
    assert res["success"] is False


def test_scatter_gather_distance_fusion():
    coord = DistributedCoordinator([
        CallableShard("s1", _shard_fn("a", 0.1)),
        CallableShard("s2", _shard_fn("b", 0.2)),
    ])
    res = coord.search([0.1] * 4, k=3)
    assert res["success"]
    assert res["shards_queried"] == 2
    assert res["total_results"] == 3
    # closest distances should come from s1 (smaller base_dist)
    assert res["results"][0]["distance"] <= res["results"][1]["distance"]
    coord.shutdown()


def test_fault_tolerance_skips_failed_shard():
    def boom(qv, k, **kw):
        raise RuntimeError("shard down")

    coord = DistributedCoordinator([
        CallableShard("ok", _shard_fn("a")),
        CallableShard("bad", boom),
    ])
    res = coord.search([0.1] * 4, k=2)
    assert res["success"] is True
    assert res["degraded"] is True
    assert "bad" in res["shards_failed"]
    assert res["total_results"] == 2
    coord.shutdown()


def test_rrf_fusion():
    coord = DistributedCoordinator([
        CallableShard("s1", _shard_fn("a")),
        CallableShard("s2", _shard_fn("b")),
    ])
    res = coord.search([0.1] * 4, k=5, fusion="rrf")
    assert res["fusion"] == "rrf"
    assert all("rrf_score" in r for r in res["results"])
    coord.shutdown()


def test_distance_fusion_dedupes_by_id():
    # both shards return the same id; keep the better (smaller) distance
    def s1(qv, k, **kw):
        return [{"vector_id": "x", "distance": 0.5, "metadata": {}}]

    def s2(qv, k, **kw):
        return [{"vector_id": "x", "distance": 0.2, "metadata": {}}]

    coord = DistributedCoordinator([CallableShard("s1", s1), CallableShard("s2", s2)])
    res = coord.search([0.1], k=5)
    assert res["total_results"] == 1
    assert res["results"][0]["distance"] == 0.2
    coord.shutdown()


def test_index_shard_wrapper():
    class FakeIndex:
        def search(self, qv, k):
            return [{"vector_id": "z", "distance": 0.0, "metadata": {}}]

    coord = DistributedCoordinator([IndexShard("idx", FakeIndex())])
    res = coord.search([0.1], k=1)
    assert res["results"][0]["vector_id"] == "z"
    coord.shutdown()
