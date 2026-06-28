"""
Tests for Phase 11 — CRDT sync, vector clocks, and region routing.
Run with: pytest test/test_crdt_sync.py -v
"""
import os
import tempfile
import time

import pytest

from utils.vector_clock import (
    VectorClock,
    VectorClockStore,
    happens_before,
    concurrent,
)
from services.crdt_sync import (
    GCounter,
    LWWElementSet,
    VectorCRDT,
    CRDTSyncService,
)
from services.region_router import RegionRouter, _haversine_km


# ===========================================================================
# VectorClock tests
# ===========================================================================

class TestVectorClock:
    def test_increment(self):
        vc = VectorClock()
        vc.increment("node-a")
        vc.increment("node-a")
        vc.increment("node-b")
        assert vc.clocks == {"node-a": 2, "node-b": 1}

    def test_merge_takes_max(self):
        a = VectorClock({"node-a": 3, "node-b": 1})
        b = VectorClock({"node-a": 1, "node-b": 5, "node-c": 2})
        merged = a.merge(b)
        assert merged.clocks == {"node-a": 3, "node-b": 5, "node-c": 2}

    def test_happens_before(self):
        a = VectorClock({"n1": 1, "n2": 0})
        b = VectorClock({"n1": 2, "n2": 1})
        assert happens_before(a, b) is True
        assert happens_before(b, a) is False

    def test_concurrent_clocks(self):
        a = VectorClock({"n1": 2, "n2": 0})
        b = VectorClock({"n1": 0, "n2": 2})
        assert concurrent(a, b) is True
        assert concurrent(b, a) is True

    def test_not_concurrent_when_ordered(self):
        a = VectorClock({"n1": 1})
        b = VectorClock({"n1": 2})
        assert concurrent(a, b) is False

    def test_to_from_dict_roundtrip(self):
        vc = VectorClock({"x": 5, "y": 3})
        assert VectorClock.from_dict(vc.to_dict()) == vc


class TestVectorClockStore:
    def test_increment_and_get(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store = VectorClockStore(path)
            store.increment("vec-1", "node-a")
            store.increment("vec-1", "node-a")
            clock = store.get("vec-1")
            assert clock.clocks == {"node-a": 2}
        finally:
            os.unlink(path)

    def test_merge_stores_max(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store = VectorClockStore(path)
            store.increment("v", "a")
            incoming = VectorClock({"a": 5, "b": 3})
            result = store.merge("v", incoming)
            assert result.clocks["a"] == 5
            assert result.clocks["b"] == 3
        finally:
            os.unlink(path)


# ===========================================================================
# GCounter tests
# ===========================================================================

class TestGCounter:
    def test_increment_and_value(self):
        gc = GCounter("n1")
        gc.increment(3)
        gc.increment(2)
        assert gc.value() == 5

    def test_merge_grows_only(self):
        a = GCounter("n1", {"n1": 5, "n2": 2})
        b = GCounter("n2", {"n1": 3, "n2": 7})
        merged = a.merge(b)
        assert merged.counts == {"n1": 5, "n2": 7}
        assert merged.value() == 12

    def test_to_from_dict(self):
        gc = GCounter("n1", {"n1": 4})
        restored = GCounter.from_dict(gc.to_dict())
        assert restored.node_id == "n1"
        assert restored.value() == 4


# ===========================================================================
# LWWElementSet tests
# ===========================================================================

class TestLWWElementSet:
    def test_add_and_lookup(self):
        lww = LWWElementSet()
        lww.add("v1", {"dim": 128}, timestamp=1000.0)
        assert lww.lookup("v1") == {"dim": 128}

    def test_later_add_wins(self):
        lww = LWWElementSet()
        lww.add("v1", "old", timestamp=1000.0)
        lww.add("v1", "new", timestamp=2000.0)
        assert lww.lookup("v1") == "new"

    def test_remove_tombstones(self):
        lww = LWWElementSet()
        lww.add("v1", "data", timestamp=1000.0)
        lww.remove("v1", timestamp=2000.0)
        assert lww.lookup("v1") is None

    def test_add_after_remove_resurfaces(self):
        lww = LWWElementSet()
        lww.add("v1", "data", timestamp=1000.0)
        lww.remove("v1", timestamp=2000.0)
        lww.add("v1", "revived", timestamp=3000.0)
        assert lww.lookup("v1") == "revived"

    def test_merge_takes_latest(self):
        a = LWWElementSet()
        a.add("v1", "from_a", timestamp=1000.0)
        b = LWWElementSet()
        b.add("v1", "from_b", timestamp=2000.0)
        merged = a.merge(b)
        assert merged.lookup("v1") == "from_b"

    def test_elements_excludes_deleted(self):
        lww = LWWElementSet()
        lww.add("v1", "alive", timestamp=1000.0)
        lww.add("v2", "dead", timestamp=1000.0)
        lww.remove("v2", timestamp=2000.0)
        elems = lww.elements()
        assert "v1" in elems
        assert "v2" not in elems


# ===========================================================================
# VectorCRDT tests
# ===========================================================================

class TestVectorCRDT:
    def test_insert_and_retrieve(self):
        crdt = VectorCRDT("region-1")
        crdt.insert_vector("vec-1", {"label": "cat"}, timestamp=1.0)
        assert crdt.get_vector("vec-1") == {"label": "cat"}

    def test_delete_removes_vector(self):
        crdt = VectorCRDT("region-1")
        crdt.insert_vector("vec-1", {"label": "cat"}, timestamp=1.0)
        crdt.delete_vector("vec-1", timestamp=2.0)
        assert crdt.get_vector("vec-1") is None

    def test_merge_two_states(self):
        a = VectorCRDT("r1")
        a.insert_vector("v1", "meta_a", timestamp=1.0)

        b = VectorCRDT("r2")
        b.insert_vector("v2", "meta_b", timestamp=1.0)

        merged_state = VectorCRDT.merge(a.to_state(), b.to_state())
        merged = VectorCRDT.from_state("r1", merged_state)
        assert merged.get_vector("v1") == "meta_a"
        assert merged.get_vector("v2") == "meta_b"

    def test_lww_conflict_resolved_by_timestamp(self):
        """The higher timestamp wins in a concurrent write."""
        a = VectorCRDT("r1")
        a.insert_vector("v1", "old", timestamp=100.0)

        b = VectorCRDT("r2")
        b.insert_vector("v1", "new", timestamp=200.0)

        merged_state = VectorCRDT.merge(a.to_state(), b.to_state())
        merged = VectorCRDT.from_state("r1", merged_state)
        assert merged.get_vector("v1") == "new"

    def test_counter_grows_through_merge(self):
        a = VectorCRDT("r1")
        a.insert_vector("v1", "x")
        a.insert_vector("v2", "y")

        b = VectorCRDT("r2")
        b.insert_vector("v3", "z")

        merged_state = VectorCRDT.merge(a.to_state(), b.to_state())
        merged = VectorCRDT.from_state("r1", merged_state)
        # r1 contributed 2, r2 contributed 1 → total ≥ 3
        assert merged.counter.value() >= 3


# ===========================================================================
# CRDTSyncService tests
# ===========================================================================

class TestCRDTSyncService:
    def _make_service(self, node_id="node-1"):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        svc = CRDTSyncService(node_id=node_id, clock_store_path=path)
        svc._clock_store_path = path  # keep ref for cleanup
        return svc

    def test_local_insert_and_retrieve(self):
        svc = self._make_service()
        svc.local_insert("v1", {"dim": 3})
        assert svc.get_vector("v1") == {"dim": 3}

    def test_local_delete(self):
        svc = self._make_service()
        svc.local_insert("v1", {"dim": 3}, timestamp=1.0)
        svc.local_delete("v1", timestamp=2.0)
        assert svc.get_vector("v1") is None

    def test_produce_delta_structure(self):
        svc = self._make_service()
        svc.local_insert("v1", "meta")
        delta = svc.produce_delta()
        assert delta["node_id"] == "node-1"
        assert "additions" in delta
        assert "v1" in delta["additions"]
        assert "counter" in delta
        assert "vector_clocks" in delta

    def test_merge_delta_no_conflict(self):
        local = self._make_service("local")
        remote = self._make_service("remote")

        # Remote inserts a different vector
        remote.local_insert("v-remote", "meta_remote", timestamp=1.0)
        delta = remote.produce_delta()

        results = local.merge_delta(delta)
        # All entries should be "ok" (no local counterpart)
        assert all(v == "ok" for v in results.values())
        assert local.get_vector("v-remote") == "meta_remote"

    def test_merge_delta_detects_conflict(self):
        local = self._make_service("local")
        remote = self._make_service("remote")

        # Both nodes write to the same vector independently (no causal link)
        local.local_insert("shared", "local_meta", timestamp=1.0)
        remote.local_insert("shared", "remote_meta", timestamp=1.0)

        delta = remote.produce_delta()
        results = local.merge_delta(delta)
        # shared vector clocks are concurrent → conflict detected
        assert results.get("shared") == "conflict"
        assert len(local.get_conflicts()) >= 1

    def test_vector_count(self):
        svc = self._make_service()
        assert svc.vector_count() == 0
        svc.local_insert("v1", "a", timestamp=1.0)
        svc.local_insert("v2", "b", timestamp=2.0)
        assert svc.vector_count() == 2
        svc.local_delete("v1", timestamp=3.0)
        assert svc.vector_count() == 1

    def test_resolve_conflict(self):
        local = self._make_service("local")
        remote = self._make_service("remote")

        local.local_insert("v-conflict", "lv", timestamp=1.0)
        remote.local_insert("v-conflict", "rv", timestamp=1.0)
        delta = remote.produce_delta()
        local.merge_delta(delta)

        assert len(local.get_conflicts()) >= 1
        local.resolve_conflict("v-conflict")
        remaining = [c for c in local.get_conflicts() if c["vector_id"] == "v-conflict"]
        assert len(remaining) == 0

    def test_merge_remote_state(self):
        local = self._make_service("local")
        remote = self._make_service("remote")

        remote.local_insert("r1", "data_r1", timestamp=1.0)
        remote_state = remote._crdt.to_state()

        local.local_insert("l1", "data_l1", timestamp=2.0)
        local.merge_remote_state(remote_state)

        assert local.get_vector("r1") == "data_r1"
        assert local.get_vector("l1") == "data_l1"


# ===========================================================================
# RegionRouter tests
# ===========================================================================

class TestRegionRouter:
    def _make_router(self):
        router = RegionRouter()
        router.register_region("us-east",  lat=37.79, lon=-122.39, endpoint="http://us-east:8000")
        router.register_region("eu-west",  lat=53.35, lon=-6.26,   endpoint="http://eu-west:8000")
        router.register_region("ap-south", lat=19.08, lon=72.88,   endpoint="http://ap-south:8000")
        return router

    def test_haversine_distance(self):
        # London to Paris is ~340 km
        d = _haversine_km(51.5, -0.12, 48.85, 2.35)
        assert 300 < d < 400

    def test_routes_to_nearest(self):
        router = self._make_router()
        # Client in Dublin (53.3°N 6.3°W) → nearest should be eu-west
        endpoint = router.route(53.3, -6.3)
        assert endpoint == "http://eu-west:8000"

    def test_fallback_when_nearest_unhealthy(self):
        router = self._make_router()
        router.mark_healthy("eu-west", False)
        # Next closest to Dublin is us-east or ap-south; at least not eu-west
        endpoint = router.route(53.3, -6.3)
        assert endpoint != "http://eu-west:8000"

    def test_returns_none_when_all_unhealthy(self):
        router = self._make_router()
        for name in ["us-east", "eu-west", "ap-south"]:
            router.mark_healthy(name, False)
        assert router.route(0.0, 0.0) is None

    def test_list_regions(self):
        router = self._make_router()
        regions = router.list_regions()
        names = {r["name"] for r in regions}
        assert {"us-east", "eu-west", "ap-south"} == names

    def test_deregister_region(self):
        router = self._make_router()
        assert router.deregister_region("ap-south") is True
        names = {r["name"] for r in router.list_regions()}
        assert "ap-south" not in names
