"""
CRDT synchronisation service for multi-region active-active vector writes.

Implements:
- GCounter: grow-only counter per region (used to count inserts)
- LWWElementSet: Last-Write-Wins element set keyed by vector_id
- VectorCRDT: high-level merge wrapper
- CRDTSyncService: tracks local state, produces deltas, merges incoming deltas
"""
from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional

from utils.vector_clock import VectorClock, VectorClockStore, concurrent


# ---------------------------------------------------------------------------
# GCounter
# ---------------------------------------------------------------------------

class GCounter:
    """Grow-only counter — each node has its own slot."""

    def __init__(self, node_id: str, counts: Optional[Dict[str, int]] = None):
        self.node_id = node_id
        self.counts: Dict[str, int] = dict(counts) if counts else {}

    def increment(self, amount: int = 1) -> None:
        self.counts[self.node_id] = self.counts.get(self.node_id, 0) + amount

    def value(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        merged: Dict[str, int] = dict(self.counts)
        for node, count in other.counts.items():
            merged[node] = max(merged.get(node, 0), count)
        return GCounter(self.node_id, merged)

    def to_dict(self) -> Dict[str, Any]:
        return {"node_id": self.node_id, "counts": self.counts}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GCounter":
        return cls(data["node_id"], data.get("counts", {}))


# ---------------------------------------------------------------------------
# LWW Element Set
# ---------------------------------------------------------------------------

class LWWElementSet:
    """Last-Write-Wins Element Set for vector metadata."""

    def __init__(self) -> None:
        self._add_set: Dict[str, Dict[str, Any]] = {}
        self._remove_set: Dict[str, float] = {}

    def add(self, element_id: str, value: Any, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        existing = self._add_set.get(element_id)
        if existing is None or ts >= existing["ts"]:
            self._add_set[element_id] = {"ts": ts, "value": value}

    def remove(self, element_id: str, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self._remove_set[element_id] = max(self._remove_set.get(element_id, 0.0), ts)

    def lookup(self, element_id: str) -> Optional[Any]:
        entry = self._add_set.get(element_id)
        if entry is None:
            return None
        remove_ts = self._remove_set.get(element_id, 0.0)
        if remove_ts >= entry["ts"]:
            return None
        return entry["value"]

    def elements(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for eid, entry in self._add_set.items():
            if self._remove_set.get(eid, 0.0) < entry["ts"]:
                result[eid] = entry["value"]
        return result

    def merge(self, other: "LWWElementSet") -> "LWWElementSet":
        merged = LWWElementSet()
        for eid, entry in self._add_set.items():
            merged._add_set[eid] = dict(entry)
        for eid, entry in other._add_set.items():
            existing = merged._add_set.get(eid)
            if existing is None or entry["ts"] >= existing["ts"]:
                merged._add_set[eid] = dict(entry)
        for eid, ts in self._remove_set.items():
            merged._remove_set[eid] = ts
        for eid, ts in other._remove_set.items():
            merged._remove_set[eid] = max(merged._remove_set.get(eid, 0.0), ts)
        return merged

    def to_dict(self) -> Dict[str, Any]:
        return {"add_set": self._add_set, "remove_set": self._remove_set}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWElementSet":
        obj = cls()
        obj._add_set = dict(data.get("add_set", {}))
        obj._remove_set = dict(data.get("remove_set", {}))
        return obj


# ---------------------------------------------------------------------------
# VectorCRDT
# ---------------------------------------------------------------------------

class VectorCRDT:
    """Combines GCounter + LWWElementSet into a complete vector CRDT state."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.counter = GCounter(node_id)
        self.lww = LWWElementSet()

    def to_state(self) -> Dict[str, Any]:
        return {
            "counter": self.counter.to_dict(),
            "lww": self.lww.to_dict(),
        }

    @classmethod
    def from_state(cls, node_id: str, state: Dict[str, Any]) -> "VectorCRDT":
        obj = cls(node_id)
        if "counter" in state:
            obj.counter = GCounter.from_dict(state["counter"])
        if "lww" in state:
            obj.lww = LWWElementSet.from_dict(state["lww"])
        return obj

    @staticmethod
    def merge(local_state: Dict[str, Any], remote_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two state dicts from different regions and return the merged state.
        """
        node_id = local_state.get("counter", {}).get("node_id", "unknown")
        local_crdt = VectorCRDT.from_state(node_id, local_state)
        remote_node_id = remote_state.get("counter", {}).get("node_id", "remote")
        remote_crdt = VectorCRDT.from_state(remote_node_id, remote_state)

        merged = VectorCRDT(node_id)
        merged.counter = local_crdt.counter.merge(remote_crdt.counter)
        merged.lww = local_crdt.lww.merge(remote_crdt.lww)
        return merged.to_state()

    def insert_vector(self, vector_id: str, metadata: Any, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self.lww.add(vector_id, metadata, ts)
        self.counter.increment()

    def delete_vector(self, vector_id: str, timestamp: Optional[float] = None) -> None:
        self.lww.remove(vector_id, timestamp or time.time())

    def get_vector(self, vector_id: str) -> Optional[Any]:
        return self.lww.lookup(vector_id)

    def all_vectors(self) -> Dict[str, Any]:
        return self.lww.elements()


# ---------------------------------------------------------------------------
# CRDTSyncService
# ---------------------------------------------------------------------------

class CRDTSyncService:
    """
    Tracks the local CRDT state for a region.
    Produces deltas for outbound sync and merges incoming deltas.
    """

    def __init__(self, node_id: str, clock_store_path: str = "crdt_clocks.json") -> None:
        self.node_id = node_id
        self._lock = threading.Lock()
        self._crdt = VectorCRDT(node_id)
        self._clock_store = VectorClockStore(clock_store_path)
        self._conflicts: List[Dict[str, Any]] = []
        self._last_snapshot: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Local operations
    # ------------------------------------------------------------------

    def local_insert(self, vector_id: str, metadata: Any, timestamp: Optional[float] = None) -> None:
        ts = timestamp or time.time()
        with self._lock:
            self._crdt.insert_vector(vector_id, metadata, ts)
            self._clock_store.increment(vector_id, self.node_id)

    def local_delete(self, vector_id: str, timestamp: Optional[float] = None) -> None:
        ts = timestamp or time.time()
        with self._lock:
            self._crdt.delete_vector(vector_id, ts)
            self._clock_store.increment(vector_id, self.node_id)

    def get_vector(self, vector_id: str) -> Optional[Any]:
        with self._lock:
            return self._crdt.get_vector(vector_id)

    def all_vectors(self) -> Dict[str, Any]:
        with self._lock:
            return self._crdt.all_vectors()

    def vector_count(self) -> int:
        return len(self.all_vectors())

    # ------------------------------------------------------------------
    # Delta production
    # ------------------------------------------------------------------

    def produce_delta(self) -> Dict[str, Any]:
        """Return a delta representing the full current state (for sync)."""
        with self._lock:
            state = self._crdt.to_state()
            lww = state["lww"]
            delta: Dict[str, Any] = {
                "node_id": self.node_id,
                "timestamp": time.time(),
                "additions": dict(lww.get("add_set", {})),
                "deletions": dict(lww.get("remove_set", {})),
                "counter": state["counter"],
                "vector_clocks": {
                    k: self._clock_store.get(k).to_dict()
                    for k in self._clock_store.all_keys()
                },
            }
            self._last_snapshot = state
            return delta

    # ------------------------------------------------------------------
    # Incoming delta merge
    # ------------------------------------------------------------------

    def merge_delta(self, delta: Dict[str, Any]) -> Dict[str, str]:
        """
        Merge an incoming delta from a remote region.
        Returns {vector_id: "ok"|"conflict"} for each vector in the delta's clocks.
        """
        results: Dict[str, str] = {}
        remote_node = delta.get("node_id", "remote")

        with self._lock:
            remote_lww = LWWElementSet()
            remote_lww._add_set = dict(delta.get("additions", {}))
            remote_lww._remove_set = dict(delta.get("deletions", {}))

            remote_vc_data: Dict[str, Dict[str, int]] = delta.get("vector_clocks", {})
            for vector_id, remote_vc_dict in remote_vc_data.items():
                remote_vc = VectorClock.from_dict(remote_vc_dict)
                local_vc = self._clock_store.get(vector_id)
                if concurrent(local_vc, remote_vc):
                    self._conflicts.append({
                        "vector_id": vector_id,
                        "local_clock": local_vc.to_dict(),
                        "remote_clock": remote_vc.to_dict(),
                        "remote_node": remote_node,
                        "detected_at": time.time(),
                    })
                    results[vector_id] = "conflict"
                else:
                    results[vector_id] = "ok"
                self._clock_store.merge(vector_id, remote_vc)

            self._crdt.lww = self._crdt.lww.merge(remote_lww)

            if "counter" in delta:
                remote_counter = GCounter.from_dict(delta["counter"])
                self._crdt.counter = self._crdt.counter.merge(remote_counter)

        return results

    # ------------------------------------------------------------------
    # Conflicts
    # ------------------------------------------------------------------

    def get_conflicts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._conflicts)

    def resolve_conflict(self, vector_id: str) -> bool:
        with self._lock:
            before = len(self._conflicts)
            self._conflicts = [c for c in self._conflicts if c["vector_id"] != vector_id]
            return len(self._conflicts) < before

    # ------------------------------------------------------------------
    # Full state merge (initial sync)
    # ------------------------------------------------------------------

    def merge_remote_state(self, remote_state: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            local_state = self._crdt.to_state()
            merged_state = VectorCRDT.merge(local_state, remote_state)
            self._crdt = VectorCRDT.from_state(self.node_id, merged_state)
            return merged_state
