"""
Vector clock implementation for distributed conflict detection.
Used by CRDTSyncService to detect concurrent vs causally ordered writes.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional


class VectorClock:
    """
    A vector clock maps node_id -> logical counter.
    Used to establish causal ordering between events across nodes.
    """

    def __init__(self, clocks: Optional[Dict[str, int]] = None):
        self.clocks: Dict[str, int] = dict(clocks) if clocks else {}

    def increment(self, node_id: str) -> "VectorClock":
        """Increment the counter for node_id and return self."""
        self.clocks[node_id] = self.clocks.get(node_id, 0) + 1
        return self

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Return a new VectorClock that is the element-wise maximum."""
        merged: Dict[str, int] = dict(self.clocks)
        for node_id, count in other.clocks.items():
            merged[node_id] = max(merged.get(node_id, 0), count)
        return VectorClock(merged)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.clocks)

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorClock":
        return cls(data)

    def __repr__(self) -> str:
        return f"VectorClock({self.clocks})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VectorClock):
            return False
        return self.clocks == other.clocks


def happens_before(a: VectorClock, b: VectorClock) -> bool:
    """
    Return True if event a causally happened before event b.
    a < b iff forall i: a[i] <= b[i] AND exists j: a[j] < b[j]
    """
    all_nodes = set(a.clocks) | set(b.clocks)
    less_or_equal = all(a.clocks.get(n, 0) <= b.clocks.get(n, 0) for n in all_nodes)
    strictly_less = any(a.clocks.get(n, 0) < b.clocks.get(n, 0) for n in all_nodes)
    return less_or_equal and strictly_less


def concurrent(a: VectorClock, b: VectorClock) -> bool:
    """Return True if neither a happened-before b nor b happened-before a."""
    return not happens_before(a, b) and not happens_before(b, a)


class VectorClockStore:
    """
    Persists named VectorClock instances to a JSON file.
    Thread-safe via a lock.
    """

    def __init__(self, path: str = "vector_clocks.json"):
        self._path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
        except OSError:
            pass  # best-effort

    def get(self, key: str) -> VectorClock:
        with self._lock:
            return VectorClock(self._data.get(key, {}))

    def set(self, key: str, clock: VectorClock) -> None:
        with self._lock:
            self._data[key] = clock.to_dict()
            self._save()

    def increment(self, key: str, node_id: str) -> VectorClock:
        with self._lock:
            clock = VectorClock(self._data.get(key, {}))
            clock.increment(node_id)
            self._data[key] = clock.to_dict()
            self._save()
            return clock

    def merge(self, key: str, incoming: VectorClock) -> VectorClock:
        """Merge incoming clock into stored clock and persist."""
        with self._lock:
            stored = VectorClock(self._data.get(key, {}))
            merged = stored.merge(incoming)
            self._data[key] = merged.to_dict()
            self._save()
            return merged

    def all_keys(self) -> list:
        with self._lock:
            return list(self._data.keys())

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._save()
