"""
Example plugin: BruteForcePlugin wraps utils/brute_force_index.py.
"""
from __future__ import annotations

import sys
import os
from typing import Any, Dict, List

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from sdk.plugin_sdk.base_index import BaseIndexPlugin
from sdk.plugin_sdk.decorators import plugin
from utils.brute_force_index import BruteForceIndex


@plugin(
    name="brute_force",
    version="1.0.0",
    plugin_type="index",
    description="Exact nearest-neighbour search via exhaustive scan.",
    author="VectorDB",
)
class BruteForcePlugin(BaseIndexPlugin):
    """
    Wraps BruteForceIndex in the plugin SDK interface.
    dimension and metric can be passed at construction time.
    """

    def __init__(self, dimension: int = 128, metric: str = "cosine") -> None:
        self._index = BruteForceIndex(dimension=dimension, metric=metric)
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._dimension = dimension

    def train(self, vectors: List[List[float]]) -> None:
        """BruteForce is always trained; this is a no-op."""

    def add(self, vector: List[float], id: str, metadata: Dict[str, Any]) -> None:
        self._index.add([vector], [id])
        self._metadata[id] = metadata or {}

    def search(self, query: List[float], k: int) -> List[Dict[str, Any]]:
        raw = self._index.search(query, k)
        return [
            {"id": vid, "score": score, "metadata": self._metadata.get(vid, {})}
            for vid, score in raw
        ]

    def delete(self, id: str) -> bool:
        existed = self._index.delete(id)
        if existed:
            self._metadata.pop(id, None)
        return existed

    def get_stats(self) -> Dict[str, Any]:
        stats = self._index.get_stats()
        stats["metadata_count"] = len(self._metadata)
        return stats

    def save(self, path: str) -> None:
        self._index.save(path)

    def load(self, path: str) -> None:
        self._index.load(path)
