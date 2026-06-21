"""
Brute-force exact nearest-neighbor search.
Merged from C:\\ann search engine on 2026-06-21.

Use for:
- Small collections (< ~50k vectors)
- Ground-truth recall benchmarking against approximate indexes
- Unit tests that need deterministic results
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from utils.base_index import BaseIndex

logger = logging.getLogger(__name__)


class BruteForceIndex(BaseIndex):
    """
    Exact nearest-neighbor via exhaustive scan.

    - 100% recall
    - O(N) per query — does not scale to millions of vectors
    - Supports cosine, euclidean, dot distance
    """

    def __init__(self, dimension: int, metric: str = "cosine"):
        super().__init__(dimension, metric)
        self._vectors: Dict[str, np.ndarray] = {}
        self._vector_list: List[np.ndarray] = []
        self._vector_ids: List[str] = []

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def index_type(self) -> str:
        return "brute_force"

    @property
    def is_trained(self) -> bool:
        return True

    @property
    def vector_count(self) -> int:
        return len(self._vectors)

    # ── mutations ────────────────────────────────────────────────────────────

    def add(self, vectors: List[List[float]], vector_ids: Optional[List[str]] = None) -> None:
        if vector_ids is None:
            vector_ids = self._default_vector_ids(len(vectors), len(self._vectors))

        for vector, vid in zip(vectors, vector_ids):
            arr = self._validate_vector(vector)
            self._vectors[vid] = arr
            self._vector_list.append(arr)
            self._vector_ids.append(vid)

        self._vector_count = len(self._vectors)
        logger.info("Added %d vectors. Total: %d", len(vectors), self.vector_count)

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._vectors:
            return False
        idx = self._vector_ids.index(vector_id)
        del self._vectors[vector_id]
        self._vector_list.pop(idx)
        self._vector_ids.pop(idx)
        self._vector_count = len(self._vectors)
        return True

    def clear(self) -> None:
        self._vectors.clear()
        self._vector_list.clear()
        self._vector_ids.clear()
        self._vector_count = 0

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, query: List[float], k: int = 10) -> List[Tuple[str, float]]:
        if self.vector_count == 0:
            return []

        query_arr = self._validate_vector(query)
        distances = self._compute_distances(query_arr)

        k = min(k, len(distances))
        top_indices = np.argpartition(distances, k - 1)[:k]
        results = [(self._vector_ids[i], float(distances[i])) for i in top_indices]
        results.sort(key=lambda x: x[1])
        return results

    def _compute_distances(self, query: np.ndarray) -> np.ndarray:
        mat = np.array(self._vector_list)  # (N, D)
        if self.metric == "cosine":
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = mat @ query / norms / (np.linalg.norm(query) or 1.0)
            return 1.0 - sims
        elif self.metric == "euclidean":
            return np.linalg.norm(mat - query, axis=1)
        elif self.metric == "dot":
            return -(mat @ query)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        data = {
            "dimension": self.dimension,
            "metric": self.metric,
            "vectors": {vid: vec.tolist() for vid, vec in self._vectors.items()},
            "vector_ids": self._vector_ids,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        logger.info("BruteForceIndex saved to %s", filepath)

    def load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            data = json.load(f)
        self.dimension = data["dimension"]
        self.metric = data["metric"]
        self._vectors = {vid: np.array(v, dtype=np.float32) for vid, v in data["vectors"].items()}
        self._vector_ids = data["vector_ids"]
        self._vector_list = [self._vectors[vid] for vid in self._vector_ids]
        self._vector_count = len(self._vectors)
        logger.info("BruteForceIndex loaded from %s (%d vectors)", filepath, self.vector_count)

    # ── stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        mem = sum(v.nbytes for v in self._vector_list)
        return {
            "index_type": self.index_type,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "metric": self.metric,
            "is_trained": self.is_trained,
            "memory_usage_bytes": mem,
        }
