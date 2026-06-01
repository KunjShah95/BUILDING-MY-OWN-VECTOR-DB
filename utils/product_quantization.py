"""
Product Quantization (PQ) Index

Compresses high-dimensional vectors into compact 8-bit codes using product
quantization, enabling 10-16x memory reduction with minimal recall loss.

How it works:
  1. Split each D-dimensional vector into M sub-vectors of dimension D/M each
  2. For each subspace, learn 256 centroids (8-bit codebook) via k-means
  3. Encode: each sub-vector -> index of nearest centroid (1 byte)
  4. Search: Asymmetric Distance Computation (ADC)
     - Precompute distance from query sub-vectors to all centroids
     - Look up distances for encoded codes => fast approximate search

Memory savings:
  - Original: D × 4 bytes (float32)
  - PQ compressed: M × 1 byte (uint8)
  - For 384-dim vectors with M=32: 1536 bytes -> 32 bytes (48x compression)

Usage:
  index = PQIndex(M=32)
  index.train(training_vectors)
  index.add(vector, "id1")
  index.add(vector, "id2")
  results = index.search(query_vector, k=5)
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


class PQIndex:
    """Product Quantization index with asymmetric distance computation.

    Parameters
    ----------
    M : int
        Number of sub-quantizers (subspaces). The vector dimension D must be
        divisible by M.  Defaults to min(32, D // 2).
    k_sub : int
        Number of centroids per sub-quantizer (must be ≤ 256 for uint8 codes).
        Default 256 (8-bit).
    n_iter : int
        Max k-means iterations per subspace. Default 20.
    distance_metric : str
        "cosine" or "euclidean". Default "cosine".
    verbose : bool
        Log training progress. Default False.
    """

    def __init__(
        self,
        M: Optional[int] = None,
        k_sub: int = 256,
        n_iter: int = 20,
        distance_metric: str = "cosine",
        verbose: bool = False,
    ):
        if k_sub > 256:
            raise ValueError("k_sub must be ≤ 256 (for uint8 encoding)")
        if k_sub < 1:
            raise ValueError("k_sub must be ≥ 1")

        self.M = M  # set during train() once D is known
        self.k_sub = k_sub
        self.n_iter = n_iter
        self.distance_metric = distance_metric
        self.verbose = verbose

        # Codebooks: list of np.ndarray, each shape (k_sub, sub_dim)
        self.codebooks: Optional[List[np.ndarray]] = None

        # Encoded vectors: vector_id -> np.uint8 codes of length M
        self.codes: Dict[str, np.ndarray] = {}
        self.vectors: Dict[str, np.ndarray] = {}  # original for rerank (optional)
        self.metadata: Dict[str, Any] = {}
        self.vector_ids: List[str] = []

        # Training stats
        self._sub_dim: int = 0
        self._dim: int = 0
        self.is_trained: bool = False

    # ---- Public API --------------------------------------------------------

    def train(self, vectors: List[List[float]]) -> PQIndex:
        """Learn codebooks from training vectors.

        Parameters
        ----------
        vectors : list of list of float
            Training set (at least ``k_sub`` vectors per subspace recommended).

        Returns
        -------
        self
        """
        arr = np.array(vectors, dtype=np.float32)
        if self.distance_metric == "cosine":
            for i in range(len(arr)):
                arr[i] = _l2_normalize(arr[i])

        n, D = arr.shape
        if n < 1:
            raise ValueError("Need at least 1 training vector")

        # Auto-infer M if not set
        if self.M is None:
            self.M = min(32, D // 2)
        self.M = max(1, min(self.M, D))
        if D % self.M != 0:
            # Round M down to nearest divisor of D
            while D % self.M != 0:
                self.M -= 1
            if self.M < 1:
                raise ValueError(f"Cannot find valid M for dimension {D}")

        self._dim = D
        self._sub_dim = D // self.M
        self.codebooks = []

        for m in range(self.M):
            start = m * self._sub_dim
            end = start + self._sub_dim
            sub_vectors = arr[:, start:end]

            # K-means for this subspace
            codebook = self._kmeans(sub_vectors, self.k_sub, self.n_iter)
            self.codebooks.append(codebook)

            if self.verbose:
                logger.info(
                    "Sub-quantizer %d/%d trained (%d centroids, dim=%d)",
                    m + 1, self.M, self.k_sub, self._sub_dim,
                )

        self.is_trained = True
        if self.verbose:
            logger.info(
                "PQ training complete: D=%d, M=%d, sub_dim=%d, k_sub=%d",
                D, self.M, self._sub_dim, self.k_sub,
            )
        return self

    def add(self, vector: List[float], vector_id: str, metadata: Any = None) -> None:
        """Compress and store a vector."""
        if not self.is_trained:
            raise RuntimeError("PQIndex must be trained before adding vectors")

        vec = np.array(vector, dtype=np.float32)
        if self.distance_metric == "cosine":
            vec = _l2_normalize(vec)

        encoded = self._encode_vector(vec)
        self.codes[vector_id] = encoded
        self.vectors[vector_id] = vec  # keep original for optional exact rerank
        self.metadata[vector_id] = metadata
        self.vector_ids.append(vector_id)

    def add_batch(self, entries: List[Dict[str, Any]]) -> None:
        """Add multiple vectors at once.

        Each entry must have ``vector``, ``vector_id``, and optionally
        ``metadata``.
        """
        for entry in entries:
            self.add(
                vector=entry["vector"],
                vector_id=entry["vector_id"],
                metadata=entry.get("metadata"),
            )

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search using Asymmetric Distance Computation (ADC).

        Parameters
        ----------
        query_vector : list of float
            Query vector (not compressed).
        k : int
            Number of results.
        rerank : bool
            If True, rerank top ``k * 3`` candidates with exact distance for
            improved accuracy.  Requires original vectors to be stored.

        Returns
        -------
        list of dict with keys ``vector_id``, ``distance``, ``metadata``.
        """
        if not self.is_trained:
            raise RuntimeError("PQIndex must be trained before searching")
        if not self.vector_ids:
            return []

        query = np.array(query_vector, dtype=np.float32)
        if self.distance_metric == "cosine":
            query = _l2_normalize(query)

        # Precompute distance table: list of (k_sub,) arrays
        # dist_table[m][c] = distance from query sub-vector to centroid c
        dist_tables = self._compute_distance_tables(query)

        # Score all encoded vectors
        candidates: List[Tuple[float, str]] = []
        for vid in self.vector_ids:
            code = self.codes[vid]
            # Sum distances from lookups in precomputed tables
            approx_dist = sum(dist_tables[m][int(code[m])] for m in range(self.M))
            candidates.append((approx_dist, vid))

        # Sort by approximate distance
        candidates.sort(key=lambda x: x[0])

        if rerank:
            if not self.vectors:
                logger.warning("Rerank unavailable: original vectors not loaded. Falling back to ADC.")
            else:
                # Take more candidates, rerank with exact distance
                rerank_count = min(k * 3, len(candidates))
                top = candidates[:rerank_count]
                reranked: List[Tuple[float, str]] = []
                for approx_dist, vid in top:
                    exact_dist = self._exact_distance(query, self.vectors[vid])
                    reranked.append((exact_dist, vid))
                reranked.sort(key=lambda x: x[0])
                results = reranked[:k]
                return [
                    {
                        "vector_id": vid,
                        "distance": float(dist),
                        "metadata": self.metadata.get(vid),
                    }
                    for dist, vid in results
                ]
        results = candidates[:k]

        return [
            {
                "vector_id": vid,
                "distance": float(dist),
                "metadata": self.metadata.get(vid),
            }
            for dist, vid in results
        ]

    def delete(self, vector_id: str) -> bool:
        """Remove a vector from the index."""
        if vector_id not in self.codes:
            return False
        del self.codes[vector_id]
        self.vectors.pop(vector_id, None)
        self.metadata.pop(vector_id, None)
        if vector_id in self.vector_ids:
            self.vector_ids.remove(vector_id)
        return True

    def clear(self) -> None:
        """Remove all vectors (keeps codebooks)."""
        self.codes.clear()
        self.vectors.clear()
        self.metadata.clear()
        self.vector_ids.clear()

    # ---- Persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the index to a JSON file."""
        data = {
            "M": self.M,
            "k_sub": self.k_sub,
            "n_iter": self.n_iter,
            "distance_metric": self.distance_metric,
            "_dim": self._dim,
            "_sub_dim": self._sub_dim,
            "is_trained": self.is_trained,
            "codebooks": [cb.tolist() for cb in self.codebooks] if self.codebooks else None,
            "codes": {vid: code.tolist() for vid, code in self.codes.items()},
            "vectors": {vid: vec.tolist() for vid, vec in self.vectors.items()},
            "metadata": {vid: _make_serializable(meta) for vid, meta in self.metadata.items()},
            "vector_ids": self.vector_ids,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> PQIndex:
        """Deserialize an index from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        obj = cls(
            M=data["M"],
            k_sub=data["k_sub"],
            n_iter=data.get("n_iter", 20),
            distance_metric=data.get("distance_metric", "cosine"),
        )
        obj._dim = data["_dim"]
        obj._sub_dim = data["_sub_dim"]
        obj.is_trained = data["is_trained"]
        obj.codebooks = [np.array(cb, dtype=np.float32) for cb in data["codebooks"]]
        obj.codes = {vid: np.array(code, dtype=np.uint8) for vid, code in data["codes"].items()}
        obj.metadata = {}
        for vid, meta in data.get("metadata", {}).items():
            obj.metadata[vid] = meta
        obj.vector_ids = data.get("vector_ids", [])
        # Restore original vectors for exact rerank support
        obj.vectors = {}
        if "vectors" in data:
            obj.vectors = {vid: np.array(vec, dtype=np.float32) for vid, vec in data["vectors"].items()}
        return obj

    # ---- Statistics --------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        n = len(self.vector_ids)
        bytes_original = n * self._dim * 4 if self._dim > 0 else 0
        bytes_compressed = n * self.M if self.M else 0
        compression_ratio = bytes_original / bytes_compressed if bytes_compressed > 0 else 0

        return {
            "total_vectors": n,
            "dimension": self._dim,
            "M": self.M,
            "sub_dim": self._sub_dim,
            "k_sub": self.k_sub,
            "is_trained": self.is_trained,
            "bytes_original": bytes_original,
            "bytes_compressed": bytes_compressed,
            "compression_ratio": round(compression_ratio, 1),
            "memory_saved_bytes": bytes_original - bytes_compressed,
        }

    def __len__(self) -> int:
        return len(self.vector_ids)

    # ---- Internal: K-means ------------------------------------------------

    @staticmethod
    def _kmeans(data: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """Simple k-means for a single subspace.

        Parameters
        ----------
        data : np.ndarray, shape (n, sub_dim)
        k : int
            Number of centroids.
        max_iter : int
            Max iterations.

        Returns
        -------
        centroids : np.ndarray, shape (k, sub_dim)
        """
        n, d = data.shape
        if n == 0:
            return np.zeros((k, d), dtype=np.float32)
        if n <= k:
            # Fewer points than centroids: pad with zeros
            centroids = np.zeros((k, d), dtype=np.float32)
            centroids[:n] = data
            return centroids

        # Forgy initialization: pick k random points
        rng = np.random.RandomState(0)
        idx = rng.choice(n, k, replace=False)
        centroids = data[idx].copy()

        for iteration in range(max_iter):
            # Assign each point to nearest centroid
            dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)  # (n, k)
            labels = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.any():
                    new_centroids[j] = data[mask].mean(axis=0)
                else:
                    new_centroids[j] = centroids[j]  # keep old

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < 1e-6:
                break

        return centroids

    # ---- Internal: Encoding / Distance ------------------------------------

    def _encode_vector(self, vec: np.ndarray) -> np.ndarray:
        """Encode a single vector into M uint8 codes."""
        codes = np.empty(self.M, dtype=np.uint8)
        for m in range(self.M):
            start = m * self._sub_dim
            end = start + self._sub_dim
            sub_vec = vec[start:end]
            # Distance to all centroids in this subspace
            cb = self.codebooks[m]  # (k_sub, sub_dim)
            dists = np.linalg.norm(sub_vec - cb, axis=1)  # (k_sub,)
            codes[m] = int(np.argmin(dists))
        return codes

    def _compute_distance_tables(self, query: np.ndarray) -> List[np.ndarray]:
        """Precompute distance from query sub-vectors to all centroids.

        Returns
        -------
        list of np.ndarray, each shape (k_sub,)
        """
        tables = []
        for m in range(self.M):
            start = m * self._sub_dim
            end = start + self._sub_dim
            q_sub = query[start:end]
            cb = self.codebooks[m]
            if self.distance_metric == "cosine":
                # dot product in normalized space -> cosine distance
                dists = 1.0 - np.dot(cb, q_sub)
            else:
                dists = np.linalg.norm(cb - q_sub, axis=1)
            tables.append(dists)
        return tables

    def _exact_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(v1 - v2))
        return float(1.0 - np.dot(v1, v2))
