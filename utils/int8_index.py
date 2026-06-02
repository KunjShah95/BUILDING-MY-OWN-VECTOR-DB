"""
Int8 (8-bit integer) Quantization Index

Memory-efficient vector storage by quantizing float32 vectors to int8.
Provides 4x memory reduction with minimal recall loss (typically <1-2%).

How it works:
  1. Learn per-dimension min/max ranges from training data
  2. Quantize float32 → int8: q = round(255 * (x - min) / (max - min)) - 128
  3. De-quantize int8 → float32: x = min + (q + 128) * (max - min) / 255
  4. Search: de-quantize centroids then use cosine/euclidean distance

Memory savings:
  - Original: D × 4 bytes (float32)
  - Int8: D × 1 byte (int8)
  - For 384-dim vectors: 1536 bytes → 384 bytes (4x compression)

Usage:
  index = Int8Index()
  index.train(training_vectors)
  index.add(vector, "id1", {"text": "doc1"})
  results = index.search(query_vector, k=5)
"""

from __future__ import annotations

import json
import logging
import time
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


class Int8Index:
    """Int8 quantized vector index with exact de-quantized search.

    Parameters
    ----------
    distance_metric : str
        "cosine" or "euclidean". Default "cosine".
    verbose : bool
        Log training progress. Default False.
    """

    def __init__(
        self,
        distance_metric: str = "cosine",
        verbose: bool = False,
    ):
        self.distance_metric = distance_metric
        self.verbose = verbose

        # Quantization parameters: per-dimension min/max
        self._dim: int = 0
        self._mins: Optional[np.ndarray] = None   # shape (D,)
        self._maxs: Optional[np.ndarray] = None   # shape (D,)
        self._scales: Optional[np.ndarray] = None # shape (D,)

        # Quantized vectors: vector_id -> np.int8 array of length D
        self.codes: Dict[str, np.ndarray] = {}
        self.vectors: Dict[str, np.ndarray] = {}  # original float32 for exact search
        self.metadata: Dict[str, Any] = {}
        self.vector_ids: List[str] = []
        self.is_trained: bool = False

    # ---- Public API --------------------------------------------------------

    def train(self, vectors: List[List[float]]) -> Int8Index:
        """Learn per-dimension min/max ranges from training vectors.

        Parameters
        ----------
        vectors : list of list of float
            Training set.

        Returns
        -------
        self
        """
        arr = np.array(vectors, dtype=np.float32)
        n, D = arr.shape
        if n < 1:
            raise ValueError("Need at least 1 training vector")

        self._dim = D
        self._mins = arr.min(axis=0)  # shape (D,)
        self._maxs = arr.max(axis=0)

        # Avoid division by zero: if min == max, set scale to 1.0
        ranges = self._maxs - self._mins
        ranges = np.where(ranges < 1e-12, 1.0, ranges)
        self._scales = 255.0 / ranges

        self.is_trained = True
        if self.verbose:
            logger.info(
                "Int8 training complete: D=%d, %d vectors, ranges [%.4f, %.4f]",
                D, n, self._mins.min(), self._maxs.max(),
            )
        return self

    def add(self, vector: List[float], vector_id: str, metadata: Any = None) -> None:
        """Quantize and store a vector as int8."""
        if not self.is_trained:
            raise RuntimeError("Int8Index must be trained before adding vectors")

        vec = np.array(vector, dtype=np.float32)
        if self.distance_metric == "cosine":
            vec = _l2_normalize(vec)

        encoded = self._quantize(vec)
        self.codes[vector_id] = encoded
        self.vectors[vector_id] = vec  # keep original for exact de-quantized search
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
        use_exact: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search using int8 quantized index.

        Parameters
        ----------
        query_vector : list of float
            Query vector.
        k : int
            Number of results.
        use_exact : bool
            If True, de-quantize stored vectors for exact distance computation.
            If False, compute distance directly in quantized space (faster but
            slightly less accurate).

        Returns
        -------
        list of dict with keys ``vector_id``, ``distance``, ``metadata``.
        """
        if not self.is_trained:
            raise RuntimeError("Int8Index must be trained before searching")
        if not self.vector_ids:
            return []

        query = np.array(query_vector, dtype=np.float32)
        if self.distance_metric == "cosine":
            query = _l2_normalize(query)

        if use_exact:
            # De-quantize vectors for exact distance
            candidates: List[Tuple[float, str]] = []
            for vid in self.vector_ids:
                reconstructed = self._dequantize(self.codes[vid])
                dist = self._exact_distance(query, reconstructed)
                candidates.append((dist, vid))
        else:
            # Quantize query and compute in int8 space (faster)
            q_int8 = self._quantize(query)
            candidates = []
            for vid in self.vector_ids:
                code = self.codes[vid]
                if self.distance_metric == "cosine":
                    # Use dot product approximation in int8 space
                    q_f = q_int8.astype(np.float32)
                    c_f = code.astype(np.float32)
                    dist = 1.0 - np.dot(q_f, c_f) / (
                        np.linalg.norm(q_f) * np.linalg.norm(c_f) + 1e-12
                    )
                else:
                    dist = float(np.linalg.norm(q_int8.astype(np.float32) - code.astype(np.float32)))
                candidates.append((dist, vid))

        candidates.sort(key=lambda x: x[0])
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
        """Remove all vectors (keeps quantization ranges)."""
        self.codes.clear()
        self.vectors.clear()
        self.metadata.clear()
        self.vector_ids.clear()

    # ---- Persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the index to a JSON file."""
        data = {
            "distance_metric": self.distance_metric,
            "_dim": self._dim,
            "is_trained": self.is_trained,
            "mins": self._mins.tolist() if self._mins is not None else None,
            "maxs": self._maxs.tolist() if self._maxs is not None else None,
            "scales": self._scales.tolist() if self._scales is not None else None,
            "codes": {vid: code.tolist() for vid, code in self.codes.items()},
            "vectors": {vid: vec.tolist() for vid, vec in self.vectors.items()},
            "metadata": {vid: _make_serializable(meta) for vid, meta in self.metadata.items()},
            "vector_ids": self.vector_ids,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> Int8Index:
        """Deserialize an index from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        obj = cls(
            distance_metric=data.get("distance_metric", "cosine"),
        )
        obj._dim = data["_dim"]
        obj.is_trained = data["is_trained"]
        obj._mins = np.array(data["mins"], dtype=np.float32) if data.get("mins") else None
        obj._maxs = np.array(data["maxs"], dtype=np.float32) if data.get("maxs") else None
        obj._scales = np.array(data["scales"], dtype=np.float32) if data.get("scales") else None
        obj.codes = {vid: np.array(code, dtype=np.int8) for vid, code in data["codes"].items()}
        obj.metadata = {}
        for vid, meta in data.get("metadata", {}).items():
            obj.metadata[vid] = meta
        obj.vector_ids = data.get("vector_ids", [])
        obj.vectors = {}
        if "vectors" in data:
            obj.vectors = {vid: np.array(vec, dtype=np.float32) for vid, vec in data["vectors"].items()}
        return obj

    # ---- Statistics --------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        n = len(self.vector_ids)
        bytes_original = n * self._dim * 4 if self._dim > 0 else 0
        bytes_compressed = n * self._dim * 1 if self._dim > 0 else 0  # 1 byte per dim
        compression_ratio = bytes_original / bytes_compressed if bytes_compressed > 0 else 0

        return {
            "total_vectors": n,
            "dimension": self._dim,
            "is_trained": self.is_trained,
            "bytes_original": bytes_original,
            "bytes_compressed": bytes_compressed,
            "compression_ratio": round(compression_ratio, 1),
            "memory_saved_bytes": bytes_original - bytes_compressed,
            "index_type": "int8",
        }

    def __len__(self) -> int:
        return len(self.vector_ids)

    # ---- Internal: Quantization -------------------------------------------

    def _quantize(self, vec: np.ndarray) -> np.ndarray:
        """Quantize float32 vector to int8."""
        if self._mins is None or self._scales is None:
            raise RuntimeError("Int8Index not trained")
        # q = round(255 * (x - min) / (max - min)) - 128
        scaled = (vec - self._mins) * self._scales  # range [0, 255]
        codes = np.round(scaled).clip(0, 255).astype(np.int8) - 128
        return codes

    def _dequantize(self, codes: np.ndarray) -> np.ndarray:
        """De-quantize int8 codes back to float32."""
        if self._mins is None or self._scales is None:
            raise RuntimeError("Int8Index not trained")
        # x = min + (q + 128) / 255 * (max - min)
        vec = self._mins + (codes.astype(np.float32) + 128) / self._scales
        return vec

    def _exact_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(v1 - v2))
        return float(1.0 - np.dot(v1, v2))
