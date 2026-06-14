"""
SIMD / AVX-512 Optimized Distance Kernels (Phase 5: Hardware Acceleration).

Provides hand-optimized distance computation functions that leverage CPU SIMD
instructions (AVX2, AVX-512) through NumPy's vectorized operations plus
optional Cython/PyBind11 extensions.

For pure Python fallback, we use numpy which already utilizes SIMD internally
for many operations. The C++ extensions provide additional speedup for
inner-loop distances during HNSW/IVF search.

The module auto-detects and selects the fastest available backend:
  1. C++/PyBind11 extension (if compiled)
  2. Numba JIT-compiled kernels
  3. NumPy vectorized (always available, uses CPU SIMD internally)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_HAS_NUMBA = False
try:
    from numba import njit, prange, float32, int32
    _HAS_NUMBA = True
except ImportError:
    pass

_HAS_CPP_EXT = False
try:
    from _simd_kernels import (
        cosine_distance_avx512,
        euclidean_distance_avx512,
        dot_product_avx512,
    )
    _HAS_CPP_EXT = True
except ImportError:
    pass


def get_available_backends() -> List[str]:
    backends = ["numpy"]
    if _HAS_CPP_EXT:
        backends.insert(0, "cpp_avx512")
    if _HAS_NUMBA:
        backends.insert(1, "numba")
    return backends


# ---------------------------------------------------------------------------
# Numba JIT kernels
# ---------------------------------------------------------------------------


if _HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _numba_batch_cosine(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Numba-parallelized batch cosine distance."""
        n = vectors.shape[0]
        result = np.empty(n, dtype=np.float32)
        for i in prange(n):
            dot = 0.0
            norm_v = 0.0
            norm_q = 0.0
            for j in range(vectors.shape[1]):
                dot += vectors[i, j] * query[j]
                norm_v += vectors[i, j] ** 2
                norm_q += query[j] ** 2
            norm_v = np.sqrt(norm_v)
            norm_q = np.sqrt(norm_q)
            if norm_v < 1e-10 or norm_q < 1e-10:
                result[i] = 1.0
            else:
                result[i] = 1.0 - dot / (norm_v * norm_q)
        return result

    @njit(parallel=True, fastmath=True)
    def _numba_batch_euclidean(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Numba-parallelized batch Euclidean distance."""
        n = vectors.shape[0]
        result = np.empty(n, dtype=np.float32)
        for i in prange(n):
            sq = 0.0
            for j in range(vectors.shape[1]):
                diff = vectors[i, j] - query[j]
                sq += diff * diff
            result[i] = np.sqrt(sq)
        return result

    @njit(parallel=True, fastmath=True)
    def _numba_batch_dot(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Numba-parallelized batch dot product."""
        n = vectors.shape[0]
        result = np.empty(n, dtype=np.float32)
        for i in prange(n):
            dot = 0.0
            for j in range(vectors.shape[1]):
                dot += vectors[i, j] * query[j]
            result[i] = -dot  # negative for distance semantics
        return result


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


class SIMDKernels:
    """Auto-selecting SIMD-optimized distance kernels."""

    def __init__(self, preferred_backend: Optional[str] = None):
        self.backend = preferred_backend or self._pick_backend()
        logger.info("SIMDKernels using backend: %s", self.backend)

    @staticmethod
    def _pick_backend() -> str:
        if _HAS_CPP_EXT:
            return "cpp_avx512"
        if _HAS_NUMBA:
            return "numba"
        return "numpy"

    @staticmethod
    def is_avx512_available() -> bool:
        return _HAS_CPP_EXT

    @staticmethod
    def is_numba_available() -> bool:
        return _HAS_NUMBA

    def batch_cosine(
        self, vectors: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        """Batch cosine distance.

        Args:
            vectors: (N, D) float32 array.
            query: (D,) float32 array.

        Returns:
            (N,) float32 array of distances.
        """
        if self.backend == "cpp_avx512":
            return cosine_distance_avx512(vectors, query)  # type: ignore
        if self.backend == "numba":
            return _numba_batch_cosine(vectors, query.astype(np.float32))
        # NumPy fallback
        dots = vectors @ query
        v_norms = np.linalg.norm(vectors, axis=1)
        q_norm = np.linalg.norm(query)
        denom = np.maximum(v_norms * q_norm, 1e-10)
        return (1.0 - dots / denom).astype(np.float32)

    def batch_euclidean(
        self, vectors: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        """Batch Euclidean distance."""
        if self.backend == "cpp_avx512":
            return euclidean_distance_avx512(vectors, query)  # type: ignore
        if self.backend == "numba":
            return _numba_batch_euclidean(vectors, query.astype(np.float32))
        # NumPy fallback using ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        vec_sq = np.sum(vectors ** 2, axis=1)
        q_sq = np.dot(query, query)
        cross = vectors @ query
        sq_dists = np.maximum(vec_sq + q_sq - 2.0 * cross, 0.0)
        return np.sqrt(sq_dists).astype(np.float32)

    def batch_dot_product(
        self, vectors: np.ndarray, query: np.ndarray
    ) -> np.ndarray:
        """Batch dot product (negative for distance semantics)."""
        if self.backend == "cpp_avx512":
            return dot_product_avx512(vectors, query)  # type: ignore
        if self.backend == "numba":
            return _numba_batch_dot(vectors, query.astype(np.float32))
        return -(vectors @ query).astype(np.float32)

    def batch_distance(
        self, vectors: np.ndarray, query: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        """Dispatch to the appropriate distance function."""
        if metric == "cosine":
            return self.batch_cosine(vectors, query)
        if metric == "euclidean":
            return self.batch_euclidean(vectors, query)
        if metric == "dot":
            return self.batch_dot_product(vectors, query)
        raise ValueError(f"Unknown metric: {metric}")
