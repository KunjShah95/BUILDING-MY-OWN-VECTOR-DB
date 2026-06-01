"""GPU-accelerated distance computation using CUDA (optional).

Requires cupy to be installed: pip install cupy-cuda12x
"""
import numpy as np

try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False


class GPUDistanceKernel:
    """GPU-accelerated distance functions using CuPy."""

    @staticmethod
    def is_available() -> bool:
        return HAS_CUDA

    @staticmethod
    def cosine_distance_gpu(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine distances between query and all vectors on GPU.

        Args:
            query: 1-D query vector of shape (D,).
            vectors: 2-D array of shape (N, D).

        Returns:
            1-D array of cosine distances of shape (N,).
        """
        if not HAS_CUDA:
            raise ImportError("cupy is not available. Install with: pip install cupy-cuda12x")
        query_gpu = cp.asarray(query)
        vectors_gpu = cp.asarray(vectors)
        dots = cp.dot(vectors_gpu, query_gpu)
        vec_norms = cp.linalg.norm(vectors_gpu, axis=1)
        query_norm = cp.linalg.norm(query_gpu)
        denominator = vec_norms * query_norm
        denominator = cp.maximum(denominator, 1e-10)
        similarity = dots / denominator
        return cp.asnumpy(1.0 - similarity)

    @staticmethod
    def euclidean_distance_gpu(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute Euclidean distances between query and all vectors on GPU.

        Args:
            query: 1-D query vector of shape (D,).
            vectors: 2-D array of shape (N, D).

        Returns:
            1-D array of Euclidean distances of shape (N,).
        """
        if not HAS_CUDA:
            raise ImportError("cupy is not available. Install with: pip install cupy-cuda12x")
        query_gpu = cp.asarray(query)
        vectors_gpu = cp.asarray(vectors)
        diff = vectors_gpu - query_gpu
        sq = cp.sum(diff ** 2, axis=1)
        return cp.asnumpy(cp.sqrt(sq))

    @staticmethod
    def inner_product_gpu(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute inner product similarity on GPU.

        Args:
            query: 1-D query vector of shape (D,).
            vectors: 2-D array of shape (N, D).

        Returns:
            1-D array of inner products of shape (N,).
        """
        if not HAS_CUDA:
            raise ImportError("cupy is not available. Install with: pip install cupy-cuda12x")
        query_gpu = cp.asarray(query)
        vectors_gpu = cp.asarray(vectors)
        return cp.asnumpy(cp.dot(vectors_gpu, query_gpu))

    @staticmethod
    def batch_cosine_distance_gpu(queries: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine distances between two sets on GPU.

        Args:
            queries: 2-D array of shape (M, D).
            vectors: 2-D array of shape (N, D).

        Returns:
            2-D array of shape (M, N).
        """
        if not HAS_CUDA:
            raise ImportError("cupy is not available. Install with: pip install cupy-cuda12x")
        q_gpu = cp.asarray(queries)
        v_gpu = cp.asarray(vectors)
        dots = cp.dot(q_gpu, v_gpu.T)
        q_norms = cp.linalg.norm(q_gpu, axis=1, keepdims=True)
        v_norms = cp.linalg.norm(v_gpu, axis=1, keepdims=True).T
        denominator = cp.maximum(q_norms * v_norms, 1e-10)
        return cp.asnumpy(1.0 - dots / denominator)
