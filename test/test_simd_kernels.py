"""Tests for SIMD / AVX-512 Optimized Distance Kernels (Phase 5)."""

import numpy as np
import pytest

from utils.simd_kernels import (
    SIMDKernels,
    get_available_backends,
)


@pytest.fixture
def vectors():
    np.random.seed(42)
    return np.random.randn(20, 8).astype(np.float32)


@pytest.fixture
def query():
    return np.random.randn(8).astype(np.float32)


class TestSIMDKernelsBackend:
    def test_get_available_backends(self):
        backends = get_available_backends()
        assert "numpy" in backends
        assert len(backends) >= 1

    def test_default_backend_selection(self):
        kernels = SIMDKernels()
        assert kernels.backend in ["numpy", "numba", "cpp_avx512"]

    def test_preferred_backend(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        assert kernels.backend == "numpy"


class TestSIMDCosine:
    def test_cosine_distance_range(self, vectors, query):
        kernels = SIMDKernels(preferred_backend="numpy")
        dists = kernels.batch_cosine(vectors, query)
        assert dists.shape == (20,)
        assert np.all(dists >= 0.0)
        assert np.all(dists <= 2.0)

    def test_cosine_identical_vectors(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dists = kernels.batch_cosine(v, q)
        assert np.allclose(dists, [0.0], atol=1e-6)

    def test_cosine_orthogonal_vectors(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        q = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dists = kernels.batch_cosine(v, q)
        assert np.allclose(dists, [1.0], atol=1e-5)

    def test_cosine_opposite_vectors(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        q = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        dists = kernels.batch_cosine(v, q)
        assert np.allclose(dists, [2.0], atol=1e-5)

    def test_cosine_empty_vectors(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        v = np.zeros((0, 8), dtype=np.float32)
        q = np.zeros(8, dtype=np.float32)
        dists = kernels.batch_cosine(v, q)
        assert dists.shape == (0,)

    def test_cosine_zero_vector(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dists = kernels.batch_cosine(v, q)
        assert dists[0] >= 0.0  # Should handle gracefully (not crash)


class TestSIMDEuclidean:
    def test_euclidean_distance_range(self, vectors, query):
        kernels = SIMDKernels(preferred_backend="numpy")
        dists = kernels.batch_euclidean(vectors, query)
        assert dists.shape == (20,)
        assert np.all(dists >= 0.0)

    def test_euclidean_identical(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dists = kernels.batch_euclidean(v, q)
        assert np.allclose(dists, [0.0], atol=1e-6)

    def test_euclidean_same_as_numpy(self, vectors, query):
        kernels = SIMDKernels(preferred_backend="numpy")
        dists = kernels.batch_euclidean(vectors, query)
        expected = np.sqrt(np.sum((vectors - query) ** 2, axis=1))
        assert np.allclose(dists, expected, atol=1e-5)


class TestSIMDDotProduct:
    def test_dot_product(self, vectors, query):
        kernels = SIMDKernels(preferred_backend="numpy")
        dists = kernels.batch_dot_product(vectors, query)
        assert dists.shape == (20,)

    def test_dot_product_values(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        v = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        q = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        dists = kernels.batch_dot_product(v, q)
        expected = -(1*4 + 2*5 + 3*6)
        assert np.allclose(dists, [expected], atol=1e-5)


class TestSIMDBatchDistance:
    def test_batch_distance_cosine(self, vectors, query):
        kernels = SIMDKernels(preferred_backend="numpy")
        dists = kernels.batch_distance(vectors, query, metric="cosine")
        assert dists.shape == (20,)

    def test_batch_distance_euclidean(self, vectors, query):
        kernels = SIMDKernels(preferred_backend="numpy")
        dists = kernels.batch_distance(vectors, query, metric="euclidean")
        assert dists.shape == (20,)

    def test_batch_distance_dot(self, vectors, query):
        kernels = SIMDKernels(preferred_backend="numpy")
        dists = kernels.batch_distance(vectors, query, metric="dot")
        assert dists.shape == (20,)

    def test_batch_distance_invalid_metric(self):
        kernels = SIMDKernels(preferred_backend="numpy")
        with pytest.raises(ValueError, match="Unknown metric"):
            kernels.batch_distance(
                np.zeros((1, 3), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                metric="invalid",
            )

    def test_numba_cosine_same_as_numpy(self, vectors, query):
        if SIMDKernels.is_numba_available():
            numba_k = SIMDKernels(preferred_backend="numba")
            numpy_k = SIMDKernels(preferred_backend="numpy")
            n_dists = numba_k.batch_cosine(vectors, query)
            np_dists = numpy_k.batch_cosine(vectors, query)
            assert np.allclose(n_dists, np_dists, atol=1e-5)


class TestSIMDAvx512Detection:
    def test_is_avx512_available(self):
        result = SIMDKernels.is_avx512_available()
        assert isinstance(result, bool)

    def test_is_numba_available(self):
        result = SIMDKernels.is_numba_available()
        assert isinstance(result, bool)
