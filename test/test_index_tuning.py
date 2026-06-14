"""
Unit tests for utils/index_tuning.py — DatasetAnalyzer & HNSWParameterRecommender.

Covers:
  - DatasetAnalyzer.analyze on various vector distributions
  - HNSWParameterRecommender.recommend with different stats
  - Edge cases: tiny datasets, uniform data, high-dimensional, normalized vs unnormalized
  - Convenience function tune_hnsw
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from utils.index_tuning import (
    DatasetAnalyzer,
    DatasetStatistics,
    HNSWParameterRecommender,
    HNSWParameterRecommendation,
    tune_hnsw,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def low_dim_gaussian() -> np.ndarray:
    """1,000 vectors, 8D — simple low-dimensional Gaussian."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1000, 8)).astype(np.float32)


@pytest.fixture
def high_dim_uniform() -> np.ndarray:
    """500 vectors, 128D uniform on hypercube — high-D, sparse."""
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, (500, 128)).astype(np.float32)


@pytest.fixture
def normalized_clusters() -> np.ndarray:
    """600 vectors, 64D, 3 well-separated L2-normalised clusters."""
    rng = np.random.default_rng(42)
    centroids = np.array([[5.0] * 64, [-3.0] * 64, [0.0] * 64], dtype=np.float32)
    vecs = []
    for c in centroids:
        cluster_vecs = c + rng.standard_normal((200, 64)).astype(np.float32) * 0.3
        # L2-normalise
        norms = np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
        cluster_vecs = cluster_vecs / norms
        vecs.append(cluster_vecs)
    return np.vstack(vecs)


@pytest.fixture
def tiny_dataset() -> np.ndarray:
    """5 vectors, 4D — edge case for small datasets."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((5, 4)).astype(np.float32)


@pytest.fixture
def uniform_sphere() -> np.ndarray:
    """300 vectors, 32D uniformly distributed on the unit sphere (L2 normalised)."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((300, 32)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# DatasetAnalyzer tests
# ---------------------------------------------------------------------------

class TestDatasetAnalyzer:
    def test_low_dim_gaussian_basic_stats(self, low_dim_gaussian):
        """Check dimension, count, and basic variance."""
        analyzer = DatasetAnalyzer(sample_size=500)
        stats = analyzer.analyze(low_dim_gaussian)

        assert stats.num_vectors == 1000
        assert stats.dimension == 8
        assert stats.dim_variances.shape == (8,)
        assert stats.dim_means.shape == (8,)
        assert stats.mean_variance > 0.5  # std normal → variance ~1.0
        assert stats.is_normalized is False  # not L2-normalised

    def test_high_dim_uniform(self, high_dim_uniform):
        """High-dimensional uniform should have higher intrinsic dim."""
        analyzer = DatasetAnalyzer(sample_size=300)
        stats = analyzer.analyze(high_dim_uniform)

        assert stats.dimension == 128
        assert stats.is_high_dimensional is True
        # Uniform data on hypercube → high intrinsic dimensionality
        assert stats.intrinsic_dim_twonn > 10
        assert stats.intrinsic_dim_pca > 10

    def test_normalized_clusters(self, normalized_clusters):
        """Normalised cluster data: should detect clusters + normalised flag."""
        analyzer = DatasetAnalyzer(sample_size=500)
        stats = analyzer.analyze(normalized_clusters)

        assert stats.is_normalized is True
        assert abs(stats.norm_mean - 1.0) < 0.02
        # 3 well-separated clusters → high separation ratio
        assert stats.cluster_separation_ratio > 2.0
        assert stats.estimated_cluster_count >= 1

    def test_uniform_sphere_is_normalized(self, uniform_sphere):
        """Uniform on sphere should be detected as normalised."""
        analyzer = DatasetAnalyzer(sample_size=200)
        stats = analyzer.analyze(uniform_sphere)

        assert stats.is_normalized is True
        assert abs(stats.norm_mean - 1.0) < 0.02
        assert stats.norm_std < 0.02

    def test_tiny_dataset(self, tiny_dataset):
        """Tiny dataset should not crash and return sensible defaults."""
        analyzer = DatasetAnalyzer(sample_size=-1)  # use all
        stats = analyzer.analyze(tiny_dataset)

        assert stats.num_vectors == 5
        assert stats.dimension == 4
        # Should still produce reasonable intrinsic dim estimates
        assert stats.intrinsic_dim_twonn > 0
        assert stats.intrinsic_dim_pca > 0

    def test_empty_dataset_raises(self):
        """Empty dataset should raise an error."""
        analyzer = DatasetAnalyzer()
        with pytest.raises((ValueError, IndexError)):
            analyzer.analyze(np.empty((0, 10)))

    def test_sample_size_larger_than_dataset(self, low_dim_gaussian):
        """sample_size > N should use all vectors without error."""
        analyzer = DatasetAnalyzer(sample_size=9999)
        stats = analyzer.analyze(low_dim_gaussian)
        assert stats.num_vectors == 1000

    def test_variance_coefficient_low_on_uniform(self):
        """Uniform data should have low variance CV (dims have similar spread)."""
        rng = np.random.default_rng(42)
        # Uniform: all dims have same variance = (b-a)²/12
        data = rng.uniform(-1, 1, (500, 16)).astype(np.float32)
        analyzer = DatasetAnalyzer(sample_size=300)
        stats = analyzer.analyze(data)

        assert stats.variance_coefficient < 0.3  # uniform dims → low CV

    def test_variance_coefficient_high_on_skewed(self):
        """Data with mixed-variance dims should have higher CV."""
        rng = np.random.default_rng(42)
        # First 4 dims high variance, rest low
        data = np.column_stack([
            rng.standard_normal(500) * 10,
            rng.standard_normal(500) * 10,
            rng.standard_normal(500) * 5,
            rng.standard_normal(500) * 5,
            rng.standard_normal(500) * 0.1,
            rng.standard_normal(500) * 0.1,
        ]).astype(np.float32)
        analyzer = DatasetAnalyzer(sample_size=300)
        stats = analyzer.analyze(data)

        assert stats.variance_coefficient > 0.5  # mixed → higher CV


# ---------------------------------------------------------------------------
# HNSWParameterRecommender tests
# ---------------------------------------------------------------------------

class TestHNSWParameterRecommender:
    def test_recommend_for_low_dim(self, low_dim_gaussian):
        """Low-dim Gaussian should get moderate M, moderate ef_construction."""
        analyzer = DatasetAnalyzer(sample_size=500)
        stats = analyzer.analyze(low_dim_gaussian)
        recommender = HNSWParameterRecommender(recall_target=0.95)
        rec = recommender.recommend(stats)

        assert isinstance(rec, HNSWParameterRecommendation)
        assert 4 <= rec.m <= 48
        assert rec.m0 == rec.m * 2
        assert 50 <= rec.ef_construction <= 600
        assert 10 <= rec.ef_search <= 300
        assert rec.expected_recall > 0.0
        assert len(rec.reasoning) >= 2
        assert rec.confidence in ("high", "medium", "low")

    def test_recommend_for_high_dim(self, high_dim_uniform):
        """High-dim should get higher M and ef_construction."""
        analyzer = DatasetAnalyzer(sample_size=300)
        stats = analyzer.analyze(high_dim_uniform)
        recommender = HNSWParameterRecommender(recall_target=0.95)
        rec = recommender.recommend(stats)

        assert rec.m >= 8
        assert rec.ef_construction >= 100

    def test_recommend_for_normalized_clusters(self, normalized_clusters):
        """Clustered normalised data — should produce reasonable parameters."""
        analyzer = DatasetAnalyzer(sample_size=500)
        stats = analyzer.analyze(normalized_clusters)
        # Confirm the analyzer correctly detected normalization
        assert stats.is_normalized is True

        recommender = HNSWParameterRecommender(recall_target=0.95)
        rec = recommender.recommend(stats)

        # Reasonable parameters for this data
        assert rec.m >= 8
        assert rec.ef_construction >= 100
        assert rec.expected_recall > 0.5

    def test_recall_target_affects_ef(self, low_dim_gaussian):
        """Higher recall target should produce higher ef_search and ef_construction."""
        analyzer = DatasetAnalyzer(sample_size=500)
        stats = analyzer.analyze(low_dim_gaussian)

        rec_low = HNSWParameterRecommender(recall_target=0.90).recommend(stats)
        rec_high = HNSWParameterRecommender(recall_target=0.99).recommend(stats)

        assert rec_high.ef_search >= rec_low.ef_search
        assert rec_high.ef_construction >= rec_low.ef_construction

    def test_tiny_dataset_recommendation(self, tiny_dataset):
        """Tiny datasets should get low confidence and conservative parameters."""
        analyzer = DatasetAnalyzer(sample_size=-1)
        stats = analyzer.analyze(tiny_dataset)
        rec = HNSWParameterRecommender(recall_target=0.95).recommend(stats)

        assert rec.confidence == "low"
        assert rec.m >= 4
        assert rec.expected_recall > 0.0

    def test_to_dict_serializable(self, low_dim_gaussian):
        """to_dict() should produce JSON-serializable output."""
        analyzer = DatasetAnalyzer(sample_size=500)
        stats = analyzer.analyze(low_dim_gaussian)
        rec = HNSWParameterRecommender(recall_target=0.95).recommend(stats)

        d = rec.to_dict()
        assert "hnsw" in d
        assert "m" in d["hnsw"]
        assert "ef_construction" in d["hnsw"]
        assert "ef_search" in d["hnsw"]
        assert "expected_recall" in d
        assert "confidence" in d
        assert "reasoning" in d
        assert isinstance(d["reasoning"], list)

    def test_stats_summary_serializable(self, low_dim_gaussian):
        """DatasetStatistics.summary should be JSON-serializable."""
        analyzer = DatasetAnalyzer(sample_size=500)
        stats = analyzer.analyze(low_dim_gaussian)

        s = stats.summary
        assert s["num_vectors"] == 1000
        assert s["dimension"] == 8
        assert isinstance(s["intrinsic_dim_twonn"], float)
        assert isinstance(s["is_normalized"], bool)
        assert isinstance(s["is_high_dimensional"], bool)


# ---------------------------------------------------------------------------
# Convenience function tests
# ---------------------------------------------------------------------------

class TestTuneHNSW:
    def test_tune_hnsw_returns_recommendation(self, low_dim_gaussian):
        """tune_hnsw convenience function should return a valid recommendation."""
        rec = tune_hnsw(low_dim_gaussian, recall_target=0.95)

        assert isinstance(rec, HNSWParameterRecommendation)
        assert rec.m > 0
        assert rec.ef_construction > 0
        assert rec.ef_search > 0

    def test_tune_hnsw_different_targets(self, high_dim_uniform):
        """tune_hnsw should respect different recall targets."""
        rec1 = tune_hnsw(high_dim_uniform, recall_target=0.90)
        rec2 = tune_hnsw(high_dim_uniform, recall_target=0.99)

        assert rec2.ef_search >= rec1.ef_search
        assert rec2.ef_construction >= rec1.ef_construction

    def test_tune_hnsw_invalid_target_clamped(self, low_dim_gaussian):
        """Extreme recall targets should be clamped.""" 
        rec_low = tune_hnsw(low_dim_gaussian, recall_target=0.0)
        rec_high = tune_hnsw(low_dim_gaussian, recall_target=1.5)

        # Both should produce reasonable results without crashing
        assert rec_low.ef_search > 0
        assert rec_high.ef_search > 0
        # The 1.5 target should be clamped to 0.999, giving high ef
        assert rec_high.ef_search >= rec_low.ef_search


# ---------------------------------------------------------------------------
# Integration property: recommendations form a smooth frontier
# ---------------------------------------------------------------------------

class TestSmoothFrontier:
    """As dataset difficulty increases, recommendations should be monotonic."""

    def test_more_vectors_increases_ef(self):
        """Larger synthetic datasets should get higher ef_construction."""
        rng = np.random.default_rng(42)

        prev_ef = 0
        for n in [500, 5_000, 50_000]:
            data = rng.standard_normal((n, 32)).astype(np.float32)
            rec = tune_hnsw(data, recall_target=0.95, sample_size=1000)
            assert rec.ef_construction >= prev_ef
            prev_ef = rec.ef_construction
