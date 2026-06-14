"""
Tests for the auto-tune integration — VectorIndexerConfig.tune() and
VectorIndexer.create_index with auto_tune=True.

Covers:
  - tune() updates hnsw_params with data-driven values
  - auto_tune=True in create_index triggers tune() automatically
  - auto_tune=False preserves existing default behaviour
  - Edge cases: tiny datasets, random data, empty vectors
"""

import numpy as np
import pytest

from services.vector_indexer import VectorIndexerConfig, VectorIndexer, IndexMethod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_vectors() -> np.ndarray:
    """500 random 64D vectors — representative real-world scale."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 64)).astype(np.float32)


@pytest.fixture
def clustered_vectors() -> np.ndarray:
    """400 vectors, 32D, with 3 natural clusters — structured data."""
    rng = np.random.default_rng(42)
    centroids = np.array([[3.0] * 32, [-2.0] * 32, [0.5] * 32], dtype=np.float32)
    vecs = []
    for c in centroids:
        cluster = c + rng.standard_normal((133, 32)).astype(np.float32) * 0.5
        vecs.append(cluster)
    return np.vstack(vecs)


@pytest.fixture
def tiny_vectors() -> np.ndarray:
    """10 vectors, 8D — tiny edge case."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((10, 8)).astype(np.float32)


# ---------------------------------------------------------------------------
# VectorIndexerConfig.tune()
# ---------------------------------------------------------------------------

class TestTune:
    def test_tune_updates_hnsw_params(self, random_vectors):
        """tune() should replace hnsw_params with data-driven values."""
        config = VectorIndexerConfig()
        old_params = dict(config.hnsw_params)  # copy

        config.tune(random_vectors)

        # Params should have changed
        assert config.hnsw_params != old_params
        assert config.hnsw_params["m"] > 0
        assert config.hnsw_params["m0"] == config.hnsw_params["m"] * 2
        assert 50 <= config.hnsw_params["ef_construction"] <= 800
        assert config.hnsw_params["ef_search"] > 0

    def test_tune_with_clustered_data(self, clustered_vectors):
        """Clustered data should produce higher ef_construction."""
        config = VectorIndexerConfig(recall_target=0.95)
        config.tune(clustered_vectors)

        # Clustered data gets a boost from cluster_separation_ratio
        assert config.hnsw_params["m"] >= 4
        assert config.hnsw_params["ef_construction"] >= 100

    def test_tune_with_high_recall_target(self, random_vectors):
        """Higher recall target should give higher ef values."""
        v1 = VectorIndexerConfig(recall_target=0.90)
        v2 = VectorIndexerConfig(recall_target=0.99)
        v1.tune(random_vectors)
        v2.tune(random_vectors)

        assert v2.hnsw_params["ef_search"] >= v1.hnsw_params["ef_search"]
        assert v2.hnsw_params["ef_construction"] >= v1.hnsw_params["ef_construction"]

    def test_tune_updates_num_vectors_and_dim(self, random_vectors):
        """tune() should update num_vectors and vector_dim from data."""
        config = VectorIndexerConfig(num_vectors=999, vector_dim=999)
        config.tune(random_vectors)

        assert config.num_vectors == 500
        assert config.vector_dim == 64

    def test_tune_preserves_other_params(self, random_vectors):
        """tune() should not affect method, speed_priority, recall_target."""
        config = VectorIndexerConfig(method="ivf", speed_priority=True, recall_target=0.9)
        config.tune(random_vectors)

        assert config.method == IndexMethod.IVF
        assert config.speed_priority is True
        assert config.recall_target == 0.9


# ---------------------------------------------------------------------------
# VectorIndexer.create_index with auto_tune=True
# ---------------------------------------------------------------------------

class TestAutoTune:
    def test_auto_tune_true_changes_params(self, random_vectors):
        """create_index with auto_tune=True should use data-driven params."""
        config = VectorIndexerConfig(method="hnsw", auto_tune=True)
        indexer = VectorIndexer(config)
        # The config has old defaults before create_index
        params_before = dict(config.hnsw_params)

        result = indexer.create_index(random_vectors.tolist())

        assert result["success"] is True
        # Config should have been updated by the auto-tune inside create_index
        assert config.hnsw_params != params_before
        assert result["config"]["hnsw_params"]["m"] == config.hnsw_params["m"]

    def test_auto_tune_false_keeps_defaults(self, random_vectors):
        """create_index with auto_tune=False (default) should keep settings defaults."""
        config = VectorIndexerConfig(method="hnsw", auto_tune=False)
        params_before = dict(config.hnsw_params)

        indexer = VectorIndexer(config)
        result = indexer.create_index(random_vectors.tolist())

        assert result["success"] is True
        # Params should be unchanged (no auto-tune applied)
        assert config.hnsw_params == params_before

    def test_auto_tune_in_result_config(self, random_vectors):
        """create_index result should include auto_tune in config dict."""
        config = VectorIndexerConfig(method="hnsw", auto_tune=True)
        indexer = VectorIndexer(config)
        result = indexer.create_index(random_vectors.tolist())

        assert result["config"]["auto_tune"] is True

    def test_auto_tune_with_tiny_data(self, tiny_vectors):
        """Tiny dataset should not crash auto-tune."""
        config = VectorIndexerConfig(method="hnsw", auto_tune=True)
        indexer = VectorIndexer(config)
        result = indexer.create_index(tiny_vectors.tolist())

        assert result["success"] is True
        assert config.hnsw_params["m"] > 0

    def test_auto_tune_empty_vectors_skips(self):
        """Empty vector list should skip auto-tune and not crash."""
        config = VectorIndexerConfig(method="hnsw", auto_tune=True)
        indexer = VectorIndexer(config)
        params_before = dict(config.hnsw_params)

        result = indexer.create_index([])

        # create_index with no vectors creates an empty index (success=True)
        # but auto-tune should NOT have been attempted
        assert result["success"] is True
        assert config.hnsw_params == params_before


# ---------------------------------------------------------------------------
# Recommendation metadata
# ---------------------------------------------------------------------------

class TestRecommendationMetadata:
    def test_tune_stores_recommendation(self, random_vectors):
        """tune() should store the full HNSWParameterRecommendation."""
        config = VectorIndexerConfig()
        assert config.recommendation is None

        config.tune(random_vectors)

        assert config.recommendation is not None
        assert config.recommendation.m > 0
        assert config.recommendation.ef_construction > 0
        assert config.recommendation.ef_search > 0
        assert config.recommendation.confidence in ("high", "medium", "low")
        assert len(config.recommendation.reasoning) >= 1
        assert config.recommendation.expected_recall > 0.0

    def test_tune_stores_dataset_stats(self, random_vectors):
        """tune() should store DatasetStatistics."""
        config = VectorIndexerConfig()
        assert config.dataset_stats is None

        config.tune(random_vectors)

        assert config.dataset_stats is not None
        assert config.dataset_stats.num_vectors == 500
        assert config.dataset_stats.dimension == 64
        assert config.dataset_stats.is_high_dimensional is True

    def test_to_dict_includes_recommendation_after_tune(self, random_vectors):
        """to_dict should include recommendation and dataset_stats after tune()."""
        config = VectorIndexerConfig()
        config.tune(random_vectors)
        d = config.to_dict()

        assert "recommendation" in d
        assert "hnsw" in d["recommendation"]
        assert "m" in d["recommendation"]["hnsw"]
        assert "expected_recall" in d["recommendation"]
        assert "confidence" in d["recommendation"]
        assert "reasoning" in d["recommendation"]
        assert "dataset_stats" in d
        assert "num_vectors" in d["dataset_stats"]
        assert "intrinsic_dim_pca" in d["dataset_stats"]

    def test_to_dict_without_tune_excludes_recommendation(self):
        """to_dict should not include recommendation when tune() not called."""
        config = VectorIndexerConfig()
        d = config.to_dict()

        assert "recommendation" not in d
        assert "dataset_stats" not in d


# ---------------------------------------------------------------------------
# Default behaviour
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_auto_tune_defaults_to_true(self):
        """auto_tune should default to True so new indexes get data-driven defaults."""
        config = VectorIndexerConfig()
        assert config.auto_tune is True

    def test_explicit_auto_tune_false_disables_tuning(self, random_vectors):
        """Setting auto_tune=False should skip data-driven tuning."""
        config = VectorIndexerConfig(method="hnsw", auto_tune=False)
        params_before = dict(config.hnsw_params)

        indexer = VectorIndexer(config)
        result = indexer.create_index(random_vectors.tolist())

        assert result["success"] is True
        # Params should be unchanged (no auto-tune applied)
        assert config.hnsw_params == params_before
        assert config.recommendation is None
        assert config.dataset_stats is None

    def test_to_dict_includes_auto_tune(self):
        """to_dict should include auto_tune field."""
        config = VectorIndexerConfig(auto_tune=True)
        d = config.to_dict()
        assert d["auto_tune"] is True

        config2 = VectorIndexerConfig(auto_tune=False)
        d2 = config2.to_dict()
        assert d2["auto_tune"] is False

    def test_config_with_explicit_params_still_works(self):
        """Explicit constructor parameters should still work as before."""
        config = VectorIndexerConfig(method="hnsw", speed_priority=True)
        assert config.method == IndexMethod.HNSW
        assert config.speed_priority is True
        assert config.num_vectors == 10000
        assert config.vector_dim == 128
