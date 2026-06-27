"""Tests for Phase 15 Vector-Native AI Features.

Covers: LTRTrainer, RLHFService, VectorExplainer, FederatedEmbeddingService.
Runs with pytest; no external deps required (numpy only).
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_features(n: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    names = ["bm25", "distance", "freshness", "popularity"][:n]
    return {k: float(rng.uniform(0, 1)) for k in names}


# ---------------------------------------------------------------------------
# LTR Trainer tests
# ---------------------------------------------------------------------------

class TestLTRTrainer:
    def setup_method(self):
        self.tmp = tempfile.mktemp(suffix=".jsonl")
        from services.ltr_trainer import LTRTrainer
        self.trainer = LTRTrainer(data_file=self.tmp)

    def teardown_method(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def _seed_data(self, n=20):
        rng = np.random.default_rng(99)
        for i in range(n):
            self.trainer.collect_training_pair(
                query_id=f"q{i % 5}",
                doc_id=f"doc{i}",
                label=float(rng.integers(0, 4)),
                features=make_features(seed=i),
            )

    def test_collect_and_load(self):
        self.trainer.collect_training_pair("q1", "d1", 2.0, {"bm25": 0.5})
        pairs = self.trainer.load_training_data()
        assert len(pairs) == 1
        assert pairs[0].query_id == "q1"
        assert pairs[0].label == 2.0

    def test_train_pointwise_linear(self):
        self._seed_data(20)
        model = self.trainer.train_pointwise("linear")
        assert model.model_type == "linear"
        assert len(model.weights) == 4  # 4 features

    def test_train_pointwise_mlp(self):
        self._seed_data(20)
        model = self.trainer.train_pointwise("mlp")
        assert model.model_type == "mlp"
        assert "W1" in model.weights

    def test_train_pairwise(self):
        self._seed_data(20)
        model = self.trainer.train_pairwise()
        assert model.model_type in ("pairwise", "linear")

    def test_evaluate_ndcg_mrr(self):
        self._seed_data(20)
        self.trainer.train_pointwise("linear")
        test_pairs = self.trainer.load_training_data()[:10]
        metrics = self.trainer.evaluate(test_pairs)
        assert "ndcg_at_10" in metrics
        assert "mrr" in metrics
        assert 0.0 <= metrics["ndcg_at_10"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0

    def test_export_load_model(self):
        self._seed_data(10)
        self.trainer.train_pointwise("linear")
        export_path = tempfile.mktemp(suffix=".json")
        try:
            self.trainer.export_model(export_path)
            from services.ltr_trainer import LTRTrainer
            t2 = LTRTrainer(data_file=self.tmp)
            m = t2.load_model(export_path)
            assert m.model_type == "linear"
        finally:
            if os.path.exists(export_path):
                os.remove(export_path)

    def test_stats(self):
        self._seed_data(5)
        stats = self.trainer.stats()
        assert stats["training_data_size"] == 5
        assert stats["model_trained"] is False


# ---------------------------------------------------------------------------
# RLHF Service tests
# ---------------------------------------------------------------------------

class TestRLHFService:
    def setup_method(self):
        self.tmp = tempfile.mktemp(suffix=".jsonl")
        from services.rlhf_service import RLHFService
        self.svc = RLHFService(signal_file=self.tmp)

    def teardown_method(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_record_click(self):
        sig = self.svc.record_signal("q1", "r1", "click")
        assert sig.implicit_relevance == 1.0

    def test_record_skip(self):
        sig = self.svc.record_signal("q1", "r2", "skip")
        assert sig.implicit_relevance == 0.0

    def test_record_dwell(self):
        sig = self.svc.record_signal("q1", "r3", "dwell", dwell_ms=5000)
        assert 0.0 < sig.implicit_relevance < 1.0

    def test_train_reward_model(self):
        # need preference pairs: one click + one skip for same query
        self.svc.record_signal("q1", "r1", "click")
        self.svc.record_signal("q1", "r2", "skip")
        model = self.svc.train_reward_model()
        assert 0.0 <= model.accuracy <= 1.0
        assert model.n_pairs > 0

    def test_predict_relevance(self):
        qv = [0.1, 0.9, 0.3]
        dv = [0.2, 0.8, 0.4]
        score = self.svc.predict_relevance(qv, dv)
        assert 0.0 < score < 1.0

    def test_stats(self):
        self.svc.record_signal("q1", "r1", "click")
        stats = self.svc.stats()
        assert stats["total_signals"] == 1
        assert stats["signal_counts"]["click"] == 1


# ---------------------------------------------------------------------------
# Vector Explainer tests
# ---------------------------------------------------------------------------

class TestVectorExplainer:
    def setup_method(self):
        from services.vector_explainer import VectorExplainer
        self.exp = VectorExplainer()

    def test_explain_basic(self):
        qv = [1.0, 0.0, 0.5, 0.0]
        rv = [0.8, 0.2, 0.6, 0.1]
        exp = self.exp.explain(qv, rv, metadata={"title": "Test doc"}, top_k_dims=3)
        assert 0.0 <= exp.semantic_similarity <= 1.0
        assert len(exp.matched_dimensions) == 3
        assert "Test doc" in exp.metadata_highlights.get("title", "")
        assert len(exp.natural_language_summary) > 10

    def test_explain_returns_top_k(self):
        qv = list(range(20))
        rv = list(range(20, 40))
        exp = self.exp.explain(qv, rv, top_k_dims=5)
        assert len(exp.matched_dimensions) == 5

    def test_explain_batch(self):
        qv = [1.0, 0.0, 0.5]
        results = [
            {"vector": [0.9, 0.1, 0.4], "vector_id": "d1"},
            {"vector": [0.1, 0.9, 0.2], "vector_id": "d2"},
        ]
        exps = self.exp.explain_batch(qv, results, top_k_dims=3)
        assert len(exps) == 2
        assert exps[0].result_id == "d1"

    def test_explain_zero_vector(self):
        exp = self.exp.explain([0.0, 0.0], [1.0, 0.0])
        assert exp.semantic_similarity == 0.0

    def test_dimension_contribution_sign(self):
        qv = [1.0, -1.0, 0.0]
        rv = [1.0,  1.0, 0.0]
        exp = self.exp.explain(qv, rv, top_k_dims=3)
        dims = {d["dim"]: d for d in exp.matched_dimensions}
        # dim 0: 1*1 = 1 (positive), dim 1: -1*1 = -1 (negative)
        assert dims[0]["direction"] == "positive"
        assert dims[1]["direction"] == "negative"


# ---------------------------------------------------------------------------
# Federated Embedding tests
# ---------------------------------------------------------------------------

class TestFederatedEmbedding:
    def setup_method(self):
        from services.federated_embedding import FederatedEmbeddingService
        self.svc = FederatedEmbeddingService()

    def test_register_client(self):
        result = self.svc.register_client("c1", "v1.0.0", {"n_samples": 100})
        assert result["success"] is True
        assert result["client_id"] == "c1"

    def test_compute_gradient_update(self):
        self.svc.register_client("c1", "v1.0.0", {})
        embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        labels = [0, 1]
        result = self.svc.compute_gradient_update("c1", embs, labels)
        assert result["success"] is True
        assert "delta_weights" in result
        assert len(result["delta_weights"]) == 3

    def test_aggregate_updates(self):
        self.svc.register_client("c1", "v1.0.0", {})
        self.svc.register_client("c2", "v1.0.0", {})
        self.svc.compute_gradient_update("c1", [[0.1, 0.2], [0.3, 0.4]], [0, 1])
        self.svc.compute_gradient_update("c2", [[0.5, 0.6], [0.7, 0.8]], [0, 1])
        result = self.svc.aggregate_updates()
        assert result["success"] is True
        assert result["rounds_completed"] == 1
        assert result["n_clients"] == 2

    def test_global_version_bumps(self):
        v0 = self.svc.get_global_model_version()
        self.svc.compute_gradient_update("c1", [[0.1, 0.2]], [0])
        self.svc.aggregate_updates()
        v1 = self.svc.get_global_model_version()
        assert v1 != v0

    def test_status(self):
        self.svc.register_client("cx", "v1.0.0", {"foo": "bar"})
        s = self.svc.status()
        assert s["connected_clients"] == 1
        assert any(c["client_id"] == "cx" for c in s["clients"])
