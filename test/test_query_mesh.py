"""Tests for Phase 14 — Intelligent Query Mesh."""
import math
import time
import pytest

# ── 1. QueryCostPredictor ─────────────────────────────────────────────────────

from services.query_cost_predictor import QueryCostPredictor


def make_predictor():
    p = QueryCostPredictor()
    p.set_collection_size("col1", 10_000)
    return p


def test_cost_estimate_brute_scan_equals_N():
    p = make_predictor()
    est = p.estimate([0.0] * 8, k=10, collection_id="col1", index_type="brute")
    assert est.estimated_scan_count == 10_000


def test_cost_estimate_hnsw_less_than_brute():
    p = make_predictor()
    hnsw = p.estimate([0.0] * 8, k=10, collection_id="col1", index_type="hnsw")
    brute = p.estimate([0.0] * 8, k=10, collection_id="col1", index_type="brute")
    assert hnsw.estimated_latency_ms < brute.estimated_latency_ms


def test_cost_estimate_complexity_score_range():
    p = make_predictor()
    est = p.estimate([0.0] * 8, k=10, collection_id="col1", index_type="hnsw")
    assert 0.0 <= est.complexity_score <= 1.0


def test_cost_estimate_filters_reduce_cost():
    p = make_predictor()
    no_filter = p.estimate([0.0] * 8, k=10, collection_id="col1", index_type="brute")
    with_filter = p.estimate([0.0] * 8, k=10, collection_id="col1", index_type="brute",
                             filters={"category": "tech"})
    assert with_filter.estimated_latency_ms < no_filter.estimated_latency_ms


def test_record_actual_and_calibrate():
    p = make_predictor()
    est = p.estimate([0.0] * 8, k=10, collection_id="col1", index_type="hnsw")
    p.record_actual(est.query_id, actual_latency_ms=5.0, actual_scan_count=700)
    result = p.calibrate()
    assert result["status"] in ("ok", "no_data")


# ── 2. QueryScheduler ─────────────────────────────────────────────────────────

from services.query_scheduler import QueryScheduler, TenantCreditPool


def test_token_bucket_consume_and_refill():
    pool = TenantCreditPool(tenant_id="t1", credits_per_second=10.0, max_burst=20.0)
    assert pool.consume(10.0) is True
    assert pool.consume(15.0) is False  # not enough credits
    pool.refill()
    # After one refill (0.1 s worth = 1.0 credit), still not enough for 15
    assert pool.current_credits < 15.0


def test_scheduler_accepts_valid_tenant():
    sched = QueryScheduler(max_workers=2)
    sched.register_tenant("tenant_a", credits_per_second=100.0, max_burst=1000.0)
    future = sched.schedule("tenant_a", lambda: 42, cost_estimate=1.0)
    result = future.result(timeout=2)
    assert result == 42
    sched.shutdown()


def test_scheduler_rejects_over_budget():
    sched = QueryScheduler(max_workers=2)
    sched.register_tenant("tenant_b", credits_per_second=1.0, max_burst=5.0)
    # Drain credits
    for _ in range(5):
        sched.schedule("tenant_b", lambda: None, cost_estimate=1.0)
    # This one should be rejected
    future = sched.schedule("tenant_b", lambda: None, cost_estimate=10.0)
    with pytest.raises(RuntimeError):
        future.result(timeout=2)
    sched.shutdown()


# ── 3. FusionTelemetry ────────────────────────────────────────────────────────

from services.fusion_telemetry import FusionTelemetry


def _make_results(ids, base_score=1.0):
    return [{"vector_id": f"v{i}", "score": base_score - i * 0.01} for i in ids]


def test_rrf_fusion_deduplicates():
    ft = FusionTelemetry()
    r1 = _make_results(range(5))
    r2 = _make_results(range(3, 8))
    fused = ft.fuse({"hnsw": r1, "ivf": r2}, k=10)
    ids = [r["vector_id"] for r in fused]
    assert len(ids) == len(set(ids))  # no duplicates


def test_rrf_fusion_returns_k_or_fewer():
    ft = FusionTelemetry()
    r1 = _make_results(range(20))
    fused = ft.fuse({"hnsw": r1}, k=5)
    assert len(fused) <= 5


def test_telemetry_records_after_fuse():
    ft = FusionTelemetry()
    ft.record_index_latency("hnsw", 3.5, 10)
    ft.record_index_latency("ivf", 1.2, 10)
    tel = ft.get_telemetry(window_seconds=60)
    assert "hnsw" in tel["per_index"]
    assert "ivf" in tel["per_index"]


def test_recommend_index_returns_string():
    ft = FusionTelemetry()
    ft.record_index_latency("hnsw", 5.0, 10)
    ft.record_index_latency("ivf", 2.0, 10)
    rec = ft.recommend_index({})
    assert isinstance(rec, str)
    assert rec == "ivf"  # ivf has lower p50


# ── 4. ViewRecommender ────────────────────────────────────────────────────────

from services.view_recommender import ViewRecommender, _cosine_similarity


def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-6


def test_view_recommender_clusters_repeated_queries():
    vr = ViewRecommender(mv_service=None)  # will skip auto-create
    vec = [1.0, 0.0, 0.0]
    for _ in range(15):
        vr.record_query(vec, k=10, filters=None, collection_id="col1")
    recs = vr.analyze(min_frequency=10)
    assert len(recs) >= 1
    assert recs[0].query_count >= 10


def test_view_recommender_ignores_rare_queries():
    vr = ViewRecommender(mv_service=None)
    for i in range(5):
        vr.record_query([float(i), 1.0, 0.0], k=10, filters=None, collection_id="col1")
    recs = vr.analyze(min_frequency=10)
    assert recs == []
