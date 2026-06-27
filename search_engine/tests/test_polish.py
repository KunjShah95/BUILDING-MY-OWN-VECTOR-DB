"""Tests for query intel, eval harness, and freshness recrawl (Polish layer)."""

from search_engine.query import normalize, expand_query, multi_query, rewrite
from search_engine.eval import ndcg_at_k, mrr, recall_at_k, precision_at_k, evaluate
from search_engine.recrawl import FreshnessTracker


# ---- Query intel -----------------------------------------------------------

def test_normalize_collapses_ws():
    assert normalize("  build   my   vector  db ") == "build my vector db"


def test_expand_query_adds_abbrev_and_synonyms():
    out = expand_query("fast vector db")
    assert out.startswith("fast vector db")          # original preserved first
    assert "database" in out                          # db -> database
    assert "embedding" in out                         # vector synonym
    assert "quick" in out or "low latency" in out     # fast synonym


def test_rewrite_strips_noise():
    assert rewrite("please show me the hnsw index") == "the hnsw index"


def test_multi_query_distinct_variants():
    v = multi_query("build ann db", n=3)
    assert v[0] == "build ann db"
    assert len(v) == len(set(v))                      # no dup variants
    assert len(v) <= 3


# ---- Eval metrics ----------------------------------------------------------

def test_metrics_perfect_ranking():
    retrieved = ["d1", "d2", "d3"]
    relevant = ["d1", "d2", "d3"]
    assert recall_at_k(retrieved, relevant, 3) == 1.0
    assert precision_at_k(retrieved, relevant, 3) == 1.0
    assert mrr(retrieved, relevant) == 1.0
    assert abs(ndcg_at_k(retrieved, relevant, 3) - 1.0) < 1e-9


def test_metrics_partial_and_mrr():
    retrieved = ["x", "d1", "y", "d2"]
    relevant = {"d1": 1.0, "d2": 1.0}
    assert mrr(retrieved, relevant) == 0.5            # first hit at rank 2
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    # NDCG should be < 1 because relevant docs are not at the top.
    assert ndcg_at_k(retrieved, relevant, 4) < 1.0


def test_evaluate_averages():
    runs = [
        {"retrieved": ["d1", "d2"], "relevant": ["d1"]},
        {"retrieved": ["z", "d3"], "relevant": ["d3"]},
    ]
    res = evaluate(runs, k=2)
    assert res.n_queries == 2
    assert 0.0 < res.mrr <= 1.0
    d = res.as_dict()
    assert "ndcg@2" in d and "recall@2" in d


# ---- Freshness recrawl -----------------------------------------------------

def test_freshness_due_after_interval():
    ft = FreshnessTracker(base_interval=100.0, min_interval=10.0)
    ft.record("https://a.com/", content_hash=111, now=1000.0)
    assert ft.due(now=1050.0) == []                   # not yet due
    assert "https://a.com/" in ft.due(now=1200.0)     # past last+interval


def test_freshness_interval_shrinks_on_change():
    ft = FreshnessTracker(base_interval=100.0, min_interval=10.0)
    ft.record("https://a.com/", content_hash=1, now=0.0)
    before = ft.next_due_at("https://a.com/")
    ft.record("https://a.com/", content_hash=2, now=100.0)   # content changed
    after_interval = ft.next_due_at("https://a.com/") - 100.0
    assert after_interval < (before - 0.0)            # interval halved -> sooner
    assert ft.stats()["total_changes"] == 1
