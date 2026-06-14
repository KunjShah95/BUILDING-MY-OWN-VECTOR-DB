"""
Validate the HNSWParameterRecommender by building HNSW indexes with recommended
parameters and measuring actual recall against brute-force ground truth.

Tests across multiple synthetic datasets with different properties:
  - Low-dimensional Gaussian (8D)
  - Mid-dimensional uniform (64D)
  - High-dimensional uniform (128D)
  - Well-separated clusters
  - L2-normalised sphere data

For each dataset:
  1. Run tune_hnsw() to get recommended M, ef_construction, ef_search
  2. Build an index with the recommended params and measure recall
  3. Compare predicted "expected_recall" vs actual recall
  4. Report how well the recommender performed

Usage:
    python scripts/validate_recommender.py          # fast: recommended + 3 baselines
    python scripts/validate_recommender.py --grid   # also brute-force grid search (slow: ~30x per dataset)
"""

import argparse
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.hnsw_index import HNSWIndex
from utils.index_tuning import DatasetAnalyzer, HNSWParameterRecommender, tune_hnsw


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_gaussian_lowdim(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Low-dimensional Gaussian blob."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def make_uniform_mid(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Mid-dimensional uniform on hypercube."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, dim)).astype(np.float32)


def make_uniform_highdim(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """High-dimensional uniform on hypercube."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, dim)).astype(np.float32)


def make_clusters(n: int, dim: int, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    """Well-separated Gaussian clusters."""
    rng = np.random.default_rng(seed)
    centroids = rng.uniform(-5, 5, (n_clusters, dim)).astype(np.float32)
    per_cluster = n // n_clusters
    vecs = []
    for c in centroids:
        cluster = c + rng.standard_normal((per_cluster, dim)).astype(np.float32) * 0.3
        vecs.append(cluster)
    # Pad / trim to exact size
    result = np.vstack(vecs)[:n]
    return result


def make_normalized_sphere(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Uniform on unit sphere (L2-normalised)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


# ---------------------------------------------------------------------------
# Brute-force ground truth
# ---------------------------------------------------------------------------

def brute_force_gt(vectors: np.ndarray, query: np.ndarray, k: int) -> set:
    """Return set of vector_id strings for the k nearest neighbours."""
    query = query.ravel().astype(np.float32)
    # Normalise query if vectors appear normalised
    v_norms = np.linalg.norm(vectors, axis=1)
    if abs(np.mean(v_norms) - 1.0) < 0.05:
        query /= np.linalg.norm(query) + 1e-10
        dists = 1.0 - vectors @ query
    else:
        diff = vectors - query
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
    nearest = np.argsort(dists)[:k]
    return set(str(int(i)) for i in nearest)


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name": "Gaussian 8D",
        "generator": make_gaussian_lowdim,
        "n": 1000,
        "dim": 8,
    },
    {
        "name": "Uniform 64D",
        "generator": make_uniform_mid,
        "n": 1000,
        "dim": 64,
    },
    {
        "name": "Uniform 128D",
        "generator": make_uniform_highdim,
        "n": 1000,
        "dim": 128,
    },
    {
        "name": "Clusters 32D",
        "generator": make_clusters,
        "n": 1000,
        "dim": 32,
    },
    {
        "name": "Sphere 64D",
        "generator": make_normalized_sphere,
        "n": 1000,
        "dim": 64,
    },
    {
        "name": "Gaussian 16D (large)",
        "generator": make_gaussian_lowdim,
        "n": 5000,
        "dim": 16,
    },
]

N_QUERIES = 100
K = 10


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------

def benchmark_dataset(name: str, vectors: np.ndarray, queries: list, gt: list,
                      recall_target: float, *, run_grid: bool = False) -> dict:
    """
    For a single dataset:
      1. Get recommender's suggested params
      2. Build HNSW index with those params, measure recall at various ef_search
      3. Build HNSW with alternative fixed params for comparison
      4. Return structured results
    """
    n, d = vectors.shape

    # -- Step 1: Recommendation ------------------------------------------------
    rec = tune_hnsw(vectors, recall_target=recall_target, sample_size=min(n, 1000))

    # -- Step 2: Test recommended config ---------------------------------------
    def test_config(m: int, efc: int, label: str) -> dict:
        t0 = time.perf_counter()
        idx = HNSWIndex(m=m, ef_construction=efc, distance_metric="cosine")
        vectors_list = vectors.tolist()
        for j in range(n):
            idx.insert(vectors_list[j], str(j))
        build_time = time.perf_counter() - t0

        ef_values = [10, 30, 50, 100, 200]
        results = {}
        for ef in ef_values:
            latencies = []
            recalls = []
            for qi, q in enumerate(queries):
                t0 = time.perf_counter()
                res = idx.search(q, k=K, ef=ef)
                lat = (time.perf_counter() - t0) * 1000
                latencies.append(lat)
                found = set(r["vector_id"] for r in res)
                recalls.append(len(found & gt[qi]) / K)
            results[ef] = {
                "recall": float(np.mean(recalls)),
                "latency_avg_ms": float(np.mean(latencies)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
            }
        return {
            "label": label,
            "m": m,
            "ef_construction": efc,
            "build_time_s": round(build_time, 3),
            "ef_results": results,
        }

    rec_config = test_config(rec.m, rec.ef_construction, "recommended")

    # -- Step 3: Comparison configs --------------------------------------------
    baselines = [
        test_config(8, 100, "M08-efc100"),
        test_config(16, 200, "M16-efc200"),
        test_config(32, 300, "M32-efc300"),
    ]

    # -- Step 4: Find which M actually gives the best recall@10 at ef=50 -------
    best_recall = rec_config["ef_results"].get(50, {}).get("recall", 0.0)
    best_m = rec.m
    best_efc = rec.ef_construction
    if run_grid:
        for m_candidate in [4, 8, 12, 16, 24, 32]:
            for efc_candidate in [50, 100, 200, 300, 400]:
                t0 = time.perf_counter()
                idx = HNSWIndex(m=m_candidate, ef_construction=efc_candidate, distance_metric="cosine")
                vectors_list = vectors.tolist()
                for j in range(n):
                    idx.insert(vectors_list[j], str(j))
                _ = time.perf_counter() - t0
                recalls_50 = []
                for qi, q in enumerate(queries):
                    res = idx.search(q, k=K, ef=50)
                    found = set(r["vector_id"] for r in res)
                    recalls_50.append(len(found & gt[qi]) / K)
                r50 = float(np.mean(recalls_50))
                if r50 > best_recall:
                    best_recall = r50
                    best_m = m_candidate
                    best_efc = efc_candidate
        best_recall = round(best_recall, 4)

    return {
        "dataset": name,
        "num_vectors": n,
        "dimension": d,
        "grid_searched": run_grid,
        "recommendation": {
            "m": rec.m,
            "m0": rec.m0,
            "ef_construction": rec.ef_construction,
            "ef_search": rec.ef_search,
            "expected_recall": rec.expected_recall,
            "confidence": rec.confidence,
            "reasoning": rec.reasoning,
        },
        "recommended": rec_config,
        "baselines": baselines,
        "best_config": {
            "m": best_m,
            "ef_construction": best_efc,
            "recall_at_ef50": best_recall,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(all_results: list):
    """Print a clean summary of all benchmark results."""

    # -- Per-dataset detail ----------------------------------------------
    for r in all_results:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {r['dataset']}  ({r['num_vectors']} x {r['dimension']}D)")
        print(f"{'=' * 70}")

        rec = r["recommendation"]
        print(f"\n  Recommendation:")
        print(f"    M={rec['m']}  ef_c={rec['ef_construction']}  ef_search={rec['ef_search']}")
        print(f"    Expected recall: {rec['expected_recall']:.1%}  (confidence: {rec['confidence']})")
        for line in rec["reasoning"]:
            print(f"    -> {line}")

        print(f"\n  Recall @ 10  (best in bold)\n")
        header = f"  {'Config':<18}"
        for ef in [10, 30, 50, 100, 200]:
            header += f" {'ef=' + str(ef):<12}"
        print(header)
        print(f"  {'-' * 78}")

        def fmt_row(label: str, ef_results: dict, is_best: bool = False) -> str:
            row = f"  {label:<18}"
            for ef in [10, 30, 50, 100, 200]:
                if ef in ef_results:
                    val = f"{ef_results[ef]['recall']:.1%}"
                    row += f" {val:<12}"
                else:
                    row += f" {'-':<12}"
            return row

        # Show recommended config
        print(fmt_row("recommended", r["recommended"]["ef_results"]))
        # Show baselines
        for b in r["baselines"]:
            print(fmt_row(b["label"], b["ef_results"]))

        # Highlight best config
        best = r["best_config"]
        print(f"\n  Best config at ef=50:  M={best['m']}, ef_c={best['ef_construction']}, "
              f"recall={best['recall_at_ef50']:.1%}")

        # Compare predicted vs actual at ef_search
        act_rec = r["recommended"]["ef_results"].get(rec["ef_search"], {}).get("recall", 0.0)
        exp_rec = rec["expected_recall"]
        delta = act_rec - exp_rec
        print(f"  Predicted recall @ ef_search={rec['ef_search']}: {exp_rec:.1%}")
        print(f"  Actual recall    @ ef_search={rec['ef_search']}: {act_rec:.1%}")
        if delta >= 0.05:
            print(f"  !  Under-predicted by {delta:.1%} (conservative -- safe!)")
        elif delta >= 0.0:
            print(f"  OK Accurate (within +/-{abs(delta):.1%} of prediction)")
        else:
            print(f"  !! Over-predicted by {abs(delta):.1%}")

    # -- Summary table ---------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("SUMMARY - Recommender accuracy across all datasets")
    print(f"{'=' * 70}")
    header = f"  {'Dataset':<22} {'M':<4} {'ef_c':<6} {'ef_s':<5} {'Exp Rcl':<9} {'Act Rcl':<9} {'Best M':<7} {'Best ef_c':<9} {'Best R@50':<9}"
    print(header)
    print(f"  {'-' * 80}")
    for r in all_results:
        rec = r["recommendation"]
        act = r["recommended"]["ef_results"].get(rec["ef_search"], {}).get("recall", 0.0)
        best = r["best_config"]
        print(
            f"  {r['dataset']:<22} "
            f"{rec['m']:<4} {rec['ef_construction']:<6} {rec['ef_search']:<5} "
            f"{rec['expected_recall']:<9.1%} {act:<9.1%} "
            f"{best['m']:<7} {best['ef_construction']:<9} {best['recall_at_ef50']:<9.1%}"
        )
    print(f"  {'-' * 80}")

    # -- Aggregate stats -------------------------------------------------
    deltas = []
    for r in all_results:
        rec = r["recommendation"]
        act = r["recommended"]["ef_results"].get(rec["ef_search"], {}).get("recall", 0.0)
        deltas.append(act - rec["expected_recall"])
    print(f"\n  Mean prediction error: {np.mean(deltas):.1%}")
    print(f"  Median prediction error: {np.median(deltas):.1%}")
    print(f"  Min error: {np.min(deltas):.1%}  Max error: {np.max(deltas):.1%}")
    print(f"  Datasets where predicted <= actual (safe): "
          f"{sum(1 for d in deltas if d >= -0.01)} / {len(deltas)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(grid: bool = False):
    print(f"Validating HNSWParameterRecommender across {len(DATASETS)} datasets\n")

    all_results = []

    for ds in DATASETS:
        name = ds["name"]
        print(f"[{name}] Generating {ds['n']} x {ds['dim']}D vectors...")

        vectors = ds["generator"](ds["n"], ds["dim"])
        # Reserve last N_QUERIES as queries
        queries_np = vectors[-N_QUERIES:]
        vectors = vectors[:-N_QUERIES]

        print(f"  Computing brute-force ground truth for {N_QUERIES} queries...")
        gt = [brute_force_gt(vectors, q, K) for q in queries_np]

        queries_list = [q.tolist() for q in queries_np]

        result = benchmark_dataset(
            name, vectors, queries_list, gt,
            recall_target=0.95,
            run_grid=grid,
        )
        all_results.append(result)

    print_results(all_results)

    print(f"\n{'=' * 70}")
    print("Done. Recommender validation complete.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate HNSWParameterRecommender")
    parser.add_argument("--grid", action="store_true",
                        help="Run brute-force grid search (slow: ~30 index builds per dataset)")
    args = parser.parse_args()
    main(grid=args.grid)
