"""
HNSW vs Vamana benchmark on structured (clustered) data.

Usage:
    python scripts/benchmark_hnsw.py
"""

import time
import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.hnsw_index import HNSWIndex
from utils.vamana_index import VamanaVectorIndex

DIM = 64
N_VECTORS = 1_000
N_QUERIES = 50
K = 10


def generate_data(dim, n_vectors, n_queries, seed=42):
    """Generate clustered data using sklearn (fallback to random)."""
    try:
        from sklearn.datasets import make_blobs
        vecs, _ = make_blobs(
            n_samples=n_vectors + n_queries,
            n_features=dim,
            centers=8,
            cluster_std=0.4,
            random_state=seed,
        )
    except ImportError:
        print("  [sklearn not installed, using random unit vectors]")
        rng = np.random.default_rng(seed)
        vecs = rng.standard_normal((n_vectors + n_queries, dim)).astype(np.float32)

    vecs = vecs.astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    queries = vecs[-n_queries:]
    vecs = vecs[:-n_queries]
    return vecs, queries


def brute_force_gt(vectors, query, k):
    query = np.array(query, dtype=np.float32)
    query /= np.linalg.norm(query)
    dists = 1.0 - vectors @ query
    nearest = np.argsort(dists)[:k]
    return set(str(i) for i in nearest)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

HNSW_CONFIGS = [
    ("M08-efc200",  8,  200),
    ("M16-efc050", 16,   50),
    ("M16-efc200", 16,  200),
]

VAMANA_CONFIGS = [
    ("L10-R04", 10,  4),
    ("L10-R08", 10,  8),
    ("L20-R08", 20,  8),
    ("L20-R16", 20, 16),
    ("L50-R32", 50, 32),
]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_hnsw(vectors, queries, gt):
    print("\n" + "=" * 60)
    print("HNSW BENCHMARK")
    print("=" * 60)
    for name, m, efc in HNSW_CONFIGS:
        t0 = time.perf_counter()
        idx = HNSWIndex(m=m, ef_construction=efc, distance_metric="cosine")
        for j in range(N_VECTORS):
            idx.insert(vectors[j].tolist(), str(j))
        build_time = time.perf_counter() - t0

        print(f"\n  {name}  (M={m}, ef_c={efc})  build={build_time:.2f}s  nodes={idx.total_inserted}")
        print(f"  {'ef':<8} {'Recall':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'QPS':<10}")
        print(f"  {'-'*48}")
        for ef in [10, 50, 200]:
            latencies = []
            recalls = []
            for qi, q in enumerate(queries):
                t0 = time.perf_counter()
                res = idx.search(q.tolist(), k=K, ef=ef)
                lat = (time.perf_counter() - t0) * 1000
                latencies.append(lat)
                found = set(r["vector_id"] for r in res)
                recalls.append(len(found & gt[qi]) / K)
            print(
                f"  {ef:<8} {float(np.mean(recalls)):<10.4f} {float(np.mean(latencies)):<10.3f} "
                f"{float(np.percentile(latencies, 95)):<10.3f} {1000.0 / float(np.mean(latencies)):<10.1f}"
            )


def run_vamana(vectors, queries, gt):
    print("\n" + "=" * 60)
    print("VAMANA BENCHMARK")
    print("=" * 60)
    for name, L, R in VAMANA_CONFIGS:
        mmap_dir = f"vamana_bench_{name}"
        if os.path.exists(mmap_dir):
            shutil.rmtree(mmap_dir)

        t0 = time.perf_counter()
        idx = VamanaVectorIndex(dim=DIM, L=L, R=R, mmap_dir=mmap_dir)
        for j in range(N_VECTORS):
            idx.insert(vectors[j].tolist(), str(j))
        build_time = time.perf_counter() - t0

        n_nodes = len(idx.ids)
        print(f"\n  {name}  (L={L}, R={R})  build={build_time:.2f}s  nodes={n_nodes}")
        print(f"  {'bw':<8} {'Recall':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'QPS':<10}")
        print(f"  {'-'*48}")
        for bw in [10, 50, 200]:
            latencies = []
            recalls = []
            for qi, q in enumerate(queries):
                t0 = time.perf_counter()
                res = idx.search(q.tolist(), k=K, beam_width=bw)
                lat = (time.perf_counter() - t0) * 1000
                latencies.append(lat)
                found = set(r["vector_id"] for r in res)
                recalls.append(len(found & gt[qi]) / K)
            print(
                f"  {bw:<8} {float(np.mean(recalls)):<10.4f} {float(np.mean(latencies)):<10.3f} "
                f"{float(np.percentile(latencies, 95)):<10.3f} {1000.0 / float(np.mean(latencies)):<10.1f}"
            )

        # Close the mmap before cleaning up (Windows file lock)
        if idx._adj is not None:
            base = getattr(idx._adj, "_mmap", None)
            if base is not None:
                base.close()
            idx._adj = None
            import gc
            gc.collect()
        if os.path.exists(mmap_dir):
            shutil.rmtree(mmap_dir)


def comparison_table(vectors, queries, gt):
    """One final table with both algorithms side-by-side at beam/ef=50."""
    print("\n\n" + "=" * 85)
    print("COMPARISON — HNSW vs Vamana at beam/ef=50")
    print("=" * 85)
    print(f"{'Algo':<9} {'Config':<12} {'Build(s)':<10} {'Recall':<10} {'Avg(ms)':<10} {'P95(ms)':<10} {'QPS':<10}")
    print("-" * 85)

    for name, m, efc in HNSW_CONFIGS:
        t0 = time.perf_counter()
        idx = HNSWIndex(m=m, ef_construction=efc, distance_metric="cosine")
        for j in range(N_VECTORS):
            idx.insert(vectors[j].tolist(), str(j))
        bt = time.perf_counter() - t0
        latencies, recalls = [], []
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            res = idx.search(q.tolist(), k=K, ef=50)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            found = set(r["vector_id"] for r in res)
            recalls.append(len(found & gt[qi]) / K)
        print(
            f"{'HNSW':<9} {name:<12} {bt:<10.2f} {float(np.mean(recalls)):<10.4f} "
            f"{float(np.mean(latencies)):<10.3f} {float(np.percentile(latencies, 95)):<10.3f} "
            f"{1000.0 / float(np.mean(latencies)):<10.1f}"
        )

    for name, L, R in VAMANA_CONFIGS:
        mmap_dir = f"vamana_comp_{name}"
        if os.path.exists(mmap_dir):
            shutil.rmtree(mmap_dir)
        t0 = time.perf_counter()
        idx = VamanaVectorIndex(dim=DIM, L=L, R=R, mmap_dir=mmap_dir)
        for j in range(N_VECTORS):
            idx.insert(vectors[j].tolist(), str(j))
        bt = time.perf_counter() - t0
        latencies, recalls = [], []
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            res = idx.search(q.tolist(), k=K, beam_width=50)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            found = set(r["vector_id"] for r in res)
            recalls.append(len(found & gt[qi]) / K)
        print(
            f"{'VAMANA':<9} {name:<12} {bt:<10.2f} {float(np.mean(recalls)):<10.4f} "
            f"{float(np.mean(latencies)):<10.3f} {float(np.percentile(latencies, 95)):<10.3f} "
            f"{1000.0 / float(np.mean(latencies)):<10.1f}"
        )
        # Close mmap before cleanup
        if idx._adj is not None:
            base = getattr(idx._adj, "_mmap", None)
            if base is not None:
                base.close()
            idx._adj = None
            import gc
            gc.collect()
        if os.path.exists(mmap_dir):
            shutil.rmtree(mmap_dir)

    print("=" * 85)
    print("Notes:")
    print("  - HNSW uses hierarchical graph navigation for fast routing to neighborhood")
    print("  - Vamana uses single-layer graph with beam search (simpler, robust to deletion)")
    print("  - Vamana scan-insert O(N) vs HNSW O(logN) on clustered data")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark():
    print(f"Generating {N_VECTORS} x {DIM}d vectors + {N_QUERIES} queries (clustered)...")
    vectors, queries = generate_data(DIM, N_VECTORS, N_QUERIES)
    print("Computing brute-force ground truth...")
    gt = [brute_force_gt(vectors, q, K) for q in queries]

    run_hnsw(vectors, queries, gt)
    run_vamana(vectors, queries, gt)
    comparison_table(vectors, queries, gt)


if __name__ == "__main__":
    run_benchmark()
