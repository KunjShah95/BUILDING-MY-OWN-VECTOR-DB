import time
import json
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hnsw_index import HNSWIndex
from utils.ivf_index import IVFIndex

SCALES = [
    {"name": "1K", "n": 1000, "dims": 128},
    {"name": "10K", "n": 10000, "dims": 128},
    {"name": "100K", "n": 100000, "dims": 128},
    {"name": "1M", "n": 1000000, "dims": 128},
]

RESULTS_DIR = Path(".benchmarks")
RESULTS_FILE = RESULTS_DIR / "scale_benchmark_results.json"


def generate_vectors(n: int, dims: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    vectors = np.random.randn(n, dims).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def benchmark_scale(n: int, dims: int, name: str, n_queries: int = 100, k: int = 10):
    print(f"\n{'='*60}")
    print(f"Scale: {name} ({n} vectors, {dims} dims)")
    print(f"{'='*60}")

    vectors = generate_vectors(n, dims)
    queries = generate_vectors(n_queries, dims, seed=99)

    results = {"scale": name, "n_vectors": n, "dimensions": dims}

    # Brute force baseline
    print("\nBrute force...")
    bf_times = []
    for q in queries:
        start = time.perf_counter()
        dots = vectors @ q
        np.argsort(-dots)[:k]
        bf_times.append(time.perf_counter() - start)

    # Build HNSW
    print("Building HNSW index...")
    hnsw = HNSWIndex(m=16, ef_construction=200)
    build_start = time.time()
    for i, v in enumerate(vectors):
        hnsw.insert(v, f"vec_{i}")
    build_time = time.time() - build_start
    print(f"HNSW build: {build_time:.2f}s")

    # HNSW search
    print("HNSW search...")
    hnsw_times = []
    hnsw_recall = []
    for qi, q in enumerate(queries):
        dots = vectors @ q
        bf_top = set(np.argsort(-dots)[:k])

        start = time.perf_counter()
        hnsw_results = hnsw.search(q, k)
        hnsw_times.append(time.perf_counter() - start)

        hnsw_top = set(r["vector_id"] for r in hnsw_results)
        recall = len(bf_top & hnsw_top) / k
        hnsw_recall.append(recall)

    # Build IVF
    print("Building IVF index...")
    ivf = IVFIndex(n_clusters=min(100, n // 10), n_probes=10)
    train_start = time.time()
    ivf.train(vectors)
    for i, v in enumerate(vectors):
        ivf.add(v, f"vec_{i}")
    ivf_build_time = time.time() - train_start
    print(f"IVF build: {ivf_build_time:.2f}s")

    # IVF search
    print("IVF search...")
    ivf_times = []
    ivf_recall = []
    for qi, q in enumerate(queries):
        dots = vectors @ q
        bf_top = set(np.argsort(-dots)[:k])

        start = time.perf_counter()
        ivf_results = ivf.search(q, k)
        ivf_times.append(time.perf_counter() - start)

        ivf_top = set(r["vector_id"] for r in ivf_results)
        recall = len(bf_top & ivf_top) / k
        ivf_recall.append(recall)

    results["brute_force"] = {
        "avg_time_ms": float(np.mean(bf_times) * 1000),
        "p50_ms": float(np.median(bf_times) * 1000),
        "p99_ms": float(np.percentile(bf_times, 99) * 1000),
    }
    results["hnsw"] = {
        "build_time_s": float(build_time),
        "avg_time_ms": float(np.mean(hnsw_times) * 1000),
        "p50_ms": float(np.median(hnsw_times) * 1000),
        "p99_ms": float(np.percentile(hnsw_times, 99) * 1000),
        "avg_recall": float(np.mean(hnsw_recall)),
    }
    results["ivf"] = {
        "build_time_s": float(ivf_build_time),
        "avg_time_ms": float(np.mean(ivf_times) * 1000),
        "p50_ms": float(np.median(ivf_times) * 1000),
        "p99_ms": float(np.percentile(ivf_times, 99) * 1000),
        "avg_recall": float(np.mean(ivf_recall)),
    }

    print(f"BF:     {results['brute_force']['avg_time_ms']:.2f}ms")
    print(f"HNSW:   {results['hnsw']['avg_time_ms']:.2f}ms  recall={results['hnsw']['avg_recall']:.4f}")
    print(f"IVF:    {results['ivf']['avg_time_ms']:.2f}ms  recall={results['ivf']['avg_recall']:.4f}")

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for scale in SCALES:
        name = scale["name"]
        n = scale["n"]
        dims = scale["dims"]
        n_queries = min(100, n // 10)
        all_results[name] = benchmark_scale(n, dims, name, n_queries=n_queries)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("SCALE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scale':<8} {'Method':<12} {'Avg (ms)':<10} {'P50 (ms)':<10} {'P99 (ms)':<10} {'Recall':<10}")
    print("-"*80)
    for name, data in all_results.items():
        for method in ["brute_force", "hnsw", "ivf"]:
            if method in data:
                m = data[method]
                recall = m.get("avg_recall", 1.0)
                print(f"{name:<8} {method:<12} {m['avg_time_ms']:<10.2f} {m['p50_ms']:<10.2f} {m['p99_ms']:<10.2f} {recall:<10.4f}")

    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
