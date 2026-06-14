"""Built-in ANN benchmark harness."""
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BenchmarkSuite:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def run_recall_benchmark(self, search_fn: Callable, queries: np.ndarray,
                             ground_truth: np.ndarray, k: int = 10,
                             methods: List[str] = None) -> List[Dict[str, Any]]:
        """Run recall@k benchmark for each method."""
        methods = methods or ["hnsw", "ivf", "brute"]
        results = []
        for method in methods:
            recalls = []
            latencies = []
            for i in range(len(queries)):
                q = queries[i].tolist()
                start = time.perf_counter()
                try:
                    result = search_fn(q, k=k, method=method)
                    elapsed = (time.perf_counter() - start) * 1000
                    ids = [r.get("vector_id") or r.get("id") for r in result.get("results", [])]
                    gt_ids = [str(g) for g in ground_truth[i][:k]]
                    matches = sum(1 for vid in ids if vid in gt_ids)
                    recall = matches / k if k > 0 else 0
                    recalls.append(recall)
                    latencies.append(elapsed)
                except Exception as e:
                    logger.warning("Benchmark error for %s at query %d: %s", method, i, e)
            avg_recall = float(np.mean(recalls)) if recalls else 0
            avg_latency = float(np.mean(latencies)) if latencies else 0
            p99_latency = float(np.percentile(latencies, 99)) if latencies else 0
            results.append({
                "method": method,
                "avg_recall": round(avg_recall, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "p99_latency_ms": round(p99_latency, 2),
                "queries_run": len(recalls),
            })
        self.results.extend(results)
        return results

    def generate_synthetic_dataset(self, n_vectors: int = 10000,
                                   n_queries: int = 100,
                                   dim: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random vectors and brute-force ground truth."""
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        # Brute-force ground truth
        gt = np.argsort(-queries @ vectors.T, axis=1)[:, :10]
        return vectors, queries, gt

    def get_results(self) -> Dict[str, Any]:
        return {"benchmarks": self.results}

    def clear(self):
        self.results = []

benchmark_suite = BenchmarkSuite()
