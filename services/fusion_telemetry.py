"""Cross-index fusion with per-index latency telemetry."""
import threading
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant for RRF


class FusionTelemetry:
    """Wraps RRF fusion with per-index latency tracking and overlap analysis."""

    def __init__(self, window_seconds: int = 60):
        self._lock = threading.Lock()
        self._window_seconds = window_seconds
        # Records: list of (timestamp, index_name, latency_ms, result_count)
        self._latency_records: deque = deque(maxlen=10_000)
        # Overlap records: list of (timestamp, overlap_ratio)
        self._overlap_records: deque = deque(maxlen=10_000)
        # Fusion time records: list of (timestamp, ms)
        self._fusion_times: deque = deque(maxlen=10_000)

    def fuse(self, results_per_index: Dict[str, List[dict]], k: int) -> List[dict]:
        """
        Run RRF fusion across multiple indexes and record telemetry.

        Args:
            results_per_index: mapping of index_name -> list of result dicts
                               each result must have a 'vector_id' or 'id' key.
            k: number of results to return.

        Returns:
            Fused and ranked list of result dicts (up to k items).
        """
        t0 = time.time()
        scores: Dict[str, float] = defaultdict(float)
        items: Dict[str, dict] = {}

        now = time.time()
        index_id_sets: Dict[str, set] = {}

        for index_name, results in results_per_index.items():
            id_set = set()
            for rank, res in enumerate(results):
                vid = res.get("vector_id") or res.get("id") or str(rank)
                items[vid] = res
                scores[vid] += 1.0 / (_RRF_K + rank + 1)
                id_set.add(vid)
            index_id_sets[index_name] = id_set

        # Compute overlap between indexes
        all_sets = list(index_id_sets.values())
        if len(all_sets) >= 2:
            intersection = set.intersection(*all_sets)
            union = set.union(*all_sets)
            overlap = len(intersection) / max(len(union), 1)
        else:
            overlap = 1.0

        fused = sorted(
            [dict(**items[vid], rrf_score=round(s, 6)) for vid, s in scores.items()],
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:k]

        fusion_ms = (time.time() - t0) * 1000

        with self._lock:
            self._overlap_records.append((now, overlap))
            self._fusion_times.append((now, fusion_ms))

        return fused

    def record_index_latency(self, index_name: str, latency_ms: float, result_count: int) -> None:
        with self._lock:
            self._latency_records.append((time.time(), index_name, latency_ms, result_count))

    def get_telemetry(self, window_seconds: Optional[int] = None) -> dict:
        window = window_seconds or self._window_seconds
        cutoff = time.time() - window

        with self._lock:
            lat_records = [(ts, idx, ms, cnt) for ts, idx, ms, cnt in self._latency_records if ts >= cutoff]
            overlap_records = [(ts, o) for ts, o in self._overlap_records if ts >= cutoff]
            fusion_times = [ms for ts, ms in self._fusion_times if ts >= cutoff]

        # Per-index stats
        per_index: Dict[str, List[float]] = defaultdict(list)
        per_index_counts: Dict[str, List[int]] = defaultdict(list)
        for _, idx, ms, cnt in lat_records:
            per_index[idx].append(ms)
            per_index_counts[idx].append(cnt)

        index_stats = {}
        for idx, latencies in per_index.items():
            latencies.sort()
            n = len(latencies)
            index_stats[idx] = {
                "p50_ms": latencies[n // 2] if n else 0,
                "p95_ms": latencies[int(n * 0.95)] if n else 0,
                "avg_result_count": sum(per_index_counts[idx]) / max(len(per_index_counts[idx]), 1),
                "sample_count": n,
            }

        avg_overlap = sum(o for _, o in overlap_records) / max(len(overlap_records), 1) if overlap_records else 0
        avg_fusion_ms = sum(fusion_times) / max(len(fusion_times), 1) if fusion_times else 0

        return {
            "window_seconds": window,
            "per_index": index_stats,
            "overlap_ratio_avg": round(avg_overlap, 4),
            "fusion_time_avg_ms": round(avg_fusion_ms, 3),
            "total_fusion_ops": len(fusion_times),
        }

    def recommend_index(self, query_stats: dict) -> str:
        """Suggest the best single index based on recent telemetry."""
        telemetry = self.get_telemetry()
        per_index = telemetry.get("per_index", {})
        if not per_index:
            return "hnsw"  # default

        # Prefer the index with lowest p50 latency
        best = min(per_index.items(), key=lambda kv: kv[1].get("p50_ms", float("inf")))
        return best[0]


fusion_telemetry = FusionTelemetry()
