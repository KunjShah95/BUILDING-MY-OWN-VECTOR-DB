"""Adaptive index selection — per-query routing to fastest index."""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class IndexPerformance:
    method: str  # hnsw, ivf, brute
    avg_latency_ms: float = 0.0
    avg_recall: float = 1.0
    sample_count: int = 0
    last_used: float = 0.0

@dataclass
class QueryProfile:
    cardinality: int = 0  # estimated result count
    filter_selectivity: float = 1.0  # 0.0 to 1.0
    latency_budget_ms: float = 100.0
    preferred_method: Optional[str] = None

class AdaptiveIndexSelector:
    """Routes queries to the best index based on runtime metrics."""
    def __init__(self):
        self._perf: Dict[str, Dict[str, IndexPerformance]] = {}  # collection -> method -> perf
        self._lock = False  # simple flag to avoid reentrancy

    def record_query(self, collection_id: str, method: str, latency_ms: float, recall: float = 1.0):
        if collection_id not in self._perf:
            self._perf[collection_id] = {}
        if method not in self._perf[collection_id]:
            self._perf[collection_id][method] = IndexPerformance(method=method)
        perf = self._perf[collection_id][method]
        perf.avg_latency_ms = (perf.avg_latency_ms * perf.sample_count + latency_ms) / (perf.sample_count + 1)
        perf.avg_recall = (perf.avg_recall * perf.sample_count + recall) / (perf.sample_count + 1)
        perf.sample_count += 1
        perf.last_used = time.time()

    def select_method(self, collection_id: str, profile: Optional[QueryProfile] = None) -> str:
        """Pick the best index method based on performance history and query profile."""
        if collection_id not in self._perf or not self._perf[collection_id]:
            return "hnsw"  # default
        profile = profile or QueryProfile()
        available = self._perf[collection_id]
        # Score each method
        scores = {}
        for method, perf in available.items():
            if perf.sample_count < 3:
                scores[method] = 100  # bootstrap — try it
                continue
            latency_score = max(0, 100 - perf.avg_latency_ms)
            recall_score = perf.avg_recall * 100
            scores[method] = latency_score * 0.4 + recall_score * 0.6
        if not scores:
            return "hnsw"
        best = max(scores, key=scores.get)
        logger.debug("Adaptive select for %s: %s (scores=%s)", collection_id, best, scores)
        return best

    def get_performance_report(self, collection_id: str) -> Dict[str, Any]:
        if collection_id not in self._perf:
            return {"collection_id": collection_id, "methods": {}}
        return {
            "collection_id": collection_id,
            "methods": {
                m: {
                    "avg_latency_ms": round(p.avg_latency_ms, 2),
                    "avg_recall": round(p.avg_recall, 4),
                    "sample_count": p.sample_count,
                }
                for m, p in self._perf[collection_id].items()
            }
        }

    def get_all_reports(self) -> Dict[str, Any]:
        return {cid: self.get_performance_report(cid)["methods"] for cid in self._perf}

adaptive_selector = AdaptiveIndexSelector()
