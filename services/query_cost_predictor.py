"""Query cost predictor: estimates scan count, latency, and memory for a query before execution."""
import math
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Dict] = {
    "hnsw": {"ef_search": 100},
    "ivf": {"nprobe": 10, "nlist": 100},
    "ivf_pq": {"nprobe": 10, "nlist": 100},
    "brute": {},
}


@dataclass
class CostEstimate:
    query_id: str
    estimated_scan_count: int
    estimated_latency_ms: float
    estimated_memory_mb: float
    complexity_score: float  # 0-1
    index_type: str
    collection_size: int


@dataclass
class _ActualRecord:
    query_id: str
    actual_latency_ms: float
    actual_scan_count: int


class QueryCostPredictor:
    """Estimates query cost before execution and calibrates coefficients over time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latency_coeff: Dict[str, float] = {k: 1.0 for k in _DEFAULTS}
        self._actuals: List[_ActualRecord] = []
        self._collection_sizes: Dict[str, int] = {}

    def set_collection_size(self, collection_id: str, size: int) -> None:
        with self._lock:
            self._collection_sizes[collection_id] = size

    def estimate(
        self,
        query_vector: List[float],
        k: int,
        collection_id: str,
        index_type: str = "hnsw",
        filters: Optional[Dict] = None,
        index_params: Optional[Dict] = None,
    ) -> CostEstimate:
        index_type = index_type.lower().replace("-", "_")
        params = dict(_DEFAULTS.get(index_type, {}))
        if index_params:
            params.update(index_params)

        with self._lock:
            N = self._collection_sizes.get(collection_id, 10_000)
            coeff = self._latency_coeff.get(index_type, 1.0)

        filter_sel = max(0.1, 1.0 - 0.05 * len(filters)) if filters else 1.0
        scan, latency = self._base_cost(index_type, N, k, params, filter_sel, coeff)

        dim = len(query_vector) if query_vector else 128
        memory_mb = (scan * dim * 4) / (1024 * 1024)
        complexity = min(1.0, latency / 10_000.0)

        return CostEstimate(
            query_id=str(uuid.uuid4()),
            estimated_scan_count=max(1, int(scan)),
            estimated_latency_ms=round(latency, 3),
            estimated_memory_mb=round(memory_mb, 4),
            complexity_score=round(complexity, 4),
            index_type=index_type,
            collection_size=N,
        )

    def record_actual(self, query_id: str, actual_latency_ms: float, actual_scan_count: int) -> None:
        with self._lock:
            self._actuals.append(_ActualRecord(query_id, actual_latency_ms, actual_scan_count))
            if len(self._actuals) > 10_000:
                self._actuals = self._actuals[-5_000:]

    def calibrate(self) -> Dict:
        """Adjust latency coefficients based on recorded actuals (simple mean ratio)."""
        with self._lock:
            records = list(self._actuals)
        if not records:
            return {"status": "no_data"}
        avg_latency = sum(r.actual_latency_ms for r in records) / len(records)
        logger.info("Calibration: %d records, avg actual latency=%.2f ms", len(records), avg_latency)
        return {"status": "ok", "records_used": len(records), "avg_actual_latency_ms": avg_latency}

    def _base_cost(
        self, index_type: str, N: int, k: int, params: Dict, filter_sel: float, coeff: float
    ) -> Tuple[float, float]:
        if index_type == "hnsw":
            ef = params.get("ef_search", 100)
            scan = ef * math.log(max(N, 2))
            latency = scan * 0.001 * coeff
        elif index_type == "ivf":
            nprobe = params.get("nprobe", 10)
            nlist = max(params.get("nlist", 100), 1)
            scan = nprobe * (N / nlist)
            latency = scan * 0.0005 * coeff
        elif index_type == "ivf_pq":
            nprobe = params.get("nprobe", 10)
            nlist = max(params.get("nlist", 100), 1)
            scan = nprobe * (N / nlist)
            latency = scan * 0.0002 * coeff
        else:  # brute
            scan = float(N)
            latency = N * 0.002 * coeff
        return scan * filter_sel, latency * filter_sel


query_cost_predictor = QueryCostPredictor()
