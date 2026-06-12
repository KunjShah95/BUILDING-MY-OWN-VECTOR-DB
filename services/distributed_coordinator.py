"""
Distributed query aggregation (ROADMAP Phase 2).

A coordinator scatter-gathers a query to N shards in parallel, then fuses the
per-shard top-K into a global top-K. Shards are pluggable via the ``Shard``
protocol, so the same coordinator drives:

  - in-process indexes (HNSW/IVF/PartitionManager) for single-box parallelism
  - remote nodes over HTTP/gRPC (wrap the call in a Shard implementation)

Fault tolerance: a shard that errors or times out is skipped; the query still
returns results from the healthy shards (with ``degraded=True`` flagged).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class Shard(Protocol):
    """A searchable shard. Any object with this shape can be a shard."""

    name: str

    def search(self, query_vector: List[float], k: int, **kwargs) -> List[Dict[str, Any]]:
        """Return a list of {vector_id, distance, metadata} dicts."""
        ...


@dataclass
class CallableShard:
    """Adapt any search callable into a Shard (handy for local indexes/tests)."""
    name: str
    fn: Callable[..., List[Dict[str, Any]]]

    def search(self, query_vector: List[float], k: int, **kwargs) -> List[Dict[str, Any]]:
        return self.fn(query_vector, k, **kwargs)


@dataclass
class IndexShard:
    """Wrap an HNSW/IVF index instance as a shard."""
    name: str
    index: Any

    def search(self, query_vector: List[float], k: int, **kwargs) -> List[Dict[str, Any]]:
        res = self.index.search(query_vector, k, **kwargs) if kwargs else self.index.search(query_vector, k)
        return res or []


@dataclass
class DistributedCoordinator:
    """
    Scatter-gather coordinator over a set of shards.

    Args:
        shards: list of Shard objects.
        max_workers: thread-pool size for fan-out (defaults to #shards).
        per_shard_timeout: seconds to wait per shard before skipping it.
        oversample: per-shard k multiplier so the global merge has enough
            candidates (top-K from each shard, then re-rank).
    """
    shards: List[Shard]
    max_workers: Optional[int] = None
    per_shard_timeout: float = 5.0
    oversample: int = 2
    _executor: ThreadPoolExecutor = field(init=False, repr=False)

    def __post_init__(self):
        workers = self.max_workers or max(1, len(self.shards))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shard")

    def _query_shard(self, shard: Shard, query_vector, per_shard_k, kwargs):
        return shard.name, shard.search(query_vector, per_shard_k, **kwargs)

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        fusion: str = "distance",
        **shard_kwargs,
    ) -> Dict[str, Any]:
        """
        Fan out the query to all shards and fuse the global top-K.

        Args:
            k: global number of results.
            fusion: "distance" (merge-sort by distance) or "rrf"
                (reciprocal rank fusion across shard result lists).
        """
        if not self.shards:
            return {"success": False, "message": "no shards configured", "results": []}

        start = time.time()
        per_shard_k = max(k * self.oversample, k)
        per_shard_results: Dict[str, List[Dict[str, Any]]] = {}
        failed: List[str] = []

        futures = {
            self._executor.submit(self._query_shard, s, query_vector, per_shard_k, shard_kwargs): s
            for s in self.shards
        }
        for fut in as_completed(futures, timeout=None):
            shard = futures[fut]
            try:
                name, results = fut.result(timeout=self.per_shard_timeout)
                per_shard_results[name] = results or []
            except Exception as exc:  # noqa: BLE001 - one shard down must not fail the query
                logger.warning("shard %s failed: %s", getattr(shard, "name", "?"), exc)
                failed.append(getattr(shard, "name", "?"))

        if fusion == "rrf":
            merged = self._fuse_rrf(per_shard_results, k)
        else:
            merged = self._fuse_distance(per_shard_results, k)

        return {
            "success": True,
            "results": merged,
            "total_results": len(merged),
            "shards_queried": len(self.shards),
            "shards_failed": failed,
            "degraded": bool(failed),
            "fusion": fusion,
            "search_time": time.time() - start,
        }

    @staticmethod
    def _fuse_distance(per_shard: Dict[str, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """Merge by ascending distance, deduplicating by vector_id (keep best)."""
        best: Dict[str, Dict[str, Any]] = {}
        for results in per_shard.values():
            for r in results:
                vid = r.get("vector_id")
                if vid is None:
                    continue
                if vid not in best or r.get("distance", 0) < best[vid].get("distance", 0):
                    best[vid] = r
        merged = sorted(best.values(), key=lambda r: r.get("distance", 0.0))
        return merged[:k]

    @staticmethod
    def _fuse_rrf(per_shard: Dict[str, List[Dict[str, Any]]], k: int, c: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal rank fusion across shard result lists."""
        scores: Dict[str, float] = {}
        keep: Dict[str, Dict[str, Any]] = {}
        for results in per_shard.values():
            for rank, r in enumerate(results):
                vid = r.get("vector_id")
                if vid is None:
                    continue
                scores[vid] = scores.get(vid, 0.0) + 1.0 / (c + rank + 1)
                keep.setdefault(vid, r)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for vid, score in ranked[:k]:
            row = dict(keep[vid])
            row["rrf_score"] = score
            out.append(row)
        return out

    def shutdown(self):
        self._executor.shutdown(wait=False)
