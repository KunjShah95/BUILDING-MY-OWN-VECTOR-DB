"""
Distributed / Partitioned Index Foundation

Provides a foundation for partitioning vector indexes across multiple shards
for horizontal scaling. Supports:

  - Hash-based partitioning by vector_id
  - Range-based partitioning by timestamp (for time-series)
  - Consistent hashing for even distribution across nodes
  - Query routing that fans out across all partitions and merges results

Usage:
  # Create a partitioned HNSW index
  partitioner = PartitionManager(strategy="hash", num_partitions=4)
  indexes = partitioner.create_partitions(vectors, method="hnsw")

  # Search across all partitions with result merging
  results = partitioner.search(query_vector, k=10)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---- Partition Strategies --------------------------------------------------


class PartitionStrategy:
    """Base class for partition strategies."""

    def partition_id(self, key: str, num_partitions: int) -> int:
        raise NotImplementedError


class HashPartitionStrategy(PartitionStrategy):
    """Consistent hash-based partitioning.

    Uses MD5 hash of the vector_id to deterministically assign
    vectors to partitions for even distribution.
    """

    def partition_id(self, key: str, num_partitions: int) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % num_partitions


class RangePartitionStrategy(PartitionStrategy):
    """Range-based partitioning by timestamp.

    Useful for time-series data where queries often target recent time windows.
    """

    def __init__(self, ranges: List[Tuple[str, str]]):
        """
        Parameters
        ----------
        ranges : list of (start, end) tuples
            ISO-format time ranges for each partition, e.g.
            [("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31")]
        """
        self.ranges = ranges

    def partition_id(self, key: str, num_partitions: int) -> int:
        # key is expected to be a timestamp string
        for i, (start, end) in enumerate(self.ranges):
            if start <= key <= end:
                return i
        return 0  # default to first partition


# ---- Partition Manager -----------------------------------------------------


class PartitionManager:
    """Manages a set of partitioned vector indexes for distributed search.

    Parameters
    ----------
    strategy : str or PartitionStrategy
        "hash" for hash-based, or a custom PartitionStrategy instance.
    num_partitions : int
        Number of partitions (shards). Default 4.
    index_type : str
        Type of index for each partition: "hnsw", "ivf", "brute", "pq", "int8".
        Default "hnsw".
    """

    def __init__(
        self,
        strategy: str = "hash",
        num_partitions: int = 4,
        index_type: str = "hnsw",
    ):
        if isinstance(strategy, str):
            if strategy == "hash":
                self.strategy = HashPartitionStrategy()
            else:
                raise ValueError(f"Unknown partition strategy: {strategy}")
        else:
            self.strategy = strategy

        self.num_partitions = num_partitions
        self.index_type = index_type
        self.partitions: Dict[int, Any] = {}  # partition_id -> index object
        self.index_params: Dict[str, Any] = {}

    def _create_index(self) -> Any:
        """Create a single partition index."""
        if self.index_type == "hnsw":
            from utils.hnsw_index import HNSWIndex
            return HNSWIndex(
                m=self.index_params.get("m", 16),
                m0=self.index_params.get("m0", 32),
                ef_construction=self.index_params.get("ef_construction", 200),
                distance_metric=self.index_params.get("distance_metric", "cosine"),
            )
        elif self.index_type == "ivf":
            from utils.ivf_index import IVFIndex
            return IVFIndex(
                n_clusters=self.index_params.get("n_clusters", 50),
                n_probes=self.index_params.get("n_probes", 5),
            )
        elif self.index_type == "pq":
            from utils.product_quantization import PQIndex
            return PQIndex(
                M=self.index_params.get("M", 16),
                k_sub=self.index_params.get("k_sub", 256),
                distance_metric=self.index_params.get("distance_metric", "cosine"),
            )
        elif self.index_type == "int8":
            from utils.int8_index import Int8Index
            return Int8Index(
                distance_metric=self.index_params.get("distance_metric", "cosine"),
            )
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def build(
        self,
        vectors: List[Dict[str, Any]],
        index_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build partitioned indexes from a list of vectors.

        Each vector dict must have at least ``vector`` and ``vector_id`` keys.

        Parameters
        ----------
        vectors : list of dict
            Vectors to index.
        index_params : dict, optional
            Parameters passed to each partition's index constructor.

        Returns
        -------
        dict with partition stats.
        """
        if index_params:
            self.index_params.update(index_params)

        # Initialize empty partitions
        self.partitions = {i: self._create_index() for i in range(self.num_partitions)}

        # Distribute vectors across partitions
        partition_vectors: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(self.num_partitions)}
        for v in vectors:
            pid = self.strategy.partition_id(v["vector_id"], self.num_partitions)
            partition_vectors[pid].append(v)

        # Train and add vectors per partition
        for pid, pvectors in partition_vectors.items():
            idx = self.partitions[pid]
            if not pvectors:
                continue

            # Train if needed (IVF, PQ, Int8 require training)
            if hasattr(idx, "train"):
                training_vecs = [v["vector"] for v in pvectors if v.get("vector")]
                if training_vecs:
                    idx.train(training_vecs)

            # Add vectors
            for v in pvectors:
                if hasattr(idx, "add"):
                    idx.add(v["vector"], v["vector_id"], v.get("metadata"))
                elif hasattr(idx, "insert"):
                    idx.insert(v["vector"], v["vector_id"], v.get("metadata"))

        # Return stats
        stats = {}
        for pid, idx in self.partitions.items():
            if hasattr(idx, "get_stats"):
                stats[str(pid)] = idx.get_stats()
            elif hasattr(idx, "get_graph_stats"):
                stats[str(pid)] = idx.get_graph_stats()
            else:
                stats[str(pid)] = {"vector_count": len(getattr(idx, "vector_ids", []))}

        return {
            "success": True,
            "num_partitions": self.num_partitions,
            "strategy": type(self.strategy).__name__,
            "index_type": self.index_type,
            "total_vectors": len(vectors),
            "partition_stats": stats,
        }

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        **search_kwargs,
    ) -> Dict[str, Any]:
        """Search across all partitions and merge results.

        Fans out the query to each partition, collects top-k candidates
        per partition, then re-ranks globally.

        Parameters
        ----------
        query_vector : list of float
            Query vector.
        k : int
            Number of results to return.
        **search_kwargs
            Additional kwargs passed to each partition's search method.

        Returns
        -------
        dict with merged results.
        """
        if not self.partitions:
            return {"success": False, "message": "No partitions built", "results": []}

        start = time.time()
        all_candidates: List[Tuple[float, str, Dict]] = []

        for pid, idx in self.partitions.items():
            try:
                if hasattr(idx, "search"):
                    per_partition_k = max(k, 20)  # oversample per partition
                    if self.index_type == "hnsw":
                        result = idx.search(
                            query_vector,
                            per_partition_k,
                            ef=search_kwargs.get("ef_search", 50),
                        )
                    elif self.index_type == "ivf":
                        result = idx.search_with_rerank(
                            query_vector, per_partition_k
                        ) if search_kwargs.get("use_rerank", True) else idx.search(
                            query_vector, per_partition_k
                        )
                    else:
                        result = idx.search(query_vector, per_partition_k)

                    for r in result:
                        all_candidates.append((
                            r.get("distance", 0),
                            r.get("vector_id", ""),
                            r.get("metadata", {}),
                        ))
            except Exception as e:
                logger.warning("Partition %d search failed: %s", pid, e)

        # Global merge: sort by distance
        all_candidates.sort(key=lambda x: x[0])
        top_k = all_candidates[:k]

        results = [
            {
                "vector_id": vid,
                "distance": float(dist),
                "metadata": meta,
            }
            for dist, vid, meta in top_k
        ]

        search_time = time.time() - start

        return {
            "success": True,
            "results": results,
            "total_results": len(results),
            "search_time": search_time,
            "method": f"partitioned_{self.index_type}",
            "partitions_queried": len(self.partitions),
        }

    def save(self, base_path: str) -> Dict[str, Any]:
        """Save all partition indexes to disk."""
        os.makedirs(base_path, exist_ok=True)
        for pid, idx in self.partitions.items():
            partition_path = os.path.join(base_path, f"partition_{pid}")
            if hasattr(idx, "save"):
                idx.save(partition_path + ".json")
        return {
            "success": True,
            "message": f"Saved {len(self.partitions)} partitions to {base_path}",
        }

    def load(self, base_path: str) -> Dict[str, Any]:
        """Load all partition indexes from disk."""
        self.partitions = {}

        # Static import mapping: index_type -> (module_path, class_name)
        _import_map = {
            "hnsw": ("utils.hnsw_index", "HNSWIndex"),
            "ivf": ("utils.ivf_index", "IVFIndex"),
            "pq": ("utils.product_quantization", "PQIndex"),
            "int8": ("utils.int8_index", "Int8Index"),
        }

        if self.index_type not in _import_map:
            return {"success": False, "message": f"Unknown index type: {self.index_type}"}

        mod_path, cls_name = _import_map[self.index_type]
        import importlib
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)

        for pid in range(self.num_partitions):
            partition_path = os.path.join(base_path, f"partition_{pid}.json")
            if os.path.exists(partition_path):
                self.partitions[pid] = cls.load(partition_path)

        return {
            "success": True,
            "message": f"Loaded {len(self.partitions)} partitions from {base_path}",
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate stats across all partitions."""
        stats = {}
        for pid, idx in self.partitions.items():
            if hasattr(idx, "get_stats"):
                stats[str(pid)] = idx.get_stats()
            elif hasattr(idx, "get_graph_stats"):
                stats[str(pid)] = idx.get_graph_stats()
        return {
            "num_partitions": len(self.partitions),
            "strategy": type(self.strategy).__name__,
            "index_type": self.index_type,
            "partition_stats": stats,
        }
