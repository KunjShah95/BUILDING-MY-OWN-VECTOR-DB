"""
Named index lifecycle management.
Merged from C:\\ann search engine on 2026-06-21.

Manages create/save/load/query across HNSW, IVF, and BruteForce indexes
using the vector DB's own index implementations.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from services.vector_service import VectorService
from utils.brute_force_index import BruteForceIndex
from utils.hnsw_index import HNSWIndex
from utils.ivf_index import IVFIndex
from utils.ivf_pq_index import IVFPQIndex

logger = logging.getLogger(__name__)

INDEX_DIR = "indexes/ann"

_INDEX_FACTORIES = {
    "hnsw": lambda dim, kw: HNSWIndex(
        m=kw.get("m", 16),
        m0=kw.get("m0"),
        ef_construction=kw.get("ef_construction", 200),
    ),
    "ivf": lambda dim, kw: IVFIndex(
        n_clusters=kw.get("n_clusters", 100),
        n_probes=kw.get("n_probes", 10),
    ),
    "ivfpq": lambda dim, kw: IVFPQIndex(
        nlist=kw.get("nlist", max(16, int(dim ** 0.5))),
        M=kw.get("M", 8),
        k_sub=kw.get("k_sub", 256),
        nprobe=kw.get("nprobe", 8),
        metric=kw.get("metric", "cosine"),
    ),
    "brute": lambda dim, kw: BruteForceIndex(
        dimension=dim,
        metric=kw.get("metric", "cosine"),
    ),
}


class AnnIndexService:
    """
    Lifecycle manager for ANN indexes.

    Keeps one instance of each index type in memory.  Callers can build
    (create), persist (save), restore (load), query info, and run
    cross-index comparison searches.
    """

    def __init__(self, db: Session):
        self.db = db
        self.vector_service = VectorService(db)
        self._indexes: Dict[str, Any] = {}
        os.makedirs(INDEX_DIR, exist_ok=True)
        self._load_saved_indexes()

    # ── init helpers ─────────────────────────────────────────────────────────

    def _load_saved_indexes(self) -> None:
        # IVF-PQ uses a directory
        ivfpq_dir = os.path.join(INDEX_DIR, "ivfpq_index")
        if os.path.isdir(ivfpq_dir):
            try:
                self._indexes["ivfpq"] = IVFPQIndex.load(ivfpq_dir)
                logger.info("Auto-loaded ivfpq index from %s/", ivfpq_dir)
            except Exception as exc:
                logger.warning("Failed to auto-load ivfpq: %s", exc)

        for filename in os.listdir(INDEX_DIR):
            if not filename.endswith(".json"):
                continue
            index_type = filename.replace("_index.json", "")
            filepath = os.path.join(INDEX_DIR, filename)
            try:
                if index_type == "hnsw":
                    idx = HNSWIndex()
                    idx.load(filepath)
                    self._indexes["hnsw"] = idx
                elif index_type == "ivf":
                    idx = IVFIndex()
                    idx.load(filepath)
                    self._indexes["ivf"] = idx
                elif index_type == "brute":
                    idx = BruteForceIndex(dimension=1)
                    idx.load(filepath)
                    self._indexes["brute"] = idx
                logger.info("Auto-loaded %s index from %s", index_type, filepath)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", filename, exc)

    # ── public API ───────────────────────────────────────────────────────────

    def create_index(self, index_type: str, **kwargs) -> Dict[str, Any]:
        """Build and populate an index from all vectors currently in the DB."""
        if index_type not in _INDEX_FACTORIES:
            return {"success": False, "message": f"Unknown index type: {index_type}"}

        vectors_resp = self.vector_service.get_all_vectors(limit=100_000)
        if not vectors_resp.get("success"):
            return {"success": False, "message": "Failed to fetch vectors from DB"}

        raw = vectors_resp.get("vectors", [])
        if not raw:
            return {"success": False, "message": "No vectors in DB to index"}

        vector_data = [v["vector_data"] for v in raw]
        vector_ids = [v["vector_id"] for v in raw]
        metadatas = [v.get("metadata") for v in raw]
        dimension = len(vector_data[0])

        try:
            index = _INDEX_FACTORIES[index_type](dimension, kwargs)

            if index_type == "hnsw":
                batch = [
                    {"vector": vec, "vector_id": vid, "metadata": meta}
                    for vec, vid, meta in zip(vector_data, vector_ids, metadatas)
                ]
                index.insert_batch(batch)

            elif index_type in ("ivf", "ivfpq"):
                index.train([[float(x) for x in v] for v in vector_data])
                batch = [
                    {"vector": vec, "vector_id": vid, "metadata": meta}
                    for vec, vid, meta in zip(vector_data, vector_ids, metadatas)
                ]
                index.add_batch(batch)

            elif index_type == "brute":
                index.add(vector_data, vector_ids)

            self._indexes[index_type] = index
            self._persist_index(index_type)

            return {
                "success": True,
                "message": f"Created {index_type} index",
                "vector_count": len(vector_ids),
                "stats": self._get_raw_stats(index_type),
            }

        except Exception as exc:
            logger.error("Error creating %s index: %s", index_type, exc)
            return {"success": False, "message": str(exc)}

    def save_index(self, index_type: str) -> Dict[str, Any]:
        if index_type not in self._indexes:
            return {"success": False, "message": f"Index '{index_type}' not loaded"}
        try:
            self._persist_index(index_type)
            return {"success": True, "message": f"Saved {index_type} index"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def load_index(self, index_type: str) -> Dict[str, Any]:
        filepath = os.path.join(INDEX_DIR, f"{index_type}_index.json")
        if not os.path.exists(filepath):
            return {"success": False, "message": f"No saved index at {filepath}"}
        try:
            if index_type == "hnsw":
                idx = HNSWIndex()
                idx.load(filepath)
            elif index_type == "ivf":
                idx = IVFIndex()
                idx.load(filepath)
            elif index_type == "ivfpq":
                idx = IVFPQIndex.load(filepath)
            elif index_type == "brute":
                idx = BruteForceIndex(dimension=1)
                idx.load(filepath)
            else:
                return {"success": False, "message": f"Unknown index type: {index_type}"}

            self._indexes[index_type] = idx
            return {
                "success": True,
                "message": f"Loaded {index_type} index",
                "stats": self._get_raw_stats(index_type),
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def get_index_info(self, index_type: Optional[str] = None) -> Dict[str, Any]:
        if index_type:
            if index_type not in self._indexes:
                return {"success": False, "message": f"Index '{index_type}' not loaded"}
            return {
                "success": True,
                "index_type": index_type,
                "stats": self._get_raw_stats(index_type),
            }

        return {
            "success": True,
            "indexes": {
                k: self._get_raw_stats(k) for k in self._indexes
            },
        }

    def get_statistics(self) -> Dict[str, Any]:
        total_resp = self.vector_service.get_all_vectors(limit=1)
        total = total_resp.get("total", 0) if total_resp.get("success") else 0
        return {
            "success": True,
            "total_vectors_in_db": total,
            "loaded_indexes": list(self._indexes.keys()),
            "index_stats": {k: self._get_raw_stats(k) for k in self._indexes},
        }

    def compare_search(self, query_vector: List[float], k: int = 10) -> Dict[str, Any]:
        """Run query_vector against every loaded index, return timing + results."""
        if not self._indexes:
            return {"success": False, "message": "No indexes loaded"}

        comparison: Dict[str, Any] = {}

        for index_type, index in self._indexes.items():
            t0 = time.perf_counter()
            try:
                if index_type == "hnsw":
                    raw = index.search(query_vector, k=k)
                    results = [
                        {"vector_id": r["vector_id"], "distance": r["distance"]}
                        for r in raw
                    ]
                elif index_type in ("ivf", "ivfpq"):
                    raw = index.search(query_vector, k=k)
                    results = [
                        {"vector_id": r["vector_id"], "distance": r["distance"]}
                        for r in raw
                    ]
                elif index_type == "brute":
                    raw = index.search(query_vector, k=k)
                    results = [{"vector_id": vid, "distance": dist} for vid, dist in raw]
                else:
                    results = []
            except Exception as exc:
                results = []
                logger.warning("compare_search error on %s: %s", index_type, exc)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            comparison[index_type] = {
                "results": results,
                "search_time_ms": round(elapsed_ms, 3),
                "result_count": len(results),
            }

        return {"success": True, "k": k, "comparison": comparison}

    # ── private helpers ──────────────────────────────────────────────────────

    def _persist_index(self, index_type: str) -> None:
        index = self._indexes[index_type]
        if index_type == "ivfpq":
            # IVF-PQ uses a directory with numpy binary files
            dirpath = os.path.join(INDEX_DIR, "ivfpq_index")
            index.save(dirpath)
            logger.info("Persisted ivfpq index → %s/", dirpath)
        else:
            filepath = os.path.join(INDEX_DIR, f"{index_type}_index.json")
            index.save(filepath)
            logger.info("Persisted %s index → %s", index_type, filepath)

    def _get_raw_stats(self, index_type: str) -> Dict[str, Any]:
        index = self._indexes[index_type]
        try:
            return index.get_stats()
        except AttributeError:
            return {"index_type": index_type}
