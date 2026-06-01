import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from services.collection_service import CollectionService
from services.vector_service import VectorService
from utils.index_paths import get_hnsw_path, get_ivf_path


class CollectionIndexService:
    """Per-collection HNSW index build, load, search, and stats."""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.collection_service = CollectionService(db_session)
        self.vector_service = VectorService(db_session)

    def _normalize(self, collection_id: str) -> str:
        return collection_id.strip().lower()

    def collection_index_path(self, collection_id: str) -> str:
        return get_hnsw_path(self._normalize(collection_id))

    def collection_index_exists(self, collection_id: str) -> bool:
        slug = self._normalize(collection_id)
        if self.vector_service.hnsw_db.get_hnsw_index(slug) is not None:
            return True
        return os.path.exists(self.collection_index_path(slug))

    def ensure_collection_index_loaded(self, collection_id: str) -> bool:
        slug = self._normalize(collection_id)
        if self.vector_service.hnsw_db.get_hnsw_index(slug) is not None:
            return True
        if not os.path.exists(self.collection_index_path(slug)):
            return False
        result = self.vector_service.load_index("hnsw", collection_id=slug)
        return bool(result.get("success"))

    def build_collection_index(
        self,
        collection_id: str,
        method: str = "hnsw",
        m: int = 16,
        m0: Optional[int] = None,
        ef_construction: int = 200,
        n_clusters: int = 100,
        n_probes: int = 10,
    ) -> Dict[str, Any]:
        slug = self._normalize(collection_id)
        coll = self.collection_service.get_collection(slug)
        if not coll.get("success"):
            return coll

        if method == "hnsw":
            create_result = self.vector_service.create_index(
                method="hnsw",
                m=m,
                m0=m0,
                ef_construction=ef_construction,
                collection_id=slug,
            )
            if not create_result.get("success"):
                return create_result
            save_result = self.vector_service.save_index("hnsw", collection_id=slug)
            if not save_result.get("success"):
                return save_result
            return {
                "success": True,
                "message": f"HNSW index built and saved for collection '{slug}'",
                "collection_id": slug,
                "index_path": self.collection_index_path(slug),
                "stats": create_result.get("stats"),
                "parameters": create_result.get("parameters"),
            }

        if method == "ivf":
            create_result = self.vector_service.create_index(
                method="ivf",
                n_clusters=n_clusters,
                n_probes=n_probes,
                collection_id=slug,
            )
            if not create_result.get("success"):
                return create_result
            save_result = self.vector_service.save_index("ivf", collection_id=slug)
            if not save_result.get("success"):
                return save_result
            return {
                "success": True,
                "message": f"IVF index built and saved for collection '{slug}'",
                "collection_id": slug,
                "index_path": create_result.get("index_path"),
                "stats": create_result.get("stats"),
                "parameters": {"n_clusters": n_clusters, "n_probes": n_probes},
            }

        return {"success": False, "message": f"Unknown indexing method: {method}"}

    def search_collection_indexed(
        self,
        collection_id: str,
        query_vector: List[float],
        k: int = 5,
        method: str = "hnsw",
        ef_search: Optional[int] = None,
        n_probes: Optional[int] = None,
        use_rerank: Optional[bool] = True,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        slug = self._normalize(collection_id)
        coll = self.collection_service.get_collection(slug)
        if not coll.get("success"):
            return coll

        metric = distance_metric or coll["collection"].get("distance_metric", "cosine")

        if method == "brute":
            return self.vector_service.search_vectors(
                query_vector=query_vector,
                k=k,
                method="brute",
                collection_id=slug,
                filters=filters,
                distance_metric=metric,
            )

        if method == "ivf":
            ivf_in_memory = self.vector_service.ivf_db.get_ivf_index(slug)
            if ivf_in_memory is None and os.path.exists(get_ivf_path(slug)):
                self.vector_service.load_index("ivf", collection_id=slug)
            effective = "ivf" if self.vector_service.ivf_db.get_ivf_index(slug) else "brute"
            return self.vector_service.search_vectors(
                query_vector=query_vector,
                k=k,
                method=effective,
                n_probes=n_probes,
                use_rerank=use_rerank,
                collection_id=slug,
                filters=filters,
                distance_metric=metric,
            )

        if self.collection_index_exists(slug):
            self.ensure_collection_index_loaded(slug)

        effective_method = "hnsw" if self.vector_service.hnsw_db.get_hnsw_index(slug) else "brute"
        return self.vector_service.search_vectors(
            query_vector=query_vector,
            k=k,
            method=effective_method,
            ef_search=ef_search,
            n_probes=n_probes,
            use_rerank=use_rerank,
            collection_id=slug,
            filters=filters,
            distance_metric=metric,
        )

    def get_collection_index_stats(self, collection_id: str) -> Dict[str, Any]:
        slug = self._normalize(collection_id)
        coll = self.collection_service.get_collection(slug)
        if not coll.get("success"):
            return coll

        hnsw_path = get_hnsw_path(slug)
        ivf_path = get_ivf_path(slug)
        hnsw_on_disk = os.path.exists(hnsw_path)
        ivf_on_disk = os.path.exists(ivf_path)
        hnsw_in_memory = self.vector_service.hnsw_db.get_hnsw_index(slug)
        ivf_in_memory = self.vector_service.ivf_db.get_ivf_index(slug)

        stats: Dict[str, Any] = {
            "collection_id": slug,
            "hnsw": {
                "index_path": hnsw_path,
                "index_on_disk": hnsw_on_disk,
                "index_loaded": hnsw_in_memory is not None,
            },
            "ivf": {
                "index_path": ivf_path,
                "index_on_disk": ivf_on_disk,
                "index_loaded": ivf_in_memory is not None,
            },
        }

        if hnsw_in_memory is not None:
            graph = hnsw_in_memory.get_graph_stats()
            stats["method"] = "hnsw"
            stats["is_indexed"] = True
            stats.update({
                "total_nodes": graph.get("total_nodes", 0),
                "total_edges": graph.get("total_edges", 0),
                "avg_connections": graph.get("avg_connections", 0),
                "max_level": graph.get("max_level", 0),
                "level_distribution": graph.get("level_distribution", {}),
            })
        elif ivf_in_memory is not None:
            ivf_stats = ivf_in_memory.get_stats()
            stats["method"] = "ivf"
            stats["is_indexed"] = True
            stats.update({
                "total_vectors": ivf_stats.get("total_vectors", 0),
                "n_clusters": ivf_stats.get("n_clusters", 0),
                "n_probes": ivf_stats.get("n_probes", 0),
                "avg_cluster_size": ivf_stats.get("avg_cluster_size", 0),
            })
        elif hnsw_on_disk:
            stats["is_indexed"] = True
            stats["method"] = "hnsw"
            stats["message"] = "HNSW index file exists on disk but is not loaded in memory"
        elif ivf_on_disk:
            stats["is_indexed"] = True
            stats["method"] = "ivf"
            stats["message"] = "IVF index file exists on disk but is not loaded in memory"
        else:
            stats["is_indexed"] = False
            stats["message"] = "No per-collection index; searches use brute force"

        return {"success": True, "stats": stats}

    def rebuild_ivf(self, collection_id: str, n_clusters: int = 100, n_probes: int = 10) -> Dict[str, Any]:
        """Rebuild the per-collection IVF index."""
        slug = self._normalize(collection_id)
        return self.vector_service.ivf_db.rebuild_ivf_index(
            n_clusters=n_clusters, n_probes=n_probes, collection_id=slug,
        )
