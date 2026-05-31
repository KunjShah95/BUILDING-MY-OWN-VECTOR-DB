import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from services.collection_service import CollectionService
from services.vector_service import VectorService
from utils.index_paths import get_hnsw_path


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

        if method == "ivf":
            return {
                "success": False,
                "message": "Per-collection IVF is not implemented yet; use method='hnsw'",
            }

        if method != "hnsw":
            return {"success": False, "message": f"Unknown indexing method: {method}"}

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
            return {
                "success": False,
                "message": "Per-collection IVF search is not implemented yet",
            }

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

        path = self.collection_index_path(slug)
        on_disk = os.path.exists(path)
        in_memory = self.vector_service.hnsw_db.get_hnsw_index(slug)

        stats: Dict[str, Any] = {
            "collection_id": slug,
            "index_path": path,
            "index_on_disk": on_disk,
            "index_loaded": in_memory is not None,
            "method": "hnsw",
        }

        if in_memory is not None:
            graph = in_memory.get_graph_stats()
            stats.update(
                {
                    "is_indexed": True,
                    "total_nodes": graph.get("total_nodes", 0),
                    "total_edges": graph.get("total_edges", 0),
                    "avg_connections": graph.get("avg_connections", 0),
                    "max_level": graph.get("max_level", 0),
                    "level_distribution": graph.get("level_distribution", {}),
                }
            )
        elif on_disk:
            stats["is_indexed"] = True
            stats["message"] = "Index file exists on disk but is not loaded in memory"
        else:
            stats["is_indexed"] = False
            stats["message"] = "No per-collection HNSW index; searches use brute force"

        return {"success": True, "stats": stats}
