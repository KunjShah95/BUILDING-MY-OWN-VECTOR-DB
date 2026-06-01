from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping, Collection
from models.vector_model import VectorModel
from database.hnsw_database import HNSWVectorDatabase
from database.ivf_database import IVFVectorDatabase
from services.collection_service import CollectionService
import time
import logging

logger = logging.getLogger(__name__)


class VectorService:
    """
    Service class for vector operations
    Provides business logic layer between API and database
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.vector_model = VectorModel(db_session)
        self.hnsw_db = HNSWVectorDatabase(db_session)
        self.ivf_db = IVFVectorDatabase(db_session)
        self.collection_service = CollectionService(db_session)

    def _collection_ids_for_tenant(self, tenant_id: str) -> List[str]:
        """Get all collection IDs belonging to a tenant."""
        records = (
            self.db.query(Collection)
            .filter(Collection.tenant_id == tenant_id)
            .all()
        )
        return [r.collection_id for r in records]

    def _insert_into_indexes(
        self,
        vector_data: List[float],
        vector_id: str,
        metadata: Optional[Dict[str, Any]],
        collection_id: Optional[str],
    ) -> None:
        global_hnsw = self.hnsw_db.get_hnsw_index()
        if global_hnsw is not None:
            global_hnsw.insert(vector_data, vector_id, metadata)

        if collection_id:
            scoped = self.hnsw_db.get_hnsw_index(collection_id)
            if scoped is not None:
                scoped.insert(vector_data, vector_id, metadata)

        if self.ivf_db.ivf_index is not None and self.ivf_db.ivf_index.is_trained:
            self.ivf_db.ivf_index.add(vector_data, vector_id, metadata)

    # ==================== Vector Operations ====================

    def create_vector(
        self,
        vector_data: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        collection_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new vector

        Args:
            vector_data: Vector data
            metadata: Optional metadata
            vector_id: Optional custom ID
            collection_id: Optional collection scope
            tenant_id: Optional tenant scope (validates collection ownership)

        Returns:
            Creation result
        """
        try:
            if collection_id:
                dim_check = self.collection_service.validate_vector_dimension(
                    collection_id, vector_data, tenant_id=tenant_id,
                )
                if not dim_check.get("success"):
                    return dim_check

            vector = self.vector_model.create_vector(
                vector_data, metadata, vector_id, collection_id
            )

            self._insert_into_indexes(
                vector_data, vector.vector_id, metadata, collection_id
            )

            logger.info(f"Created vector: {vector.vector_id}")

            return {
                "success": True,
                "message": "Vector created successfully",
                "vector_id": vector.vector_id,
                "vector": vector.to_dict(),
            }
        except Exception as e:
            logger.error(f"Error creating vector: {str(e)}")
            return {"success": False, "message": f"Error creating vector: {str(e)}"}

    def create_vector_batch(
        self,
        vectors: List[Dict[str, Any]],
        batch_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create multiple vectors in a batch

        Args:
            vectors: List of vector dictionaries
            batch_name: Optional batch name
            description: Optional description

        Returns:
            Batch creation result
        """
        try:
            # Insert into database
            result = self.vector_model.create_vector_batch(
                vectors, batch_name, description
            )

            if self.hnsw_db.hnsw_index is not None or self.ivf_db.ivf_index is not None:
                for vector_data in vectors:
                    cid = vector_data.get("collection_id")
                    self._insert_into_indexes(
                        vector_data["vector"],
                        vector_data["vector_id"],
                        vector_data.get("metadata"),
                        cid,
                    )

            logger.info(f"Created batch with {result['vector_count']} vectors")

            return {
                "success": True,
                "message": "Batch created successfully",
                "batch_id": result.get("batch_id"),
                "batch_name": result.get("batch_name"),
                "vector_count": result.get("vector_count"),
            }
        except Exception as e:
            logger.error(f"Error creating batch: {str(e)}")
            return {"success": False, "message": f"Error creating batch: {str(e)}"}

    def get_vector(self, vector_id: str,
                   tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a vector by ID

        Args:
            vector_id: Vector ID
            tenant_id: Optional tenant scope (validates collection ownership)

        Returns:
            Vector data or error
        """
        try:
            vector = self.vector_model.get_vector(vector_id)

            if vector:
                if tenant_id and vector.collection_id:
                    t_ids = self._collection_ids_for_tenant(tenant_id)
                    if vector.collection_id not in t_ids:
                        return {"success": False, "message": "Vector not found"}
                return {"success": True, "vector": vector.to_dict()}
            else:
                return {"success": False, "message": "Vector not found"}
        except Exception as e:
            logger.error(f"Error getting vector {vector_id}: {str(e)}")
            return {"success": False, "message": f"Error getting vector: {str(e)}"}

    def get_all_vectors(self, limit: int = 1000, offset: int = 0,
                        tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get all vectors with pagination

        Args:
            limit: Maximum number of vectors
            offset: Offset for pagination
            tenant_id: Optional tenant scope

        Returns:
            List of vectors
        """
        try:
            if tenant_id:
                vectors = self.vector_model.get_all_vectors(
                    limit, offset,
                    collection_ids=self._collection_ids_for_tenant(tenant_id),
                )
            else:
                vectors = self.vector_model.get_all_vectors(limit, offset)

            return {
                "success": True,
                "vectors": [vector.to_dict() for vector in vectors],
                "count": len(vectors),
                "limit": limit,
                "offset": offset,
                "total": self.vector_model.get_vector_count(),
            }
        except Exception as e:
            logger.error(f"Error getting vectors: {str(e)}")
            return {"success": False, "message": f"Error getting vectors: {str(e)}"}

    def update_vector(
        self,
        vector_id: str,
        vector_data: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update a vector

        Args:
            vector_id: Vector ID
            vector_data: Optional new vector data
            metadata: Optional new metadata

        Returns:
            Update result
        """
        try:
            success = self.vector_model.update_vector(vector_id, metadata, vector_data)

            if success:
                logger.info(f"Updated vector: {vector_id}")
                return {"success": True, "message": "Vector updated successfully"}
            else:
                return {"success": False, "message": "Vector not found"}
        except Exception as e:
            logger.error(f"Error updating vector {vector_id}: {str(e)}")
            return {"success": False, "message": f"Error updating vector: {str(e)}"}

    def delete_vector(self, vector_id: str,
                      tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete a vector

        Args:
            vector_id: Vector ID
            tenant_id: Optional tenant scope (validates collection ownership)

        Returns:
            Delete result
        """
        try:
            vector = self.vector_model.get_vector(vector_id)
            if not vector:
                return {"success": False, "message": "Vector not found"}

            if tenant_id and vector.collection_id:
                t_ids = self._collection_ids_for_tenant(tenant_id)
                if vector.collection_id not in t_ids:
                    return {"success": False, "message": "Vector not found"}

            success = self.vector_model.delete_vector(vector_id)

            if success:
                if self.hnsw_db.hnsw_index is not None:
                    self.hnsw_db.hnsw_index.delete(vector_id)

                if vector and vector.collection_id:
                    scoped = self.hnsw_db.get_hnsw_index(vector.collection_id)
                    if scoped is not None:
                        scoped.delete(vector_id)

                if self.ivf_db.ivf_index is not None:
                    self.ivf_db.ivf_index.delete_vector(vector_id)

                logger.info(f"Deleted vector: {vector_id}")

                return {
                    "success": True,
                    "message": f"Vector {vector_id} deleted successfully",
                }
            else:
                return {"success": False, "message": "Vector not found"}
        except Exception as e:
            logger.error(f"Error deleting vector {vector_id}: {str(e)}")
            return {"success": False, "message": f"Error deleting vector: {str(e)}"}

    # ==================== Search Operations ====================

    def search_vectors(
        self,
        query_vector: List[float],
        k: int = 5,
        method: str = "hnsw",
        ef_search: Optional[int] = None,
        n_probes: Optional[int] = None,
        use_rerank: Optional[bool] = True,
        collection_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar vectors

        Args:
            query_vector: Query vector
            k: Number of results
            method: Search method
            ef_search: HNSW search parameter
            tenant_id: Optional tenant scope (validates collection ownership)

        Returns:
            Search results
        """
        try:
            start_time = time.time()
            metric = distance_metric or "cosine"

            # Tenant scope validation
            if tenant_id:
                if collection_id:
                    coll_check = self.collection_service.get_collection(
                        collection_id, tenant_id=tenant_id,
                    )
                    if not coll_check.get("success"):
                        return coll_check
                else:
                    # No collection specified — search across tenant's collections
                    t_ids = self._collection_ids_for_tenant(tenant_id)
                    if not t_ids:
                        return {
                            "success": True, "results": [],
                            "total_results": 0, "method": "brute_force",
                        }
                    results = self.vector_model.search_vectors(
                        query_vector, k, metric, filters, tenant_id=tenant_id,
                    )
                    search_time = time.time() - start_time
                    return {
                        "success": True,
                        "query_vector": query_vector,
                        "results": results,
                        "total_results": len(results),
                        "search_time": search_time,
                        "method": "brute_force_tenant",
                    }

            # Collection-scoped search: prefer collection HNSW, else brute force.
            if collection_id:
                if method == "hnsw" and self.hnsw_db.get_hnsw_index(collection_id):
                    result = self.hnsw_db.search_hnsw(
                        query_vector,
                        k,
                        ef_search,
                        filters=filters,
                        distance_metric=metric,
                        collection_id=collection_id,
                    )
                else:
                    results = self.vector_model.search_vectors(
                        query_vector, k, metric, filters, collection_id
                    )
                    search_time = time.time() - start_time
                    result = {
                        "success": True,
                        "query_vector": query_vector,
                        "results": results,
                        "total_results": len(results),
                        "search_time": search_time,
                        "method": "brute_force",
                        "collection_id": collection_id,
                    }
                return result

            if method == "hnsw":
                # Check if HNSW index exists, fallback to brute force if not
                if self.hnsw_db.hnsw_index is None:
                    result = self.hnsw_db.search_brute_force(
                        query_vector, k, filters=filters, distance_metric=metric
                    )
                else:
                    result = self.hnsw_db.search_hnsw(
                        query_vector, k, ef_search, filters=filters, distance_metric=metric
                    )
            elif method == "ivf":
                effective_n_probes = (
                    n_probes
                    if n_probes is not None
                    else (
                        self.ivf_db.ivf_index.n_probes if self.ivf_db.ivf_index else 10
                    )
                )
                result = self.ivf_db.search(
                    query_vector,
                    k,
                    use_ivf=True,
                    use_rerank=use_rerank,
                    n_probes=effective_n_probes,
                )
            elif method == "brute":
                result = self.hnsw_db.search_brute_force(
                    query_vector, k, filters=filters, distance_metric=metric
                )
            else:
                return {"success": False, "message": f"Unknown search method: {method}"}

            search_time = time.time() - start_time

            # Combine timing
            if result.get("success"):
                result["search_time"] = search_time

            return result

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return {"success": False, "message": f"Error during search: {str(e)}"}

    def compare_search_methods(
        self, query_vector: List[float], k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare different search methods

        Args:
            query_vector: Query vector
            k: Number of results

        Returns:
            Comparison results
        """
        try:
            result = self.hnsw_db.compare_search_methods(query_vector, k)
            return result
        except Exception as e:
            logger.error(f"Error comparing methods: {str(e)}")
            return {"success": False, "message": f"Error comparing methods: {str(e)}"}

    # ==================== Index Operations ====================

    def create_index(
        self,
        method: str = "hnsw",
        m: int = 16,
        m0: Optional[int] = None,
        ef_construction: int = 200,
        n_clusters: int = 100,
        n_probes: int = 10,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an index

        Args:
            method: Indexing method
            m: HNSW parameter
            m0: HNSW parameter
            ef_construction: HNSW parameter
            n_clusters: IVF parameter
            n_probes: IVF parameter

        Returns:
            Index creation result
        """
        try:
            if method == "hnsw":
                return self.hnsw_db.create_hnsw_index(
                    m, m0, ef_construction, collection_id=collection_id
                )
            elif method == "ivf":
                return self.ivf_db.create_ivf_index(
                    n_clusters, n_probes, collection_id=collection_id
                )
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}",
                }
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return {"success": False, "message": f"Error creating index: {str(e)}"}

    def save_index(
        self, method: str = "hnsw", collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save index to disk

        Args:
            method: Indexing method

        Returns:
            Save result
        """
        try:
            if method == "hnsw":
                return self.hnsw_db.save_hnsw_index(collection_id=collection_id)
            elif method == "ivf":
                return self.ivf_db.save_ivf_index(collection_id=collection_id)
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}",
                }
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return {"success": False, "message": f"Error saving index: {str(e)}"}

    def load_index(
        self, method: str = "hnsw", collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load index from disk

        Args:
            method: Indexing method

        Returns:
            Load result
        """
        try:
            if method == "hnsw":
                return self.hnsw_db.load_hnsw_index(collection_id=collection_id)
            elif method == "ivf":
                return self.ivf_db.load_ivf_index(collection_id=collection_id)
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}",
                }
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return {"success": False, "message": f"Error loading index: {str(e)}"}

    def get_index_info(self, method: str = "hnsw") -> Dict[str, Any]:
        """
        Get index information

        Args:
            method: Indexing method

        Returns:
            Index information
        """
        try:
            if method == "hnsw":
                return self.hnsw_db.get_hnsw_index_info()
            elif method == "ivf":
                return self.ivf_db.get_index_stats()
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}",
                }
        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            return {"success": False, "message": f"Error getting index info: {str(e)}"}

    # ==================== Statistics ====================

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Database statistics
        """
        try:
            stats = self.vector_model.get_vector_statistics()

            # Add HNSW index stats
            if self.hnsw_db.hnsw_index is not None:
                stats["hnsw_index"] = self.hnsw_db.hnsw_index.get_graph_stats()

            return {"success": True, "stats": stats}
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"success": False, "message": f"Error getting stats: {str(e)}"}

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status

        Returns:
            Health status
        """
        try:
            total_vectors = self.vector_model.get_vector_count()
            index_available = (
                self.hnsw_db.hnsw_index is not None or self.ivf_db.ivf_index is not None
            )

            return {
                "status": "healthy" if total_vectors >= 0 else "unhealthy",
                "database": "connected",
                "index_available": index_available,
                "total_vectors": total_vectors,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "index_available": False,
                "total_vectors": 0,
            }
