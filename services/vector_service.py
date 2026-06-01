from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping
from models.vector_model import VectorModel, VectorPgVectorModel
from database.hnsw_database import HNSWVectorDatabase
from database.ivf_database import IVFVectorDatabase
from services.collection_service import CollectionService
from utils.bm25_index import BM25Index
from utils.hybrid_search import reciprocal_rank_fusion
from utils.product_quantization import PQIndex
from services.reranker_service import RerankerService
from config.settings import get_settings
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)
settings = get_settings()


# Module-level cache so the cross-encoder model is loaded once across requests
_reranker_instance: Optional[RerankerService] = None


def _get_reranker() -> RerankerService:
    """Return a cached RerankerService singleton."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance


class VectorService:
    """
    Service class for vector operations
    Provides business logic layer between API and database
    """

    def __init__(self, db_session: Session):
        self.db = db_session
        self.vector_model = VectorModel(db_session)
        self.pgvector_model = VectorPgVectorModel(db_session) if settings.USE_PGVECTOR else None
        self.hnsw_db = HNSWVectorDatabase(db_session)
        self.ivf_db = IVFVectorDatabase(db_session)
        self.collection_service = CollectionService(db_session)
        self.bm25_index: Optional[BM25Index] = None
        self.pq_index: Optional[PQIndex] = None

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

        if collection_id:
            scoped_ivf = self.ivf_db.get_ivf_index(collection_id)
            if scoped_ivf is not None and scoped_ivf.is_trained:
                scoped_ivf.add(vector_data, vector_id, metadata)

        if self.pq_index is not None and self.pq_index.is_trained:
            self.pq_index.add(vector_data, vector_id, metadata)

    # ==================== Vector Operations ====================

    def create_vector(
        self,
        vector_data: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new vector

        Args:
            vector_data: Vector data
            metadata: Optional metadata
            vector_id: Optional custom ID

        Returns:
            Creation result
        """
        try:
            if collection_id:
                dim_check = self.collection_service.validate_vector_dimension(
                    collection_id, vector_data
                )
                if not dim_check.get("success"):
                    return dim_check

            vector = self.vector_model.create_vector(
                vector_data, metadata, vector_id, collection_id
            )

            self._insert_into_indexes(
                vector_data, vector.vector_id, metadata, collection_id
            )

            # Index text metadata for BM25 sparse retrieval
            if metadata and isinstance(metadata, dict):
                text = metadata.get("text") or metadata.get("content") or ""
                if text:
                    self._ensure_bm25_initialized()
                    if self.bm25_index is not None:
                        self.bm25_index.add_document(text, vector.vector_id)

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

    def _ensure_bm25_initialized(self):
        """Lazy-init the BM25 index from all vectors with text metadata."""
        if self.bm25_index is not None:
            return
        self.bm25_index = BM25Index()
        try:
            vectors = self.vector_model.get_all_vectors(limit=10000)
            for v in vectors:
                meta = v.to_dict().get("metadata", {})
                if isinstance(meta, dict):
                    text = meta.get("text") or meta.get("content") or ""
                    if text:
                        self.bm25_index.add_document(text, v.vector_id)
        except Exception:
            pass

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

            # Index text metadata for BM25 sparse retrieval
            for vector_data in vectors:
                meta = vector_data.get("metadata")
                if meta and isinstance(meta, dict):
                    text = meta.get("text") or meta.get("content") or ""
                    if text:
                        self._ensure_bm25_initialized()
                        if self.bm25_index is not None:
                            self.bm25_index.add_document(text, vector_data["vector_id"])

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

    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Get a vector by ID

        Args:
            vector_id: Vector ID

        Returns:
            Vector data or error
        """
        try:
            vector = self.vector_model.get_vector(vector_id)

            if vector:
                return {"success": True, "vector": vector.to_dict()}
            else:
                return {"success": False, "message": "Vector not found"}
        except Exception as e:
            logger.error(f"Error getting vector {vector_id}: {str(e)}")
            return {"success": False, "message": f"Error getting vector: {str(e)}"}

    def get_all_vectors(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """
        Get all vectors with pagination

        Args:
            limit: Maximum number of vectors
            offset: Offset for pagination

        Returns:
            List of vectors
        """
        try:
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

    def delete_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Delete a vector

        Args:
            vector_id: Vector ID

        Returns:
            Delete result
        """
        try:
            vector = self.vector_model.get_vector(vector_id)
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

                if vector and vector.collection_id:
                    scoped_ivf = self.ivf_db.get_ivf_index(vector.collection_id)
                    if scoped_ivf is not None:
                        scoped_ivf.delete_vector(vector_id)

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
        cross_encoder_rerank: bool = False,
        rerank_top_k: Optional[int] = None,
        query_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search for similar vectors

        Args:
            query_vector: Query vector
            k: Number of results
            method: Search method
            ef_search: HNSW search parameter
            cross_encoder_rerank: Enable cross-encoder re-ranking
            rerank_top_k: Candidates to send to cross-encoder
            query_text: Original text query (required for cross-encoder)

        Returns:
            Search results
        """
        try:
            start_time = time.time()
            metric = distance_metric or "cosine"

            # Determine effective k for initial ANN search
            # If cross-encoder reranking is enabled, fetch more candidates
            # so re-ranking has a larger pool to pick from
            effective_k = k
            if cross_encoder_rerank and query_text:
                effective_k = rerank_top_k or (k * 3)
                effective_k = min(effective_k, 500)  # cap for safety

            if settings.USE_PGVECTOR and self.pgvector_model is not None:
                results = self.pgvector_model.search_vectors(
                    query_vector, effective_k, collection_id=collection_id
                )
                search_time = time.time() - start_time
                return {
                    "success": True,
                    "query_vector": query_vector,
                    "results": results,
                    "total_results": len(results),
                    "search_time": search_time,
                    "method": "pgvector",
                    "collection_id": collection_id,
                }

            # Collection-scoped search
            if collection_id:
                if method == "hnsw" and self.hnsw_db.get_hnsw_index(collection_id):
                    result = self.hnsw_db.search_hnsw(
                        query_vector,
                        effective_k,
                        ef_search,
                        filters=filters,
                        distance_metric=metric,
                        collection_id=collection_id,
                    )
                elif method == "ivf":
                    ivf_idx = self.ivf_db.get_ivf_index(collection_id)
                    if ivf_idx is not None:
                        effective_n_probes = (
                            n_probes if n_probes is not None else ivf_idx.n_probes
                        )
                        result = self.ivf_db.search(
                            query_vector,
                            effective_k,
                            use_ivf=True,
                            use_rerank=bool(use_rerank),
                            n_probes=effective_n_probes,
                            collection_id=collection_id,
                        )
                    else:
                        results = self.vector_model.search_vectors(
                            query_vector, effective_k, metric, filters, collection_id
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
                else:
                    results = self.vector_model.search_vectors(
                        query_vector, effective_k, metric, filters, collection_id
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
                        query_vector, effective_k, filters=filters, distance_metric=metric
                    )
                else:
                    result = self.hnsw_db.search_hnsw(
                        query_vector, effective_k, ef_search, filters=filters, distance_metric=metric
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
                    effective_k,
                    use_ivf=True,
                    use_rerank=use_rerank,
                    n_probes=effective_n_probes,
                )
            elif method == "brute":
                result = self.hnsw_db.search_brute_force(
                    query_vector, effective_k, filters=filters, distance_metric=metric
                )
            elif method == "pq":
                if self.pq_index is not None and self.pq_index.is_trained:
                    results = self.pq_index.search(
                        query_vector,
                        k=effective_k,
                        rerank=bool(use_rerank) if use_rerank is not None else False,
                    )
                    result = {
                        "success": True,
                        "results": results,
                        "total_results": len(results),
                        "method": "pq",
                    }
                else:
                    return {
                        "success": False,
                        "message": "PQ index not available. Create one first with method='pq'.",
                    }
            elif method == "hybrid":
                result = self._hybrid_search(
                    query_vector=query_vector,
                    query_text=filters.get("query_text", "") if filters else "",
                    k=k,
                    ef_search=ef_search,
                    filters=filters,
                    distance_metric=metric,
                )
            else:
                return {"success": False, "message": f"Unknown search method: {method}"}

            search_time = time.time() - start_time

            # Combine timing
            if result.get("success"):
                result["search_time"] = search_time

            if filters and any(isinstance(v, dict) for v in filters.values()):
                from services.metadata_filter import MetadataFilter
                result["results"] = MetadataFilter.post_filter(result.get("results", []), filters)
                result["total_results"] = len(result["results"])

            # Cross-encoder re-ranking after initial ANN search
            if (
                cross_encoder_rerank
                and query_text
                and result.get("success")
                and result.get("results")
            ):
                try:
                    reranker = _get_reranker()
                    reranked = reranker.rerank(
                        query=query_text,
                        results=result["results"],
                        top_k=k,
                    )
                    result["results"] = reranked
                    result["total_results"] = len(reranked)
                    result["method"] = result.get("method", method) + "+rerank"
                    result["cross_encoder_reranked"] = True
                except Exception as rerank_err:
                    logger.warning("Cross-encoder re-ranking failed: %s", rerank_err)
                    result["cross_encoder_reranked"] = False
                    result["rerank_error"] = str(rerank_err)

            if result.get("success") and not filters and not cross_encoder_rerank:
                try:
                    cache.cache_search(query_vector, k, collection_id or "", result)
                except Exception:
                    pass

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
            elif method == "pq":
                return self._create_pq_index(collection_id=collection_id)
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
            elif method == "pq":
                return self._save_pq_index(collection_id=collection_id)
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
            elif method == "pq":
                return self._load_pq_index(collection_id=collection_id)
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
            elif method == "pq":
                return self._get_pq_stats()
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}",
                }
        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            return {"success": False, "message": f"Error getting index info: {str(e)}"}

    # ==================== PQ Index Operations ====================

    def _create_pq_index(
        self,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and train a PQ index from existing vectors."""
        try:
            # Get all vectors for training
            vecs = self.vector_model.get_all_vectors(limit=10000)
            if not vecs:
                return {"success": False, "message": "No vectors to train PQ index on. Insert vectors first."}

            # Determine dimension from first vector
            first = vecs[0].to_dict()
            vector_data = first.get("vector", [])
            if not vector_data:
                return {"success": False, "message": "Vectors have no data."}

            dim = len(vector_data)
            # Collect all vectors for training
            training_vectors = []
            for v in vecs:
                data = v.to_dict().get("vector", [])
                if len(data) == dim:
                    training_vectors.append(data)

            # Determine M: try to find divisors of dim that give good compression
            # Good M values: divisors of dim close to dim // 16 or dim // 8
            M = max(1, dim // 16)
            while dim % M != 0:
                M -= 1
            M = max(1, min(M, 64))

            self.pq_index = PQIndex(M=M, k_sub=256, n_iter=20, distance_metric="cosine", verbose=True)
            self.pq_index.train(training_vectors)

            # Insert all existing vectors into the PQ index
            for v in vecs:
                data = v.to_dict()
                vid = data.get("vector_id", "")
                vec = data.get("vector", [])
                meta = data.get("metadata", {})
                if len(vec) == dim:
                    self.pq_index.add(vec, vid, meta)

            stats = self.pq_index.get_stats()
            logger.info(
                "PQ index created: D=%d, M=%d, vectors=%d, ratio=%.1f:1",
                dim, M, len(vecs), stats["compression_ratio"],
            )

            return {
                "success": True,
                "message": f"PQ index created and trained on {len(vecs)} vectors",
                "stats": stats,
            }
        except Exception as e:
            logger.error(f"Error creating PQ index: {str(e)}")
            return {"success": False, "message": f"Error creating PQ index: {str(e)}"}

    def _save_pq_index(
        self, collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save PQ index to disk."""
        if self.pq_index is None:
            return {"success": False, "message": "PQ index not initialized"}
        try:
            import os
            from utils.index_paths import get_pq_path
            path = get_pq_path(collection_id=collection_id)
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            self.pq_index.save(path)
            return {"success": True, "message": f"PQ index saved to {path}"}
        except Exception as e:
            logger.error(f"Error saving PQ index: {str(e)}")
            return {"success": False, "message": f"Error saving PQ index: {str(e)}"}

    def _load_pq_index(
        self, collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load PQ index from disk."""
        try:
            from utils.index_paths import get_pq_path
            path = get_pq_path(collection_id=collection_id)
            self.pq_index = PQIndex.load(path)
            return {
                "success": True,
                "message": f"PQ index loaded from {path}",
                "stats": self.pq_index.get_stats(),
            }
        except Exception as e:
            logger.error(f"Error loading PQ index: {str(e)}")
            return {"success": False, "message": f"Error loading PQ index: {str(e)}"}

    def _get_pq_stats(self) -> Dict[str, Any]:
        if self.pq_index is None:
            return {"success": False, "message": "PQ index not initialized"}
        return {"success": True, "stats": self.pq_index.get_stats()}

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

    def _hybrid_search(
        self,
        query_vector: List[float],
        query_text: str,
        k: int = 5,
        ef_search: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Hybrid search: fuse dense vector search + sparse BM25 via RRF.
        """
        import time
        start = time.time()

        # 1. Dense search using best available index
        dense_method = "hnsw" if self.hnsw_db.hnsw_index is not None else "brute"
        dense_result = self.search_vectors(
            query_vector=query_vector,
            k=k * 2,  # get more candidates for fusion
            method=dense_method,
            ef_search=ef_search,
            filters=filters,
            distance_metric=distance_metric,
        )
        dense_results = dense_result.get("results", [])

        # 2. Sparse search with BM25
        sparse_results = []
        if query_text and self.bm25_index is not None:
            try:
                sparse_raw = self.bm25_index.search(query_text, k=k * 2)
                # bm25 returns list of (doc_id, score) tuples
                for doc_id, score in sparse_raw:
                    sparse_results.append({"vector_id": doc_id, "distance": 1.0 - min(score / 10.0, 1.0)})
            except Exception:
                pass

        # 3. Fusion via Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_n=k)

        search_time = time.time() - start
        return {
            "success": True,
            "results": fused,
            "total_results": len(fused),
            "search_time": search_time,
            "method": "hybrid",
            "dense_method": dense_method,
            "has_bm25": self.bm25_index is not None and bool(query_text),
        }

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
