import os
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping
from models.vector_model import VectorModel
from utils.hnsw_index import HNSWIndex
from utils.ivf_index import IVFIndex
from utils.index_paths import ensure_index_dir, get_hnsw_path
import numpy as np
from datetime import datetime
import json
import time


class HNSWVectorDatabase:
    """
    Vector Database with HNSW indexing capabilities

    Combines PostgreSQL for persistent storage with HNSW for fast search
    """

    _GLOBAL_KEY = "__global__"

    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.vector_model = VectorModel(db_session)

        # Indexes (global + per-collection)
        self.hnsw_index = None
        self.ivf_index = None
        self._scoped_hnsw: Dict[str, HNSWIndex] = {}

        # Index paths (global default)
        self.hnsw_index_path = get_hnsw_path()
        self.ivf_index_path = "ivf_index_data.json"

    def _scope_key(self, collection_id: Optional[str] = None) -> str:
        return collection_id if collection_id else self._GLOBAL_KEY

    def get_hnsw_index(self, collection_id: Optional[str] = None) -> Optional[HNSWIndex]:
        """Return in-memory HNSW index for global or collection scope."""
        return self._scoped_hnsw.get(self._scope_key(collection_id))

    def insert_vector(
        self,
        vector_data: List[float],
        metadata: Dict[str, Any] = None,
        vector_id: str = None,
    ) -> Dict[str, Any]:
        """
        Insert a vector into the database
        """
        try:
            vector = self.vector_model.create_vector(vector_data, metadata, vector_id)

            # Add to HNSW index if it exists
            if self.hnsw_index is not None:
                self.hnsw_index.insert(vector_data, vector.vector_id, metadata)

            return {
                "success": True,
                "message": "Vector inserted successfully",
                "vector_id": vector.vector_id,
                "vector": vector.to_dict(),
            }
        except Exception as e:
            return {"success": False, "message": f"Error inserting vector: {str(e)}"}

    def insert_vector_batch(
        self,
        vectors: List[Dict[str, Any]],
        batch_name: str = None,
        description: str = None,
    ) -> Dict[str, Any]:
        """
        Insert a batch of vectors
        """
        try:
            result = self.vector_model.create_vector_batch(
                vectors, batch_name, description
            )

            # Add to HNSW index if it exists
            if self.hnsw_index is not None:
                for vector_data in vectors:
                    self.hnsw_index.insert(
                        vector_data["vector"],
                        vector_data["vector_id"],
                        vector_data.get("metadata"),
                    )

            return {
                "success": True,
                "message": "Vector batch inserted successfully",
                "result": result,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error inserting vector batch: {str(e)}",
            }

    def create_hnsw_index(
        self,
        m: int = None,
        m0: int = None,
        ef_construction: int = None,
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an HNSW index with optimized parameters

        Args:
            m: Number of neighbors per node (default from settings)
            m0: Number of neighbors in layer 0 (default: 2*m)
            ef_construction: Construction parameter (default from settings)
            collection_id: Optional collection scope for the index

        Returns:
            Index creation result
        """
        try:
            from config.settings import get_settings

            settings = get_settings()

            if m is None:
                m = settings.DEFAULT_M
            if ef_construction is None:
                ef_construction = settings.DEFAULT_EF_CONSTRUCTION
            if m0 is None:
                m0 = settings.DEFAULT_M0

            if collection_id:
                vectors_data = self.vector_model.get_vectors_by_collection(collection_id)
            else:
                vectors_data = self.vector_model.get_all_vectors(limit=None)

            if len(vectors_data) == 0:
                scope = f"collection '{collection_id}'" if collection_id else "database"
                return {
                    "success": False,
                    "message": f"No vectors found in {scope} to create index",
                }

            index = HNSWIndex(
                m=m,
                m0=m0,
                ef_construction=ef_construction,
                distance_metric=settings.DEFAULT_DISTANCE_METRIC,
            )

            for vector in vectors_data:
                index.insert(
                    vector.vector_data, vector.vector_id, vector.meta_data
                )

            key = self._scope_key(collection_id)
            self._scoped_hnsw[key] = index
            if not collection_id:
                self.hnsw_index = index

            self.hnsw_index_path = get_hnsw_path(collection_id)
            stats = index.get_graph_stats()

            scope_msg = f" for collection '{collection_id}'" if collection_id else ""
            return {
                "success": True,
                "message": f"HNSW Index created{scope_msg} with m={m}, ef_construction={ef_construction}",
                "stats": stats,
                "parameters": {"m": m, "m0": m0, "ef_construction": ef_construction},
                "collection_id": collection_id,
                "index_path": self.hnsw_index_path,
            }
        except Exception as e:
            return {"success": False, "message": f"Error creating HNSW index: {str(e)}"}

    def save_hnsw_index(self, collection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Save HNSW index to disk

        Returns:
            Save result
        """
        try:
            index = self.get_hnsw_index(collection_id)
            if index is None:
                return {"success": False, "message": "No HNSW index to save"}

            path = get_hnsw_path(collection_id)
            ensure_index_dir(collection_id)
            index.save(path)

            return {
                "success": True,
                "message": f"HNSW Index saved to {path}",
                "collection_id": collection_id,
                "index_path": path,
            }
        except Exception as e:
            return {"success": False, "message": f"Error saving HNSW index: {str(e)}"}

    def load_hnsw_index(self, collection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load HNSW index from disk

        Returns:
            Load result
        """
        try:
            path = get_hnsw_path(collection_id)
            if not os.path.exists(path):
                return {
                    "success": False,
                    "message": f"HNSW Index file not found: {path}",
                }

            from config.settings import get_settings

            settings = get_settings()
            index = HNSWIndex(distance_metric=settings.DEFAULT_DISTANCE_METRIC)
            index.load(path)

            key = self._scope_key(collection_id)
            self._scoped_hnsw[key] = index
            if not collection_id:
                self.hnsw_index = index
            self.hnsw_index_path = path

            stats = index.get_graph_stats()

            return {
                "success": True,
                "message": "HNSW Index loaded successfully",
                "stats": stats,
                "collection_id": collection_id,
                "index_path": path,
            }
        except Exception as e:
            return {"success": False, "message": f"Error loading HNSW index: {str(e)}"}

    def _filter_search_results(
        self,
        results: List[Dict[str, Any]],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not filters:
            return results[:k]

        vector_ids = [item.get("vector_id") for item in results if item.get("vector_id")]
        if not vector_ids:
            return []

        vectors = (
            self.db_session.query(Vector)
            .filter(Vector.vector_id.in_(vector_ids))
            .all()
        )
        vector_map = {vector.vector_id: vector for vector in vectors}

        filtered = []
        for item in results:
            vector_id = item.get("vector_id")
            if not vector_id:
                continue
            vector = vector_map.get(vector_id)
            if vector and self.vector_model._metadata_matches(vector.meta_data, filters):
                if "metadata" not in item:
                    item["metadata"] = vector.meta_data
                filtered.append(item)
            if len(filtered) >= k:
                break
        return filtered

    def search_hnsw(
        self,
        query_vector: List[float],
        k: int = 5,
        ef_search: int = None,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
        collection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search using HNSW index with optimized parameters

        Args:
            query_vector: Query vector
            k: Number of results to return
            ef_search: Search parameter (higher = better recall, lower = faster)

        Returns:
            Search results with timing information
        """
        try:
            index = self.get_hnsw_index(collection_id)
            if index is None:
                scope = f"collection '{collection_id}'" if collection_id else "global"
                return {
                    "success": False,
                    "message": f"No HNSW index for {scope}. Create or load one first.",
                }

            from config.settings import get_settings

            if ef_search is None:
                ef_search = get_settings().DEFAULT_EF_SEARCH

            start_time = time.time()
            fetch_k = k * 20 if filters else k
            raw_results = index.search(query_vector, fetch_k, ef=ef_search)
            results = self._filter_search_results(raw_results, k, filters)
            search_time = time.time() - start_time

            return {
                "success": True,
                "query_vector": query_vector,
                "results": results,
                "total_results": len(results),
                "search_time": search_time,
                "ef_search": ef_search,
                "method": "hnsw",
                "collection_id": collection_id,
            }
        except Exception as e:
            return {"success": False, "message": f"Error during HNSW search: {str(e)}"}

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        method: str = "hnsw",
        ef_search: int = None,
    ) -> Dict[str, Any]:
        """
        Search using specified method

        Args:
            query_vector: Query vector
            k: Number of results to return
            method: Search method ('hnsw', 'ivf', 'brute')
            ef_search: HNSW search parameter

        Returns:
            Search results
        """
        if method == "hnsw":
            # If no HNSW index, auto-fallback to brute force
            if self.hnsw_index is None:
                return self.search_brute_force(query_vector, k)
            return self.search_hnsw(query_vector, k, ef_search)
        elif method == "brute":
            return self.search_brute_force(query_vector, k)
        else:
            return {"success": False, "message": f"Unknown search method: {method}"}

    def search_brute_force(
        self,
        query_vector: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
    ) -> Dict[str, Any]:
        """
        Search using brute force (fallback)

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            Search results
        """
        try:
            start_time = time.time()
            results = self.vector_model.search_vectors(
                query_vector, k, distance_metric, filters
            )
            search_time = time.time() - start_time

            return {
                "success": True,
                "query_vector": query_vector,
                "results": results,
                "total_results": len(results),
                "search_time": search_time,
                "method": "brute_force",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during brute force search: {str(e)}",
            }

    def compare_search_methods(
        self, query_vector: List[float], k: int = 5
    ) -> Dict[str, Any]:
        """
        Compare different search methods

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            Comparison results
        """
        results = {"query_vector": query_vector, "methods": {}}

        # HNSW Search
        if self.hnsw_index is not None:
            hnsw_result = self.search_hnsw(query_vector, k)
            results["methods"]["hnsw"] = {
                "results": hnsw_result.get("results", []),
                "time": hnsw_result.get("search_time", 0),
                "count": hnsw_result.get("total_results", 0),
            }

        # Brute Force Search
        brute_result = self.search_brute_force(query_vector, k)
        results["methods"]["brute_force"] = {
            "results": brute_result.get("results", []),
            "time": brute_result.get("search_time", 0),
            "count": brute_result.get("total_results", 0),
        }

        return results

    def delete_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Delete a vector from database and index

        Args:
            vector_id: Vector ID to delete

        Returns:
            Delete result
        """
        try:
            success = self.vector_model.delete_vector(vector_id)

            # Delete from HNSW index
            if success and self.hnsw_index is not None:
                self.hnsw_index.delete(vector_id)

            if success:
                return {
                    "success": True,
                    "message": f"Vector {vector_id} deleted successfully",
                }
            else:
                return {"success": False, "message": "Vector not found"}
        except Exception as e:
            return {"success": False, "message": f"Error deleting vector: {str(e)}"}

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database and index statistics
        """
        try:
            stats = self.vector_model.get_vector_statistics()

            # Add HNSW stats
            if self.hnsw_index is not None:
                stats["hnsw_index"] = self.hnsw_index.get_graph_stats()

            return {"success": True, "stats": stats}
        except Exception as e:
            return {"success": False, "message": f"Error getting stats: {str(e)}"}

    def get_hnsw_index_info(self) -> Dict[str, Any]:
        """
        Get HNSW index information
        """
        try:
            if self.hnsw_index is None:
                return {"success": False, "message": "No HNSW index created yet"}

            stats = self.hnsw_index.get_graph_stats()

            return {
                "success": True,
                "index_info": {
                    "is_indexed": True,
                    "total_nodes": stats["total_nodes"],
                    "total_edges": stats["total_edges"],
                    "avg_connections": stats["avg_connections"],
                    "max_level": stats["max_level"],
                    "level_distribution": stats["level_distribution"],
                },
            }
        except Exception as e:
            return {"success": False, "message": f"Error getting index info: {str(e)}"}

    def batch_search_hnsw(
        self, query_vectors: List[List[float]], k: int = 5, ef_search: int = None
    ) -> Dict[str, Any]:
        """
        Perform batch searches using HNSW index

        Args:
            query_vectors: List of query vectors
            k: Number of results to return per query
            ef_search: Search parameter

        Returns:
            Batch search results with timing
        """
        try:
            if self.hnsw_index is None:
                return {"success": False, "message": "No HNSW index created yet"}

            from config.settings import get_settings

            if ef_search is None:
                ef_search = get_settings().DEFAULT_EF_SEARCH

            start_time = time.time()
            batch_results = []

            for query_vec in query_vectors:
                results = self.hnsw_index.search(query_vec, k, ef=ef_search)
                batch_results.append(results)

            batch_time = time.time() - start_time

            return {
                "success": True,
                "total_queries": len(query_vectors),
                "results": batch_results,
                "batch_time": batch_time,
                "avg_query_time": batch_time / len(query_vectors)
                if query_vectors
                else 0,
                "queries_per_second": len(query_vectors) / batch_time
                if batch_time > 0
                else 0,
                "ef_search": ef_search,
                "method": "hnsw_batch",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during batch HNSW search: {str(e)}",
            }

    def rebuild_hnsw_index(
        self, m: int = None, m0: int = None, ef_construction: int = None
    ) -> Dict[str, Any]:
        """
        Rebuild HNSW index with optimized parameters

        Args:
            m: Number of neighbors per node (default from settings)
            m0: Number of neighbors in layer 0
            ef_construction: Construction parameter (default from settings)

        Returns:
            Rebuild result
        """
        try:
            self.hnsw_index = None

            return self.create_hnsw_index(m, m0, ef_construction)
        except Exception as e:
            return {
                "success": False,
                "message": f"Error rebuilding HNSW index: {str(e)}",
            }
