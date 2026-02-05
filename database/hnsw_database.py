import os
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping
from models.vector_model import VectorModel
from utils.hnsw_index import HNSWIndex
from utils.ivf_index import IVFIndex
import numpy as np
from datetime import datetime
import json
import time

class HNSWVectorDatabase:
    """
    Vector Database with HNSW indexing capabilities
    
    Combines PostgreSQL for persistent storage with HNSW for fast search
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.vector_model = VectorModel(db_session)
        
        # Indexes
        self.hnsw_index = None
        self.ivf_index = None
        
        # Index paths
        self.hnsw_index_path = "hnsw_index_data.json"
        self.ivf_index_path = "ivf_index_data.json"
        
    def insert_vector(self, vector_data: List[float], metadata: Dict[str, Any] = None, 
                     vector_id: str = None) -> Dict[str, Any]:
        """
        Insert a vector into the database
        """
        try:
            vector = self.vector_model.create_vector(vector_data, metadata, vector_id)
            
            # Add to HNSW index if it exists
            if self.hnsw_index is not None:
                self.hnsw_index.insert(
                    vector_data, 
                    vector.vector_id, 
                    metadata
                )
            
            return {
                "success": True,
                "message": "Vector inserted successfully",
                "vector_id": vector.vector_id,
                "vector": vector.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error inserting vector: {str(e)}"
            }
    
    def insert_vector_batch(self, vectors: List[Dict[str, Any]], batch_name: str = None,
                           description: str = None) -> Dict[str, Any]:
        """
        Insert a batch of vectors
        """
        try:
            result = self.vector_model.create_vector_batch(vectors, batch_name, description)
            
            # Add to HNSW index if it exists
            if self.hnsw_index is not None:
                for vector_data in vectors:
                    self.hnsw_index.insert(
                        vector_data["vector"],
                        vector_data["vector_id"],
                        vector_data.get("metadata")
                    )
            
            return {
                "success": True,
                "message": "Vector batch inserted successfully",
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error inserting vector batch: {str(e)}"
            }
    
    def create_hnsw_index(self, m: int = 16, m0: int = None, 
                         ef_construction: int = 200) -> Dict[str, Any]:
        """
        Create an HNSW index
        
        Args:
            m: Number of neighbors per node
            m0: Number of neighbors in layer 0
            ef_construction: Construction parameter
            
        Returns:
            Index creation result
        """
        try:
            # Get all vectors from database
            vectors_data = self.vector_model.get_all_vectors()
            
            if len(vectors_data) == 0:
                return {
                    "success": False,
                    "message": "No vectors found to create index"
                }
            
            # Create HNSW index
            self.hnsw_index = HNSWIndex(
                m=m, 
                m0=m0, 
                ef_construction=ef_construction
            )
            
            # Insert all vectors
            for vector in vectors_data:
                self.hnsw_index.insert(
                    vector.vector_data,
                    vector.vector_id,
                    vector.metadata
                )
            
            # Get index statistics
            stats = self.hnsw_index.get_graph_stats()
            
            return {
                "success": True,
                "message": f"HNSW Index created",
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating HNSW index: {str(e)}"
            }
    
    def save_hnsw_index(self) -> Dict[str, Any]:
        """
        Save HNSW index to disk
        
        Returns:
            Save result
        """
        try:
            if self.hnsw_index is None:
                return {
                    "success": False,
                    "message": "No HNSW index to save"
                }
            
            self.hnsw_index.save(self.hnsw_index_path)
            
            return {
                "success": True,
                "message": f"HNSW Index saved to {self.hnsw_index_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error saving HNSW index: {str(e)}"
            }
    
    def load_hnsw_index(self) -> Dict[str, Any]:
        """
        Load HNSW index from disk
        
        Returns:
            Load result
        """
        try:
            if not os.path.exists(self.hnsw_index_path):
                return {
                    "success": False,
                    "message": f"HNSW Index file not found: {self.hnsw_index_path}"
                }
            
            self.hnsw_index = HNSWIndex()
            self.hnsw_index.load(self.hnsw_index_path)
            
            stats = self.hnsw_index.get_graph_stats()
            
            return {
                "success": True,
                "message": "HNSW Index loaded successfully",
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading HNSW index: {str(e)}"
            }
    
    def search_hnsw(self, query_vector: List[float], k: int = 5, 
                   ef_search: int = None) -> Dict[str, Any]:
        """
        Search using HNSW index
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            ef_search: Search parameter (higher = better recall)
            
        Returns:
            Search results
        """
        try:
            if self.hnsw_index is None:
                return {
                    "success": False,
                    "message": "No HNSW index created yet"
                }
            
            start_time = time.time()
            results = self.hnsw_index.search(query_vector, k, ef=ef_search)
            search_time = time.time() - start_time
            
            return {
                "success": True,
                "query_vector": query_vector,
                "results": results,
                "total_results": len(results),
                "search_time": search_time,
                "method": "hnsw"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during HNSW search: {str(e)}"
            }
    
    def search(self, query_vector: List[float], k: int = 5, 
              method: str = 'hnsw', ef_search: int = None) -> Dict[str, Any]:
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
        if method == 'hnsw':
            return self.search_hnsw(query_vector, k, ef_search)
        elif method == 'brute':
            return self.search_brute_force(query_vector, k)
        else:
            return {
                "success": False,
                "message": f"Unknown search method: {method}"
            }
    
    def search_brute_force(self, query_vector: List[float], k: int = 5) -> Dict[str, Any]:
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
            results = self.vector_model.search_vectors(query_vector, k, 'cosine')
            search_time = time.time() - start_time
            
            return {
                "success": True,
                "query_vector": query_vector,
                "results": results,
                "total_results": len(results),
                "search_time": search_time,
                "method": "brute_force"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during brute force search: {str(e)}"
            }
    
    def compare_search_methods(self, query_vector: List[float], k: int = 5) -> Dict[str, Any]:
        """
        Compare different search methods
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            Comparison results
        """
        results = {
            "query_vector": query_vector,
            "methods": {}
        }
        
        # HNSW Search
        if self.hnsw_index is not None:
            hnsw_result = self.search_hnsw(query_vector, k)
            results["methods"]["hnsw"] = {
                "results": hnsw_result.get("results", []),
                "time": hnsw_result.get("search_time", 0),
                "count": hnsw_result.get("total_results", 0)
            }
        
        # Brute Force Search
        brute_result = self.search_brute_force(query_vector, k)
        results["methods"]["brute_force"] = {
            "results": brute_result.get("results", []),
            "time": brute_result.get("search_time", 0),
            "count": brute_result.get("total_results", 0)
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
                    "message": f"Vector {vector_id} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "message": "Vector not found"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting vector: {str(e)}"
            }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database and index statistics
        """
        try:
            stats = self.vector_model.get_vector_statistics()
            
            # Add HNSW stats
            if self.hnsw_index is not None:
                stats["hnsw_index"] = self.hnsw_index.get_graph_stats()
            
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting stats: {str(e)}"
            }
    
    def get_hnsw_index_info(self) -> Dict[str, Any]:
        """
        Get HNSW index information
        """
        try:
            if self.hnsw_index is None:
                return {
                    "success": False,
                    "message": "No HNSW index created yet"
                }
            
            stats = self.hnsw_index.get_graph_stats()
            
            return {
                "success": True,
                "index_info": {
                    "is_indexed": True,
                    "total_nodes": stats["total_nodes"],
                    "total_edges": stats["total_edges"],
                    "avg_connections": stats["avg_connections"],
                    "max_level": stats["max_level"],
                    "level_distribution": stats["level_distribution"]
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting index info: {str(e)}"
            }
    
    def rebuild_hnsw_index(self, m: int = 16, m0: int = None, 
                          ef_construction: int = 200) -> Dict[str, Any]:
        """
        Rebuild HNSW index
        
        Args:
            m: Number of neighbors per node
            m0: Number of neighbors in layer 0
            ef_construction: Construction parameter
            
        Returns:
            Rebuild result
        """
        try:
            self.hnsw_index = None
            
            return self.create_hnsw_index(m, m0, ef_construction)
        except Exception as e:
            return {
                "success": False,
                "message": f"Error rebuilding HNSW index: {str(e)}"
            }
