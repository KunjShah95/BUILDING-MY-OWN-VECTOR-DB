from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping
from models.vector_model import VectorModel
from utils.ivf_index import IVFIndex
from utils.clustering import KMeans
import numpy as np
from datetime import datetime
import json

class IVFVectorDatabase:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.vector_model = VectorModel(db_session)
        self.ivf_index = None
        self.index_path = "index_data.json"
    
    def clear_database(self) -> Dict[str, Any]:
        """
        Clear all vectors from the database
        """
        try:
            result = self.vector_model.clear_all_vectors()
            # Reset index
            self.ivf_index = None
            return result
        except Exception as e:
            return {
                "success": False,
                "message": f"Error clearing database: {str(e)}"
            }
        
    def insert_vector(self, vector_data: List[float], metadata: Dict[str, Any] = None, 
                     vector_id: str = None) -> Dict[str, Any]:
        """
        Insert a vector into the database
        """
        try:
            vector = self.vector_model.create_vector(vector_data, metadata, vector_id)
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
    
    def create_ivf_index(self, n_clusters: int = 100, n_probes: int = 10, 
                        force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Create an IVF index
        
        Args:
            n_clusters: Number of clusters
            n_probes: Number of clusters to probe during search
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            Index creation result
        """
        try:
            # Check if index already exists
            if self.ivf_index is not None and not force_rebuild:
                return {
                    "success": False,
                    "message": "IVF Index already exists. Use force_rebuild=True to rebuild."
                }
            
            # Get all vectors from database
            vectors_data = self.vector_model.get_all_vectors()
            
            if len(vectors_data) == 0:
                return {
                    "success": False,
                    "message": "No vectors found to create index"
                }
            
            # Prepare data for indexing
            vectors = [vector.vector_data for vector in vectors_data]
            vector_ids = [vector.vector_id for vector in vectors_data]
            
            # Create IVF index
            self.ivf_index = IVFIndex(n_clusters=n_clusters, n_probes=n_probes)
            
            # Train on all vectors
            self.ivf_index.train(vectors)
            
            # Add all vectors to index
            for vector, vector_id in zip(vectors, vector_ids):
                original_vector = self.vector_model.get_vector(vector_id)
                metadata = original_vector.metadata if original_vector else None
                self.ivf_index.add(vector, vector_id, metadata)
            
            # Save index to disk
            self.ivf_index.save(self.index_path)
            
            # Get index statistics
            stats = self.ivf_index.get_stats()
            
            return {
                "success": True,
                "message": f"IVF Index created with {n_clusters} clusters",
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating IVF index: {str(e)}"
            }
    
    def load_ivf_index(self) -> Dict[str, Any]:
        """
        Load IVF index from disk
        
        Returns:
            Load result
        """
        try:
            if not os.path.exists(self.index_path):
                return {
                    "success": False,
                    "message": f"Index file not found: {self.index_path}"
                }
            
            self.ivf_index = IVFIndex()
            self.ivf_index.load(self.index_path)
            
            stats = self.ivf_index.get_stats()
            
            return {
                "success": True,
                "message": "IVF Index loaded successfully",
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading IVF index: {str(e)}"
            }
    
    def save_ivf_index(self) -> Dict[str, Any]:
        """
        Save IVF index to disk
        
        Returns:
            Save result
        """
        try:
            if self.ivf_index is None:
                return {
                    "success": False,
                    "message": "No IVF index to save"
                }
            
            self.ivf_index.save(self.index_path)
            
            return {
                "success": True,
                "message": f"IVF Index saved to {self.index_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error saving IVF index: {str(e)}"
            }
    
    def search(self, query_vector: List[float], k: int = 5, 
              use_ivf: bool = True, use_rerank: bool = True,
              n_probes: int = 10) -> Dict[str, Any]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            use_ivf: Whether to use IVF index (if available)
            use_rerank: Whether to use reranking for better accuracy
            n_probes: Number of clusters to probe
            
        Returns:
            Search results
        """
        try:
            if use_ivf and self.ivf_index is not None:
                # Use IVF search
                if n_probes != self.ivf_index.n_probes:
                    self.ivf_index.n_probes = n_probes
                
                if use_rerank:
                    results = self.ivf_index.search_with_rerank(query_vector, k)
                else:
                    results = self.ivf_index.search(query_vector, k)
                
                method = "ivf_with_rerank" if use_rerank else "ivf"
            else:
                # Fall back to brute force search
                results = self.vector_model.search_vectors(query_vector, k, 'cosine')
                method = "brute_force"
            
            return {
                "success": True,
                "query_vector": query_vector,
                "results": results,
                "total_results": len(results),
                "method": method
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during search: {str(e)}"
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
        import time
        
        results = {
            "query_vector": query_vector,
            "methods": {}
        }
        
        # IVF Search
        if self.ivf_index is not None:
            start_time = time.time()
            ivf_results = self.ivf_index.search(query_vector, k)
            ivf_time = time.time() - start_time
            
            results["methods"]["ivf"] = {
                "results": ivf_results,
                "time": ivf_time,
                "count": len(ivf_results)
            }
        
        # Brute Force Search
        start_time = time.time()
        brute_force_results = self.vector_model.search_vectors(query_vector, k, 'cosine')
        brute_force_time = time.time() - start_time
        
        results["methods"]["brute_force"] = {
            "results": brute_force_results,
            "time": brute_force_time,
            "count": len(brute_force_results)
        }
        
        return results
    
    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Get a specific vector by ID
        """
        try:
            vector = self.vector_model.get_vector(vector_id)
            if vector:
                return {
                    "success": True,
                    "vector": vector.to_dict()
                }
            else:
                return {
                    "success": False,
                    "message": "Vector not found"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving vector: {str(e)}"
            }
    
    def get_all_vectors(self, limit: int = 1000, offset: int = 0) -> Dict[str, Any]:
        """
        Get all vectors in the database with pagination
        """
        try:
            vectors = self.vector_model.get_all_vectors(limit, offset)
            return {
                "success": True,
                "vectors": [vector.to_dict() for vector in vectors],
                "count": len(vectors),
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving vectors: {str(e)}"
            }
    
    def update_vector(self, vector_id: str, metadata: Dict[str, Any] = None, 
                     vector_data: List[float] = None) -> Dict[str, Any]:
        """
        Update an existing vector
        """
        try:
            success = self.vector_model.update_vector(vector_id, metadata, vector_data)
            if success:
                return {
                    "success": True,
                    "message": "Vector updated successfully"
                }
            else:
                return {
                    "success": False,
                    "message": "Vector not found"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error updating vector: {str(e)}"
            }
    
    def delete_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Delete a vector by ID
        """
        try:
            success = self.vector_model.delete_vector(vector_id)
            if success and self.ivf_index is not None:
                self.ivf_index.delete_vector(vector_id)
                self.save_ivf_index()
            
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
        Get database statistics
        """
        try:
            stats = self.vector_model.get_vector_statistics()
            
            # Add IVF index stats
            if self.ivf_index is not None:
                ivf_stats = self.ivf_index.get_stats()
                stats["ivf_index"] = ivf_stats
            
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting stats: {str(e)}"
            }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get IVF index statistics
        """
        try:
            if self.ivf_index is None:
                return {
                    "success": False,
                    "message": "No IVF index created yet"
                }
            
            stats = self.ivf_index.get_stats()
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting index stats: {str(e)}"
            }
    
    def rebuild_ivf_index(self, n_clusters: int = 100, n_probes: int = 10) -> Dict[str, Any]:
        """
        Rebuild the IVF index
        
        Args:
            n_clusters: Number of clusters
            n_probes: Number of clusters to probe
            
        Returns:
            Rebuild result
        """
        try:
            # Clear existing index
            self.ivf_index = None
            
            # Create new index
            return self.create_ivf_index(n_clusters=n_clusters, n_probes=n_probes, force_rebuild=True)
        except Exception as e:
            return {
                "success": False,
                "message": f"Error rebuilding IVF index: {str(e)}"
            }
