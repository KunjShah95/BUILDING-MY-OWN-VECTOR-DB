from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping
from models.vector_model import VectorModel
from utils.clustering import VectorIndexer
import numpy as np
from datetime import datetime

class VectorDatabase:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.vector_model = VectorModel(db_session)
        self.indexer = None
        self.is_indexed = False
        
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
    
    def create_index(self, k: int = 100, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Create an index using K-Means clustering
        
        Args:
            k: Number of clusters
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            Index creation result
        """
        try:
            # Check if index already exists
            if self.is_indexed and not force_rebuild:
                return {
                    "success": False,
                    "message": "Index already exists. Use force_rebuild=True to rebuild."
                }
            
            # Get all vectors from database
            vectors_data = self.vector_model.get_all_vectors()
            
            if len(vectors_data) == 0:
                return {
                    "success": False,
                    "message": "No vectors found to create index"
                }
            
            # Prepare data for clustering
            vector_arrays = [vector.vector_data for vector in vectors_data]
            vector_ids = [vector.vector_id for vector in vectors_data]
            
            # Create indexer
            self.indexer = VectorIndexer(k=k)
            self.indexer.fit(vector_arrays, vector_ids)
            
            self.is_indexed = True
            
            # Get cluster information
            cluster_info = self.indexer.get_cluster_info()
            
            return {
                "success": True,
                "message": f"Index created with {k} clusters",
                "cluster_info": cluster_info,
                "total_vectors": len(vectors_data)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error creating index: {str(e)}"
            }
    
    def search(self, query_vector: List[float], k: int = 5, 
              distance_metric: str = 'cosine', filters: Dict[str, Any] = None,
              use_index: bool = True, n_probes: int = 10) -> Dict[str, Any]:
        """
        Search for similar vectors using brute force or indexed approach
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            distance_metric: Distance metric to use
            filters: Metadata filters (not implemented in this version)
            use_index: Whether to use the index (if available)
            n_probes: Number of clusters to probe for approximate search
            
        Returns:
            Search results
        """
        try:
            query_vector = np.array(query_vector)
            
            # Use index if available and requested
            if self.is_indexed and use_index:
                results = self._search_indexed(query_vector, k, n_probes)
            else:
                # Fall back to brute force search
                results = self.vector_model.search_vectors(query_vector.tolist(), k, distance_metric)
            
            return {
                "success": True,
                "query_vector": query_vector.tolist(),
                "results": results,
                "total_results": len(results),
                "method": "indexed" if self.is_indexed and use_index else "brute_force"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during search: {str(e)}"
            }
    
    def _search_indexed(self, query_vector: np.ndarray, k: int = 5, 
                       n_probes: int = 10) -> List[Dict[str, Any]]:
        """
        Search using the created index
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            n_probes: Number of clusters to probe
            
        Returns:
            Search results from indexed approach
        """
        if self.indexer is None:
            raise ValueError("No index created. Call create_index() first.")
            
        # Use the indexer for search
        results = self.indexer.search(query_vector, k, n_probes)
        
        # Enhance results with metadata from database
        enhanced_results = []
        for result in results:
            vector_id = result["vector_id"]
            vector = self.vector_model.get_vector(vector_id)
            if vector:
                result["metadata"] = vector.metadata
                result["created_at"] = vector.created_at
                enhanced_results.append(result)
            else:
                # If vector not found in database, still include the result
                enhanced_results.append(result)
        
        return enhanced_results
    
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
    
    def get_vector_by_id(self, vector_id: int) -> Dict[str, Any]:
        """
        Get a specific vector by database ID
        """
        try:
            vector = self.vector_model.get_vector_by_id(vector_id)
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
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting stats: {str(e)}"
            }
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index
        
        Returns:
            Index information
        """
        if not self.is_indexed:
            return {
                "success": False,
                "message": "No index created yet"
            }
        
        try:
            cluster_info = self.indexer.get_cluster_info()
            return {
                "success": True,
                "index_info": {
                    "is_indexed": self.is_indexed,
                    "total_clusters": cluster_info["total_clusters"],
                    "cluster_sizes": cluster_info["cluster_sizes"],
                    "total_vectors": self.vector_model.get_vector_count()
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting index info: {str(e)}"
            }
    
    def get_cluster_vectors(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get all vectors in a specific cluster
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Vectors in the cluster
        """
        if not self.is_indexed:
            return {
                "success": False,
                "message": "No index created yet"
            }
        
        try:
            vectors = self.indexer.get_cluster_vectors(cluster_id)
            return {
                "success": True,
                "cluster_id": cluster_id,
                "vectors": vectors,
                "count": len(vectors)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting cluster vectors: {str(e)}"
            }
    
    def get_all_clusters(self) -> Dict[str, Any]:
        """
        Get all clusters with their vectors
        
        Returns:
            All clusters
        """
        if not self.is_indexed:
            return {
                "success": False,
                "message": "No index created yet"
            }
        
        try:
            clusters = self.indexer.get_all_clusters()
            return {
                "success": True,
                "clusters": clusters,
                "total_clusters": len(clusters)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting all clusters: {str(e)}"
            }
    
    def get_vector_metadata_fields(self) -> Dict[str, Any]:
        """
        Get all unique metadata fields across all vectors
        """
        try:
            # Get all unique metadata keys
            vectors = self.vector_model.get_all_vectors(limit=1000)  # Limit for performance
            
            all_fields = set()
            for vector in vectors:
                if vector.metadata:
                    if isinstance(vector.metadata, dict):
                        all_fields.update(vector.metadata.keys())
            
            return {
                "success": True,
                "metadata_fields": list(all_fields)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting metadata fields: {str(e)}"
            }
    
    def rebuild_index(self, k: int = 100) -> Dict[str, Any]:
        """
        Rebuild the index with new parameters
        
        Args:
            k: Number of clusters
            
        Returns:
            Rebuild result
        """
        try:
            # Clear existing index
            self.indexer = None
            self.is_indexed = False
            
            # Create new index
            return self.create_index(k=k, force_rebuild=True)
        except Exception as e:
            return {
                "success": False,
                "message": f"Error rebuilding index: {str(e)}"
            }
