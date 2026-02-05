from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping
from models.vector_model import VectorModel
from database.hnsw_database import HNSWVectorDatabase
from database.ivf_database import IVFVectorDatabase
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
    
    # ==================== Vector Operations ====================
    
    def create_vector(self, vector_data: List[float], 
                     metadata: Optional[Dict[str, Any]] = None,
                     vector_id: Optional[str] = None) -> Dict[str, Any]:
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
            # Insert into database
            vector = self.vector_model.create_vector(vector_data, metadata, vector_id)
            
            # Add to HNSW index if it exists
            if self.hnsw_db.hnsw_index is not None:
                self.hnsw_db.hnsw_index.insert(
                    vector_data,
                    vector.vector_id,
                    metadata
                )
            
            logger.info(f"Created vector: {vector.vector_id}")
            
            return {
                "success": True,
                "message": "Vector created successfully",
                "vector_id": vector.vector_id,
                "vector": vector.to_dict()
            }
        except Exception as e:
            logger.error(f"Error creating vector: {str(e)}")
            return {
                "success": False,
                "message": f"Error creating vector: {str(e)}"
            }
    
    def create_vector_batch(self, vectors: List[Dict[str, Any]], 
                           batch_name: Optional[str] = None,
                           description: Optional[str] = None) -> Dict[str, Any]:
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
            result = self.vector_model.create_vector_batch(vectors, batch_name, description)
            
            # Add to HNSW index if it exists
            if self.hnsw_db.hnsw_index is not None:
                for vector_data in vectors:
                    self.hnsw_db.hnsw_index.insert(
                        vector_data["vector"],
                        vector_data["vector_id"],
                        vector_data.get("metadata")
                    )
            
            logger.info(f"Created batch with {result['vector_count']} vectors")
            
            return {
                "success": True,
                "message": "Batch created successfully",
                "batch_id": result.get("batch_id"),
                "batch_name": result.get("batch_name"),
                "vector_count": result.get("vector_count")
            }
        except Exception as e:
            logger.error(f"Error creating batch: {str(e)}")
            return {
                "success": False,
                "message": f"Error creating batch: {str(e)}"
            }
    
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
            logger.error(f"Error getting vector {vector_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error getting vector: {str(e)}"
            }
    
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
                "total": self.vector_model.get_vector_count()
            }
        except Exception as e:
            logger.error(f"Error getting vectors: {str(e)}")
            return {
                "success": False,
                "message": f"Error getting vectors: {str(e)}"
            }
    
    def update_vector(self, vector_id: str, 
                     vector_data: Optional[List[float]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            logger.error(f"Error updating vector {vector_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error updating vector: {str(e)}"
            }
    
    def delete_vector(self, vector_id: str) -> Dict[str, Any]:
        """
        Delete a vector
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Delete result
        """
        try:
            # Delete from database
            success = self.vector_model.delete_vector(vector_id)
            
            if success:
                # Delete from HNSW index if it exists
                if self.hnsw_db.hnsw_index is not None:
                    self.hnsw_db.hnsw_index.delete(vector_id)
                
                logger.info(f"Deleted vector: {vector_id}")
                
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
            logger.error(f"Error deleting vector {vector_id}: {str(e)}")
            return {
                "success": False,
                "message": f"Error deleting vector: {str(e)}"
            }
    
    # ==================== Search Operations ====================
    
    def search_vectors(self, query_vector: List[float], k: int = 5,
                      method: str = 'hnsw', 
                      ef_search: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of results
            method: Search method
            ef_search: HNSW search parameter
            
        Returns:
            Search results
        """
        try:
            start_time = time.time()
            
            if method == 'hnsw':
                result = self.hnsw_db.search_hnsw(query_vector, k, ef_search)
            elif method == 'brute':
                result = self.hnsw_db.search_brute_force(query_vector, k)
            else:
                return {
                    "success": False,
                    "message": f"Unknown search method: {method}"
                }
            
            search_time = time.time() - start_time
            
            # Combine timing
            if result.get("success"):
                result["search_time"] = search_time
                
            return result
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return {
                "success": False,
                "message": f"Error during search: {str(e)}"
            }
    
    def compare_search_methods(self, query_vector: List[float], 
                              k: int = 5) -> Dict[str, Any]:
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
            return {
                "success": False,
                "message": f"Error comparing methods: {str(e)}"
            }
    
    # ==================== Index Operations ====================
    
    def create_index(self, method: str = 'hnsw', m: int = 16,
                    m0: Optional[int] = None, ef_construction: int = 200,
                    n_clusters: int = 100, n_probes: int = 10) -> Dict[str, Any]:
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
            if method == 'hnsw':
                return self.hnsw_db.create_hnsw_index(m, m0, ef_construction)
            elif method == 'ivf':
                return self.ivf_db.create_ivf_index(n_clusters, n_probes)
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}"
                }
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return {
                "success": False,
                "message": f"Error creating index: {str(e)}"
            }
    
    def save_index(self, method: str = 'hnsw') -> Dict[str, Any]:
        """
        Save index to disk
        
        Args:
            method: Indexing method
            
        Returns:
            Save result
        """
        try:
            if method == 'hnsw':
                return self.hnsw_db.save_hnsw_index()
            elif method == 'ivf':
                return self.ivf_db.save_ivf_index()
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}"
                }
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return {
                "success": False,
                "message": f"Error saving index: {str(e)}"
            }
    
    def load_index(self, method: str = 'hnsw') -> Dict[str, Any]:
        """
        Load index from disk
        
        Args:
            method: Indexing method
            
        Returns:
            Load result
        """
        try:
            if method == 'hnsw':
                return self.hnsw_db.load_hnsw_index()
            elif method == 'ivf':
                return self.ivf_db.load_ivf_index()
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}"
                }
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return {
                "success": False,
                "message": f"Error loading index: {str(e)}"
            }
    
    def get_index_info(self, method: str = 'hnsw') -> Dict[str, Any]:
        """
        Get index information
        
        Args:
            method: Indexing method
            
        Returns:
            Index information
        """
        try:
            if method == 'hnsw':
                return self.hnsw_db.get_hnsw_index_info()
            elif method == 'ivf':
                return self.ivf_db.get_index_stats()
            else:
                return {
                    "success": False,
                    "message": f"Unknown indexing method: {method}"
                }
        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            return {
                "success": False,
                "message": f"Error getting index info: {str(e)}"
            }
    
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
            
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {
                "success": False,
                "message": f"Error getting stats: {str(e)}"
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status
        
        Returns:
            Health status
        """
        try:
            total_vectors = self.vector_model.get_vector_count()
            index_available = self.hnsw_db.hnsw_index is not None
            
            return {
                "status": "healthy" if total_vectors >= 0 else "unhealthy",
                "database": "connected",
                "index_available": index_available,
                "total_vectors": total_vectors
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "index_available": False,
                "total_vectors": 0
            }
