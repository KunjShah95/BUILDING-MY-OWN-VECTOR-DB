from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping
from models.vector_model import VectorModel
import numpy as np
from datetime import datetime

class VectorDatabase:
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.vector_model = VectorModel(db_session)
    
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
    
    def search(self, query_vector: List[float], k: int = 5, 
              distance_metric: str = 'cosine', filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Search for similar vectors using brute force
        """
        try:
            if filters:
                results = self.vector_model.search_vectors_with_filters(
                    query_vector, k, filters, distance_metric
                )
            else:
                results = self.vector_model.search_vectors(query_vector, k, distance_metric)
            
            return {
                "success": True,
                "query_vector": query_vector,
                "results": results,
                "total_results": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error during search: {str(e)}"
            }
    
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
    
    def get_vector_batch(self, batch_id: int) -> Dict[str, Any]:
        """
        Get all vectors belonging to a specific batch
        """
        try:
            vectors = self.vector_model.get_vectors_by_batch(batch_id)
            return {
                "success": True,
                "batch_id": batch_id,
                "vectors": [vector.to_dict() for vector in vectors],
                "count": len(vectors)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving batch: {str(e)}"
            }
    
    def get_all_batches(self) -> Dict[str, Any]:
        """
        Get all batches
        """
        try:
            batches = self.db_session.query(VectorBatch).all()
            return {
                "success": True,
                "batches": [batch.to_dict() for batch in batches],
                "count": len(batches)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving batches: {str(e)}"
            }
    
    def delete_batch(self, batch_id: int) -> Dict[str, Any]:
        """
        Delete an entire batch of vectors
        """
        try:
            success = self.vector_model.delete_batch(batch_id)
            if success:
                return {
                    "success": True,
                    "message": f"Batch {batch_id} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "message": "Batch not found"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error deleting batch: {str(e)}"
            }
    
    def get_vector_metadata_fields(self) -> Dict[str, Any]:
        """
        Get all unique metadata fields across all vectors
        """
        try:
            # Get all unique metadata keys
            # This is a simplified approach - in production, you'd want more sophisticated analysis
            vectors = self.vector_model.get_all_vectors(limit=1000)  # Limit for performance
            
            all_fields = set()
            for vector in vectors:
                if vector.meta_data:
                    if isinstance(vector.meta_data, dict):
                        all_fields.update(vector.meta_data.keys())
            
            return {
                "success": True,
                "metadata_fields": list(all_fields)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting metadata fields: {str(e)}"
            }
