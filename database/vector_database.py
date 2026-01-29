from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from database.schema import Vector
from models.vector_model import VectorModel
import numpy as np

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
    
    def search(self, query_vector: List[float], k: int = 5, 
              distance_metric: str = 'cosine') -> Dict[str, Any]:
        """
        Search for similar vectors using brute force
        """
        try:
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
    
    def get_all_vectors(self) -> Dict[str, Any]:
        """
        Get all vectors in the database
        """
        try:
            vectors = self.vector_model.get_all_vectors()
            return {
                "success": True,
                "vectors": [vector.to_dict() for vector in vectors],
                "count": len(vectors)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving vectors: {str(e)}"
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
            vectors = self.vector_model.get_all_vectors()
            if vectors:
                avg_dimension = len(vectors[0].vector_data) if vectors else 0
                return {
                    "success": True,
                    "total_vectors": len(vectors),
                    "avg_dimension": avg_dimension,
                    "database_size": f"{len(vectors) * avg_dimension * 8} bytes"  # Approximate
                }
            else:
                return {
                    "success": True,
                    "total_vectors": 0,
                    "avg_dimension": 0,
                    "database_size": "0 bytes"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error getting stats: {str(e)}"
            }
