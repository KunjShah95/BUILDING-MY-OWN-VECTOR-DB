from typing import List, Dict, Any, Optional
from database.schema import Vector
import numpy as np
from sqlalchemy.orm import Session

class VectorModel:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_vector(self, vector_data: List[float], metadata: Dict[str, Any] = None, 
                     vector_id: str = None) -> Vector:
        """
        Create a new vector in the database
        """
        if vector_id is None:
            vector_id = f"vec_{len(self.get_all_vectors()) + 1}"
        
        vector = Vector(
            vector_data=vector_data,
            meta_data=str(metadata) if metadata else None,
            vector_id=vector_id
        )
        
        self.db.add(vector)
        self.db.commit()
        self.db.refresh(vector)
        
        return vector
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Get a vector by its ID
        """
        return self.db.query(Vector).filter(Vector.vector_id == vector_id).first()
    
    def get_all_vectors(self) -> List[Vector]:
        """
        Get all vectors
        """
        return self.db.query(Vector).all()
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector by ID
        """
        vector = self.get_vector(vector_id)
        if vector:
            self.db.delete(vector)
            self.db.commit()
            return True
        return False
    
    def search_vectors(self, query_vector: List[float], k: int = 5, 
                      distance_metric: str = 'cosine') -> List[Dict[str, Any]]:
        """
        Search for similar vectors using brute force
        """
        vectors = self.get_all_vectors()
        results = []
        
        for vector in vectors:
            distance = self._calculate_distance(query_vector, vector.vector_data, distance_metric)
            results.append({
                "distance": distance,
                "vector_id": vector.vector_id,
                "metadata": vector.meta_data,
                "vector": vector.vector_data
            })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x["distance"])
        return results[:k]
    
    def _calculate_distance(self, vector1: List[float], vector2: List[float], 
                          metric: str = 'cosine') -> float:
        """
        Calculate distance between two vectors
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        if metric == 'euclidean':
            return float(np.linalg.norm(v1 - v2))
        elif metric == 'cosine':
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            return float(1 - cosine_sim)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
