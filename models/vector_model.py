from typing import List, Dict,Any,Optional,Union
import numpy as np
import json
import uuid
from sqlalchemy import func
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping

class VectorModel:
    def __init__(self, db: Session):
        self.db = db

    def create_vector(self, vector_data: List[float], metadata: Dict[str, Any] = None, 
                     vector_id: str = None) -> Vector:
        """
        Create a new vector in the database
        """
        if vector_id is None:
            # Generate a unique vector_id using UUID for guaranteed uniqueness
            vector_id = f"vec_{uuid.uuid4().hex[:12]}"
        
        # Validate vector data
        if not isinstance(vector_data, list):
            raise ValueError("Vector data must be a list")
        
        if len(vector_data) == 0:
            raise ValueError("Vector data cannot be empty")
        
        # Convert to proper format if needed
        vector_data = [float(x) for x in vector_data]
        
        vector = Vector(
            vector_data=vector_data,
            meta_data=metadata,
            vector_id=vector_id
        )
        
        self.db.add(vector)
        self.db.commit()
        self.db.refresh(vector)
        
        return vector
    
    def create_vector_batch(self, vectors: List[Dict[str, Any]], batch_name: str = None, 
                           description: str = None) -> Dict[str, Any]:
        """
        Create a batch of vectors
        """
        if not vectors:
            return {"success": False, "message": "No vectors provided"}
        
        # Create batch record
        batch = VectorBatch(
            batch_name=batch_name or f"batch_{uuid.uuid4().hex[:12]}",
            batch_size=len(vectors),
            description=description
        )
        
        self.db.add(batch)
        self.db.commit()
        self.db.refresh(batch)
        
        # Create vector records and mappings
        vector_ids = []
        for vector_data in vectors:
            vector = self.create_vector(
                vector_data=vector_data.get('vector'),
                metadata=vector_data.get('metadata'),
                vector_id=vector_data.get('vector_id')
            )
            vector_ids.append(vector.vector_id)
            
            # Create mapping
            mapping = VectorBatchMapping(
                batch_id=batch.id,
                vector_id=vector.vector_id
            )
            self.db.add(mapping)
        
        self.db.commit()
        
        return {
            "success": True,
            "batch_id": batch.id,
            "batch_name": batch.batch_name,
            "vector_count": len(vectors),
            "vector_ids": vector_ids
        }
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """
        Get a vector by its ID
        """
        return self.db.query(Vector).filter(Vector.vector_id == vector_id).first()
    
    def get_vector_by_id(self, vector_id: int) -> Optional[Vector]:
        """
        Get a vector by its database ID
        """
        return self.db.query(Vector).filter(Vector.id == vector_id).first()
    
    def get_all_vectors(self, limit: int = 1000, offset: int = 0) -> List[Vector]:
        """
        Get all vectors with pagination
        """
        return self.db.query(Vector).offset(offset).limit(limit).all()
    
    def get_vectors_by_batch(self, batch_id: int) -> List[Vector]:
        """
        Get all vectors belonging to a specific batch
        """
        mappings = self.db.query(VectorBatchMapping).filter(
            VectorBatchMapping.batch_id == batch_id
        ).all()
        
        vector_ids = [mapping.vector_id for mapping in mappings]
        return self.db.query(Vector).filter(Vector.vector_id.in_(vector_ids)).all()
    
    def get_vector_count(self) -> int:
        """
        Get total count of vectors
        """
        return self.db.query(func.count(Vector.id)).scalar()
    
    def update_vector(self, vector_id: str, metadata: Dict[str, Any] = None, 
                     vector_data: List[float] = None) -> bool:
        """
        Update an existing vector
        """
        vector = self.get_vector(vector_id)
        if not vector:
            return False
        
        if metadata is not None:
            vector.meta_data = metadata
        
        if vector_data is not None:
            vector.vector_data = [float(x) for x in vector_data]
        
        vector.updated_at = func.now()
        self.db.commit()
        return True
    
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
    
    def delete_batch(self, batch_id: int) -> bool:
        """
        Delete an entire batch of vectors
        """
        # Get all mappings for this batch
        mappings = self.db.query(VectorBatchMapping).filter(
            VectorBatchMapping.batch_id == batch_id
        ).all()
        
        # Delete vectors
        vector_ids = [mapping.vector_id for mapping in mappings]
        for vector_id in vector_ids:
            self.delete_vector(vector_id)
        
        # Delete mappings
        self.db.query(VectorBatchMapping).filter(
            VectorBatchMapping.batch_id == batch_id
        ).delete()
        
        # Delete batch
        batch = self.db.query(VectorBatch).filter(VectorBatch.id == batch_id).first()
        if batch:
            self.db.delete(batch)
            self.db.commit()
            return True
        
        return False
    
    def search_vectors(self, query_vector: List[float], k: int = 5, 
                      distance_metric: str = 'cosine', 
                      filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using brute force with optional filters
        """
        vectors = self.get_all_vectors()
        results = []
        
        for vector in vectors:
            distance = self._calculate_distance(query_vector, vector.vector_data, distance_metric)
            results.append({
                "distance": distance,
                "vector_id": vector.vector_id,
                "meta_data": vector.meta_data,
                "vector": vector.vector_data,
                "created_at": vector.created_at
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
    
    def get_vector_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database
        """
        total_vectors = self.get_vector_count()
        
        # Get some sample vectors for dimension analysis
        sample_vectors = self.get_all_vectors(limit=10)
        avg_dimension = 0
        if sample_vectors:
            avg_dimension = np.mean([len(v.vector_data) for v in sample_vectors])
        
        return {
            "total_vectors": total_vectors,
            "avg_dimension": int(avg_dimension),
            "database_size": f"{total_vectors * avg_dimension * 8} bytes"  # Approximate
        }
    
    def search_vectors_with_filters(self, query_vector: List[float], k: int = 5,
                                   filters: Dict[str, Any] = None,
                                   distance_metric: str = 'cosine') -> List[Dict[str, Any]]:
        """
        Search with metadata filters
        """
        # This is a simplified version - in a real implementation, you'd need
        # more sophisticated filtering
        if filters:
            print(f"Filters provided (simplified): {filters}")
        
        return self.search_vectors(query_vector, k, distance_metric)