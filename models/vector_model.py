from typing import List, Dict,Any,Optional,Union
import numpy as np
import json
import uuid
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from database.schema import Vector, VectorBatch, VectorBatchMapping, VectorPgVector
from config.settings import get_settings

settings = get_settings()

class VectorModel:
    def __init__(self, db: Session):
        self.db = db

    def _tenant_collection_ids(self, tenant_id: str) -> List[str]:
        """Get all collection IDs belonging to a tenant."""
        from database.schema import Collection
        rows = self.db.query(Collection.collection_id).filter(
            Collection.tenant_id == tenant_id
        ).all()
        return [r[0] for r in rows]
    
    def clear_all_vectors(self) -> Dict[str, Any]:
        """
        Clear all vectors from the database
        """
        try:
            # Delete all mappings first (due to foreign key constraints)
            self.db.query(VectorBatchMapping).delete()
            # Delete all batches
            self.db.query(VectorBatch).delete()
            # Delete all vectors
            deleted_count = self.db.query(Vector).delete()
            self.db.commit()
            
            return {
                "success": True,
                "message": f"Cleared {deleted_count} vectors from database",
                "deleted_count": deleted_count
            }
        except Exception as e:
            self.db.rollback()
            return {
                "success": False,
                "message": f"Error clearing vectors: {str(e)}"
            }

    def create_vector(self, vector_data: List[float], metadata: Dict[str, Any] = None, 
                     vector_id: str = None, collection_id: str = None) -> Vector:
        """
        Create a new vector in the database
        """
        try:
            if vector_id is None:
                # Generate a unique vector_id using UUID for guaranteed uniqueness
                vector_id = f"vec_{uuid.uuid4().hex[:12]}"
            
            # Check if vector with this ID already exists
            existing = self.get_vector(vector_id)
            if existing:
                # Return existing vector instead of creating duplicate
                return existing
            
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
                vector_id=vector_id,
                collection_id=collection_id,
            )
            
            self.db.add(vector)
            self.db.commit()
            self.db.refresh(vector)
            
            return vector
        except Exception as e:
            self.db.rollback()
            raise
    
    def create_vector_batch(self, vectors: List[Dict[str, Any]], batch_name: str = None, 
                           description: str = None) -> Dict[str, Any]:
        """
        Create a batch of vectors
        """
        try:
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
        except Exception as e:
            self.db.rollback()
            raise
    
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
    
<<<<<<< HEAD
    def get_all_vectors(self, collection_id: str = None, offset: int = 0, limit: int = 10000, filters: dict = None) -> List[Vector]:
        """
        Get all vectors with pagination and optional collection filter
        """
        query = self.db.query(Vector)
        if collection_id:
            query = query.filter(Vector.collection_id == collection_id)
        query = query.offset(offset).limit(limit)

=======
    def get_all_vectors(self, collection_id: str = None, offset: int = 0,
                        limit: int = 10000, filters: dict = None,
                        collection_ids: Optional[List[str]] = None) -> List[Vector]:
        """
        Get all vectors with pagination and optional collection/tenant filter.

        Args:
            collection_id: Filter by a single collection.
            offset: Pagination offset.
            limit: Max results.
            filters: Metadata filter dict.
            collection_ids: Filter by multiple collection IDs (for tenant scoping).
        """
        query = self.db.query(Vector)
        if collection_id:
            query = query.filter(Vector.collection_id == collection_id)
        elif collection_ids is not None:
            query = query.filter(Vector.collection_id.in_(collection_ids))
        query = query.offset(offset).limit(limit)

>>>>>>> main
        results = query.all()

        if filters:
            results = [v for v in results if self._metadata_matches(v.meta_data, filters)]

        return results

    def get_vectors_by_collection(self, collection_id: str, offset: int = 0, limit: int = 10000) -> List[Vector]:
        """Get all vectors belonging to a collection with pagination."""
        return (
            self.db.query(Vector)
            .filter(Vector.collection_id == collection_id)
            .offset(offset)
            .limit(limit)
            .all()
        )
    
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
                      filters: Dict[str, Any] = None,
                      collection_id: str = None,
                      tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
<<<<<<< HEAD
        Search for similar vectors using brute force with optional filters
        Loads vectors in batches of 1000 to avoid OOM on large datasets
=======
        Search for similar vectors using brute force with optional filters.
        Accepts tenant_id to scope search to tenant-owned collections.
        Loads vectors in batches of 1000 to avoid OOM on large datasets.
>>>>>>> main
        """
        batch_size = 1000
        offset = 0
        all_results = []

<<<<<<< HEAD
        while True:
            batch = self.get_all_vectors(collection_id, offset=offset, limit=batch_size, filters=filters)
=======
        # Determine collection IDs to filter by
        ids_to_filter = None
        if collection_id:
            ids_to_filter = [collection_id]
        elif tenant_id:
            ids_to_filter = self._tenant_collection_ids(tenant_id)

        while True:
            batch = self.get_all_vectors(
                collection_id=None if ids_to_filter is None or len(ids_to_filter) != 1 else ids_to_filter[0],
                offset=offset, limit=batch_size, filters=filters,
                collection_ids=ids_to_filter,
            )
>>>>>>> main
            if not batch:
                break

            for vector in batch:
                distance = self._calculate_distance(query_vector, vector.vector_data, distance_metric)
                all_results.append({
                    "distance": distance,
                    "vector_id": vector.vector_id,
                    "metadata": vector.meta_data,
                    "meta_data": vector.meta_data,
                    "collection_id": vector.collection_id,
                    "vector": vector.vector_data,
                    "created_at": vector.created_at
                })

            offset += batch_size

        all_results.sort(key=lambda x: x["distance"])
        return all_results[:k]

    def _metadata_matches(self, meta_data: Optional[Dict[str, Any]], filters: Dict[str, Any]) -> bool:
        if not meta_data:
            return False
        for key, value in filters.items():
            if isinstance(value, dict):
                return True
            if meta_data.get(key) != value:
                return False
        return True
    
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
                                   distance_metric: str = 'cosine',
                                   collection_id: str = None) -> List[Dict[str, Any]]:
        """
        Search with metadata filters and optional collection scope
        """
        return self.search_vectors(
            query_vector, k, distance_metric, filters, collection_id
        )


class VectorPgVectorModel:
    def __init__(self, db: Session):
        self.db = db

    def create_vector(self, vector_data: List[float], metadata: Dict[str, Any] = None,
                      vector_id: str = None, collection_id: str = None,
                      content_type: str = None) -> VectorPgVector:
        if vector_id is None:
            vector_id = f"vec_{uuid.uuid4().hex[:12]}"

        vector = VectorPgVector(
            vector=vector_data,
            meta_data=metadata,
            vector_id=vector_id,
            collection_id=collection_id,
            content_type=content_type,
        )
        self.db.add(vector)
        self.db.commit()
        self.db.refresh(vector)
        return vector

    def get_vector(self, vector_id: str) -> Optional[VectorPgVector]:
        return self.db.query(VectorPgVector).filter(VectorPgVector.vector_id == vector_id).first()

    def delete_vector(self, vector_id: str) -> bool:
        vector = self.get_vector(vector_id)
        if vector:
            self.db.delete(vector)
            self.db.commit()
            return True
        return False

    def get_vector_count(self) -> int:
        return self.db.query(func.count(VectorPgVector.id)).scalar()

    def search_vectors(self, query_vector: List[float], k: int = 10,
                       collection_id: str = None) -> List[Dict[str, Any]]:
        qv = json.dumps(query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector)
        sql = "SELECT *, vector <=> :query AS distance FROM vectors_pgvector ORDER BY distance LIMIT :k"
        params = {"query": qv, "k": k}

        if collection_id:
            sql = "SELECT *, vector <=> :query AS distance FROM vectors_pgvector WHERE collection_id = :cid ORDER BY distance LIMIT :k"
            params = {"query": qv, "k": k, "cid": collection_id}

        rows = self.db.execute(text(sql), params).fetchall()
        results = []
        for row in rows:
            results.append({
                "distance": float(row.distance),
                "vector_id": row.vector_id,
                "metadata": row.meta_data,
                "meta_data": row.meta_data,
                "collection_id": row.collection_id,
                "created_at": row.created_at,
            })
        return results