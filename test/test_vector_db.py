import pytest
import numpy as np
from database.vector_database import VectorDatabase
from config.database import SessionLocal, engine
from database.schema import Base

# Create tables for testing
Base.metadata.create_all(bind=engine)

def test_insert_vector():
    """Test vector insertion"""
    db = SessionLocal()
    vector_db = VectorDatabase(db)
    
    # Create test vector
    vector_data = np.random.rand(128).tolist()
    metadata = {"test": "data", "id": 1}
    
    result = vector_db.insert_vector(vector_data, metadata, "test_vector_1")
    
    assert result["success"] == True
    assert result["vector_id"] == "test_vector_1"
    
    db.close()

def test_search_vector():
    """Test vector search"""
    db = SessionLocal()
    vector_db = VectorDatabase(db)
    
    # Create test vector
    vector_data = np.random.rand(128).tolist()
    metadata = {"test": "data", "id": 2}
    
    vector_db.insert_vector(vector_data, metadata, "test_vector_2")
    
    # Search with same vector
    search_result = vector_db.search(vector_data, k=1)
    
    assert search_result["success"] == True
    assert search_result["total_results"] == 1
    
    db.close()

def test_database_stats():
    """Test database statistics"""
    db = SessionLocal()
    vector_db = VectorDatabase(db)
    
    stats = vector_db.get_database_stats()
    
    assert stats["success"] == True
    
    db.close()

if __name__ == "__main__":
    pytest.main([__file__])
