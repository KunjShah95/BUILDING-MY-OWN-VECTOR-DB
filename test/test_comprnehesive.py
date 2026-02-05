import pytest
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.schema import Base

# Test configuration
TEST_DATABASE_URL = "postgresql://user:password@localhost:5432/vector_db_test"

@pytest.fixture(scope="session")
def setup_database():
    """Setup test database"""
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield engine, TestingSessionLocal

@pytest.fixture
def db_session(setup_database):
    """Create database session"""
    _, SessionLocal = setup_database
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def client(db_session):
    """Create test client"""
    from api.main import app
    from config.database import get_db
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

# ==================== Core Functionality Tests ====================

class TestVectorCRUD:
    """Test vector CRUD operations"""
    
    def test_create_single_vector(self, client):
        """Test creating a single vector"""
        vector_data = {
            "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "metadata": {"category": "test", "id": 1},
            "vector_id": "test_crud_1"
        }
        
        response = client.post("/vectors", json=vector_data)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] == True
        assert data["vector_id"] == "test_crud_1"
    
    def test_create_vector_batch(self, client):
        """Test creating a batch of vectors"""
        batch_data = {
            "vectors": [
                {"vector": [0.1, 0.2], "vector_id": "batch_1"},
                {"vector": [0.3, 0.4], "vector_id": "batch_2"},
                {"vector": [0.5, 0.6], "vector_id": "batch_3"}
            ],
            "batch_name": "test_batch",
            "description": "Test batch for CRUD tests"
        }
        
        response = client.post("/vectors/batch", json=batch_data)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] == True
        assert data["vector_count"] == 3
    
    def test_get_vector(self, client):
        """Test retrieving a vector"""
        # Create a vector first
        client.post("/vectors", json={
            "vector": [0.1, 0.2],
            "vector_id": "get_test"
        })
        
        # Retrieve it
        response = client.get("/vectors/get_test")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert data["vector"]["vector_id"] == "get_test"
    
    def test_update_vector(self, client):
        """Test updating a vector"""
        # Create a vector
        client.post("/vectors", json={
            "vector": [0.1, 0.2],
            "vector_id": "update_test"
        })
        
        # Update it
        update_data = {
            "vector": [0.9, 0.8],
            "metadata": {"updated": True}
        }
        
        response = client.put("/vectors/update_test", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_delete_vector(self, client):
        """Test deleting a vector"""
        # Create a vector
        client.post("/vectors", json={
            "vector": [0.1, 0.2],
            "vector_id": "delete_test"
        })
        
        # Delete it
        response = client.delete("/vectors/delete_test")
        assert response.status_code == 200
        
        # Verify deletion
        get_response = client.get("/vectors/delete_test")
        assert get_response.status_code == 404
    
    def test_get_all_vectors_pagination(self, client):
        """Test getting all vectors with pagination"""
        # Create some vectors
        for i in range(5):
            client.post("/vectors", json={
                "vector": [0.1 * i, 0.2 * i],
                "vector_id": f"paginate_test_{i}"
            })
        
        # Get with pagination
        response = client.get("/vectors?limit=2&offset=1")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["limit"] == 2
        assert data["offset"] == 1


class TestSearchFunctionality:
    """Test search functionality"""
    
    @pytest.fixture
    def setup_indexed_data(self, client):
        """Setup data with index for search tests"""
        # Create vectors
        for i in range(20):
            client.post("/vectors", json={
                "vector": [float(i) / 20] * 10,  # Slightly different vectors
                "vector_id": f"search_test_{i:02d}"
            })
        
        # Create index
        client.post("/index", json={
            "method": "hnsw",
            "m": 8,
            "ef_construction": 100
        })
    
    def test_hnsw_search(self, client, setup_indexed_data):
        """Test HNSW search"""
        search_data = {
            "query_vector": [0.5] * 10,
            "k": 5,
            "method": "hnsw"
        }
        
        response = client.post("/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert data["total_results"] == 5
    
    def test_brute_force_search(self, client, setup_indexed_data):
        """Test brute force search"""
        search_data = {
            "query_vector": [0.5] * 10,
            "k": 5,
            "method": "brute"
        }
        
        response = client.post("/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["method"] == "brute_force"
    
    def test_search_with_ef_parameter(self, client, setup_indexed_data):
        """Test search with ef_search parameter"""
        search_data = {
            "query_vector": [0.5] * 10,
            "k": 5,
            "method": "hnsw",
            "ef_search": 100
        }
        
        response = client.post("/search", json=search_data)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
    
    def test_search_comparison(self, client, setup_indexed_data):
        """Test search method comparison"""
        response = client.get(
            "/search/compare?query_vector=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0&k=5"
        )
        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        assert "hnsw" in data["methods"]
        assert "brute_force" in data["methods"]


class TestIndexManagement:
    """Test index management"""
    
    def test_create_hnsw_index(self, client):
        """Test creating HNSW index"""
        # Create some vectors first
        for i in range(10):
            client.post("/vectors", json={
                "vector": [float(i) / 10] * 8,
                "vector_id": f"index_test_{i}"
            })
        
        # Create index
        index_data = {
            "method": "hnsw",
            "m": 8,
            "ef_construction": 50
        }
        
        response = client.post("/index", json=index_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
    
    def test_get_index_info(self, client):
        """Test getting index information"""
        # Create vectors and index
        for i in range(10):
            client.post("/vectors", json={
                "vector": [float(i) / 10] * 8,
                "vector_id": f"info_test_{i}"
            })
        
        client.post("/index", json={"method": "hnsw", "m": 8})
        
        # Get info
        response = client.get("/index?method=hnsw")
        assert response.status_code == 200
        data = response.json()
        assert "index_info" in data or "stats" in data


class TestStatisticsAndHealth:
    """Test statistics and health endpoints"""
    
    def test_get_statistics(self, client):
        """Test getting statistics"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "total_vectors" in data.get("stats", {})
    
    def test_health_check(self, client):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "database" in data
    
    def test_readiness_check(self, client):
        """Test readiness check"""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_vector_dimension(self, client):
        """Test with mismatched vector dimensions"""
        # Create a vector
        client.post("/vectors", json={
            "vector": [0.1, 0.2],
            "vector_id": "dim_test_1"
        })
        
        # Try to search with different dimension
        search_data = {
            "query_vector": [0.1, 0.2, 0.3],  # Different dimension
            "k": 5
        }
        
        response = client.post("/search", json=search_data)
        # Should fail due to dimension mismatch
        assert response.status_code in [400, 422, 500]
    
    def test_empty_vector_list(self, client):
        """Test with empty vector list"""
        batch_data = {"vectors": []}
        
        response = client.post("/vectors/batch", json=batch_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_search_k(self, client):
        """Test with invalid k value"""
        search_data = {
            "query_vector": [0.1, 0.2],
            "k": 0  # Invalid k
        }
        
        response = client.post("/search", json=search_data)
        assert response.status_code == 422  # Validation error
    
    def test_nonexistent_vector(self, client):
        """Test getting nonexistent vector"""
        response = client.get("/vectors/nonexistent_xyz_123")
        assert response.status_code == 404
    
    def test_nonexistent_index(self, client):
        """Test getting info for nonexistent index"""
        response = client.get("/index?method=hnsw")
        # May return success or error depending on implementation
        assert response.status_code in [200, 400, 404]


# ==================== Performance Tests ====================

class TestPerformance:
    """Performance and load tests"""
    
    def test_batch_insert_performance(self, client):
        """Test batch insert performance"""
        vectors = [
            {
                "vector": [float(i % 100) / 100] * 16,
                "vector_id": f"perf_{i:04d}",
                "metadata": {"index": i}
            }
            for i in range(100)
        ]
        
        batch_data = {
            "vectors": vectors,
            "batch_name": "performance_test"
        }
        
        import time
        start = time.time()
        response = client.post("/vectors/batch", json=batch_data)
        end = time.time()
        
        assert response.status_code == 201
        data = response.json()
        assert data["vector_count"] == 100
        
        # Should complete in reasonable time (< 30 seconds)
        assert end - start < 30
        print(f"Batch insert time: {end - start:.2f} seconds")
    
    def test_search_performance(self, client):
        """Test search performance"""
        # Setup data
        for i in range(100):
            client.post("/vectors", json={
                "vector": [float(i) / 100] * 16,
                "vector_id": f"search_perf_{i:03d}"
            })
        
        client.post("/index", json={"method": "hnsw", "m": 8})
        
        # Run multiple searches
        import time
        
        start = time.time()
        for _ in range(10):
            response = client.post("/search", json={
                "query_vector": [0.5] * 16,
                "k": 5,
                "method": "hnsw"
            })
            assert response.status_code == 200
        
        end = time.time()
        
        # Average search time should be reasonable (< 1 second per query)
        avg_time = (end - start) / 10
        assert avg_time < 1.0
        print(f"Average search time: {avg_time:.4f} seconds")


# ==================== Integration Tests ====================

class TestFullWorkflow:
    """Test complete workflow scenarios"""
    
    def test_vector_db_workflow(self, client):
        """Test complete vector database workflow"""
        # 1. Insert vectors
        for i in range(20):
            client.post("/vectors", json={
                "vector": [float(i) / 20] * 8,
                "vector_id": f"workflow_{i:02d}",
                "metadata": {"category": "test", "index": i}
            })
        
        # 2. Verify count
        stats_response = client.get("/stats")
        assert stats_response.json()["stats"]["total_vectors"] >= 20
        
        # 3. Create index
        client.post("/index", json={
            "method": "hnsw",
            "m": 8,
            "ef_construction": 50
        })
        
        # 4. Search
        search_response = client.post("/search", json={
            "query_vector": [0.5] * 8,
            "k": 5,
            "method": "hnsw"
        })
        assert search_response.json()["success"] == True
        
        # 5. Update a vector
        update_response = client.put("/vectors/workflow_10", json={
            "vector": [0.9] * 8,
            "metadata": {"updated": True}
        })
        assert update_response.json()["success"] == True
        
        # 6. Verify update
        get_response = client.get("/vectors/workflow_10")
        assert get_response.json()["vector"]["metadata"]["updated"] == True
        
        # 7. Delete a vector
        delete_response = client.delete("/vectors/workflow_15")
        assert delete_response.json()["success"] == True
        
        # 8. Verify deletion
        get_deleted = client.get("/vectors/workflow_15")
        assert get_deleted.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
