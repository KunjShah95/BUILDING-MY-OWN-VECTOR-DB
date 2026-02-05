import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from database.schema import Base

# Test database setup - skip API tests if PostgreSQL is unavailable
TEST_DATABASE_URL = "postgresql://user:password@localhost:5432/vector_db_test"
engine = create_engine(TEST_DATABASE_URL)
_DB_AVAILABLE = False
try:
    with engine.connect():
        _DB_AVAILABLE = True
except SQLAlchemyError:
    _DB_AVAILABLE = False
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables only when needed (inside fixtures)
def setup_test_db():
    """Setup test database tables"""
    if not _DB_AVAILABLE:
        pytest.skip("PostgreSQL test database unavailable; skipping API tests.")
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as e:
        pytest.skip(f"Could not create test database tables: {e}. Skipping API tests.")

@pytest.fixture
def db_session():
    """Create a database session for testing"""
    setup_test_db()  # Setup tables before each test
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def client(db_session):
    """Create a test client"""
    # Import app within fixture to avoid circular imports
    from api.main import app
    from config.database import get_db
    
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    test_client = TestClient(app)
    yield test_client
    
    app.dependency_overrides.clear()

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code in [200, 503]
    assert "status" in response.json()

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data

def test_create_vector(client):
    """Test vector creation"""
    vector_data = {
        "vector": [0.1, 0.2, 0.3, 0.4],
        "metadata": {"text": "test"},
        "vector_id": "test_vector_1"
    }
    
    response = client.post("/vectors", json=vector_data)
    assert response.status_code == 201
    data = response.json()
    assert data["success"] == True
    assert data["vector_id"] == "test_vector_1"

def test_create_vector_batch(client):
    """Test batch vector creation"""
    batch_data = {
        "vectors": [
            {"vector": [0.1, 0.2, 0.3], "vector_id": "batch_vec_1"},
            {"vector": [0.4, 0.5, 0.6], "vector_id": "batch_vec_2"}
        ],
        "batch_name": "test_batch"
    }
    
    response = client.post("/vectors/batch", json=batch_data)
    assert response.status_code == 201
    data = response.json()
    assert data["success"] == True

def test_get_vector(client):
    """Test getting a vector"""
    # First create a vector
    vector_data = {
        "vector": [0.1, 0.2, 0.3],
        "vector_id": "get_test_vector"
    }
    client.post("/vectors", json=vector_data)
    
    # Then retrieve it
    response = client.get("/vectors/get_test_vector")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True

def test_get_all_vectors(client):
    """Test getting all vectors"""
    response = client.get("/vectors")
    assert response.status_code == 200
    data = response.json()
    assert "vectors" in data
    assert "count" in data

def test_update_vector(client):
    """Test updating a vector"""
    # First create a vector
    vector_data = {
        "vector": [0.1, 0.2, 0.3],
        "vector_id": "update_test_vector"
    }
    client.post("/vectors", json=vector_data)
    
    # Then update it
    update_data = {
        "vector": [0.9, 0.8, 0.7],
        "metadata": {"updated": True}
    }
    
    response = client.put("/vectors/update_test_vector", json=update_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True

def test_delete_vector(client):
    """Test deleting a vector"""
    # First create a vector
    vector_data = {
        "vector": [0.1, 0.2, 0.3],
        "vector_id": "delete_test_vector"
    }
    client.post("/vectors", json=vector_data)
    
    # Then delete it
    response = client.delete("/vectors/delete_test_vector")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True
    
    # Verify deletion
    get_response = client.get("/vectors/delete_test_vector")
    assert get_response.status_code == 404

def test_search_vectors(client):
    """Test vector search"""
    # First create some vectors
    for i in range(5):
        vector_data = {
            "vector": [0.1 * i, 0.2 * i, 0.3 * i],
            "vector_id": f"search_vec_{i}"
        }
        client.post("/vectors", json=vector_data)
    
    # Create index
    index_data = {
        "method": "hnsw",
        "m": 4,
        "ef_construction": 50
    }
    client.post("/index", json=index_data)
    
    # Search
    search_data = {
        "query_vector": [0.1, 0.2, 0.3],
        "k": 3,
        "method": "hnsw"
    }
    
    response = client.post("/search", json=search_data)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "total_results" in data

def test_create_index(client):
    """Test index creation"""
    # First create some vectors
    for i in range(10):
        vector_data = {
            "vector": [0.1 * i, 0.2 * i, 0.3 * i],
            "vector_id": f"index_vec_{i}"
        }
        client.post("/vectors", json=vector_data)
    
    # Create index
    index_data = {
        "method": "hnsw",
        "m": 4,
        "ef_construction": 50
    }
    
    response = client.post("/index", json=index_data)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] == True

def test_get_statistics(client):
    """Test getting statistics"""
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data

def test_invalid_vector(client):
    """Test creating vector with invalid data"""
    invalid_data = {
        "vector": [],  # Empty vector
        "vector_id": "invalid_vector"
    }
    
    response = client.post("/vectors", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_nonexistent_vector(client):
    """Test getting nonexistent vector"""
    response = client.get("/vectors/nonexistent_vector")
    assert response.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
