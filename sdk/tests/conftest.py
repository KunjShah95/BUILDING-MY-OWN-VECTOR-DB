import pytest
from vector_db_client import VectorDBClient


@pytest.fixture
def client():
    c = VectorDBClient("http://localhost:8000")
    yield c
    c.close()


@pytest.fixture
def mock_base_url():
    return "http://localhost:8000"


@pytest.fixture
def sample_vector():
    return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def sample_metadata():
    return {"text": "hello world", "source": "test"}


@pytest.fixture
def sample_collection_data():
    return {
        "name": "test-collection",
        "collection_id": "test-coll-1",
        "modality": "text",
        "dimension": 384,
    }
