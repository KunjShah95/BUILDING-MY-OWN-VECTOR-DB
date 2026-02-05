# Vector Database API

A production-ready Vector Database API built with FastAPI, featuring HNSW and IVF indexing algorithms.

## Features

- 🚀 **High Performance**: Built on HNSW and IVF indexing algorithms
- 📊 **PostgreSQL Storage**: Persistent storage with metadata support
- 🌐 **RESTful API**: Complete CRUD operations and search endpoints
- 📚 **Auto Documentation**: Swagger UI and ReDoc available
- 🔒 **Production Ready**: Error handling, logging, and validation
- 🧪 **Test Coverage**: Unit tests for core functionality

## Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL 12+
- pip

### Installation

1. Clone the repository

```bash
git clone <repository-url>
cd vector_db_project
Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Configure environment
cp .env.example .env
# Edit .env with your database credentials
Initialize database
python -c "from database.schema import Base, engine; Base.metadata.create_all(engine)"
Running the API
# Development mode
python -m uvicorn api.main:app --reload

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000
Access Documentation
Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
API Endpoints
Vectors
Method Endpoint Description
POST /vectors Create a vector
POST /vectors/batch Create multiple vectors
GET /vectors Get all vectors
GET /vectors/{id} Get a vector
PUT /vectors/{id} Update a vector
DELETE /vectors/{id} Delete a vector
Search
Method Endpoint Description
POST /search Search for similar vectors
GET /search/compare Compare search methods
POST /search/batch Batch search
Index
Method Endpoint Description
POST /index Create an index
POST /index/save Save index to disk
POST /index/load Load index from disk
GET /index Get index information
Statistics
Method Endpoint Description
GET /stats Get database statistics
GET /health Health check
GET /ready Readiness check
Usage Examples
Create a Vector
import requests

vector = {
    "vector": [0.1, 0.2, 0.3],
    "metadata": {"text": "Hello World"},
    "vector_id": "doc_1"
}

response = requests.post("http://localhost:8000/vectors", json=vector)
print(response.json())
Search for Similar Vectors
import requests

search_request = {
    "query_vector": [0.1, 0.2, 0.3],
    "k": 5,
    "method": "hnsw",
    "ef_search": 50
}

response = requests.post("http://localhost:8000/search", json=search_request)
print(response.json())
Create an Index
import requests

index_request = {
    "method": "hnsw",
    "m": 16,
    "ef_construction": 200
}

response = requests.post("http://localhost:8000/index", json=index_request)
print(response.json())
Testing
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=.
Performance Tuning
HNSW Parameters
Parameter Description Recommended Values
m Neighbors per node 8-64
m0 Neighbors in layer 0 2*m
ef_construction Construction quality 100-400
ef_search Search breadth 10-1000
IVF Parameters
Parameter Description Recommended Values
n_clusters Number of clusters 100-10000
n_probes Clusters to search 1-100
