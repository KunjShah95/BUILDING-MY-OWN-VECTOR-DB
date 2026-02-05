# DAY 6 — Building My Own Vector DB

Date: Day 6

## Goal

Build a complete FastAPI REST API for the vector database, implement comprehensive benchmarking across all three indexing methods (HNSW, IVF, K-Means), and create production-ready endpoints with request validation and error handling.

## Highlights / Summary

- Developed full-featured FastAPI application with CRUD operations for vectors
- Implemented benchmarking suite comparing HNSW, IVF, and brute-force search methods
- Added request/response validation using Pydantic models
- Created endpoints for index management, search, and statistics
- Implemented proper error handling and logging throughout API
- Added batch operations endpoint for bulk vector insertion
- Created benchmarking dashboard insights with performance metrics

## Core Features Implemented

### 1. **FastAPI REST API** (`api/main.py`)

**Endpoints Implemented:**

```
POST   /api/v1/vectors              - Insert a single vector with metadata
POST   /api/v1/vectors/batch        - Insert multiple vectors in batch
GET    /api/v1/vectors/{vector_id}  - Retrieve a specific vector
GET    /api/v1/vectors              - List all vectors (with pagination)
DELETE /api/v1/vectors/{vector_id}  - Delete a vector
POST   /api/v1/search               - Search using query vector
POST   /api/v1/search/hybrid        - Hybrid search with filters
GET    /api/v1/stats                - Get database statistics
POST   /api/v1/index/create         - Create index (HNSW, IVF, or K-Means)
POST   /api/v1/index/load           - Load persisted index
GET    /api/v1/index/info           - Get index metadata
DELETE /api/v1/index/delete         - Delete index
POST   /api/v1/benchmark            - Run benchmark comparison
GET    /api/v1/health               - Health check endpoint
```

**Request/Response Models:**

```python
class VectorInsertRequest:
    vector: List[float]
    metadata: Dict[str, Any]
    vector_id: str

class VectorSearchRequest:
    query_vector: List[float]
    k: int
    method: str  # "hnsw", "ivf", "kmeans", "brute"
    ef_search: Optional[int]

class VectorSearchResponse:
    total_results: int
    search_time: float
    method: str
    results: List[VectorSearchResult]

class IndexCreateRequest:
    method: str  # "hnsw", "ivf", "kmeans"
    parameters: Dict[str, Any]
    save_path: Optional[str]
```

### 2. **Comprehensive Benchmarking** (`test/test_api.py` or `benchmark.py`)

**Benchmark Metrics:**

- **Query Speed**: Latency for k-nearest neighbor search
- **Memory Usage**: Index size and memory footprint
- **Index Construction Time**: Time to build index
- **Recall Rate**: Accuracy vs. brute-force baseline
- **Throughput**: Queries per second
- **Scalability**: Performance with varying dataset sizes

**Benchmark Scenarios:**

```python
class BenchmarkSuite:
    def test_100_vectors_5d()      - Small dataset, low dimensionality
    def test_1k_vectors_128d()     - Medium dataset, typical embeddings
    def test_10k_vectors_768d()    - Large dataset, high-dimensional
    def test_query_latency()       - Single query response time
    def test_batch_throughput()    - Batch insert and search throughput
    def test_memory_consumption()  - Index size analysis
```

**Expected Results Report:**

```
=== BENCHMARK RESULTS ===
Dataset: 10,000 vectors × 768 dimensions

Method      | Index Time | Query Time | Memory | Recall | Throughput
------------|------------|------------|--------|--------|------------
Brute Force |    0.000s  |  0.250s    |  58MB  | 100%   |    4 q/s
K-Means     |    0.500s  |  0.015s    |  12MB  |  98%   |   66 q/s
IVF         |    1.200s  |  0.020s    |  15MB  |  96%   |   50 q/s
HNSW        |    2.100s  |  0.008s    |  22MB  |  99%   | 125 q/s
```

### 3. **Error Handling & Validation**

**Custom Exceptions:**

```python
class VectorDBException(Exception): pass
class VectorNotFoundError(VectorDBException): pass
class InvalidVectorDimensionError(VectorDBException): pass
class IndexNotFoundError(VectorDBException): pass
class InvalidSearchParametersError(VectorDBException): pass
```

**Validation Rules:**

- Vector dimension consistency check
- Metadata size limits (max 1MB per vector)
- Query parameter validation (k must be < total vectors)
- Index existence checks before search
- Duplicate vector ID detection with options to update or skip

### 4. **Logging & Monitoring**

**Logging Configuration** (`config/logging.py`):

- Request/response logging with timestamps
- Error tracking with stack traces
- Performance metrics logging (query times, index stats)
- Database operation audit trail

**Metrics Tracked:**

```python
- total_vectors_inserted
- total_searches_performed
- average_query_time
- peak_memory_usage
- index_creation_times
- error_count_by_type
```

## Files Modified/Created

### New Files

- **`api/main.py`** - Complete FastAPI application with all endpoints
- **`test/test_benchmark.py`** - Comprehensive benchmarking suite
- **`utils/benchmark.py`** - Benchmark utilities and comparison logic
- **`utils/metrics.py`** - Performance metrics collection and analysis

### Modified Files

- **`models/pydantic_models.py`** - Added Pydantic request/response models
- **`config/settings.py`** - API configuration (host, port, rate limits)
- **`database/vector_database.py`** - Optional enhancements for consistency
- **`models/vector_model.py`** - Optional additions for bulk operations

## How to Run

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Start API Server

```powershell
# Development server with auto-reload
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or using the module directly
python api/main.py
```

API will be available at: `http://localhost:8000`

API Documentation (Swagger UI): `http://localhost:8000/docs`
Alternative docs (ReDoc): `http://localhost:8000/redoc`

### 3. Run Benchmarks

```powershell
# Run comprehensive benchmark suite
pytest test/test_benchmark.py -v

# Run specific benchmark
pytest test/test_benchmark.py::test_10k_vectors_768d -v

# Generate benchmark report
python utils/benchmark.py --output=benchmark_report.json
```

### 4. Test API Endpoints

```powershell
# Using curl
curl -X POST http://localhost:8000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "metadata": {"text": "example"},
    "vector_id": "test_1"
  }'

# Using Python requests
python -c "
import requests
response = requests.post('http://localhost:8000/api/v1/health')
print(response.json())
"

# Using Postman or Insomnia (import from /docs)
```

### 5. Run All Tests

```powershell
# Run complete test suite
pytest -v

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest test/test_api.py -v
```

## API Usage Examples

### Insert Vector

```bash
curl -X POST http://localhost:8000/api/v1/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "metadata": {
      "text": "Sample document",
      "category": "news",
      "source": "example.com"
    },
    "vector_id": "doc_001"
  }'
```

### Batch Insert

```bash
curl -X POST http://localhost:8000/api/v1/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [
      {
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"text": "doc1"},
        "vector_id": "doc_1"
      },
      {
        "vector": [0.2, 0.3, 0.4],
        "metadata": {"text": "doc2"},
        "vector_id": "doc_2"
      }
    ],
    "batch_name": "batch_1"
  }'
```

### Search Vectors

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [0.1, 0.2, 0.3, 0.4, 0.5],
    "k": 5,
    "method": "hnsw",
    "ef_search": 100
  }'
```

### Create Index

```bash
curl -X POST http://localhost:8000/api/v1/index/create \
  -H "Content-Type: application/json" \
  -d '{
    "method": "hnsw",
    "parameters": {
      "m": 16,
      "m0": 32,
      "ef_construction": 200
    },
    "save_path": "hnsw_index.json"
  }'
```

### Get Statistics

```bash
curl -X GET http://localhost:8000/api/v1/stats
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐    ┌──────────────────┐             │
│  │   API Endpoints  │    │   Error Handlers │             │
│  │                  │    │   Middleware     │             │
│  └────────┬─────────┘    └──────────────────┘             │
│           │                                               │
│  ┌────────▼─────────────────────────────────────────┐    │
│  │         Pydantic Models & Validation             │    │
│  └────────┬─────────────────────────────────────────┘    │
│           │                                               │
│  ┌────────▼──────────────────────────────────────┐       │
│  │   Vector Service Layer                        │       │
│  │  - HNSW, IVF, K-Means, Brute-Force Methods   │       │
│  └────────┬──────────────────────────────────────┘       │
│           │                                               │
│  ┌────────▼──────────────────────────────────────┐       │
│  │   Database Layer                              │       │
│  │  - PostgreSQL + Vector Storage               │       │
│  │  - HNSW / IVF Index Persistence              │       │
│  └──────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Performance Optimization Tips

1. **Index Selection**
   - Use **HNSW** for queries requiring <5ms latency (best recall)
   - Use **IVF** for balanced speed/recall with medium datasets
   - Use **K-Means** for very fast approximate search (budget constraints)

2. **Batch Operations**
   - Insert vectors in batches of 1000-5000 for 10-50x speedup
   - Create index after bulk inserts complete

3. **Caching**
   - Cache query results for identical search requests
   - Pre-compute common queries during off-peak hours

4. **Hardware**
   - Use SSD for PostgreSQL (faster I/O)
   - Allocate sufficient RAM for in-memory index copies
   - Consider GPU acceleration for large-scale operations

## Known Issues & Next Steps

### Known Issues

- ⚠️ Large index persistence (>1GB) may be slow
- ⚠️ No built-in request rate limiting (add in production)
- ⚠️ Single-threaded database session (add connection pooling)

### Next Steps (Day 7+)

- [ ] Implement authentication/authorization (JWT tokens)
- [ ] Add request rate limiting and quotas
- [ ] Implement caching layer (Redis)
- [ ] Add WebSocket support for real-time updates
- [ ] Implement sharding for distributed vector storage
- [ ] Add support for multiple metric types (L2, cosine, IP)
- [ ] Create admin dashboard for monitoring
- [ ] Implement vector field indexing (e.g., sparse vectors)
- [ ] Add support for vector quantization (PQ, binary)
- [ ] Develop deployment guides (Docker, Kubernetes)

## Testing Checklist

- [ ] All API endpoints respond with correct status codes
- [ ] Pydantic validation rejects invalid input
- [ ] Search results match expected nearest neighbors
- [ ] Index creation/loading works correctly
- [ ] Benchmarks show expected performance characteristics
- [ ] Error handling gracefully handles edge cases
- [ ] Logging captures all important events
- [ ] Database connection pooling is efficient
- [ ] Memory usage stays within expected bounds
- [ ] API documentation is complete and accurate

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [HNSW Paper](https://arxiv.org/abs/1802.02413)
- [IVF Algorithm](https://www.pinecone.io/learn/vector-database/)
- [Benchmarking Best Practices](https://easyperf.net/blog/)

## Summary

Day 6 transforms the vector database from a collection of independent indexing methods into a production-ready REST API with comprehensive benchmarking capabilities. The API provides a unified interface for all three search methods, making it easy to choose the best approach for different use cases. Benchmarking insights inform deployment decisions and optimizations.

The foundation is now ready for production deployment, distributed scaling, and advanced features in subsequent days.
