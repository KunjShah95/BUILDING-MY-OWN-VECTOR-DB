# VectorIndexer REST API Integration Guide

## 📌 Overview

The VectorIndexer REST API has been successfully integrated into your FastAPI application. You now have 9+ endpoints for managing and searching vector indexes using HNSW, IVF, or Hybrid methods.

## 🚀 Getting Started

### Start the API Server

```bash
# From project root
python -m uvicorn api.main:app --reload

# Or
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the API

- **Interactive Documentation**: <http://localhost:8000/docs> (Swagger UI)
- **Alternative Documentation**: <http://localhost:8000/redoc> (ReDoc)
- **API Root**: <http://localhost:8000>

## 📚 Available Endpoints

### 1. Configure Indexer

**POST** `/api/indexer/config`

Configure the VectorIndexer with specific settings.

**Request Body:**

```json
{
  "method": "hnsw",              // "hnsw", "ivf", or "hybrid"
  "num_vectors": 10000,          // Expected number of vectors
  "vector_dim": 128,             // Vector dimensionality
  "recall_target": 0.95,         // Target recall (0-1)
  "speed_priority": false        // Optimize for speed if true
}
```

**Response:**

```json
{
  "success": true,
  "config": {
    "method": "hnsw",
    "num_vectors": 10000,
    "vector_dim": 128,
    "recall_target": 0.95,
    "hnsw_params": {
      "m": 32,
      "m0": 64,
      "ef_construction": 300,
      "ef_search": 50
    }
  }
}
```

**Example with curl:**

```bash
curl -X POST "http://localhost:8000/api/indexer/config" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "hnsw",
    "num_vectors": 10000,
    "recall_target": 0.95
  }'
```

### 2. Create Index

**POST** `/api/indexer/create`

Build an index from a set of vectors.

**Request Body:**

```json
[
  [1.0, 2.0, 3.0, ...],     // Vector 1
  [4.0, 5.0, 6.0, ...],     // Vector 2
  [7.0, 8.0, 9.0, ...]      // Vector 3
]
```

**Response:**

```json
{
  "success": true,
  "method": "hnsw",
  "vectors_indexed": 10000,
  "creation_time": 45.23,
  "hnsw": {
    "success": true,
    "creation_time": 45.23,
    "parameters": {
      "m": 32,
      "m0": 64,
      "ef_construction": 300
    },
    "graph_stats": {
      "total_nodes": 10000,
      "total_edges": 600000,
      "avg_connections": 60,
      "max_level": 8
    }
  }
}
```

**Example with curl:**

```bash
# Create a simple test (use Python to generate proper vectors)
curl -X POST "http://localhost:8000/api/indexer/create" \
  -H "Content-Type: application/json" \
  -d '[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]'
```

### 3. Search

**POST** `/api/indexer/search`

Find nearest neighbors for a query vector.

**Request Body:**

```json
{
  "query_vector": [1.0, 2.0, 3.0, ...],  // Query vector
  "k": 10,                                 // Number of results
  "method": "hnsw",                        // Optional: override method
  "ef_search": 50                          // Optional: HNSW parameter
}
```

**Response:**

```json
{
  "success": true,
  "search_time": 0.125,
  "method": "hnsw",
  "ef_search": 50,
  "results": [
    {
      "vector_id": "vec_0",
      "distance": 0.05,
      "metadata": null
    },
    {
      "vector_id": "vec_1",
      "distance": 0.12,
      "metadata": null
    }
  ],
  "count": 10
}
```

**Example with curl:**

```bash
curl -X POST "http://localhost:8000/api/indexer/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [1.0, 2.0, 3.0],
    "k": 5
  }'
```

### 4. Batch Search

**POST** `/api/indexer/search-batch`

Search multiple queries efficiently in one request.

**Request Body:**

```json
{
  "queries": [
    [1.0, 2.0, 3.0, ...],      // Query 1
    [4.0, 5.0, 6.0, ...],      // Query 2
    [7.0, 8.0, 9.0, ...]       // Query 3
  ],
  "k": 10
}
```

**Response:**

```json
{
  "success": true,
  "total_queries": 100,
  "batch_time": 1.53,
  "avg_query_time": 0.0153,
  "queries_per_second": 65.36,
  "results": [
    [  // Results for query 1
      {"vector_id": "vec_0", "distance": 0.05},
      {"vector_id": "vec_1", "distance": 0.12}
    ],
    [  // Results for query 2
      {"vector_id": "vec_5", "distance": 0.08},
      {"vector_id": "vec_9", "distance": 0.15}
    ]
    // ... more results
  ]
}
```

**Example with curl:**

```bash
curl -X POST "http://localhost:8000/api/indexer/search-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]
    ],
    "k": 5
  }'
```

### 5. Get Statistics

**GET** `/api/indexer/stats`

Get comprehensive indexer statistics.

**Response:**

```json
{
  "success": true,
  "data": {
    "config": {
      "method": "hnsw",
      "num_vectors": 10000,
      "recall_target": 0.95,
      "hnsw_params": {...}
    },
    "vector_count": 10000,
    "created_at": "2026-02-05T14:20:30.123456",
    "last_updated": "2026-02-05T14:22:15.987654",
    "statistics": {
      "total_indexed": 10000,
      "total_searched": 245,
      "total_search_time": 31.2,
      "avg_search_time": 0.127
    },
    "hnsw_available": true,
    "ivf_available": false
  }
}
```

**Example with curl:**

```bash
curl "http://localhost:8000/api/indexer/stats"
```

### 6. Health Check

**GET** `/api/indexer/health`

Check if indexer is ready.

**Response:**

```json
{
  "status": "healthy",
  "indexer_ready": true,
  "vectors_indexed": 10000,
  "hnsw_available": true,
  "ivf_available": false
}
```

**Example with curl:**

```bash
curl "http://localhost:8000/api/indexer/health"
```

### 7. Suggest Parameters

**POST** `/api/indexer/parameters/suggest`

Get recommended index parameters for your dataset.

**Request Body:**

```json
{
  "database_size": 100000,       // Number of vectors
  "recall_target": 0.95          // Target recall (0-1)
}
```

**Response:**

```json
{
  "success": true,
  "database_size": 100000,
  "recall_target": 0.95,
  "suggestions": {
    "hnsw": {
      "m": 16,
      "m0": 32,
      "ef_construction": 300,
      "ef_search": 50
    },
    "ivf": {
      "n_clusters": 316,
      "n_probes": 31
    }
  }
}
```

**Example with curl:**

```bash
curl -X POST "http://localhost:8000/api/indexer/parameters/suggest" \
  -H "Content-Type: application/json" \
  -d '{
    "database_size": 100000,
    "recall_target": 0.95
  }'
```

### 8. Save Index

**POST** `/api/indexer/save`

Save the index to disk for later use.

**Query Parameters:**

- `path` (optional): Custom save path

**Response:**

```json
{
  "success": true,
  "index_path": "./indexes/index_20260205_142030",
  "results": {
    "hnsw": {
      "saved": true,
      "path": "./indexes/index_20260205_142030/hnsw.pkl"
    },
    "metadata": {
      "saved": true,
      "path": "./indexes/index_20260205_142030/metadata.json"
    }
  }
}
```

**Example with curl:**

```bash
curl -X POST "http://localhost:8000/api/indexer/save?path=./my_index"
```

### 9. Load Index

**POST** `/api/indexer/load`

Load a previously saved index.

**Query Parameters:**

- `path` (required): Path to index directory

**Response:**

```json
{
  "success": true,
  "results": {
    "hnsw": {"loaded": true},
    "metadata": {"loaded": true}
  },
  "vector_count": 10000
}
```

**Example with curl:**

```bash
curl -X POST "http://localhost:8000/api/indexer/load?path=./my_index"
```

### 10. Get Supported Methods

**GET** `/api/indexer/methods`

List all supported indexing methods.

**Response:**

```json
{
  "methods": [
    {
      "name": "hnsw",
      "description": "Hierarchical Navigable Small World - Fast, balanced",
      "use_cases": ["General purpose", "Balanced performance"]
    },
    {
      "name": "ivf",
      "description": "Inverted File - Fast building, simple parameters",
      "use_cases": ["Large datasets", "Speed critical"]
    },
    {
      "name": "hybrid",
      "description": "Combined HNSW+IVF - Best recall",
      "use_cases": ["High precision", "Recall > 0.99"]
    }
  ]
}
```

**Example with curl:**

```bash
curl "http://localhost:8000/api/indexer/methods"
```

## 🔄 Complete Workflow Example

### Step 1: Configure

```bash
curl -X POST "http://localhost:8000/api/indexer/config" \
  -H "Content-Type: application/json" \
  -d '{"method": "hnsw", "num_vectors": 10000, "recall_target": 0.95}'
```

### Step 2: Create Index

```bash
curl -X POST "http://localhost:8000/api/indexer/create" \
  -H "Content-Type: application/json" \
  -d @vectors.json
```

### Step 3: Search

```bash
curl -X POST "http://localhost:8000/api/indexer/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [1.0, 2.0, 3.0],
    "k": 10
  }'
```

### Step 4: Check Stats

```bash
curl "http://localhost:8000/api/indexer/stats"
```

### Step 5: Save Index

```bash
curl -X POST "http://localhost:8000/api/indexer/save?path=./production_index"
```

## 🐍 Python Client Example

```python
import requests
import json
import numpy as np

BASE_URL = "http://localhost:8000/api/indexer"

# Configure indexer
config = {
    "method": "hnsw",
    "num_vectors": 10000,
    "recall_target": 0.95
}
response = requests.post(f"{BASE_URL}/config", json=config)
print(response.json())

# Create test vectors
vectors = np.random.randn(100, 128).tolist()

# Create index
response = requests.post(f"{BASE_URL}/create", json=vectors)
print(f"Indexed {response.json()['vectors_indexed']} vectors")

# Search
query = np.random.randn(128).tolist()
search_result = {
    "query_vector": query,
    "k": 10
}
response = requests.post(f"{BASE_URL}/search", json=search_result)
results = response.json()
print(f"Found {results['count']} results in {results['search_time']*1000:.2f}ms")

# Batch search
queries = np.random.randn(100, 128).tolist()
batch_result = {
    "queries": queries,
    "k": 10
}
response = requests.post(f"{BASE_URL}/search-batch", json=batch_result)
data = response.json()
print(f"Throughput: {data['queries_per_second']:.2f} QPS")

# Get stats
response = requests.get(f"{BASE_URL}/stats")
stats = response.json()['data']
print(f"Total searches: {stats['statistics']['total_searched']}")
print(f"Avg search time: {stats['statistics']['avg_search_time']*1000:.2f}ms")

# Save index
response = requests.post(f"{BASE_URL}/save?path=./my_index")
save_result = response.json()
print(f"Index saved to: {save_result['index_path']}")
```

## 🎯 Use Case Examples

### High-Throughput Batch Processing

```json
POST /api/indexer/config
{
  "method": "ivf",
  "speed_priority": true
}

POST /api/indexer/search-batch
{
  "queries": [[...], [...], ...],  // 1000 queries
  "k": 5
}
```

### High-Precision Search

```json
POST /api/indexer/config
{
  "method": "hybrid",
  "recall_target": 0.99
}

POST /api/indexer/search
{
  "query_vector": [...],
  "k": 10,
  "method": "hybrid"
}
```

### Auto-Optimized for Your Dataset

```json
POST /api/indexer/parameters/suggest
{
  "database_size": 1000000,
  "recall_target": 0.95
}
```

## 📊 Performance Tips

1. **Use batch search** for multiple queries (10x faster)
2. **Monitor QPS** using `/api/indexer/stats`
3. **Adjust parameters** based on suggestions
4. **Save indexes** for reuse with `/api/indexer/save`
5. **Check health** before operations with `/api/indexer/health`

## 🔍 Debugging

### Check API Status

```bash
curl "http://localhost:8000/health"        # Global health
curl "http://localhost:8000/api/indexer/health"  # Indexer health
```

### View All Endpoints

Visit: <http://localhost:8000/docs>

### Check Logs

```bash
# Uvicorn logs show VectorIndexer integration status
# Look for: "✅ VectorIndexer API routes successfully integrated"
```

## 📋 Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/indexer/config` | POST | Configure indexer |
| `/api/indexer/create` | POST | Create index |
| `/api/indexer/search` | POST | Single search |
| `/api/indexer/search-batch` | POST | Batch search |
| `/api/indexer/stats` | GET | Get statistics |
| `/api/indexer/health` | GET | Health check |
| `/api/indexer/parameters/suggest` | POST | Parameter suggestions |
| `/api/indexer/save` | POST | Save index |
| `/api/indexer/load` | POST | Load index |
| `/api/indexer/methods` | GET | Supported methods |

---

Your VectorIndexer REST API is now fully integrated and ready to use! 🚀
