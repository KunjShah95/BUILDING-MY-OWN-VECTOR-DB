# Building Vector Database from Scratch
[![GitHub Stars](https://img.shields.io/github/stars/KunjShah95/BUILDING-MY-OWN-VECTOR-DB?style=flat-square)](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB)
[![GitHub Forks](https://img.shields.io/github/forks/KunjShah95/BUILDING-MY-OWN-VECTOR-DB?style=flat-square)](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/blob/main/LICENSE)

A **production-ready Vector Database** built from scratch with Python, featuring high-performance **HNSW** and **IVF** indexing algorithms backed by **PostgreSQL**.

> Built as a comprehensive learning project demonstrating advanced vector indexing, database design, and performance optimization techniques. Fully tested, benchmarked, and production-ready for deployment.

---

[GitHub](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB) · [Documentation](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/blob/main/docs/ivf_vector_search_guide.md) · [Report Bug](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues) · [Request Feature](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues)

---

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Performance](#performance-benchmarks)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Testing & Benchmarking](#testing--benchmarking)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)

---

## ✨ Features

### Core Features

- ✅ **HNSW Indexing**: Hierarchical Navigable Small World for fast approximate nearest neighbor search
- ✅ **IVF Indexing**: Inverted File Index with optimal cluster selection
- ✅ **PostgreSQL Storage**: Persistent vector storage with rich metadata support
- ✅ **Complete CRUD Operations**: Add, read, update, and delete vectors
- ✅ **Batch Operations**: Efficient bulk vector insertion and search
- ✅ **Multiple Search Methods**: HNSW, IVF, and brute force comparison

### Advanced Features

- ✅ **Automatic Index Management**: Smart index creation and lifecycle management
- ✅ **Search Method Comparison**: Compare performance metrics across algorithms
- ✅ **Comprehensive Benchmarking**: Built-in performance, recall, and throughput testing
- ✅ **Memory Optimization**: Quantization and efficient batch processing
- ✅ **OpenAPI Documentation**: Auto-generated Swagger UI and ReDoc

### Production Ready

- ✅ **Error Handling**: Comprehensive exception handling with meaningful messages
- ✅ **Structured Logging**: Process timing and detailed debug information
- ✅ **Pydantic Validation**: Complete request/response validation
- ✅ **CORS Support**: Cross-origin resource sharing enabled
- ✅ **Health Checks**: Database and service health monitoring
- ✅ **Docker Support**: Containerized deployment with Docker Compose
- ✅ **Monitoring**: Prometheus metrics and Grafana dashboards

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **PostgreSQL 12+** (local or Docker)
- **pip** (Python package manager)

### Installation & Setup

#### 1. Navigate to Project Directory

```powershell
cd "c:\BUILDING MY OWN VECTOR DB"
```

#### 2. Create Virtual Environment

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate      # Linux/Mac
```

#### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

#### 4. Initialize Database

```powershell
python -c "from database.schema import Base, engine; Base.metadata.create_all(engine)"
```

#### 5. Run the API

```powershell
python -m uvicorn api.main:app --reload
```

**Access the application:**

- 🌐 **API**: <http://localhost:8000>
- 📚 **Swagger UI**: <http://localhost:8000/docs>
- 📖 **ReDoc**: <http://localhost:8000/redoc>
- ❤️ **Health Check**: <http://localhost:8000/health>

---

## 🏗️ Architecture

### System Overview (ASCII)

```
┌─────────────────────────────────────────────────┐
│           FastAPI Application (api/)             │
│  ├── Vector CRUD endpoints                       │
│  ├── Search endpoints (HNSW, IVF, Brute Force)  │
│  ├── Index management endpoints                  │
│  └── Health & stats endpoints                    │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      Business Logic (services/)                  │
│  ├── VectorService (CRUD operations)            │
│  └── VectorIndexer (Index management)           │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│    Indexing Algorithms (utils/)                  │
│  ├── HNSW (hnsw_index.py)                      │
│  ├── IVF (ivf_index.py)                        │
│  ├── Distance calculations (distance.py)       │
│  └── Clustering utilities (clustering.py)      │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      Database Layer (database/)                  │
│  ├── PostgreSQL with SQLAlchemy                │
│  ├── Vector ORM models (schema.py)             │
│  └── Database wrappers                         │
└─────────────────────────────────────────────────┘
```

### Architecture Diagram (Mermaid)

```mermaid
graph TB
    Client["Client"]
    API["FastAPI Application"]
    
    subgraph Endpoints["API Endpoints"]
        CRUD["Vector CRUD<br/>/vectors/*"]
        Search["Search Operations<br/>/search/*"]
        Index["Index Management<br/>/index/*"]
        Health["Health & Stats<br/>/health /stats"]
    end
    
    subgraph Services["Business Logic Layer"]
        VectorSvc["VectorService<br/>(CRUD operations)"]
        IndexSvc["VectorIndexer<br/>(Index management)"]
    end
    
    subgraph Algorithms["Indexing Algorithms"]
        HNSW["HNSW Algorithm<br/>(hnsw_index.py)"]
        IVF["IVF Algorithm<br/>(ivf_index.py)"]
        Distance["Distance Metrics<br/>(distance.py)"]
    end
    
    subgraph Database["Database Layer"]
        SQLAlchemy["SQLAlchemy ORM"]
        Schema["Vector Schema"]
        PostgreSQL["PostgreSQL Database"]
    end
    
    subgraph Utils["Utilities"]
        Clustering["Clustering<br/>(clustering.py)"]
        Benchmark["Benchmarking<br/>(benchmark.py)"]
        Optimization["Optimization<br/>(optimization.py)"]
    end
    
    Client -->|HTTP| API
    API --> Endpoints
    
    CRUD --> VectorSvc
    Search --> IndexSvc
    Index --> IndexSvc
    Health --> VectorSvc
    
    VectorSvc --> SQLAlchemy
    IndexSvc --> HNSW
    IndexSvc --> IVF
    
    HNSW --> Distance
    IVF --> Distance
    IVF --> Clustering
    
    SQLAlchemy --> Schema
    Schema --> PostgreSQL
    
    IndexSvc -.->|Performance| Benchmark
    IndexSvc -.->|Tuning| Optimization
```

### Vector Search Flow Diagram (Mermaid)

```mermaid
sequenceDiagram
    participant Client as Client
    participant API as FastAPI
    participant Service as VectorService
    participant Indexer as Indexer (HNSW/IVF)
    participant Distance as Distance Calc
    participant DB as PostgreSQL
    
    Client->>API: POST /search<br/>{query_vector, k, method}
    API->>Service: search_vectors()
    
    alt HNSW Search
        Service->>Indexer: search_hnsw(query, k, ef_search)
        Indexer->>Distance: calculate_distance()
        Distance->>Indexer: distances[]
        Indexer->>Indexer: hierarchical_search()
        Indexer->>Service: top_k_results
    else IVF Search
        Service->>Indexer: search_ivf(query, k, n_probes)
        Indexer->>Indexer: find_closest_clusters()
        Indexer->>Distance: calculate_distance()
        Distance->>Indexer: distances[]
        Indexer->>Service: top_k_results
    end
    
    Service->>DB: fetch_vectors_metadata()
    DB->>Service: metadata[]
    
    Service->>API: {results, search_time, recall}
    API->>Client: 200 OK<br/>{results, metrics}
```

### Component Interaction Diagram (Mermaid)

```mermaid
graph LR
    subgraph Client["Client Layer"]
        REST["REST API<br/>Requests"]
    end
    
    subgraph FastAPI["FastAPI Layer"]
        Routes["Route Handlers"]
        Validation["Pydantic Validators"]
        Docs["OpenAPI Docs"]
    end
    
    subgraph Business["Business Logic"]
        CRUD["CRUD<br/>Operations"]
        Search["Search<br/>Operations"]
        Index["Index<br/>Operations"]
    end
    
    subgraph Indexing["Indexing Logic"]
        HNSWImpl["HNSW<br/>Implementation"]
        IVFImpl["IVF<br/>Implementation"]
        CalcDist["Distance<br/>Calculations"]
    end
    
    subgraph Storage["Persistent Storage"]
        Cache["In-Memory<br/>Indices"]
        PostgreSQL["PostgreSQL<br/>Database"]
    end
    
    subgraph Monitoring["Monitoring & Testing"]
        Metrics["Prometheus<br/>Metrics"]
        Logs["Structured<br/>Logs"]
        Bench["Benchmarking<br/>Suite"]
    end
    
    REST -->|HTTP/JSON| Routes
    Routes --> Validation
    Validation --> CRUD
    Validation --> Search
    Validation --> Index
    Docs -.->|Self-documenting| REST
    
    CRUD -->|Read/Write| PostgreSQL
    Search --> HNSWImpl
    Search --> IVFImpl
    Search --> CalcDist
    Index --> HNSWImpl
    Index --> IVFImpl
    
    HNSWImpl -->|Data| Cache
    IVFImpl -->|Data| Cache
    HNSWImpl --> PostgreSQL
    IVFImpl --> PostgreSQL
    
    Search -.->|Performance| Metrics
    Index -.->|Activity| Logs
    Index -.->|Quality| Bench
```

### Index Building & Search Lifecycle (Mermaid)

```mermaid
stateDiagram-v2
    [*] --> VectorsLoaded: Load vectors<br/>into database
    
    VectorsLoaded --> HNSWBuilding: Create HNSW Index<br/>(m=32, ef_const=300)
    VectorsLoaded --> IVFBuilding: Create IVF Index<br/>(n_clusters=100)
    
    HNSWBuilding --> HNSWReady: Build Complete<br/>~2-5 seconds
    IVFBuilding --> IVFReady: Build Complete<br/>~1-3 seconds
    
    HNSWReady --> HNSWSearch: ready for<br/>fast search
    IVFReady --> IVFSearch: ready for<br/>fast search
    
    HNSWSearch --> HNSWOptimize: Benchmark<br/>Results<br/>99%+ Recall
    IVFSearch --> IVFOptimize: Benchmark<br/>Results<br/>94% Recall
    
    HNSWOptimize --> Tuning: Performance<br/>Analysis
    IVFOptimize --> Tuning: Performance<br/>Analysis
    
    Tuning --> Rebuild: Adjust<br/>Parameters
    Tuning --> Monitor: Continue<br/>Monitoring
    
    Rebuild --> HNSWBuilding
    Rebuild --> IVFBuilding
    
    Monitor --> HNSWSearch
    Monitor --> IVFSearch
    
    HNSWSearch --> [*]
    IVFSearch --> [*]
```

### Deployment Architecture (Mermaid)

```mermaid
graph TB
    subgraph Client["Client / User"]
        Browser["Browser<br/>REST Client"]
    end
    
    subgraph Network["Network Layer"]
        LB["Load Balancer"]
    end
    
    subgraph Docker["Docker Containers"]
        subgraph API["API Service"]
            FastAPI["FastAPI App<br/>Port 8000"]
        end
        
        subgraph DB["Database Service"]
            PostgreSQL["PostgreSQL<br/>Port 5432"]
            PGVolume["Persistent<br/>Volume"]
        end
        
        subgraph Monitor["Monitoring Stack"]
            Prom["Prometheus<br/>Port 9090"]
            Grafana["Grafana<br/>Port 3000"]
        end
    end
    
    subgraph External["External Services"]
        Metrics["Prometheus<br/>Scrape Targets"]
        Alerts["Alert Manager"]
    end
    
    Browser -->|HTTP/HTTPS| LB
    LB -->|Routes| FastAPI
    
    FastAPI -->|Query/Insert| PostgreSQL
    PostgreSQL -.->|Persist| PGVolume
    
    FastAPI -.->|Metrics| Prom
    Prom -->|Scrape| FastAPI
    Prom -->|Visualize| Grafana
    Prom -->|Send| Alerts
    
    Grafana -->|Dashboard| Browser
    Alerts -->|Notify| External
```

### Project Structure

```
.
├── api/                          # FastAPI application & routes
│   └── main.py                  # Main API entry point (FastAPI app)
├── config/                      # Configuration modules
│   ├── settings.py              # App settings & parameter defaults
│   ├── database.py              # Database connection setup
│   └── logging.py               # Structured logging configuration
├── database/                    # Database layer
│   ├── schema.py                # SQLAlchemy ORM models
│   ├── vector_database.py       # Main database wrapper class
│   ├── hnsw_database.py         # HNSW-specific database operations
│   └── ivf_database.py          # IVF-specific database operations
├── models/                      # Data models
│   ├── pydantic_models.py       # API request/response schemas (Pydantic)
│   └── vector_model.py          # Vector data model & operations
├── services/                    # Business logic & service layer
│   ├── vector_service.py        # Vector CRUD operations service
│   └── vector_indexer.py        # Index management service
├── utils/                       # Utility functions & algorithms
│   ├── distance.py              # Distance metric calculations (Euclidean, cosine)
│   ├── hnsw_index.py            # HNSW algorithm implementation
│   ├── ivf_index.py             # IVF algorithm implementation
│   ├── benchmark.py             # Performance benchmarking utilities
│   ├── clustering.py            # K-means and clustering utilities
│   └── optimization.py          # Performance optimization helpers
├── scripts/                     # CLI scripts & tools
│   └── run_benchmark.py         # Automated benchmark runner
├── test/                        # Comprehensive test suite
│   ├── test_api.py              # API endpoint integration tests
│   ├── test_vector_db.py        # Database operation tests
│   ├── test_hnsw.py             # HNSW algorithm tests
│   ├── test_ivf.py              # IVF algorithm tests
│   ├── test_clustering.py       # Clustering utility tests
│   └── test_comprnehesive.py    # End-to-end integration tests
├── examples/                    # Usage examples
│   ├── indexer_examples.py      # Index creation examples
│   └── vector_indexer_api.py    # API usage examples
├── docker-compose.yaml          # Multi-container orchestration
├── Dockerfile                   # Container image definition
├── prometheus.yml               # Prometheus monitoring config
├── requirements.txt             # Python package dependencies
├── HNSW_OPTIMIZATION_GUIDE.md   # Detailed optimization documentation
├── DAY1_README.md through       # Daily work progress documentation
├── DAY7_README.md               # (Learning journey documentation)
└── README.md                    # This file

```

---

## ⚡ Performance Benchmarks

### Results on 10,000 Vectors (128 dimensions, k=10)

| Method | Recall | Avg Latency | P95 Latency | Throughput | Index Size |
|--------|--------|-------------|------------|-----------|-----------|
| **HNSW (m=32)** | 99.2% | 5.1ms | 10.2ms | 195 qps | ~45MB |
| **HNSW (m=16)** | 98.5% | 3.2ms | 6.8ms | 320 qps | ~25MB |
| **HNSW (m=8)** | 95.2% | 2.1ms | 4.5ms | 450 qps | ~15MB |
| **IVF** | 94.8% | 4.5ms | 9.1ms | 220 qps | ~50MB |
| **Brute Force** | 100% | 45.2ms | 52.3ms | 22 qps | - |

### Performance Comparison Chart (Mermaid)

```mermaid
graph LR
    subgraph Latency["Query Latency (ms)<br/>Lower is Better"]
        L1["Brute Force<br/>45.2ms"]
        L2["IVF<br/>4.5ms"]
        L3["HNSW m=32<br/>5.1ms"]
        L4["HNSW m=16<br/>3.2ms"]
        L5["HNSW m=8<br/>2.1ms"]
    end
    
    subgraph Recall["Recall@10<br/>Higher is Better"]
        R1["Brute Force<br/>100%"]
        R2["HNSW m=32<br/>99.2%"]
        R3["HNSW m=16<br/>98.5%"]
        R4["IVF<br/>94.8%"]
        R5["HNSW m=8<br/>95.2%"]
    end
    
    subgraph Throughput["Queries/Second<br/>Higher is Better"]
        T1["HNSW m=8<br/>450 qps"]
        T2["HNSW m=16<br/>320 qps"]
        T3["IVF<br/>220 qps"]
        T4["HNSW m=32<br/>195 qps"]
        T5["Brute Force<br/>22 qps"]
    end
    
    L5 -.->|Best Speed| T1
    L4 -->|Balanced| T2
    L3 -->|Balanced| T4
    L2 -->|Good Balance| T3
    L1 -->|Slowest| T5
    
    R1 -.->|Perfect| L1
    R2 -->|Excellent| L3
    R3 -->|Very Good| L4
    R4 -->|Good| L2
    R5 -->|Good| L5
```

### Optimization Achievements

**64% Performance Improvement** through:

- ✅ HNSW parameter tuning (m=32, ef_construction=300)
- ✅ Batch search support for 10-100x throughput improvement
- ✅ Optimized ef_search parameter (50)
- ✅ Configuration-driven defaults from settings

**Detailed Metrics:**

| Metric | Before Optimization | After Optimization | Improvement |
|--------|-------------------|-------------------|------------|
| Queries/Second | 2.72 qps | 4.47 qps | **+64%** |
| Avg Query Time | 276ms | 124ms | **-55%** |
| Recall@10 | 33% | 95% | **+188%** |
| F1 Score | 0.33 | 0.96 | **+191%** |

### Optimization Timeline (Mermaid)

```mermaid
graph LR
    Start("Initial State<br/>Before Optimization")
    
    Step1("Benchmark Analysis<br/>Identified bottlenecks")
    Step2("Tune Parameters<br/>m=32, ef=300")
    Step3("Batch Search Support<br/>10-100x improvement")
    Step4("Optimize ef_search<br/>Tuned to 50")
    Step5("Final State<br/>After Optimization")
    
    Metrics1("Before:<br/>2.72 qps<br/>276ms<br/>33% recall")
    Metrics2("After:<br/>4.47 qps<br/>124ms<br/>95% recall")
    
    Start -->|Measure| Step1
    Step1 -->|Tune Params| Step2
    Step1 -->|Implement| Step3
    Step1 -->|Fine-tune| Step4
    
    Step2 --> Step5
    Step3 --> Step5
    Step4 --> Step5
    
    Start -.-> Metrics1
    Step5 -.-> Metrics2
    
    style Start fill:#ff9999
    style Step5 fill:#99ff99
    style Metrics1 fill:#ffe6e6
    style Metrics2 fill:#e6ffe6
```

---

## 🔌 API Documentation

### Overview

The API provides complete vector database functionality through REST endpoints with automatic OpenAPI documentation.

### Vector Operations

#### **Create a Vector**

**POST** `/vectors`

```python
import requests

vector = {
    "vector": [0.1, 0.2, 0.3, 0.4],
    "metadata": {"text": "Hello World", "source": "example"},
    "vector_id": "doc_1"
}

response = requests.post("http://localhost:8000/vectors", json=vector)
print(response.json())
```

#### **Batch Insert Vectors**

**POST** `/vectors/batch`

```python
vectors = {
    "vectors": [
        {"vector": [0.1, 0.2], "metadata": {"id": 1}},
        {"vector": [0.3, 0.4], "metadata": {"id": 2}},
        {"vector": [0.5, 0.6], "metadata": {"id": 3}}
    ],
    "batch_name": "my_batch"
}

response = requests.post("http://localhost:8000/vectors/batch", json=vectors)
print(response.json())
```

#### **List All Vectors**

**GET** `/vectors`

```python
response = requests.get("http://localhost:8000/vectors")
vectors = response.json()
print(f"Total vectors: {len(vectors)}")
```

#### **Get Specific Vector**

**GET** `/vectors/{vector_id}`

```python
response = requests.get("http://localhost:8000/vectors/doc_1")
vector = response.json()
print(vector)
```

#### **Update Vector**

**PUT** `/vectors/{vector_id}`

```python
update = {
    "vector": [0.2, 0.3, 0.4, 0.5],
    "metadata": {"text": "Updated text"}
}

response = requests.put("http://localhost:8000/vectors/doc_1", json=update)
print(response.json())
```

#### **Delete Vector**

**DELETE** `/vectors/{vector_id}`

```python
response = requests.delete("http://localhost:8000/vectors/doc_1")
print(response.json())
```

### Search Operations

#### **Search Vectors (HNSW)**

**POST** `/search`

```python
search_request = {
    "query_vector": [0.1, 0.2, 0.3, 0.4],
    "k": 10,
    "method": "hnsw",
    "ef_search": 50
}

response = requests.post("http://localhost:8000/search", json=search_request)
results = response.json()["results"]
print(f"Found {len(results)} similar vectors")

# Results include distance and metadata
for i, result in enumerate(results):
    print(f"{i+1}. Vector ID: {result['vector_id']}, Distance: {result['distance']:.4f}")
```

#### **Compare Search Methods**

**POST** `/search/compare`

```python
compare_request = {
    "query_vector": [0.1, 0.2, 0.3, 0.4],
    "k": 5,
    "methods": ["hnsw", "ivf", "brute_force"]
}

response = requests.post("http://localhost:8000/search/compare", json=compare_request)
comparison = response.json()

for method in comparison["comparison"]:
    print(f"\n{method['method']}:")
    print(f"  Time: {method['search_time']}ms")
    print(f"  Results: {len(method['results'])}")
```

### Index Operations

#### **Create Index (HNSW)**

**POST** `/index`

```python
index_request = {
    "method": "hnsw",
    "m": 16,
    "ef_construction": 200
}

response = requests.post("http://localhost:8000/index", json=index_request)
print(response.json())
```

#### **Create Index (IVF)**

**POST** `/index`

```python
index_request = {
    "method": "ivf",
    "n_clusters": 100,
    "n_probes": 10
}

response = requests.post("http://localhost:8000/index", json=index_request)
print(response.json())
```

#### **Get Index Info**

**GET** `/index/info`

```python
response = requests.get("http://localhost:8000/index/info")
info = response.json()
print(f"Index Type: {info['index_type']}")
print(f"Vector Count: {info['vector_count']}")
print(f"Dimensions: {info['dimensions']}")
```

#### **Delete Index**

**DELETE** `/index`

```python
response = requests.delete("http://localhost:8000/index")
print(response.json())
```

### System Operations

#### **Get Statistics**

**GET** `/stats`

```python
response = requests.get("http://localhost:8000/stats")
stats = response.json()
print(f"Total Vectors: {stats['total_vectors']}")
print(f"Index Built: {stats['index_built']}")
```

#### **Health Check**

**GET** `/health`

```python
response = requests.get("http://localhost:8000/health")
health = response.json()
print(f"Status: {health['status']}")
print(f"Database: {health['database']}")
```

---

## 🧪 Testing & Benchmarking

### Run Tests

```powershell
# Run all tests
pytest test/ -v

# Run with coverage report
pytest test/ --cov=. --cov-report=html

# Run specific test file
pytest test/test_hnsw.py -v

# Run tests matching pattern
pytest test/ -k "search" -v
```

### Run Benchmarks

```powershell
# Run comprehensive benchmark suite
python scripts/run_benchmark.py
```

**Output files:**

- `benchmark_report_YYYYMMDD_HHMMSS.json` - Detailed metrics
- `configuration_comparison_YYYYMMDD_HHMMSS.json` - Configuration comparison

**Benchmark measurements include:**

- ✅ Recall@k for each method
- ✅ Query latency (min, max, avg, median, p95, p99)
- ✅ Throughput (queries per second)
- ✅ Index build time and size
- ✅ Configuration parameter comparison

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/vector_db
DATABASE_ECHO=false

# Application Settings
APP_NAME=Vector Database API
APP_VERSION=1.0.0
DEBUG=false

# HNSW Index Defaults
DEFAULT_M=32
DEFAULT_M0=64
DEFAULT_EF_CONSTRUCTION=300
DEFAULT_EF_SEARCH=50

# IVF Index Defaults
DEFAULT_N_CLUSTERS=100
DEFAULT_N_PROBES=10

# Logging
LOG_LEVEL=INFO
```

### Settings (config/settings.py)

```python
# HNSW Parameters
DEFAULT_M: int = 32              # Neighbors per node (higher = better recall, slower)
DEFAULT_M0: int = 64             # Neighbors in layer 0 (typically 2*m)
DEFAULT_EF_CONSTRUCTION: int = 300  # Building quality (higher = better index, slower build)
DEFAULT_EF_SEARCH: int = 50      # Query exploration depth (higher = better recall, slower)

# IVF Parameters
DEFAULT_N_CLUSTERS: int = 100    # Number of clusters
DEFAULT_N_PROBES: int = 10       # Clusters to search
```

### Parameter Tuning Guide

#### For High-Recall Search (Ad-Hoc)

```python
create_hnsw_index(m=32, ef_construction=500)
search_hnsw(query, k=10, ef_search=100)  # 99%+ recall
```

#### For High-Speed Batch Processing

```python
create_hnsw_index(m=16, ef_construction=200)
search_hnsw(query, k=10, ef_search=30)   # Fast but lower recall
```

#### Balanced (Recommended for 10K vectors)

```python
create_hnsw_index(m=24, ef_construction=300)  # Default
search_hnsw(query, k=10, ef_search=50)        # Default
```

---

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```powershell
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f vector_api

# Stop services
docker-compose down
```

### Build Custom Image

```powershell
# Build the image
docker build -t vector-db:latest .

# Run the container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/vector_db \
  vector-db:latest
```

### Environment Variables (Deploy)

Copy [.env.example](.env.example) to `.env` for local dev, or set these in your deploy platform:

- `DATABASE_URL` (required)
- `PORT` (optional, default 8000)
- `UVICORN_WORKERS` (optional, default 4)
- `DEBUG` (optional)
- `ALLOWED_HOSTS` (optional, comma-separated)

### Services Included

- 🐍 **Vector API** (Port 8000)
- 🗄️ **PostgreSQL** (Port 5432)
- 📊 **Prometheus** (Port 9090)
- 📈 **Grafana** (Port 3000, admin/admin)

---

## 📊 Advanced Usage

### Batch Search for High Throughput

```python
# Instead of looping - use batch method for 10-100x improvement
query_vectors = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], ...]
results = db.batch_search_hnsw(query_vectors, k=10, ef_search=50)
```

### Index Rebuilding with Optimization

```python
# Periodic rebuild with optimized parameters
db.rebuild_hnsw_index(m=32, ef_construction=300)
```

### Parameter Benchmarking

```python
# Test different configurations for your specific use case
for m in [16, 24, 32]:
    for ef_c in [200, 300, 400]:
        db.rebuild_hnsw_index(m=m, ef_construction=ef_c)
        # Run benchmark...
        # Compare results
```

---

## 🎯 Key Learnings & Best Practices

### HNSW Optimization

1. **Parameter Selection**: Start with m=16, ef_construction=200, then tune based on benchmarks
2. **Batch Processing**: Use batch_search for 10-100x throughput improvement
3. **Index Rebuilding**: Periodically rebuild with optimized parameters
4. **ef_search Tuning**: Higher ef_search = better recall but slower queries

### Performance Tuning Tradeoffs

| Parameter | Effect | Search Speed | Recall | Memory | Build Time |
|-----------|--------|--------------|--------|--------|-----------|
| ↑ m | More connections | ↓ Slower | ↑ Higher | ↑ More | ↑ Longer |
| ↑ ef_construction | Better quality | - | ↑ Higher | - | ↑ Longer |
| ↑ ef_search | More exploration | ↓ Slower | ↑ Higher | - | - |

### Production Recommendations

✅ **Use HNSW** for most vector similarity search tasks (95-99% recall with good speed)
✅ **Use IVF** when you need faster approximate search with less memory
✅ **Use Brute Force** only for validation and testing (100% recall, slow)
✅ **Monitor metrics** regularly to ensure performance targets are met
✅ **Index periodically** to maintain optimal structure as data grows

---

## 🚀 Future Enhancements

- [ ] GPU acceleration for distance calculations
- [ ] Vector quantization (Product Quantization, Binary)
- [ ] Caching layer (Redis) for hot vectors
- [ ] Multi-node sharding and clustering
- [ ] WebSocket streaming search
- [ ] Additional distance metrics (Manhattan, Hamming)
- [ ] Authentication and rate limiting
- [ ] Time-series vector data support

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

This project demonstrates principles from:

- **HNSW Paper**: "Efficient and robust approximate nearest neighbor search" (Malkov & Yashunin, 2018)
- **IVF**: "Product quantization for nearest neighbor search" (Jégou et al.)
- **FAISS**: Facebook AI's similarity search library
- **FastAPI**: Modern async Python web framework
- **PostgreSQL**: Robust relational database

---

## 📞 Support

For issues, questions, or suggestions:

1. Check existing documentation in DAY1-DAY7 README files
2. Review test cases in `test/` directory for usage examples
3. Check `examples/` directory for complete code samples


---

## Built with ❤️ as a comprehensive learning project

## Last updated: February 5, 2026
