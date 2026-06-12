# Building a Vector Database from Scratch

[![GitHub Stars](https://img.shields.io/github/stars/KunjShah95/BUILDING-MY-OWN-VECTOR-DB?style=flat-square)](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB)
[![GitHub Forks](https://img.shields.io/github/forks/KunjShah95/BUILDING-MY-OWN-VECTOR-DB?style=flat-square)](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-12%2B-336791?style=flat-square&logo=postgresql)](https://postgresql.org)

A **production-ready vector database** built from scratch in Python with **FastAPI**, **PostgreSQL**, and custom implementations of **HNSW**, **IVF**, **PQ**, **LSH**, **KD-Tree**, **VP-Tree**, and **BM25** indexing algorithms. Supports **multimodal ingestion** (text, image, audio), **RAG pipelines**, **hybrid search**, **cross-encoder re-ranking**, **gRPC**, **WebSocket streaming**, **OpenAI-compatible endpoints**, **multi-tenancy**, **metric monitoring**, and **Kubernetes deployment**.

> This repo is a hands-on learning project for vector search, approximate nearest neighbors, database design, multimodal retrieval, RAG, and API performance tuning.

---

[GitHub](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB) · [Docs](docs/) · [Roadmap](ROADMAP.md) · [API Guide](API_INTEGRATION_GUIDE.md) · [Issues](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues)

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture Overview](#architecture-overview)
- [Indexing Algorithms](#indexing-algorithms)
- [API Reference](#api-reference)
- [Services Layer](#services-layer)
- [SDK & Clients](#sdk--clients)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Performance Benchmarks](#performance-benchmarks)
- [Testing](#testing)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security](#security)

---

## Highlights

- **9 ANN algorithms** — HNSW, IVF, PQ, Int8, LSH, KD-Tree, VP-Tree, BM25, Hybrid (RRF)
- **Crash durability** — Write-Ahead Log with fsync + recovery replay; checkpoint/truncate after snapshot
- **Background compaction** — HNSW soft-delete tombstones reclaimed by a daemon thread
- **Larger-than-RAM storage** — memory-mapped vector store with dynamic growth and row reclamation
- **Cost-based query planner** — AST parser for hybrid metadata + `semantic_match` queries with filter-first/vector-first optimization, exposed at `/search-engine/hybrid-query`
- **Row-level RBAC** — per-key operation gates + metadata-predicate document security (`utils/rbac.py`)
- **Distributed query aggregation** — parallel scatter-gather coordinator with global top-K fusion (distance/RRF) and shard fault-tolerance
- **Dynamic quantization** — memory-pressure-driven precision policy (fp32 → Int8 → PQ → Binary)
- **Startup crash recovery** — pending WALs auto-replayed into HNSW/IVF indexes on boot
- **Hybrid search** — Dense vector + BM25 sparse retrieval fused via Reciprocal Rank Fusion (RRF)
- **Cross-encoder re-ranking** — Local `cross-encoder/ms-marco-MiniLM-L-6-v2` model + Cohere API fallback
- **Multimodal ingestion** — Text (Sentence-Transformers), Image (CLIP ViT-B/32), Audio (librosa MFCC w/ chunking)
- **Long-audio chunking** — Split long audio into segment-level vectors with configurable chunk duration
- **RAG pipeline** — PDF/DOCX/HTML/Markdown ingestion → chunk → embed → store → retrieve → LLM answer
- **Streaming RAG** — Server-Sent Events (SSE) token-by-token responses
- **gRPC API** — Protobuf-defined service mirroring the REST API
- **WebSocket search** — Real-time streaming search results
- **GraphQL API** — Strawberry-based GraphQL schema with queries, mutations, auto-generated docs
- **OpenAI-compatible endpoints** — Drop-in replacement for `/v1/embeddings` and `/v1/chat/completions`
- **Multi-tenancy** — Isolated tenants, API key authentication, per-tenant rate limiting
- **Time-series vectors** — Timestamped vectors with series grouping, time-range search, and window aggregation
- **SQL pre-filtering** — JSONB metadata filtering at the database level before ANN search
- **Async ingestion queue** — In-memory batch queue with periodic flush for large uploads
- **Cloud storage** — S3/Azure Blob storage backend wired into media store
- **Int8 quantization** — 4x memory compression with per-dimension min/max quantization
- **Distributed/partitioned indexes** — Hash/range partitioning across HNSW/IVF/PQ/Int8 shards
- **CLIP quality tuning** — Temperature scaling and normalization control for cross-modal text→image search
- **GPU acceleration** — Optional CuPy kernels for batch distance computation
- **Prometheus metrics** — `/metrics` endpoint with request counters and latency histograms
- **Dashboard UI** — Real-time monitoring with Chart.js visualizations
- **Python SDK** — `vector_db_client` PyPI package with full API coverage
- **Docker Compose** — API + PostgreSQL + Prometheus + Grafana stack
- **Helm chart** — Kubernetes deployment with HPA, PVCs, and probes
- **Terraform** — AWS and Azure infrastructure templates
- **CI/CD** — GitHub Actions pipeline with lint, test, and build stages
- **190+ tests** — Covering API endpoints, index algorithms, services, durability (WAL/compaction/mmap), query planner, RBAC, distributed coordinator, dynamic quantization, and utilities

---

## Architecture Overview

### System Architecture

```mermaid
graph TB
    Client[Client / App / SDK]
    API[FastAPI REST + WebSocket]
    gRPC[gRPC Server]
    Services[Service Layer]
    DB[(PostgreSQL + pgvector)]
    HNSWIndex[HNSW Graph]
    IVFIndex[IVF Inverted File]
    PQIndex[Product Quantization]
    BM25[BM25 Sparse Index]
    Media[Media Storage<br/>Local / S3 / Azure Blob]
    Redis[(Redis Cache)]
    LLM[OpenAI / LLM]
    Monitoring[Prometheus + Grafana]

    Client --> API
    Client --> gRPC
    API --> Services
    gRPC --> Services
    Services --> DB
    Services --> HNSWIndex
    Services --> IVFIndex
    Services --> PQIndex
    Services --> BM25
    Services --> Media
    Services --> Redis
    Services --> LLM
    API --> Monitoring
```

### Layer Architecture

```
┌─────────────────────────────────────────────────┐
│                  API Layer                       │
│  REST (FastAPI) │ gRPC │ WebSocket │ Dashboard  │
├─────────────────────────────────────────────────┤
│               Middleware Layer                   │
│  Auth │ CORS │ Rate Limiting │ Request Size     │
├─────────────────────────────────────────────────┤
│               Service Layer                     │
│  Vector │ Collection │ Multimodal │ RAG │ Auth  │
│  Embedding │ Hybrid Search │ Reranker │ Cache   │
├─────────────────────────────────────────────────┤
│              Database Layer                     │
│  SQLAlchemy ORM │ pgvector │ Vector Model       │
├─────────────────────────────────────────────────┤
│              Algorithm Layer                    │
│  HNSW │ IVF │ PQ │ LSH │ KD-Tree │ VP-Tree     │
│  BM25 │ K-Means │ Distance │ Optimization       │
├─────────────────────────────────────────────────┤
│              Storage Layer                      │
│  PostgreSQL │ Local FS │ S3 │ Azure Blob        │
└─────────────────────────────────────────────────┘
```

### Request Flow

1. Client sends request (REST/gRPC/WebSocket) with API key
2. Auth middleware validates key, resolves tenant, checks rate limit
3. Request reaches appropriate router (Vector, Collection, Multimodal, RAG, etc.)
4. Service layer orchestrates business logic (embed, search, index, store)
5. Database layer persists/retrieves vectors from PostgreSQL
6. Algorithm layer performs ANN search (HNSW/IVF/PQ) or exact search
7. Optional post-processing: metadata filtering, cross-encoder re-ranking, hybrid fusion
8. Response returned to client with timing metrics

---

## Indexing Algorithms

| Algorithm | File | Type | Parameters | Best For |
|-----------|------|------|------------|----------|
| **HNSW** | `utils/hnsw_index.py` | Hierarchical graph | `m`, `m0`, `ef_construction`, `ef_search` | General purpose, high recall |
| **IVF** | `utils/ivf_index.py` | Inverted file + PQ | `n_clusters`, `n_probes` | Large datasets, fast build |
| **PQ** | `utils/product_quantization.py` | Product Quantization | `M` (sub-quantizers), `k_sub` | Memory-constrained, 10-48x compression |
| **Int8** | `utils/int8_index.py` | 8-bit quantization | `distance_metric` | 4x compression with minimal recall loss |
| **LSH** | `utils/lsh_index.py` | Locality-Sensitive Hashing | `n_hash_tables`, `n_hash_bits` | Approximate, probabilistic |
| **KD-Tree** | `utils/kdtree_index.py` | Space-partitioning tree | `leaf_size` | Low-dimensional (<20) exact search |
| **VP-Tree** | `utils/vp_tree.py` | Metric tree | `leaf_size` | Arbitrary distance metrics |
| **BM25** | `utils/bm25_index.py` | Sparse text retrieval | `k1`, `b` | Text search, hybrid search |
| **K-Means** | `utils/clustering.py` | Clustering | `k`, `max_iters` | IVF training, PQ codebook learning |

### HNSW (Hierarchical Navigable Small World)

The flagship index — a multi-layer graph where upper layers have fewer nodes with longer edges for fast navigation, and lower layers have dense connections for fine-grained search.

Key features:
- **Hierarchical structure**: Nodes assigned to random levels via exponential distribution
- **Greedy best-first search**: Navigates from top layer down to layer 0
- **Dynamic insertion**: Single-vector insert without full rebuild
- **Configurable recall/speed**: `ef_search` controls search breadth; `ef_construction` controls build quality
- **Bidirectional connections**: Each edge is symmetric for robust graph traversal

### IVF + PQ (Inverted File with Product Quantization)

Two-stage quantization:
1. **Coarse quantizer** (K-Means): Assigns vectors to nearest cluster centroid
2. **Fine quantizer** (Product Quantization): Compresses residuals (vector − centroid) into compact codes

Search uses **Asymmetric Distance Computation (ADC)** — query is not compressed, while database vectors are. This gives excellent recall with 10-48x memory compression (384-dim → 32 bytes with M=32).

### Hybrid Search (Dense + Sparse)

Combines dense vector search (HNSW/IVF/brute) with sparse BM25 retrieval via **Reciprocal Rank Fusion (RRF)**:

```
RRF(d) = Σ 1/(k + rank_i(d))   for each result set i
```

Where `k = 60` (default) is the ranking constant. Results are fused, normalized, and sorted by RRF score.

### Benchmark Snapshot (10K vectors, 128-dim)

| Metric | Value |
|--------|-------|
| Average recall@10 | 0.981 |
| Average precision@10 | 0.999 |
| Average F1 score | 0.983 |
| Avg query time (HNSW) | 0.129 s |
| Throughput | 4.46 qps |
| Build time (HNSW, m=32) | 45.2 s |

> These numbers are from the saved benchmark reports. See `scripts/run_benchmark.py` and `scripts/scale_benchmark.py` for reproduction.

---

## API Reference

The API serves **80+ REST endpoints** plus gRPC, GraphQL, and WebSocket. Full OpenAPI docs at `/docs` (Swagger) and `/redoc` (ReDoc).

### Root & Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | App info (name, version, docs, health) |
| `GET` | `/health` | Health check (DB connection, index, vector count) |
| `GET` | `/ready` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

### Collections

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections` | Create a named collection (text/image/audio/multimodal) |
| `GET` | `/collections` | List all collections (pagination, tenant-scoped) |
| `GET` | `/collections/{id}` | Get collection details |
| `DELETE` | `/collections/{id}` | Delete collection (cascade deletes vectors, index files) |
| `POST` | `/collections/{id}/index` | Build & persist per-collection HNSW/IVF index |
| `POST` | `/collections/{id}/index/save` | Save per-collection index to disk |
| `POST` | `/collections/{id}/index/load` | Load per-collection index from disk |
| `POST` | `/collections/{id}/index/ivf/rebuild` | Rebuild IVF index with new parameters |
| `GET` | `/collections/{id}/index/stats` | Collection index status & graph statistics |
| `POST` | `/collections/{id}/index/sparse` | Build BM25 sparse index |

### Multimodal Ingestion

| Method | Path | Modality | Description |
|--------|------|----------|-------------|
| `POST` | `/collections/{id}/ingest/text` | Text | Embed text server-side, store as vector |
| `POST` | `/collections/{id}/ingest/image` | Image | Upload image → CLIP embed → store |
| `POST` | `/collections/{id}/ingest/audio` | Audio | Upload audio → librosa MFCC → store (supports `chunk_seconds` for long-audio segmentation) |
| `POST` | `/collections/{id}/ingest/pdf` | Document | PDF → extract text → chunk → embed → store |

### Multimodal Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/{id}/search/text` | Natural language text search |
| `POST` | `/collections/{id}/search/image` | Image similarity search (upload query image) |
| `POST` | `/collections/{id}/search/audio` | Audio similarity search (upload query audio) |
| `POST` | `/collections/{id}/search/hybrid` | Hybrid dense + sparse search (RRF fusion) |
| `POST` | `/collections/{id}/search/rerank` | Search + cross-encoder re-ranking |
| `POST` | `/collections/{id}/vectors/timeseries` | Insert time-series vector with timestamp & series_id |
| `GET` | `/collections/{id}/search/timeseries` | Time-range vector search with series filtering |
| `GET` | `/collections/{id}/series` | List distinct time-series in a collection |
| `GET` | `/collections/{id}/series/{sid}/latest` | Get most recent vectors in a series |
| `GET` | `/collections/{id}/series/{sid}/aggregate` | Aggregate vectors over time windows |

### Vector CRUD

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/vectors` | Create single vector |
| `POST` | `/vectors/batch` | Batch insert vectors |
| `GET` | `/vectors` | List vectors (paginated) |
| `GET` | `/vectors/{id}` | Get vector by ID |
| `PUT` | `/vectors/{id}` | Update vector data/metadata |
| `DELETE` | `/vectors/{id}` | Delete vector |

### Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Search with method selection (hnsw/ivf/brute/pq/hybrid/int8) |
| `GET` | `/search/compare` | Compare all available search methods |
| `POST` | `/search/batch` | Batch search multiple queries |
| `WS` | `/ws/search/{collection_id}` | WebSocket streaming search |

### Index Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/index` | Create index (HNSW/IVF/PQ, global or per-collection) |
| `POST` | `/index/save` | Persist index to disk |
| `POST` | `/index/load` | Load index from disk |
| `GET` | `/index` | Index info & statistics |
| `POST` | `/collections/{id}/index/sparse` | Build BM25 sparse index |

### RAG & LLM

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/{id}/query` | RAG query (retrieve + LLM answer) |
| `GET` | `/collections/{id}/query/stream` | Streaming RAG query (SSE) |
| `POST` | `/llm/stream` | Generic LLM streaming completion |

### OpenAI-Compatible

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List available embedding & chat models |
| `POST` | `/v1/embeddings` | Text → embedding vector (OpenAI-compatible) |
| `POST` | `/v1/chat/completions` | Chat with optional RAG context (OpenAI-compatible) |

### Authentication & Tenants

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/keys/create` | Create API key (scoped to tenant+collection) |
| `DELETE` | `/api/keys/revoke` | Revoke API key |
| `GET` | `/api/keys/list/{tenant_id}` | List keys for a tenant |
| `POST` | `/api/tenants` | Create tenant |
| `GET` | `/api/tenants` | List tenants |
| `GET` | `/api/tenants/{id}` | Get tenant |
| `PATCH` | `/api/tenants/{id}` | Update tenant |
| `DELETE` | `/api/tenants/{id}` | Delete tenant |
| `POST` | `/api/tenants/{id}/keys` | Create tenant-scoped API key |
| `GET` | `/api/tenants/{id}/keys` | List tenant API keys |
| `DELETE` | `/api/tenants/{id}/keys/{key}` | Revoke tenant API key |

### Dashboard & Media

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard` | Dashboard UI (HTML) |
| `GET` | `/api/dashboard/stats` | Aggregated stats |
| `GET` | `/api/dashboard/latency` | Query latency metrics |
| `GET` | `/api/dashboard/index-info` | Index status info |
| `GET` | `/media?content_uri=...` | Serve stored media files |

### Playground

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/playground/templates` | List API templates |
| `POST` | `/playground/templates` | Create API template |
| `POST` | `/playground/feedback` | Submit feedback |

### Vector Indexer (Legacy)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/indexer/config` | Configure indexer |
| `POST` | `/api/indexer/create` | Create index from vectors |
| `POST` | `/api/indexer/search` | Single search |
| `POST` | `/api/indexer/search-batch` | Batch search |
| `GET` | `/api/indexer/stats` | Get indexer statistics |
| `GET` | `/api/indexer/health` | Indexer health check |
| `POST` | `/api/indexer/parameters/suggest` | Auto-parameter suggestions |
| `POST` | `/api/indexer/save` | Save index to disk |
| `POST` | `/api/indexer/load` | Load index from disk |

### gRPC API

The gRPC server runs on `0.0.0.0:50051` and provides these RPCs:

| RPC | Description |
|-----|-------------|
| `Search` | Vector similarity search |
| `Insert` | Insert single vector |
| `BatchInsert` | Batch vector insertion |
| `Delete` | Delete a vector |
| `GetVector` | Get vector by ID |
| `ListCollections` | List all collections |
| `Health` | Health check |

Protobuf definition: `api/grpc/vector_service.proto`

### Full Endpoint Summary

```
GET    /                              App info
GET    /health                        Health check
GET    /ready                         Readiness probe
GET    /metrics                       Prometheus metrics

POST   /collections                   Create collection
GET    /collections                   List collections
GET    /collections/{id}              Get collection
DELETE /collections/{id}              Delete collection
POST   /collections/{id}/index        Build per-collection index
GET    /collections/{id}/index/stats  Collection index stats
POST   /collections/{id}/index/sparse Build sparse index

POST   /collections/{id}/ingest/text     Ingest text
POST   /collections/{id}/ingest/image    Ingest image
POST   /collections/{id}/ingest/audio    Ingest audio
POST   /collections/{id}/ingest/pdf      Ingest PDF

POST   /collections/{id}/search/text     Search text
POST   /collections/{id}/search/image    Search image
POST   /collections/{id}/search/audio    Search audio
POST   /collections/{id}/search/hybrid   Hybrid search
POST   /collections/{id}/search/rerank   Search + rerank
POST   /collections/{id}/query           RAG query
GET    /collections/{id}/query/stream    Streaming RAG

POST   /vectors                      Create vector
POST   /vectors/batch                Batch create
GET    /vectors                      List vectors
GET    /vectors/{id}                 Get vector
PUT    /vectors/{id}                 Update vector
DELETE /vectors/{id}                 Delete vector

POST   /search                       Search vectors
GET    /search/compare               Compare methods
POST   /search/batch                 Batch search

POST   /index                        Create index
POST   /index/save                   Save index
POST   /index/load                   Load index
GET    /index                        Index info

GET    /v1/models                    List models (OpenAI compat)
POST   /v1/embeddings                Embeddings (OpenAI compat)
POST   /v1/chat/completions          Chat completions (OpenAI compat)

POST   /llm/stream                   Streaming LLM

POST   /api/keys/create              Create API key
DELETE /api/keys/revoke              Revoke API key
GET    /api/keys/list/{tenant}       List keys

POST   /api/tenants                  Create tenant
GET    /api/tenants                  List tenants
GET    /api/tenants/{id}             Get tenant
PATCH  /api/tenants/{id}             Update tenant
DELETE /api/tenants/{id}             Delete tenant
POST   /api/tenants/{id}/keys        Create tenant key
GET    /api/tenants/{id}/keys        List tenant keys
DELETE /api/tenants/{id}/keys/{key}  Revoke tenant key

WS     /ws/search/{collection_id}    WebSocket search
WS     /ws/health                    WebSocket health

GET    /dashboard                    Dashboard UI
GET    /api/dashboard/stats          Dashboard stats
GET    /api/dashboard/latency        Dashboard latency
GET    /api/dashboard/index-info     Dashboard index info

GET    /media                        Serve media files
GET    /playground/templates         List templates
POST   /playground/templates         Create template
POST   /playground/feedback          Submit feedback

POST   /api/indexer/config           Indexer config
POST   /api/indexer/create           Indexer create
POST   /api/indexer/search           Indexer search
POST   /api/indexer/search-batch     Indexer batch search
GET    /api/indexer/stats            Indexer stats
GET    /api/indexer/health           Indexer health
POST   /api/indexer/parameters/suggest  Parameter suggestions
POST   /api/indexer/save             Indexer save
POST   /api/indexer/load             Indexer load
GET    /api/indexer/methods          Supported methods
```

---

## Services Layer

The `services/` directory contains the core business logic:

| Service | File | Responsibilities |
|---------|------|-----------------|
| `VectorService` | `services/vector_service.py` | Vector CRUD, search (HNSW/IVF/PQ/brute/hybrid), index management, PQ operations |
| `CollectionService` | `services/collection_service.py` | Collection CRUD, modality validation, dimension enforcement, multi-tenancy |
| `CollectionIndexService` | `services/collection_index_service.py` | Per-collection HNSW/IVF index build, load, search, stats |
| `MultimodalService` | `services/multimodal_service.py` | Text/image/audio ingest and search, modality-gated routing |
| `EmbeddingService` | `services/embedding_service.py` | Text (Sentence-Transformers), Image (CLIP), Audio (librosa MFCC) embedding |
| `RAGService` | `services/rag_service.py` | PDF/document ingestion, chunking, RAG query with LLM |
| `StreamingRAGService` | `services/streaming_rag_service.py` | SSE token-by-token streaming RAG |
| `HybridSearchService` | `services/hybrid_search_service.py` | BM25 sparse index build + dense/sparse fusion via RRF |
| `RerankerService` | `services/reranker_service.py` | Cross-encoder re-ranking (local + Cohere API fallback) |
| `AuthService` | `services/auth_service.py` | API key generation (SHA-256 hashed), validation, revocation |
| `TenantService` | `services/tenant_service.py` | Tenant CRUD, token-bucket rate limiting |
| `IngestionService` | `services/ingestion_service.py` | Async bulk ingestion queue with batched flushes + REST API |
| `CacheService` | `services/cache_service.py` | Redis-backed search result & embedding caching |
| `MediaStore` | `services/media_store.py` | File persistence via StorageBackend (Local FS / S3 / Azure Blob) |
| `StorageBackend` | `services/storage_backend.py` | Abstract storage (Local FS / S3 / Azure Blob) |
| `MetadataFilter` | `services/metadata_filter.py` | SQL pre-filtering + post-filtering with 10+ operators |
| `AutoReindexService` | `services/auto_reindex.py` | Threshold-based automatic index rebuild scheduling |
| `VectorIndexer` | `services/vector_indexer.py` | Unified indexer API with parameter auto-optimization |

---

## SDK & Clients

### Python SDK (`sdk/vector_db_client/`)

Install from the repo:

```bash
pip install -e sdk
```

```python
from vector_db_client import VectorDBClient

with VectorDBClient("http://localhost:8000") as client:
    # Collections
    client.collections.create(
        name="Docs",
        collection_id="product-docs",
        modality="text",
        dimension=384,
    )

    # Text ingest & search
    client.multimodal.ingest_text(
        "product-docs",
        "Returns are accepted within 30 days.",
        vector_id="policy-returns",
    )
    hits = client.multimodal.search_text("product-docs", "how do I return an item?")
    print(hits.results[0].vector_id, hits.results[0].distance)

    # Image ingest & search
    client.multimodal.ingest_image("photos", path="cat.jpg")
    results = client.multimodal.search_image("photos", path="query.jpg", k=5)

    # Audio ingest & search
    client.multimodal.ingest_audio("voice-clips", path="clip.wav")

    # CRUD
    client.vectors.create([0.1, 0.2, 0.3], metadata={"text": "sample"})
    client.vectors.get("vec_abc123")
    client.vectors.delete("vec_abc123")
```

SDK API surface:

| Resource | Methods |
|----------|---------|
| `client.collections` | `create`, `list`, `get`, `delete` |
| `client.vectors` | `create`, `get`, `delete`, `search` |
| `client.multimodal` | `ingest_text`, `search_text`, `ingest_image`, `search_image`, `ingest_audio`, `search_audio` |

Errors raise `VectorDBHTTPError` with `status_code` and `detail`.

### LangChain Integration

The SDK includes a `LangChainVectorStore` at `sdk/vector_db_client/langchain_vectorstore.py` for seamless integration with LangChain workflows.

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **PostgreSQL 12+** with `pgvector` extension (optional)
- **PowerShell** (Windows) or bash (macOS/Linux)

### Local Setup

1. **Clone and enter the repository:**

```powershell
git clone https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB.git
cd BUILDING-MY-OWN-VECTOR-DB
```

2. **Create a virtual environment:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Install dependencies:**

```powershell
pip install -r requirements.txt
pip install -e sdk   # Install SDK in editable mode
```

4. **Configure environment:**

Copy `.env.example` to `.env` and set your `DATABASE_URL`:

```env
DATABASE_URL=postgresql://vector_user:vector_password@localhost:5432/vector_db
```

5. **Create database tables:**

```powershell
python -c "from config.database import Base, engine; Base.metadata.create_all(bind=engine)"
```

6. **Start the API:**

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Points

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Dashboard | http://localhost:8000/dashboard |
| Health | http://localhost:8000/health |
| Metrics | http://localhost:8000/metrics |

### Usage Examples

**Create a collection and ingest text:**

```python
import requests

BASE = "http://localhost:8000"

# Create collection
requests.post(f"{BASE}/collections", json={
    "name": "Product docs",
    "collection_id": "product-docs",
    "modality": "text",
    "dimension": 384,
})

# Ingest text
requests.post(f"{BASE}/collections/product-docs/ingest/text", json={
    "text": "Returns are accepted within 30 days with receipt.",
    "metadata": {"source": "policy"},
    "vector_id": "policy-returns",
})

# Search
hits = requests.post(f"{BASE}/collections/product-docs/search/text", json={
    "query": "how do I return an item?",
    "k": 5,
}).json()
```

**Multimodal image search:**

```python
# Create image collection
requests.post(f"{BASE}/collections", json={
    "name": "Product photos",
    "collection_id": "product-photos",
    "modality": "image",
    "dimension": 512,
})

# Ingest image
with open("shirt.jpg", "rb") as f:
    requests.post(
        f"{BASE}/collections/product-photos/ingest/image",
        files={"file": ("shirt.jpg", f, "image/jpeg")},
        data={"metadata": '{"sku": "A1"}'},
    )

# Search by image
with open("query.jpg", "rb") as f:
    hits = requests.post(
        f"{BASE}/collections/product-photos/search/image",
        files={"file": ("query.jpg", f, "image/jpeg")},
        data={"k": 5},
    ).json()
```

**Hybrid search (dense + BM25):**

```python
hybrid = requests.post(
    f"{BASE}/collections/product-docs/search/hybrid",
    json={"query": "return policy", "k": 10, "alpha": 0.5},
).json()
```

**RAG query:**

```python
rag = requests.post(
    f"{BASE}/collections/product-docs/query",
    data={"query": "What is the return policy?", "k": 5},
).json()
print(rag["answer"])
```

---

## Configuration

Key configuration is in `config/settings.py` loaded from environment variables (`.env`). All settings have sensible defaults.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://vector_user:vector_password@localhost:5432/vector_db` | PostgreSQL connection string |
| `APP_NAME` | `Vector Database API` | Application name |
| `APP_VERSION` | `1.0.0` | Application version |
| `DEBUG` | `False` | Enable debug mode |
| `API_KEY` | `your-api-key-here` | Admin API key for full access |

### HNSW Parameters

| Variable | Default | Optimized For | Description |
|----------|---------|---------------|-------------|
| `DEFAULT_M` | `32` | 10K vectors | Number of neighbors per node |
| `DEFAULT_M0` | `64` | 10K vectors | Neighbors in layer 0 (2× M) |
| `DEFAULT_EF_CONSTRUCTION` | `300` | 10K vectors | Search breadth during construction |
| `DEFAULT_EF_SEARCH` | `50` | 10K vectors | Search breadth during query |

### IVF Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_N_CLUSTERS` | `100` | Number of clusters (sqrt(N) recommended) |
| `DEFAULT_N_PROBES` | `10` | Clusters to probe during search |

### Embedding Models

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Text embedding model |
| `DEFAULT_TEXT_DIMENSION` | `384` | Text vector dimension |
| `DEFAULT_IMAGE_MODEL` | `clip-ViT-B-32` | Image embedding model (CLIP) |
| `DEFAULT_IMAGE_DIMENSION` | `512` | Image vector dimension |
| `DEFAULT_AUDIO_MODEL` | `librosa-mfcc-128` | Audio embedding (librosa MFCC) |
| `DEFAULT_AUDIO_DIMENSION` | `128` | Audio vector dimension |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding models |

### Multimodal & Media

| Variable | Default | Description |
|----------|---------|-------------|
| `MEDIA_STORAGE_PATH` | `media_storage` | Local path for uploaded files |
| `STORAGE_PROVIDER` | `local` | Storage backend (local/s3/azure) |
| `AUDIO_SAMPLE_RATE` | `22050` | Audio sample rate (Hz) |
| `AUDIO_MAX_DURATION_SEC` | `30.0` | Max audio duration (seconds) |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_TIME` | `60` | Window size (seconds) |
| `RATE_LIMIT_ENABLED` | `True` | Enable rate limiting |
| `RATE_LIMIT_BACKEND` | `memory` | Rate limit backend (memory/redis) |

### Other

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_PGVECTOR` | `False` | Use pgvector extension for HNSW search |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `MAX_REQUEST_SIZE_MB` | `10` | Max upload size (MB) |

---

## Deployment

### Docker Compose (Recommended for Local)

```powershell
docker compose up --build
```

This starts:
- **vector-api**: FastAPI app (port 8000)
- **postgres**: PostgreSQL 15 (port 5432)
- **prometheus**: Metrics collection (port 9090)
- **grafana**: Metrics visualization (port 3000)

### Dockerfile

Build and run manually:

```powershell
docker build -t vector-db-api .
docker run -p 8000:8000 --env-file .env vector-db-api
```

### Kubernetes (Helm)

```bash
helm install vector-db ./helm/vector-db --values ./helm/vector-db/values.yaml
```

The Helm chart includes:
- Deployment with configurable replicas, resource limits, and probes
- Service (ClusterIP), HPA, PVC for media/indexes
- PostgreSQL sub-chart
- Optional Redis, Ingress, ServiceMonitor

### Terraform

Infrastructure-as-code for AWS and Azure:

```bash
cd terraform/aws
terraform init && terraform apply
```

---

## Roadmap

Actively evolving toward a fully distributed, enterprise-grade vector database. See [ROADMAP.md](ROADMAP.md) for the full plan.

**Recently shipped (Phase 1 + Phase 4):**
- ✅ Write-Ahead Logging with crash-recovery replay (`utils/wal.py`)
- ✅ Background tombstone compaction for HNSW (`utils/compaction.py`)
- ✅ Memory-mapped vector storage for larger-than-RAM datasets (`utils/mmap_store.py`)
- ✅ AST hybrid query parser + cost-based optimizer (`utils/query_planner.py`)

**Coming next:** DiskANN on-disk graphs, horizontal sharding + distributed query aggregation, Raft replication, CDC ingestion (Kafka), and row-level RBAC.

---

## Monitoring & Observability

### Prometheus Metrics

Available at `/metrics` with these custom metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `vector_search_requests_total` | Counter | Total search requests |
| `vector_ingest_requests_total` | Counter | Total ingest requests |
| `vector_query_latency_seconds` | Histogram | Query latency distribution |

### Dashboard UI

A real-time dashboard at `/dashboard` provides:
- Overview: total vectors, collections, avg latency, index status
- Search: text search with method selection and result visualization
- Similarity explorer: 2D projection of result vectors with Chart.js
- Collections: quick-create and browse collections

### Structured Logging

All services use Python's `logging` module with consistent format:

```
2026-06-02 10:00:00,000 - vector_service - INFO - Created vector: vec_abc123
```

---

## Testing

The project has **130+ tests** covering API endpoints, indexing algorithms, services, and utilities.

```powershell
# Run all tests
pytest test/ -v

# Run SDK tests
pytest sdk/tests/ -v

# Run by category
pytest test/ -k "hnsw" -v      # HNSW-specific tests
pytest test/ -k "ivf" -v       # IVF-specific tests
pytest test/ -k "api" -v       # API integration tests

# With coverage
pytest test/ --cov=. --cov-report=html
```

### Test Files

| File | Focus |
|------|-------|
| `test/test_api.py` | REST API endpoint integration |
| `test/test_vector_db.py` | Database operations |
| `test/test_hnsw.py` | HNSW algorithm correctness |
| `test/test_ivf.py` | IVF algorithm correctness |
| `test/test_kdtree.py` | KD-Tree search correctness |
| `test/test_clustering.py` | K-Means clustering |
| `test/test_multimodal.py` | Multimodal ingest & search |
| `test/test_rag.py` | RAG pipeline |
| `test/test_reranker.py` | Cross-encoder re-ranking |
| `test/test_hybrid_search.py` | Hybrid dense+sparse search |
| `test/test_langchain_vectorstore.py` | LangChain integration |
| `test/test_ws_search.py` | WebSocket search |
| `test/test_openai_compat.py` | OpenAI-compatible endpoints |
| `test/test_cache_service.py` | Redis caching |
| `test/test_storage_backend.py` | Local/S3/Azure storage |
| `test/test_multi_tenancy.py` | Tenant isolation |
| `test/test_auth_middleware.py` | Auth middleware |
| `test/test_auth_service.py` | API key management |
| `test/test_comprehensive.py` | End-to-end integration |
| `test/test_wal_recovery.py` | WAL checkpoint, replay, corruption tolerance |
| `test/test_compaction.py` | HNSW tombstone soft-delete + background compaction |
| `test/test_mmap_store.py` | Memory-mapped vector storage |
| `test/test_query_planner.py` | AST parser + cost-based optimizer |
| `test/test_rbac.py` | Row-level RBAC policy + permission gates |
| `test/test_distributed_coordinator.py` | Parallel scatter-gather + fusion + fault tolerance |
| `test/test_dynamic_quantization.py` | Memory-pressure precision policy |
| `test/test_startup_recovery.py` | Pending-WAL detection for boot recovery |

---

## Project Layout

```text
.
├── api/                          # FastAPI application
│   ├── main.py                  # App factory, middleware, all route definitions
│   ├── grpc/                    # gRPC server & protobuf definition
│   │   ├── server.py           # gRPC servicer delegating to VectorService
│   │   └── vector_service.proto# Protobuf service definition
│   ├── middleware/               # Auth middleware (API key, tenant, rate limit)
│   ├── graphql/                  # GraphQL API schema
│   │   └── schema.py           # Strawberry schema (7 queries + 5 mutations)
│   ├── routers/                  # Route modules
│   │   ├── rag.py              # PDF ingest & RAG query endpoints
│   │   ├── streaming_rag.py    # SSE streaming RAG
│   │   ├── ingestion.py        # Async bulk ingestion queue API
│   │   ├── timeseries.py       # Time-series vector endpoints
│   │   ├── auth_api.py         # API key management endpoints
│   │   ├── tenants.py          # Tenant CRUD + per-tenant keys
│   │   ├── dashboard.py        # Dashboard UI & data endpoints
│   │   ├── openai_compat.py    # OpenAI-compatible REST endpoints
│   │   ├── search_enhanced.py  # Hybrid + rerank endpoints
│   │   └── ws_search.py        # WebSocket search
│   └── static/                  # Dashboard frontend (HTML/CSS/JS)
├── config/                      # Configuration modules
│   ├── settings.py             # Pydantic BaseSettings (env vars)
│   ├── database.py             # SQLAlchemy engine & session factory
│   └── logging.py              # Structured logging setup
├── database/                    # Database layer
│   ├── schema.py               # SQLAlchemy ORM models (10 tables)
│   ├── vector_database.py      # Base database wrapper (K-Means indexer)
│   ├── hnsw_database.py        # HNSW-specific DB operations
│   └── ivf_database.py         # IVF-specific DB operations
├── models/                      # Pydantic & data models
│   ├── pydantic_models.py      # Request/response schemas (40+ models)
│   └── vector_model.py         # Vector CRUD model with brute-force search
├── services/                    # Business logic layer (18 services)
│   ├── vector_service.py       # Main vector CRUD + orchestration
│   ├── collection_service.py   # Collection CRUD + dimension validation
│   ├── collection_index_service.py  # Per-collection index management
│   ├── embedding_service.py    # Text/Image/Audio embedding backends
│   ├── multimodal_service.py   # Multimodal ingest & search
│   ├── rag_service.py          # RAG pipeline
│   ├── streaming_rag_service.py# Streaming RAG (SSE)
│   ├── hybrid_search_service.py# BM25 + dense hybrid search
│   ├── reranker_service.py     # Cross-encoder re-ranking
│   ├── auth_service.py         # API key management
│   ├── tenant_service.py       # Tenant CRUD + token-bucket rate limiting
│   ├── cache_service.py        # Redis caching
│   ├── ingestion_service.py    # Async bulk ingestion queue
│   ├── media_store.py          # File persistence
│   ├── storage_backend.py      # Local/S3/Azure storage abstraction
│   ├── metadata_filter.py      # JSONB metadata post-filtering
│   ├── auto_reindex.py         # Automatic index rebuild scheduler
│   └── vector_indexer.py       # Unified indexer with auto-optimization
├── utils/                       # Algorithms & utilities
│   ├── hnsw_index.py           # HNSW graph (Node, search, insert, save/load, tombstone delete/compact)
│   ├── wal.py                  # Write-Ahead Log: durability, checkpoint, crash-recovery replay
│   ├── compaction.py           # Background tombstone compaction daemon
│   ├── mmap_store.py           # Memory-mapped larger-than-RAM vector store
│   ├── query_planner.py        # AST hybrid query parser + cost-based optimizer
│   ├── rbac.py                 # Row-level RBAC (permission gates + metadata predicates)
│   ├── dynamic_quantization.py # Memory-pressure precision policy (fp32→int8→pq→binary)
│   ├── ivf_index.py            # IVF + coarse/fine quantizers
│   ├── product_quantization.py # PQ index with ADC
│   ├── int8_index.py           # Int8 quantization (4x compression)
│   ├── partitioned_index.py    # Distributed/partitioned index manager
│   ├── lsh_index.py            # LSH for cosine distance
│   ├── vp_tree.py              # VP-Tree metric index
│   ├── kdtree_index.py         # KD-Tree space-partitioning index
│   ├── bm25_index.py           # Okapi BM25 sparse retrieval
│   ├── hybrid_search.py        # RRF fusion engine
│   ├── clustering.py           # K-Means++ implementation
│   ├── distance.py             # Euclidean & cosine distance functions
│   ├── optimization.py         # Numba JIT, batch processing, memory optimization
│   ├── benchmark.py            # Comprehensive benchmark suite
│   ├── gpu_kernels.py          # CuPy GPU distance computation
│   ├── pdf_processor.py        # PyMuPDF text extraction
│   ├── document_processor.py   # Multi-format (DOCX, HTML, MD, TXT) extraction
│   ├── text_chunker.py         # Recursive, sentence, & token chunking
│   ├── image_metadata.py       # EXIF/metadata extraction from images
│   ├── metadata_contract.py    # Collection metadata building
│   ├── index_paths.py          # File path helpers for index save/load
│   └── index_serializer.py     # JSON/binary serialization
├── sdk/                         # Python client SDK
│   ├── vector_db_client/       # Client package
│   │   ├── client.py          # Main VectorDBClient
│   │   ├── collections.py     # CollectionsAPI
│   │   ├── vectors.py         # VectorsAPI
│   │   ├── multimodal.py      # MultimodalAPI
│   │   ├── models.py          # Response models
│   │   ├── exceptions.py      # Custom exceptions
│   │   ├── langchain_vectorstore.py  # LangChain integration
│   │   └── _http.py           # HTTP transport layer
│   └── tests/                  # SDK test suite
├── test/                        # Server test suite (130+ tests)
├── scripts/                     # Utility scripts
│   ├── run_benchmark.py        # Comprehensive benchmark runner
│   ├── scale_benchmark.py      # Scale benchmark (1K–1M vectors)
│   └── run_local.ps1           # Local dev startup script
├── examples/                    # Example code
│   ├── vector_indexer_api.py   # VectorIndexer API router
│   ├── indexer_examples.py     # Index usage examples
│   └── multimodal_demo.py      # Multimodal demo
├── docs/                        # Documentation
│   ├── LOCAL_SETUP.md          # Local development guide
│   ├── ivf_vector_search_guide.md  # IVF deep-dive
│   └── blog-*.md              # Architecture blog posts
├── helm/                        # Kubernetes Helm chart
├── terraform/                   # AWS & Azure Terraform templates
├── .github/workflows/           # CI/CD GitHub Actions
├── docker-compose.yaml          # Multi-container stack
├── dockerfile                   # Container image
├── prometheus.yml               # Prometheus scrape config
├── nginx.conf                   # Reverse proxy config
├── pyproject.toml               # Project metadata + Ruff config
└── requirements.txt             # Python dependencies
```

---

## All Database Tables

Defined in `database/schema.py`:

| Table | Description |
|-------|-------------|
| `vectors` | Vector data (ARRAY(Float)), metadata (JSONB), collection_id |
| `vectors_pgvector` | Optional pgvector-backed storage with `<=>` distance operator |
| `collections` | Multimodal collection namespaces, modality, dimension, model |
| `tenants` | Multi-tenant organizations with rate limits |
| `api_keys` | SHA-256 hashed API keys, scoped per tenant/collection |
| `vector_batches` | Batch insertion records |
| `vector_batch_mappings` | Many-to-many batch ↔ vector mapping |
| `api_templates` | Saved API playground templates |
| `feedback_entries` | User feedback submissions |

---

## Contributing

We welcome contributions of all sizes! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

**Quick overview:**

1. Fork and clone the repo
2. Set up virtual environment and install dependencies
3. Create a feature branch: `git checkout -b feat/your-feature`
4. Make changes and add tests
5. Ensure tests pass and lint is clean: `pytest test/ -v && ruff check .`
6. Commit with Conventional Commits format
7. Open a Pull Request against `main`

### Code Style

- **Linter**: [Ruff](https://docs.astral.sh/ruff/) (`ruff check .`)
- **Formatter**: Ruff (`ruff format .`)
- **Type hints**: Required for all public function signatures
- **Line length**: 100 characters
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes

---

## Security

See [SECURITY.md](SECURITY.md) for the full security policy.

**Key points:**
- API keys stored as SHA-256 hashes (plaintext never persisted)
- Per-collection key permissions (`read`, `write`, `read_write`)
- Rate limiting (configurable requests/time window)
- CORS origin restrictions
- Request size limits (10 MB default)
- Path traversal protection on media file serving
- Always use HTTPS in production

---

## Roadmap

### Implemented
- [x] HNSW index with configurable parameters (m, ef_construction, ef_search)
- [x] IVF index with coarse+fine quantization
- [x] KD-Tree, VP-Tree, LSH indexes
- [x] Product Quantization (10-48x compression)
- [x] Hybrid search (dense + BM25 via RRF)
- [x] Cross-encoder re-ranking (local + Cohere API)
- [x] Multimodal ingestion (text, image, audio)
- [x] RAG pipeline with PDF/document processing
- [x] Streaming RAG (SSE)
- [x] 68 REST API endpoints + gRPC + WebSocket
- [x] Multi-tenancy with API key auth
- [x] Per-collection HNSW/IVF indexes
- [x] OpenAI-compatible endpoints
- [x] Prometheus metrics + dashboard UI
- [x] GPU-accelerated distance (CuPy)
- [x] Redis caching
- [x] Docker Compose + Kubernetes Helm chart
- [x] AWS + Azure Terraform templates
- [x] 112+ tests
- [x] Python SDK + LangChain integration

### Completed (Latest)
- [x] **Write-Ahead Logging (WAL)** with fsync durability + crash-recovery replay (`utils/wal.py`)
- [x] **Background compaction** — HNSW tombstone soft-delete + daemon reclaim thread (`utils/compaction.py`)
- [x] **Memory-mapped vector storage** for larger-than-RAM datasets (`utils/mmap_store.py`)
- [x] **AST query planner + cost-based optimizer** for hybrid queries (`utils/query_planner.py`) + `/search-engine/hybrid-query` REST endpoint
- [x] **Row-level RBAC** — operation gates + per-key metadata predicates (`utils/rbac.py`)
- [x] **Distributed query aggregation** — parallel scatter-gather coordinator with global fusion + fault tolerance (`services/distributed_coordinator.py`)
- [x] **Dynamic quantization policy** — memory-pressure precision tiers (`utils/dynamic_quantization.py`)
- [x] **Startup crash recovery** — auto-replay pending WALs (HNSW + IVF) on boot (`services/startup_recovery.py`)
- [x] Per-collection IVF index persistence (save/load/rebuild + REST endpoints)
- [x] Cross-modal CLIP text→image search quality tuning (temperature + normalization)
- [x] Long-audio chunking and segment-level vectors (`chunk_seconds` param)
- [x] S3/Azure Blob media storage (wired into MediaStore via StorageBackend)
- [x] Async ingest job queue for large uploads (REST API + periodic flush)
- [x] SQL/metadata filters (JSONB) with pre-filtering (`MetadataFilter.pre_filter`)
- [x] Int8 vector quantization (4x compression, `utils/int8_index.py`)
- [x] Time-series vector support (timestamp/series_id schema + 5 API endpoints)
- [x] GraphQL API (Strawberry, mounted at `/graphql`)
- [x] Distributed/partitioned indexes (`utils/partitioned_index.py`)

### Upcoming (See [ROADMAP.md](ROADMAP.md))
- [ ] Phase 1 (finish): DiskANN/Vamana on-disk graph layout; auto-replay all WALs on startup
- [ ] Phase 2: Horizontal sharding, distributed query aggregation, Raft consensus
- [ ] Phase 3: CDC ingestion (Kafka/Debezium), dynamic quantization, LlamaIndex + MCP connectors
- [ ] Phase 4 (finish): Row-level RBAC; query planner exposed over REST with live cardinality stats
- [ ] Phase 5: SIMD/AVX-512 distance kernels; full GPU index construction
- [ ] Phase 6: GCN node-embedding training, temporal graph dynamics, link prediction
- [ ] Phase 7: Learning-to-Rank, personalization, real-time web connectors
- [ ] Phase 8 (new): Distributed tracing, backup/restore, index auto-tuning
- [ ] Phase 9 (new): Encryption at rest, audit logging, mTLS, data residency


## License

[MIT](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/blob/main/LICENSE)

Built by [KunjShah95](https://github.com/KunjShah95) — contributions and feedback welcome!
