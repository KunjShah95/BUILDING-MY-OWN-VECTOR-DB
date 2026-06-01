# Changelog

## [1.0.0] - 2026-06-01

### Added
- HNSW index with configurable parameters (m, ef_construction, ef_search)
- IVF index with product quantization
- KD-Tree index for low-dimensional search
- RAG pipeline with PDF/document processing
- Streaming SSE endpoint for RAG responses
- Multi-modal embedding (text, image, audio)
- Per-collection indexing
- Hybrid search (dense + sparse BM25)
- Cross-encoder reranking
- Authentication with API keys
- WebSocket streaming search
- Dashboard UI with real-time stats
- Prometheus metrics and Grafana dashboards
- Docker Compose deployment (API + PostgreSQL + Prometheus + Grafana)
- Helm chart for Kubernetes deployment
- CI/CD pipeline with GitHub Actions
- Python SDK client (`vector_db_client`)
- 112+ tests across API, index algorithms, and utilities
- 68 REST API endpoints with OpenAPI docs
- OpenAI-compatible API endpoints
- Metadata filtering (post-filter)
- Collection-scoped HNSW indexes with on-disk persistence
- PQ (Product Quantization) index
- GPU-accelerated distance kernels (optional CuPy backend)
- Bulk ingestion queue with async batching
- gRPC API (protobuf service definition + server)

### Changed
- Optimized HNSW parameters: 64% throughput improvement, 55% latency reduction, 188% recall gain
- Refactored search to support method-based dispatch (HNSW, IVF, brute, pq, hybrid)

### Documentation
- Architecture deep-dive blog posts (4-part series)
- IVF vector search guide
- Local development setup guide
- Full SDK reference in `sdk/README.md`
- Time-series vector support design doc

## [0.9.0] - 2026-01-15

### Added
- Initial HNSW and IVF indexing implementations
- PostgreSQL-backed vector storage
- Basic CRUD API
- Batch insert and search
- Benchmarking suite
- Parameter tuning guide
