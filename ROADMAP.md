# Vector Database Roadmap

This roadmap tracks the evolution of this project from a robust, single-node vector database into a distributed, enterprise-ready data infrastructure system.

**Legend:** ✅ done · 🟡 partial · ⚪ planned

## Status at a Glance

| Phase | Theme | Status |
|-------|-------|--------|
| 1 | Storage & Durability | ✅ done (HNSW+IVF WAL recovery, startup replay, DiskANN/Vamana graph built) |
| 2 | Distributed Systems & Scalability | ✅ done (Raft consensus, multi-node registry, scatter-gather coordinator) |
| 3 | Advanced Ingestion & Integrations | ✅ done (CDC Kafka, auto-quantization, LlamaIndex, MCP server) |
| 4 | Query Planner & Optimization | ✅ done (planner + REST endpoint + row-level RBAC) |
| 5 | Hardware Acceleration | ✅ done (GPU indexing, SIMD/AVX-512 kernels) |
| 6 | Graph Neural Networks | ✅ done (GCN training, temporal dynamics, link prediction) |
| 7 | Production Search Orchestration | ✅ done (LTR, personalization engine, real-time connectors) |
| 8 | Observability & Operations | ✅ done (OpenTelemetry tracing, backup/restore, graceful shutdown) |
| 9 | Security & Compliance | ✅ done (RBAC enforcement, encryption, audit logging, PII, mTLS) |

---

## Phase 1: Storage & Durability ✅
Survive crashes without data loss and manage datasets larger than available RAM.

- ✅ **Write-Ahead Logging (WAL)**: Append-only log with fsync durability, checkpoint/truncate after snapshot, and crash-recovery replay wired into HNSW index load (`utils/wal.py`, `database/hnsw_database.py`).
- ✅ **Memory-Mapped Vector Storage**: `numpy.memmap`-backed store with dynamic growth, row reclamation, and persistence so only the working set stays resident (`utils/mmap_store.py`).
- ✅ **Background Compaction**: Soft-delete tombstones in HNSW preserve graph connectivity; a daemon thread hard-removes them by interval/ratio threshold (`utils/compaction.py`, `HNSWIndex.compact`).
- ✅ **IVF WAL Recovery**: WAL replay wired into IVF index load via method-agnostic replay (`add`/`delete_vector` fallback) (`database/ivf_database.py`, `utils/wal.py`).
- ✅ **Startup Auto-Recovery**: On app boot, every collection with a pending WAL + persisted index is replayed automatically (`services/startup_recovery.py`, FastAPI `startup` hook in `api/main.py`).
- ✅ **DiskANN / Vamana Graph Layout**: SSD-optimized on-disk graph (beam search over memory-mapped adjacency) for billion-scale serving (`utils/vamana_index.py`).

## Phase 2: Distributed Systems & Scalability ✅
A single node has physical limits. We need horizontal scalability.

- ✅ **Horizontal Sharding**: Consistent hashing to shard collections across nodes (`utils/partitioned_index.py`). Multi-node placement via `ClusterRegistry` (`services/raft_coordinator.py`). 
- ✅ **Distributed Query Aggregation**: Coordinator scatter-gathers queries to shards in parallel, fuses global top-K by distance or RRF, and tolerates shard failure (`services/distributed_coordinator.py`).
- ✅ **Raft Consensus Engine**: Leader election + distributed WAL replication with ClusterRegistry for high availability (`services/raft_coordinator.py`).

## Phase 3: Advanced Ingestion & Integrations ✅
Production environments demand seamless data pipelines, not just REST APIs.

- ✅ **Change Data Capture (CDC)**: Kafka/Debezium integration for real-time streaming ingestion from upstream databases (`services/cdc_connector.py`).
- ✅ **Dynamic Quantization**: Memory-pressure/budget-driven precision policy + background QuantizationMonitor that auto-applies Int8/PQ/Binary under real memory pressure (`utils/dynamic_quantization.py`, `services/auto_reindex.py`).
- ✅ **Agentic Connectors**: LangChain VectorStore (`sdk/...`), LlamaIndex adapter (`utils/llama_index_adapter.py`), MCP server (`services/mcp_server.py`).
- ✅ **Async Ingestion Queue**: In-memory batched queue with periodic flush + REST API (`services/ingestion_service.py`).

## Phase 4: Query Planner & Optimization ✅
Move from static search pipelines to intelligent, cost-based execution.

- ✅ **AST-Based Query Planner**: Recursive-descent parser for hybrid queries (`utils/query_planner.py`).
- ✅ **Cost-Based Optimizer**: Selectivity heuristics with optional per-field cardinality stats (`utils/query_planner.py`).
- ✅ **REST Endpoint**: `/search-engine/hybrid-query` endpoint (`api/main.py`).
- ✅ **Row-Level RBAC**: Operation gates + row-level metadata predicates (`utils/rbac.py`). Persisted `row_filter` on API-key records and enforced in auth middleware (`services/auth_service.py`, `api/middleware/auth_middleware.py`).

## Phase 5: Hardware Acceleration ✅
- ✅ **GPU Indexing**: Full CuPy index construction (k-means, batch distance, HNSW neighbor selection, PQ training) via `utils/gpu_indexing.py`.
- ✅ **SIMD/AVX-512 Optimization**: Auto-detecting SIMD kernels with Numba JIT and C++/PyBind11 backend support (`utils/simd_kernels.py`).

## Phase 6: Graph Neural Networks ✅
Apply ML directly over the HNSW/Vamana index structure.

- ✅ **Graph Rerank**: Personalized PageRank-based candidate reranking (`services/gnn_service.py`).
- ✅ **GCN Node-Embedding Training**: 2-layer Graph Convolutional Network for learning low-dimensional node embeddings with self-supervised link prediction loss (`services/gnn_service.py`).
- ✅ **Temporal Graph Dynamics**: Track edge formation over time to surface trending/bursty clusters (`services/gnn_service.py`).
- ✅ **Link Prediction**: Infer missing metadata tags by predicting edges between similar sub-graphs (`services/gnn_service.py`).

## Phase 7: Production Search Orchestration ✅
Build a deployment-ready search orchestrator on top of the vector foundation.

- ✅ **Learning-to-Rank (LTR)**: XGBoost LambdaMART trained on click-through feedback to replace static RRF (`services/ltr_service.py`).
- ✅ **Personalization Engine**: Inject user-profile embeddings into query formulation with preference learning from interaction history (`services/personalization_service.py`).
- ✅ **Real-Time Data Connectors**: Web crawler, RSS feed, and REST API connectors for live data ingestion (`services/realtime_connector.py`).

## Phase 8: Observability & Operations ✅
Make the system debuggable and operable at scale.

- ✅ **Distributed Tracing**: OpenTelemetry spans across API → service → index → DB with automatic decorators and context managers (`utils/opentelemetry_tracing.py`, extends `utils/telemetry.py`).
- ✅ **Index Health & Auto-Tuning**: Surface recall drift, tombstone ratio, and fragmentation; auto-trigger compaction/reindex (`utils/index_health.py`, `services/auto_reindex.py`).
- ✅ **Backup & Restore**: Point-in-time recovery from WAL + snapshot; scheduled S3/Azure backups of index files (`services/backup_service.py`).
- ✅ **Graceful Shutdown / Startup Recovery**: On boot, replay all collection WALs and rebuild in-memory indexes automatically (`services/startup_recovery.py`).

## Phase 9: Security & Compliance ✅
Enterprise readiness.

- ✅ **Row-Level Security**: RBAC policy with per-key `row_filter` predicates persisted in database + operation gates enforced in auth middleware (`utils/rbac.py`, `database/schema.py`, `services/auth_service.py`, `api/middleware/auth_middleware.py`).
- ✅ **Encryption at Rest**: AES-256-GCM encryption of index files and mmap store on disk with key derivation and rotation support (`utils/encryption.py`).
- ✅ **Audit Logging**: Immutable append-only log of every mutation and access for compliance with tamper detection (`utils/audit_log.py`).
- ✅ **PII Redaction & Data Residency**: Field-level and pattern-based PII redaction in metadata, per-tenant region pinning (`utils/pii_redaction.py`).
- ✅ **mTLS between nodes**: Mutual TLS with auto-generated dev certificates for secure inter-node communication (`utils/mtls_service.py`).

---

## Phase 10: AI-Native & Intelligent Search ⚪
Move beyond fixed index parameters to self-optimizing, natural-language-driven retrieval.

- ⚪ **AI-powered index tuning**: Automated HNSW M/ef construction, IVF cluster count, and PQ sub-quantizer recommendations based on dataset statistics and query pattern analysis.
- ⚪ **Natural language query interface**: Text-to-vector-search — translate plain English queries into structured hybrid queries (semantic + metadata filters + time ranges).
- ⚪ **Automatic metadata enrichment**: On ingestion, auto-extract entities, topics, and summaries from text/image content to populate metadata fields for filtering.
- ⚪ **Embedding model lifecycle**: Registry with versioning, A/B testing between models, and gradual re-indexing when switching encoders.
- ⚪ **Multi-vector per document**: ColBERT-style late interaction scoring — store multiple vectors per document and compute MaxSim at query time for finer-grained relevance.

## Phase 11: Performance at Scale ⚪
Target benchmarks at 100M–1B vectors while keeping recall >95% and latency <10ms.

- ⚪ **Billion-scale benchmark optimization**: Tune Vamana/DiskANN parameters (L, R, beam width) on 100M–1B scale datasets (BIGANN, SPACEV, DEEP1B) with quantified recall/latency tradeoffs.
- ⚪ **Tiered storage**: Hot (RAM/mmap), warm (NVMe/SSD), cold (S3/Blob) vector tiers with automatic promotion/demotion based on access frequency.
- ⚪ **Adaptive index selection**: Per-query routing to the fastest index (HNSW/IVF/PQ/brute-force) based on query cardinality, filter selectivity, and latency budget.
- ⚪ **Query result caching**: Multi-level cache (L1 in-memory, L2 Redis) with TTL-based invalidation and popularity-aware pre-warming.
- ⚪ **High-concurrency connection pooling**: Multiplexed gRPC connections, connection pooling for PostgreSQL, and async I/O tuning for 10K+ concurrent queries.

## Phase 12: Enterprise & Compliance ⚪
Production-hardening for regulated industries.

- ⚪ **Multi-region active-active replication**: CRDT-based vector synchronization across geographic regions with conflict resolution and local consistency guarantees.
- ⚪ **Compliance reporting (SOC2, GDPR)**: Automated audit report generation from audit logs, data residency attestation, and right-to-erasure workflows.
- ⚪ **Cost-based query budget enforcement**: Prevent runaway queries by capping max scanning (max probes, max ef_search, max HNSW visits) per API key/tenant.
- ⚪ **Data retention & lifecycle policies**: TTL-based vector expiration, tiered archival (hot → warm → cold → delete), and automated garbage collection.
- ⚪ **Tenant-level isolation certification**: Separate database schemas, dedicated index files, and CPU/memory cgroups per tenant for true hardware-level isolation.

## Phase 13: Ecosystem & Integrations ⚪
Extend beyond Python into the broader developer ecosystem.

- ⚪ **Client SDKs — JS/TS, Go, Java, Rust, .NET**: Idiomatic clients for each language covering full API surface (CRUD, search, hybrid, RAG, admin).
- ⚪ **Apache Arrow / Flight integration**: Zero-copy vector transfer via Arrow Flight for high-throughput data loading and export between ML pipelines.
- ⚪ **Haystack integration**: Custom component for Haystack 2.x document store and embedding retrieval.
- ⚪ **Semantic Kernel integration**: Memory store connector for Microsoft Semantic Kernel vector memory.
- ⚪ **Vector benchmark suite**: Automated ANN benchmark harness (Bees-Balls, recall@k, throughput vs. build time) for regression testing index changes.
- ⚪ **Plugin / extension system**: Hot-loadable custom index algorithms, embedding models, and storage backends via a registry interface.

## Phase 14: Real-Time & Streaming ⚪
Continuous data and query processing for time-sensitive applications.

- ⚪ **Real-time index updates without blocking reads**: Lock-free concurrent index writes (HNSW/IVF) so ingestion never blocks search queries.
- ⚪ **Streaming vector search (continuous query)**: Long-lived query subscriptions that emit new results as vectors are ingested — push-based retrieval.
- ⚪ **Event-driven webhook notifications**: On vector insert/update/delete matching a registered query, fire a webhook with the result payload.
- ⚪ **Materialized views for common queries**: Pre-compute and incrementally maintain result sets for high-frequency queries (trending, popular, recent).
- ⚪ **Additional CDC sources**: PostgreSQL logical replication and MongoDB change streams as additional ingestion pipelines alongside Kafka.
