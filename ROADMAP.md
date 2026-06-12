# Vector Database Roadmap

This roadmap tracks the evolution of this project from a robust, single-node vector database into a distributed, enterprise-ready data infrastructure system.

**Legend:** ✅ done · 🟡 partial · ⚪ planned

## Status at a Glance

| Phase | Theme | Status |
|-------|-------|--------|
| 1 | Storage & Durability | ✅ done (HNSW+IVF WAL recovery, startup replay; DiskANN graph pending) |
| 2 | Distributed Systems & Scalability | 🟡 partial (parallel scatter-gather coordinator done; Raft pending) |
| 3 | Advanced Ingestion & Integrations | 🟡 partial (dynamic quantization policy added) |
| 4 | Query Planner & Optimization | ✅ done (planner + REST endpoint + row-level RBAC) |
| 5 | Hardware Acceleration | 🟡 partial (GPU batch distance) |
| 6 | Graph Neural Networks | 🟡 partial (graph rerank only) |
| 7 | Production Search Orchestration | ⚪ planned |
| 8 | Observability & Operations | ⚪ planned (new) |
| 9 | Security & Compliance | 🟡 partial (RBAC row-level security) |

---

## Phase 1: Storage & Durability ✅ (Core Done)
Survive crashes without data loss and manage datasets larger than available RAM.

- ✅ **Write-Ahead Logging (WAL)**: Append-only log with fsync durability, checkpoint/truncate after snapshot, and crash-recovery replay wired into HNSW index load (`utils/wal.py`, `database/hnsw_database.py`).
- ✅ **Memory-Mapped Vector Storage**: `numpy.memmap`-backed store with dynamic growth, row reclamation, and persistence so only the working set stays resident (`utils/mmap_store.py`).
- ✅ **Background Compaction**: Soft-delete tombstones in HNSW preserve graph connectivity; a daemon thread hard-removes them by interval/ratio threshold (`utils/compaction.py`, `HNSWIndex.compact`).
- ✅ **IVF WAL Recovery**: WAL replay wired into IVF index load via method-agnostic replay (`add`/`delete_vector` fallback) (`database/ivf_database.py`, `utils/wal.py`).
- ✅ **Startup Auto-Recovery**: On app boot, every collection with a pending WAL + persisted index is replayed automatically (`services/startup_recovery.py`, FastAPI `startup` hook in `api/main.py`).
- ⚪ **DiskANN / Vamana Graph Layout**: SSD-optimized on-disk graph (beam search over memory-mapped adjacency) for billion-scale serving. _Remaining work._

**Next up here:** integrate `MmapVectorStore` as the backing store for HNSW vectors; build the Vamana on-disk graph.

## Phase 2: Distributed Systems & Scalability ⚪
A single node has physical limits. We need horizontal scalability.

- 🟡 **Horizontal Sharding**: Consistent hashing to shard collections across nodes. _Foundation: `utils/partitioned_index.py` does hash/range partitioning; multi-node placement remains._
- ✅ **Distributed Query Aggregation**: Coordinator scatter-gathers queries to shards **in parallel** (thread pool), fuses global top-K by distance or RRF, and tolerates shard failure (`degraded` flag). Pluggable `Shard` protocol drives in-process or remote shards (`services/distributed_coordinator.py`).
- ⚪ **Raft Consensus Engine**: Leader election + distributed WAL replication (e.g. `pysyncobj`) for high availability. _Builds directly on the Phase 1 WAL._

## Phase 3: Advanced Ingestion & Integrations 🟡
Production environments demand seamless data pipelines, not just REST APIs.

- ⚪ **Change Data Capture (CDC)**: Kafka/Debezium integration for real-time streaming ingestion from upstream databases.
- 🟡 **Dynamic Quantization**: Memory-pressure / budget-driven precision policy picks fp32 → Int8 → PQ → Binary and can materialize Int8/Binary encodings (`utils/dynamic_quantization.py`). _Auto-apply hook into live indexes under runtime pressure remains._
- 🟡 **Agentic Connectors**: LangChain VectorStore done (`sdk/.../langchain_vectorstore.py`). LlamaIndex adapter and an MCP server remain.
- ✅ **Async Ingestion Queue**: In-memory batched queue with periodic flush + REST API (`services/ingestion_service.py`).

## Phase 4: Query Planner & Optimization ✅ (Planner Done)
Move from static search pipelines to intelligent, cost-based execution.

- ✅ **AST-Based Query Planner**: Recursive-descent parser turns hybrid queries like `(category = 'tech' AND price < 100) OR semantic_match("laptops")` into a typed AST (`utils/query_planner.py`).
- ✅ **Cost-Based Optimizer**: Selectivity heuristics (with optional per-field cardinality stats) pick `filter_first` / `vector_first` / `filter_only` / `vector_only`, exposed via `SearchEngineService.planned_search`.
- ✅ **REST Endpoint**: `/search-engine/hybrid-query` accepts the query DSL, embeds the `semantic_match()` text, and runs the cost-based plan (`api/main.py`).
- ✅ **Role-Based Access Control (RBAC)**: Operation gates (read/write/delete/admin) + row-level metadata predicates compiled from the same DSL, usable as a search `metadata_filter` or result filter (`utils/rbac.py`). Forward-compatible with an API-key `row_filter` field.

**Next up here:** collect live per-field cardinality stats from PostgreSQL to feed the optimizer; persist `row_filter` on the API-key record and enforce in auth middleware.

## Phase 5: Hardware Acceleration 🟡
- 🟡 **GPU Indexing**: CuPy batch distance kernels exist (`utils/gpu_kernels.py`). Offload full index construction next (RAPIDS cuVS style).
- ⚪ **SIMD/AVX-512 Optimization**: Hand-optimized C++/Cython bindings (PyBind11) for inner-loop distance.

## Phase 6: Graph Neural Networks (GNN) 🟡
Apply ML directly over the HNSW index structure.

- 🟡 **Graph Rerank**: `services/gnn_service.py` re-ranks candidates over the graph. Full **GCN** node-embedding training remains.
- ⚪ **Temporal Graph Dynamics**: Track edge formations over time to surface trending/bursty clusters. _Foundation: time-series vectors already exist._
- ⚪ **Link Prediction**: Infer missing metadata tags by predicting edges between semantically similar sub-graphs.

## Phase 7: Production Search Orchestration ⚪
Build a deployment-ready search orchestrator on top of the vector foundation.

- ⚪ **Learning-to-Rank (LTR)**: Replace static RRF with XGBoost/LightGBM trained on click-through feedback. _Foundation: `/playground/feedback` already captures signals._
- ⚪ **Personalization Engine**: Inject user-profile embeddings into query formulation.
- ⚪ **Real-Time Data Connectors**: Built-in crawlers / headless-browser ingestion of live web data.

## Phase 8: Observability & Operations ⚪ (New)
Make the system debuggable and operable at scale.

- ⚪ **Distributed Tracing**: OpenTelemetry spans across API → service → index → DB (extends existing `utils/telemetry.py`).
- ⚪ **Index Health & Auto-Tuning**: Surface recall drift, tombstone ratio, and fragmentation; auto-trigger compaction/reindex (extends `services/auto_reindex.py`).
- ⚪ **Backup & Restore**: Point-in-time recovery from WAL + snapshot; scheduled S3/Azure backups of index files.
- ⚪ **Graceful Shutdown / Startup Recovery**: On boot, replay all collection WALs and rebuild in-memory indexes automatically.

## Phase 9: Security & Compliance 🟡 (New)
Enterprise readiness.

- 🟡 **Row-Level Security**: RBAC policy with per-key row predicates + operation gates (`utils/rbac.py`). Middleware enforcement + key-record persistence remain.
- ⚪ **Encryption at Rest**: Encrypt index files and mmap store on disk.
- ⚪ **Audit Logging**: Immutable log of every mutation and access for compliance.
- ⚪ **PII Redaction & Data Residency**: Per-tenant region pinning and field-level redaction in metadata.
- ⚪ **mTLS between nodes**: Secure inter-node traffic once sharding lands (Phase 2).

---

## Recommended Next Sprint

Recently completed: Phase 1 durability loop (IVF WAL + startup replay), query-planner REST endpoint, row-level RBAC, parallel distributed query aggregation.

Highest-leverage work remaining, ordered:

1. **Persist + enforce RBAC** — add a `row_filter` column to the API-key record and apply `RBACPolicy.metadata_filter()` inside the auth middleware / search path so row-level security is enforced end-to-end.
2. **Live cardinality stats for the optimizer** — feed per-field `distinct` counts from PostgreSQL into `plan_query(stats=...)` so selectivity estimates are data-driven, not heuristic.
3. **Auto-apply dynamic quantization** — a background monitor that triggers `QuantizationPolicy` under real memory pressure and swaps live indexes to Int8/PQ.
4. **Raft / multi-node placement** (Phase 2) — turn the in-process coordinator into a true multi-node cluster with WAL replication and leader election.
5. **DiskANN / Vamana on-disk graph** (Phase 1) — billion-scale serving on SSD using the existing `MmapVectorStore`.
