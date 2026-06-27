# Vector DB Roadmap

**Legend:** ✅ done · 🟡 in progress · ⚪ planned

## Phase 10: Billion-Scale Performance ✅
Target 100M–1B vectors with recall >95% and latency <10ms.
- ✅ Vamana/DiskANN: real RobustPrune (Algorithm 3, DiskANN NeurIPS 2019) + bidirectional pruning on insert — `utils/vamana_index.py`; `MmapVamanaIndex` backed by `MmapVectorStore` for SSD-resident vectors
- ✅ IVF-PQ: coarse IVF k-means + PQ residual encoding (10-48x compression), ADC search, numpy binary persistence — `utils/ivf_pq_index.py`; wired into `AnnIndexService` as `index_type=ivfpq`
- ✅ Multi-level cache hierarchy: L1 LRU RAM → L2 NVMe mmap files → L3 S3 object storage, with promotion/demotion and hit-rate stats — `services/multilevel_cache.py`
- ✅ Adaptive batch pool: asyncio queue + adaptive batch sizing (scales min→max based on queue depth), `max_workers` semaphore, backpressure via `QueueFullError`, p50/p95/p99 latency stats — `services/adaptive_batch_pool.py`

## Phase 11: Multi-Region Active-Active ✅
Global scale with CRDT-based vector sync across regions.
- ✅ CRDT merge logic for concurrent vector writes (`services/crdt_sync.py` — GCounter + LWWElementSet + VectorCRDT + CRDTSyncService)
- ✅ Region-aware query routing (`services/region_router.py` — haversine distance, health-check fallback, REGIONS env var)
- ✅ Conflict resolution with vector-clock timestamps (`utils/vector_clock.py` — VectorClock, happens_before, concurrent, VectorClockStore)
- ✅ Cross-region replication monitoring API (`api/routers/replication.py` — status, regions, sync, conflicts endpoints)

## Phase 12: Enterprise Hardening ✅
Production isolation and compliance for regulated industries.
- ✅ Tenant-level dedicated schemas + namespace isolation (`services/tenant_isolation.py`)
- ✅ SQL-over-vector-db via pgvector wire-compatible layer (`services/pgvector_compat.py`)
- ✅ SAML/SSO/OIDC integration for auth (`services/sso_service.py`, `/api/auth/sso/*`)
- ✅ Audit dashboard with SOC2/GDPR export (`services/audit_service.py`, `/api/enterprise/audit/*`)

## Phase 13: Plugin SDK & Ecosystem ✅
Hot-loadable custom index algorithms, encoders, and storage backends.
- ✅ Plugin registry with versioned manifest (`services/plugin_registry.py`)
- ✅ Plugin SDK base classes (`sdk/plugin_sdk/`) — BaseIndexPlugin, BaseEncoderPlugin, BaseStoragePlugin, @plugin decorator, BruteForcePlugin example
- ✅ Marketplace API (`api/routers/plugins.py`) — list, load, unregister, search, stats endpoints
- ✅ Plugin sandbox (tracemalloc memory guard, subprocess isolation, 30s timeout)

## Phase 14: Intelligent Query Mesh ✅
Self-optimizing query routing and cost governance.
- ⚪ Query cost predictor (estimated scan count before execution)
- ⚪ Budget-aware query scheduler with tenant credit pools
- ⚪ Cross-index fusion with real-time latency/recall telemetry
- ⚪ Materialized view auto-recommendation engine

## Phase 15: Vector-Native AI Features ✅
Differentiated AI capabilities beyond basic ANN search.
- ⚪ Fine-tuning interface for ranking models (LTR/ColBERT)
- ⚪ RLHF feedback loop from user click/skip signals
- ⚪ On-device embedding with federated update sync
- ⚪ Vector explanation (why this result?) via attention attribution

## Phase 16: Managed Cloud Platform ✅
Self-service deployment, monitoring, and billing.
- ⚪ Helm chart auto-scaling with spot/preemptible nodes
- ⚪ Usage metering and per-tenant billing API
- ⚪ Web admin console with query analyzer and cost explorer
- ⚪ One-click restore from backup with point-in-time recovery

## Phase 17: Self-Hosted Web Search Engine ✅
Full search engine on top of the vector DB — Exa-style (neural) + SerpAPI-style (keyword) capabilities, entirely self-hosted with zero external search APIs. Once we own the index, third-party SERP/neural-search APIs become redundant. Code lives in `search_engine/`; API in `api/routers/web_search.py`.

### 17.1 Crawler & ingestion ✅
- ✅ Async fetcher (httpx) with politeness rate-limiting and `robots.txt` compliance — `search_engine/crawler/{fetcher,robots,crawler}.py`
- ✅ URL frontier (SQLite, BFS + priority scoring) with canonicalization — `crawler/frontier.py`
- ✅ Near-duplicate detection (SimHash + Hamming) before indexing — `crawler/dedup.py`
- ✅ HTML extraction (trafilatura w/ stdlib fallback) — clean text, title, links — `crawler/parser.py`

### 17.2 Dual index (keyword + neural) ✅
- ✅ BM25 keyword path — reuses `utils/bm25_index.py`, persisted to `indexes/{col}/sparse.json`
- ✅ Local embedder (sentence-transformers, offline) — reuses `services/embedding_service.py`
- ✅ Dual-write pipeline: page → BM25 + ANN vector, shared doc_id for fusion — `search_engine/ingest/web_ingest.py`

### 17.3 Retrieval & ranking ✅
- ✅ RRF fusion + rerank — reuses existing `SearchEngineService`
- ✅ Query understanding: normalize, expand, rewrite, multi-query — `search_engine/query/intel.py`
- ✅ Query router: `route()` in `intel.py` — auto-detects keyword/neural/hybrid per query
- ✅ HyDE hook — `rewrite(llm=...)` present; LLM can be injected without touching core

### 17.4 Serving ✅
- ✅ Search + crawl API — `GET /api/web/search`, `POST /api/web/crawl`
- ✅ Result cache — text-hash L1 cache (300 s TTL) in `web_search.py`; `DELETE /api/web/cache` to flush
- ✅ Eval harness: NDCG/MRR/Recall@k/Precision@k — `search_engine/eval/harness.py`
- ✅ Freshness/recrawl scheduler (adaptive interval) — `search_engine/recrawl.py`
- ✅ Background recrawl worker — `search_engine/worker.py`; `POST /api/web/recrawl/start|stop`, `GET /api/web/recrawl/status`

### 17.5 Search UI ✅
- ✅ Google/Perplexity-style React search page — `frontend/src/pages/SearchPage.tsx`
- ✅ Empty state: centered logo + large search bar + hint chips
- ✅ Results: AI Answer card (synthesized from top snippets + inline citations) + ranked web result cards
- ✅ Tab selector: All / Neural / Keyword
- ✅ `/search` route added to `main.tsx` via `react-router-dom`; "Search" link in landing nav

**Tests:** `search_engine/tests/` — 18 passing (all green).
**Bonus fix:** repaired pre-existing `lifespan` NameError in `api/main.py` that made the whole API unimportable.

**Phase 17 status: ✅ Complete**
