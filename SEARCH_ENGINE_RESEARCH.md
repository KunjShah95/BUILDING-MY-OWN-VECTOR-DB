# Search Engine on Top of Vector DB — Research Dossier

Curated reading/watch list for building a search engine layer on top of this vector DB.
Loop-maintained. Pass 1 = 2026-06-27.

---

## 1. Architecture — full-stack semantic search

- [Vector Database Tutorial: Build a Semantic Search Engine (DEV)](https://dev.to/infrasity-learning/vector-database-tutorial-build-a-semantic-search-engine-27kb) — embeddings → index → query, end to end.
- [How to Build Vector Database Architecture (OneUptime, Jan 2026)](https://oneuptime.com/blog/post/2026-01-30-vector-database-architecture/view) — ingestion, embedding, index, storage, search engine components.
- [Vector Databases Explained: Architecture & System Design (DEV)](https://dev.to/matt_frank_usa/vector-databases-explained-architecture-and-system-design-for-ai-apps-41pg)
- [Elastic vector DB practical example](https://www.elastic.co/search-labs/blog/elastic-vector-database-practical-example) — 6 vector-search tips, opinionated architectural rationale.
- [Building Scalable AI-Powered Apps with Cloud DBs (arXiv PDF)](https://arxiv.org/pdf/2504.18793) — scaling/perf considerations.
- [Beyond Vector Search: 3-Tiered Graph-RAG (MLMastery)](https://machinelearningmastery.com/beyond-vector-search-building-a-deterministic-3-tiered-graph-rag-system/) — quad store + vector DB hybrid.

## 2. Hybrid search — BM25 + dense + fusion + rerank

- [Hybrid Search: BM25, Vector & Reranking 2026 (Digital Applied)](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026) — BM25 params, ANN types, RRF vs linear, per-vendor (Pinecone/Weaviate/Qdrant/ES), cross-encoder rerank. **Most complete.**
- [Hybrid Search Explained (Big Data Boutique)](https://bigdataboutique.com/blog/hybrid-search-explained) — parallel lexical+dense, fuse ranked lists. Production default.
- [Hybrid Search for RAG (Denser.ai)](https://denser.ai/blog/hybrid-search-for-rag/) — RRF fusion; WANDS benchmark 0.7497 NDCG (+7.4% over either alone).
- [Better RAG Accuracy with Hybrid BM25 + Dense (Medium, Patricia)](https://medium.com/@pbronck/better-rag-accuracy-with-hybrid-bm25-dense-vector-search-ea99d48cba93) — 48% lift, cascading BM25+FAISS+cross-encoder, code included.
- [Building Hybrid Search That Actually Works (Ranjan Kumar)](https://ranjankumar.in/building-a-full-stack-hybrid-search-system-bm25-vectors-cross-encoders-with-docker) — full-stack + Docker, prod tradeoffs.
- [Hybrid Search 101 (Max Petrusenko)](https://www.maxpetrusenko.com/blog/hybrid-search-101-bm25-vector-reranking)

## 3. ANN index internals (relevant to OUR DB core — phase 10)

- [ANN Search Explained: IVF vs HNSW vs PQ (TiDB/PingCAP)](https://www.pingcap.com/article/approximate-nearest-neighbor-ann-search-explained-ivf-vs-hnsw-vs-pq/)
- [ANN Index Types: Flat/HNSW/IVF/IVF-PQ — when to choose (AbstractAlgorithms, Jan 2026)](https://www.abstractalgorithms.dev/ann-index-types-when-to-choose-hnsw-ivf-pq-flat)
- [HNSW vs IVF (Milvus blog)](https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md)
- [Understanding Modern Vector Search: HNSW → IVF-PQ (Medium)](https://medium.com/@mehrcodeland/understanding-modern-vector-search-from-hnsw-to-ivf-pq-f582b7e9ee89)
- [NVIDIA cuVS IVF-PQ deep dive](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/) — GPU accel, PQ compression mechanics.
- [HNSW vs IVF (MyScale)](https://www.myscale.com/blog/hnsw-vs-ivf-explained-powerful-comparison/)

## 4. Chunking + embeddings (ingestion quality)

- [Best Chunking Strategies for RAG 2026 (Firecrawl)](https://www.firecrawl.dev/blog/best-chunking-strategies-rag)
- [Document Chunking: 9 Strategies, size & overlap (LangCopilot)](https://langcopilot.com/posts/2025-10-11-document-chunking-for-rag-practical-guide) — 256-512 tok sweet spot, 10-20% overlap.
- [Advanced Chunking Methods 2026 (TecAdRise)](https://tecadrise.ai/blog/advanced-chunking-methods-2026) — late chunking.
- [Systematic Investigation of Chunking + Embedding Sensitivity (arXiv)](https://arxiv.org/html/2603.06976)
- [Ultimate Guide to Chunking (Databricks community)](https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-chunking-strategies-for-rag-applications/ba-p/113089) — code examples.
- Key data point: recursive 512-tok ranked #1 (69%) vs semantic 54% in Vecta Feb-2026 benchmark; adaptive/topic-boundary chunking 87% vs 13% fixed in MDPI clinical study.

## 5. Query understanding — rewrite / expand / HyDE / multi-query

- [Retrieval Is the Bottleneck: HyDE, Query Expansion, Multi-Query RAG (Medium, Hakim)](https://medium.com/@mudassar.hakim/retrieval-is-the-bottleneck-hyde-query-expansion-and-multi-query-rag-explained-for-production-c1842bed7f8a)
- [Query rewriting for RAG (Meilisearch)](https://www.meilisearch.com/blog/query-rewrite-rag)
- [Query Rewriting & Multi-Query: improve recall fast](https://thegeocommunity.com/blogs/generative-engine-optimization/query-rewriting-multiquery-rag/)
- HyDE/Query2Doc → 14-37% retrieval accuracy gains (bridge query↔doc semantic gap).

## 6. From-scratch build tutorials (hands-on, copy patterns)

- [Build a Semantic Search Engine from scratch (Storyblok)](https://www.storyblok.com/mp/build-a-semantic-search-engine-from-scratch) — Weaviate, schema, ANN.
- [Semantic Search with Vector DBs — Practical Guide (Medium, Bensalah)](https://medium.com/@amdj3dax/building-a-semantic-search-engine-with-vector-databases-a-practical-guide-4829fc934e53)
- [From RAG to Riches — OpenSearch (Xyonix)](https://www.xyonix.com/blog/from-rag-to-riches-a-practical-guide-to-building-semantic-search-using-embeddings-and-the-opensearch-vector-database)
- [Build Your First Semantic Search System w/ code (MLOps Community)](https://mlops.community/how-to-build-your-first-semantic-search-system-my-step-by-step-guide-with-code/)
- [Semantic Search + RAG @ CERN blog](https://db-blog.web.cern.ch/node/191)
- [Knowledge Base Search project (Dataquest)](https://www.dataquest.io/blog/vector-database-practice-project/) — chunking + hybrid practice.

## 7. Video

- [Building Search Apps with Vector Databases (Microsoft Learn, GenAI for Beginners pt 8/18)](https://learn.microsoft.com/en-us/shows/generative-ai-for-beginners/building-search-apps-vector-databases-generative-ai-for-beginners)
- _YouTube channel deep-dives queued for next pass (James Briggs/Pinecone, Sam Witteveen, Underfitted, Vespa/Weaviate talks)._

---

## Synthesis → what to build on top of this DB
1. **Hybrid retrieval** (BM25 sparse + own HNSW dense) fused via **RRF** — production default, +7% NDCG.
2. **Cross-encoder rerank** top-k — biggest accuracy lift after fusion.
3. **Query layer**: rewrite + multi-query/HyDE before hitting index.
4. **Ingestion**: recursive 512-tok chunks, 10-20% overlap, structure-aware splitters.
5. Leverage existing phases 10/15 (DiskANN, ColBERT/LTR ranking) for the ranking stack.

## Next-pass gaps _(cleared in Pass 2 below)_

---

## Pass 2 — 2026-06-27 (rerankers · eval · video · serving)

### 8. Rerankers & late interaction

- [An Overview of Late Interaction: ColBERT, ColPali, ColQwen (Weaviate)](https://weaviate.io/blog/late-interaction-overview) — MaxSim scoring explained; ColBERT retrieves 100-500 candidates then reranks.
- [jina-reranker-v3: Last-but-Not-Late-Interaction (arXiv 2509.25085)](https://arxiv.org/html/2509.25085v2) — beats ColBERTv2 on BEIR at lower cost, ONNX-exportable.
- [Top 5 Re-Ranking Models Under 150 ms (Medium/HashBlock)](https://medium.com/@connect.hashblock/top-5-re-ranking-models-under-150-ms-de675a78f4ed) — ms-marco-MiniLM L-6/L-12, cross-encoder/nli-deberta; budget < 150 ms achievable.
- [Reranker on the request path — latency budget (ZeroEntropy)](https://zeroentropy.dev/playbooks/reranker-on-the-request-path/) — rule: rerank ≤ top-50; skip on cached; async pre-warm.
- [ColBERT FastEmbed multivectors (Qdrant)](https://qdrant.tech/documentation/fastembed/fastembed-colbert/) — ColBERT via FastEmbed (no HF hub needed), ONNX runtime, dim=128.
- [LIR Workshop @ ECIR 2026 (arXiv 2511.00444)](https://arxiv.org/pdf/2511.00444) — academic state-of-the-art survey.
- **Key data**: ColBERT 3-5× throughput vs cross-encoder; ~1-2 NDCG@10 pts worse. Use cross-encoder (our `RerankerService`) for quality, ColBERT for speed-recall trade.

### 9. Eval harnesses & BEIR

- [BEIR Benchmark Leaderboard 2025/2026 (Ailog RAG)](https://app.ailog.fr/en/blog/news/beir-benchmark-update) — 18 datasets (MS MARCO, NQ, TREC-COVID, FEVER, …), nDCG@10 primary.
- [Evaluating Search Relevance with BEIR (Elastic)](https://www.elastic.co/search-labs/blog/evaluating-search-relevance-part-1) — practical BEIR integration walkthrough.
- [MRR vs MAP vs NDCG in 2026 (FutureAGI)](https://futureagi.com/blog/what-is-mrr-map-ndcg-2026/) — nDCG chosen over MRR/MAP because it handles graded relevance; needed for multi-doc ranking.
- [BEIR glossary (Zilliz)](https://zilliz.com/glossary/beir) — concise; good for onboarding.
- **Key**: our `eval/harness.py` already implements NDCG/MRR/Recall/Precision. To run BEIR datasets, swap in BEIR's `GenericDataLoader` → our retriever wrapper.

### 10. YouTube channels

- [Pinecone Official YouTube](https://www.youtube.com/channel/UC02IWPisqHk2nsWfurnl-mQ) — NLP for Semantic Search series; dense vector, sparse retrieval, hybrid.
- [Pinecone NLP for Semantic Search series](https://www.pinecone.io/learn/series/nlp/) — 10-part written+video course; covers bi-encoder → cross-encoder → ColBERT.
- James Briggs (author of above series) — search YouTube for "James Briggs vector" for hands-on notebooks.
- Sam Witteveen, Yannic Kilcher — cover embedding models, BEIR, LTR for ML practitioners.

### 11. Latency & serving

- [RAG Latency E2E Optimization (dasroot.net, Feb 2026)](https://dasroot.net/posts/2026/02/rag-latency-optimization-vector-database-caching-hybrid-search/) — caching > index tuning for p50; Redis for hot queries.
- [Memory Management & Caching for Vector Search (apxml.com)](https://apxml.com/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance/memory-management-caching)
- [QVCache: Query-Aware Vector Cache (arXiv 2602.02057)](https://arxiv.org/pdf/2602.02057) — backend-agnostic ANN cache; **p50 latency 40–1000× reduction**, memory < 1% of full index.
- [Scaling Vector Search: Sharding & Latency (Owlbuddy)](https://owlbuddy.com/scaling-a-vector-search-pipeline-sharding-and-latency-optimization/) — shard by collection, route by hash.
- [Cloud-Native Vector Search Performance (arXiv 2511.14748)](https://arxiv.org/pdf/2511.14748) — comparison of managed vs self-hosted at scale.
- [Semantic Retrieval at Walmart Scale (arXiv 2412.04637)](https://arxiv.org/pdf/2412.04637) — production hybrid retrieval with caching layer.
- **Case study**: Notion reduced latency 70-100ms → 50-70ms via embedding cache + serverless + self-hosted models (10× capacity, 90% cost cut).
- **Tied to our code**: `_search_cache` (5-min TTL) in `web_search.py` is the first layer; QVCache approach can be Phase 10 addition.

---

# DETAILED PLAN & ARCHITECTURE — Self-Hosted Web Search Engine

**Goal:** Build a full search engine on top of the existing vector DB that delivers both
Exa-style (neural) and SerpAPI-style (keyword) capabilities — **entirely self-hosted, zero
external search APIs**. Once we own the crawl + index, third-party SERP/neural APIs are
redundant. Maps to ROADMAP Phase 17.

## 0. Core principle
SerpAPI exists only to piggyback Google's index; Exa is neural search over its own crawl.
With our own crawler + dual index, both collapse into ONE engine. We are not a client of a
search engine — we *are* the search engine.

## 1. High-level architecture

```
                          ┌──────────────────────────────────────┐
                          │            INGEST PIPELINE            │
                          │                                       │
   seed URLs ───► Frontier ──► Fetcher ──► Parser ──► Dedup ──► Embedder
                  (priority)   (httpx,      (traf-     (Sim-      (local
                   queue)      robots,      ilatura)   Hash)      ST model)
                          │        │            │                    │       │
                          │        └─ discover new links ───────────►│       │
                          │                                          ▼       ▼
                          │                              ┌────────────────────────┐
                          │                              │   DUAL INDEX (ours)    │
                          │                              │  ┌─────────┐ ┌───────┐ │
                          │                              │  │ BM25    │ │ ANN   │ │
                          │                              │  │ inverted│ │ HNSW/ │ │
                          │                              │  │ index   │ │ DiskANN│ │
                          │                              │  └─────────┘ └───────┘ │
                          └──────────────────────────────└────────────────────────┘
                                                                  ▲
   ┌──────────────────────────────────────────────────────────┐ │
   │                       QUERY PIPELINE                       │ │
   │                                                            │ │
 user query ─► Query Understanding ─► Router ─┬─ keyword path ──┘ │
              (rewrite/HyDE/multi-q)          └─ neural path ──────┘
                                              │
                                   ┌──────────▼──────────┐
                                   │  RRF Fusion + dedup  │
                                   └──────────┬──────────┘
                                   ┌──────────▼──────────┐
                                   │ Cross-encoder rerank │
                                   └──────────┬──────────┘
                                   ┌──────────▼──────────┐
                                   │  Result cache + API  │
                                   └──────────┬──────────┘
                                          results
```

## 2. Component breakdown

### 2.1 Crawler (Phase 17.1) — biggest new piece
- **Fetcher**: async `httpx`/`aiohttp`, connection pool, per-host concurrency cap, retry+backoff, timeout.
- **Politeness**: `robots.txt` parse + cache (`urllib.robotparser`/`reppy`), crawl-delay honored, per-domain rate limit token bucket.
- **Frontier**: URL queue with priority (depth, domain authority, freshness). BFS default. Persistent (survive restart) — back by Redis or own KV.
- **Dedup**: URL canonicalization (strip utm, sort params) + content near-dup via SimHash/MinHash LSH.
- **Politeness/legal**: respect `noindex`, robots; identifiable User-Agent; opt-out honored.

### 2.2 Parser / extractor
- HTML → clean main text: `trafilatura` (best boilerplate removal) or `readability-lxml`.
- Extract: title, meta description, canonical URL, outbound links (feed frontier), lang detect.
- Optional: PDF/doc extract (already have `pymupdf` in deps).

### 2.3 Dual index (Phase 17.2)
- **BM25 keyword path** (SerpAPI-style): tokenizer (unicode, lowercase, stopword, stemming), postings lists, term/doc-freq stats. Use embeddable `tantivy` (Rust, fast) OR own impl for full control. Stores doc_id → score.
- **Neural path** (Exa-style): local embedder `sentence-transformers` (e.g. `bge-base`/`gte`), fully offline. Vectors → existing HNSW/DiskANN core.
- **Dual write**: each parsed doc → (a) tokenize→postings, (b) embed→ANN insert. Same doc_id keyspace → fusion can join.
- **Doc store**: doc_id → {url, title, snippet, text, crawl_ts} for result hydration.

### 2.4 Query pipeline (Phase 17.3)
- **Query understanding**: spell/normalize → rewrite → optional HyDE (embed hypothetical doc) → multi-query expansion (N variants).
- **Router**: classify intent → keyword-only / neural-only / both. Heuristic first (quoted/exact → keyword; natural lang → neural), ML later.
- **Fusion**: Reciprocal Rank Fusion `score = Σ 1/(k + rank_i)` (k≈60). Merge BM25 + ANN ranked lists, dedup by doc_id/SimHash.
- **Rerank**: local cross-encoder `ms-marco-MiniLM-L6-v2` over top-K (≈50-100). ONNX/quantized for speed. Ties to Phase 15 ColBERT/LTR.

### 2.5 Serving (Phase 17.4)
- Query API (FastAPI — already in stack): `/search?q=&mode=&k=`.
- Result cache (hot queries) — TTL + invalidate on recrawl.
- Recrawl scheduler: freshness score per doc → re-fetch stale.
- Eval harness: NDCG@k, MRR, Recall@k vs golden set; BEIR datasets for regression.

## 3. Module / file layout (proposed)

```
search_engine/
├── crawler/
│   ├── fetcher.py          # async HTTP, retry, UA
│   ├── frontier.py         # priority URL queue (persistent)
│   ├── robots.py           # robots.txt parse+cache, rate limit
│   ├── dedup.py            # SimHash/MinHash LSH
│   └── scheduler.py        # recrawl/freshness
├── ingest/
│   ├── parser.py           # trafilatura extract
│   ├── embedder.py         # local sentence-transformers
│   └── pipeline.py         # dual-write orchestration
├── index/
│   ├── bm25.py             # inverted index + BM25 (or tantivy bind)
│   ├── ann.py              # adapter → existing vector DB core
│   └── docstore.py         # doc_id → metadata/text
├── query/
│   ├── understanding.py    # rewrite/HyDE/multi-query
│   ├── router.py           # intent → source selection
│   ├── fusion.py           # RRF + dedup
│   └── rerank.py           # cross-encoder
├── api/
│   └── server.py           # FastAPI /search
└── eval/
    └── harness.py          # NDCG/MRR/Recall, BEIR
```

## 4. Tech stack (all local, no paid API)
| Layer | Choice | Note |
|-------|--------|------|
| Crawl | `httpx` async + `selectolax` | fast parse |
| Extract | `trafilatura` | boilerplate removal |
| Dedup | SimHash + `datasketch` MinHash LSH | |
| BM25 | `tantivy` (Rust) or own | embeddable, fast |
| Embed | `sentence-transformers` (bge/gte) | offline |
| ANN | **existing vector DB core** ✅ | HNSW/DiskANN |
| Rerank | `ms-marco-MiniLM` cross-encoder, ONNX | local |
| Serve | FastAPI ✅ + Redis cache | |
| Eval | BEIR, custom golden set | |

## 5. Build milestones (incremental, each shippable)
- **M1 — Vertical slice**: crawl 1 seed domain → extract → embed → ANN only → neural search API. Proves end-to-end.
- **M2 — Keyword path**: add BM25 inverted index + dual write. Keyword search works standalone.
- **M3 — Hybrid**: RRF fusion of both paths + dedup. Measure NDCG lift.
- **M4 — Rerank**: cross-encoder over top-K. Measure lift again.
- **M5 — Query intelligence**: rewrite/HyDE/multi-query + router.
- **M6 — Scale & freshness**: persistent frontier, recrawl scheduler, cache, broaden crawl.
- **M7 — Eval & tune**: BEIR harness, golden sets, tune BM25 params / ANN ef / RRF k / rerank depth.

## 6. Honest constraints
- **Coverage**: matching Google needs web-scale crawl (billions → ROADMAP Phase 10). For vertical/niche search (docs, domain), solo-buildable now.
- **Cost shift**: no API $, but you own crawl infra + storage + compute. Tradeoff = control + no per-query fee.
- **Legal/ethical**: honor robots.txt, rate limits, copyright, opt-out. Identifiable UA.
- **Freshness**: own crawl = staleness risk; recrawl scheduler mandatory for fresh results.

## 7. Open decisions (resolve before M1)
- BM25: build own vs `tantivy` binding?
- Embedder model: bge-base vs gte vs e5 (recall vs latency).
- Frontier backing store: Redis vs own KV vs SQLite.
- Crawl scope first target: which seed domain/vertical for M1?

---

# USER ACCESS LAYER — how people/apps actually use it

**Big win: access plumbing already exists.** Repo has FastAPI REST (`api/`), GraphQL, gRPC,
WebSocket streaming, OpenAI-compat endpoint, and SDKs in 6 languages. The search engine does
NOT need new infra — it exposes itself through these existing surfaces. New = one `/search`
route + a search UI on the existing React frontend.

## 8. User personas → access surface

| Persona | Need | Surface |
|---------|------|---------|
| **Humans** | type query, browse results | Web search UI (extend `frontend/`) |
| **Developers** | programmatic search in apps | REST `/search`, GraphQL, gRPC, 6 SDKs |
| **AI agents / LLMs** | search-as-a-tool for RAG | MCP server + OpenAI-compat endpoint (`api/routers/openai_compat.py`) + `rag.py` |
| **Power/data users** | bulk, analytics, exports | CLI + batch API |
| **Embedders** | drop search into their site | JS widget / iframe / browser extension |

## 9. The five access modes

### 9.1 Web search UI (humans) — extend existing React `frontend/`
- Google/Exa-style: search box → results page (title, URL, snippet, favicon).
- Controls: mode toggle (neural / keyword / hybrid), filters (domain, date, lang), pagination/infinite scroll.
- Autocomplete (prefix on BM25 terms), "did you mean", instant results.
- Result actions: open, cite, "find similar" (neural neighbors of a result).
- Streaming results via existing WebSocket (`ws_search.py`) — show fast keyword hits, stream reranked.

### 9.2 REST/GraphQL/gRPC API (developers) — extend existing `api/`
- New route `GET/POST /api/search` in a `search_engine` router (sibling to `search_enhanced.py`).
  - params: `q`, `mode` (neural|keyword|hybrid), `k`, `filters`, `rerank` (bool), `page`.
  - returns: ranked `[{doc_id, url, title, snippet, score, source}]` + timing/debug.
- GraphQL field in existing `api/graphql/schema.py` for typed clients.
- gRPC method in `api/grpc/server.py` for low-latency internal callers.
- WebSocket `ws_search.py` for streaming/live-as-you-type.

### 9.3 SDKs (developers, 6 langs) — extend existing `sdk/`
- Add `search()` method to each client (Py/TS/Go/Java/Rust/.NET), mirroring REST.
- Python: also a LangChain `Retriever` (already have `langchain_vectorstore.py`) so it drops into any RAG chain.
- Example: `client.search("query", mode="hybrid", k=10)`.

### 9.4 AI-agent / LLM access — search-as-a-tool
- **MCP server**: expose `web_search(query, k)` tool so Claude/agents call it directly. Highest-leverage for the agent era.
- **OpenAI-compat** (`openai_compat.py` already exists): wire search into the RAG/completions flow so any OpenAI-SDK app gets grounded answers.
- **RAG endpoint** (`rag.py`): search → retrieve → stuff context → LLM answer + citations. This is the "answer engine" (Perplexity-style) layer on top of raw search.

### 9.5 CLI + embeddable
- CLI: `vdb search "query" --mode hybrid -k 10` for scripts/data users.
- JS widget: `<script>` drop-in search box → calls REST. Browser extension for "search my index from anywhere".

## 10. Two product shapes (same engine, different output)
- **Search engine**: returns ranked links + snippets (Exa/Google shape). Surfaces 9.1-9.3.
- **Answer engine**: search → rerank → LLM synthesizes answer with citations (Perplexity shape). Surface 9.4 (`rag.py`).
Both share the retrieval core; answer engine = search engine + generation step.

## 11. Cross-cutting (apply to all surfaces)
- **Auth/tenancy**: reuse existing `auth_middleware.py` + `tenants.py` — API keys, per-tenant indexes, rate limits.
- **Caching**: existing `query_cache.py` for hot queries.
- **Observability**: existing `monitoring.py` + `dashboard.py` — QPS, latency, recall, click-through.
- **Feedback loop**: log clicks/skips → learning-to-rank (ROADMAP Phase 15 RLHF).

## 12. Minimal user-facing MVP (after engine M1-M3)
1. `GET /api/search?q=&mode=hybrid&k=10` → JSON results.
2. Search box + results page on existing React frontend.
3. `client.search()` in Python + TS SDK.
4. MCP `web_search` tool.
That's all four persona groups served from one engine.
