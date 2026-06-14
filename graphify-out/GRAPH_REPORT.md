# Graph Report - .  (2026-06-12)

## Corpus Check
- 150 files · ~83,834 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2228 nodes · 3389 edges · 94 communities detected
- Extraction: 76% EXTRACTED · 24% INFERRED · 0% AMBIGUOUS · INFERRED: 814 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_API Middleware & OpenAI Compat|API Middleware & OpenAI Compat]]
- [[_COMMUNITY_Auth & API Keys|Auth & API Keys]]
- [[_COMMUNITY_HNSW Search Service|HNSW Search Service]]
- [[_COMMUNITY_Admin CLI & Dashboard|Admin CLI & Dashboard]]
- [[_COMMUNITY_gRPC Server & GNN Service|gRPC Server & GNN Service]]
- [[_COMMUNITY_Python SDK Client|Python SDK Client]]
- [[_COMMUNITY_WebSocket Search & Embeddings|WebSocket Search & Embeddings]]
- [[_COMMUNITY_REST API Endpoints|REST API Endpoints]]
- [[_COMMUNITY_K-Means Clustering Tests|K-Means Clustering Tests]]
- [[_COMMUNITY_Product Quantization|Product Quantization]]
- [[_COMMUNITY_Settings & Media Storage|Settings & Media Storage]]
- [[_COMMUNITY_LangChain Integration|LangChain Integration]]
- [[_COMMUNITY_HNSW Index & Scale Benchmarks|HNSW Index & Scale Benchmarks]]
- [[_COMMUNITY_Comprehensive API Tests|Comprehensive API Tests]]
- [[_COMMUNITY_BM25 Hybrid Search|BM25 Hybrid Search]]
- [[_COMMUNITY_Int8 Quantized Index|Int8 Quantized Index]]
- [[_COMMUNITY_Vector Model & Repository|Vector Model & Repository]]
- [[_COMMUNITY_Cloud Storage Backends|Cloud Storage Backends]]
- [[_COMMUNITY_Cross-Encoder Reranker|Cross-Encoder Reranker]]
- [[_COMMUNITY_Benchmark Runner|Benchmark Runner]]
- [[_COMMUNITY_Redis Cache Service|Redis Cache Service]]
- [[_COMMUNITY_Core Vector Database|Core Vector Database]]
- [[_COMMUNITY_Async Ingestion Queue|Async Ingestion Queue]]
- [[_COMMUNITY_API Tests|API Tests]]
- [[_COMMUNITY_OpenAI-Compat Tests|OpenAI-Compat Tests]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 92|Community 92]]
- [[_COMMUNITY_Community 93|Community 93]]
- [[_COMMUNITY_Community 94|Community 94]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 104|Community 104]]

## God Nodes (most connected - your core abstractions)
1. `VectorService` - 75 edges
2. `CollectionService` - 51 edges
3. `HNSWIndex` - 44 edges
4. `PQIndex` - 43 edges
5. `KDTreeIndex` - 39 edges
6. `VectorIndexer` - 37 edges
7. `RequestSizeLimitMiddleware` - 35 edges
8. `IVFVectorDatabase` - 35 edges
9. `TenantService` - 33 edges
10. `VectorModel` - 32 edges

## Surprising Connections (you probably didn't know these)
- `get_collection_service()` --calls--> `CollectionService`  [INFERRED]
  api\main.py → services\collection_service.py
- `get_multimodal_service()` --calls--> `MultimodalService`  [INFERRED]
  api\main.py → services\multimodal_service.py
- `get_tenant_service()` --calls--> `TenantService`  [INFERRED]
  api\routers\tenants.py → services\tenant_service.py
- `get_key_manager()` --calls--> `APIKeyManager`  [INFERRED]
  api\routers\tenants.py → services\auth_service.py
- `main()` --calls--> `HNSWVectorDatabase`  [INFERRED]
  main.py → database\hnsw_database.py

## Communities

### Community 0 - "API Middleware & OpenAI Compat"
Cohesion: 0.02
Nodes (193): RequestSizeLimitMiddleware, BaseHTTPMiddleware, BaseModel, Enum, example_auto_parameters(), example_batch_processing(), example_complete_workflow(), example_hnsw_basic() (+185 more)

### Community 1 - "Auth & API Keys"
Cohesion: 0.03
Nodes (46): create_api_template(), submit_feedback(), Base, ApiKey, ApiTemplate, Collection, ensure_pgvector_extension(), FeedbackEntry (+38 more)

### Community 2 - "HNSW Search Service"
Cohesion: 0.03
Nodes (57): get_collection_index_service(), main(), HNSWVectorDatabase, Create an HNSW index with optimized parameters          Args:             m:, Vector Database with HNSW indexing capabilities      Combines PostgreSQL for p, Save HNSW index to disk          Returns:             Save result, Load HNSW index from disk          Returns:             Load result, Search using HNSW index with optimized parameters          Args: (+49 more)

### Community 3 - "Admin CLI & Dashboard"
Cohesion: 0.03
Nodes (72): migrate(), Admin Command Line Interface for the Vector Database.  Usage:   python cli.py st, Print the overall status of the vector database., Print index and vector statistics., Rebuild the specified vector index., Run Alembic database migrations., rebuild_index(), stats() (+64 more)

### Community 4 - "gRPC Server & GNN Service"
Cohesion: 0.03
Nodes (44): get_gnn_service(), gRPC server that wraps the existing VectorService., Start the gRPC server., gRPC servicer delegating to the existing VectorService., serve(), VectorDBServicer, GNNService, GNN Service: Graph Neural Network features built over the HNSW index graph.  Pro (+36 more)

### Community 5 - "Python SDK Client"
Cohesion: 0.04
Nodes (26): Exception, Tests for the vector-db SDK client., TestCollectionsAPI, TestHTTPHelpers, TestModels, TestMultimodalAPI, TestVectorDBClientInit, TestVectorsAPI (+18 more)

### Community 6 - "WebSocket Search & Embeddings"
Cohesion: 0.03
Nodes (57): get_db(), websocket_search(), embed_text(), Embed a single text string into a dense vector., openai_chat_completion(), RAGService, RAG pipeline: PDF -> Chunk -> Embed -> Vector DB -> Retrieve -> LLM completion., Ingest any supported document, optionally extracting embedded images. (+49 more)

### Community 7 - "REST API Endpoints"
Cohesion: 0.02
Nodes (86): add_process_time_header(), batch_search(), build_collection_index(), check_index_health(), collection_index_stats(), compare_search_methods(), create_collection(), create_index() (+78 more)

### Community 8 - "K-Means Clustering Tests"
Cohesion: 0.03
Nodes (49): Test K-Means prediction, Test KMeansIndexer functionality, Test cluster assignment, Test basic K-Means Functionality, test_cluster_assignment(), test_kmeans_basic(), test_kmeans_indexer(), test_vector_indexer() (+41 more)

### Community 9 - "Product Quantization"
Cohesion: 0.03
Nodes (36): populated_pq(), Tests for Product Quantization (PQ) index., Create a small PQ index with 8-dim vectors, M=4, k_sub=16., PQ approximation should find results close to exact search., Rerank should yield same or better (lower) distances., Cosine distance should be bounded [0, 2]., A trained PQ index with 50 vectors added., Searching for an indexed vector should find itself at distance ~0. (+28 more)

### Community 10 - "Settings & Media Storage"
Cohesion: 0.05
Nodes (51): get_stored_media(), Serve a file previously stored during image/audio ingest.      Pass the `conte, BaseSettings, Config, get_settings(), Get application settings with caching., Application settings loaded from environment variables, Settings (+43 more)

### Community 11 - "LangChain Integration"
Cohesion: 0.04
Nodes (30): FakeEmbeddings, mock_client(), _mock_response(), Tests for the LangChain VectorStore integration.  Uses mocks so no running ser, Build a fake ``httpx.Response`` so ``raise_for_status`` can parse it., Simulates LangChain Embeddings — returns deterministic vectors., Patch the underlying VectorDBClient so no real HTTP calls happen., VectorStore without a collection_id (uses global endpoints). (+22 more)

### Community 12 - "HNSW Index & Scale Benchmarks"
Cohesion: 0.04
Nodes (39): benchmark_scale(), generate_vectors(), main(), Test HNSW index save/load functionality, Test HNSW graph statistics, Test HNSW search with different parameters, Test basic HNSW functionality, Test HNSW node deletion (+31 more)

### Community 13 - "Comprehensive API Tests"
Cohesion: 0.03
Nodes (36): db_session(), Test updating a vector, Test deleting a vector, Test getting all vectors with pagination, Test search functionality, Test brute force search, Test search with ef_search parameter, Test search method comparison (+28 more)

### Community 14 - "BM25 Hybrid Search"
Cohesion: 0.05
Nodes (23): Tests for BM25 index and hybrid search engine., Multi-occurrence query terms should boost scores., Documents appearing in both ranked lists should rank higher., TestBM25Index, TestHybridSearchEngine, TestReciprocalRankFusion, BM25Index, doc_ids() (+15 more)

### Community 15 - "Int8 Quantized Index"
Cohesion: 0.05
Nodes (31): Int8Index, _l2_normalize(), _make_serializable(), Int8 (8-bit integer) Quantization Index  Memory-efficient vector storage by qu, Quantize and store a vector as int8., Add multiple vectors at once.          Each entry must have ``vector``, ``vect, Search using int8 quantized index.          Parameters         ----------, Remove a vector from the index. (+23 more)

### Community 16 - "Vector Model & Repository"
Cohesion: 0.05
Nodes (27): Vector, Get a vector by its ID, Get a vector by its database ID, Get all vectors with pagination and optional collection/tenant filter., Get all collection IDs belonging to a tenant., Get all vectors belonging to a collection with pagination., Get all vectors belonging to a specific batch, Get total count of vectors (+19 more)

### Community 17 - "Cloud Storage Backends"
Cohesion: 0.06
Nodes (17): ABC, AzureStorageBackend, create(), get_storage_backend(), LocalStorageBackend, Abstract storage backend for media and index files., Azure Blob Storage backend., Factory for creating storage backends based on config. (+9 more)

### Community 18 - "Cross-Encoder Reranker"
Cohesion: 0.06
Nodes (23): _extract_text(), is_available(), Cross-encoder re-ranking for improved search accuracy.  Uses a cross-encoder m, Re-rank using local cross-encoder model., Extract the text content from a search result dict, checking common keys., Re-rank using Cohere's re-rank API., Re-rank multiple query/result pairs.          Parameters         ----------, Cross-encoder re-ranker with optional Cohere API fallback.      Parameters (+15 more)

### Community 19 - "Benchmark Runner"
Cohesion: 0.05
Nodes (35): generate_test_data(), Vector Database Benchmark Runner Run comprehensive benchmarks and generate repo, Generate test data for benchmarking          Args:         num_vectors: Numbe, Run comprehensive benchmark, run_benchmark(), BenchmarkReport, BenchmarkResult, BenchmarkSuite (+27 more)

### Community 20 - "Redis Cache Service"
Cohesion: 0.07
Nodes (19): get_search_engine_service(), CacheService, _get_redis(), _make_key(), Invalidate all cached searches for a collection., Get Redis client (lazy init)., Caching for search results and embeddings., Cache search results keyed by query hash + k + collection. (+11 more)

### Community 21 - "Core Vector Database"
Cohesion: 0.06
Nodes (22): Search for similar vectors using brute force or indexed approach          Args, Search using the created index          Args:             query_vector: Query, Get a specific vector by ID, Get a specific vector by database ID, Get all vectors in the database with pagination, Insert a vector into the database, Update an existing vector, Delete a vector by ID (+14 more)

### Community 22 - "Async Ingestion Queue"
Cohesion: 0.07
Nodes (21): enqueue_many(), enqueue_vector(), flush_queue(), get_queue(), ingestion_status(), Async ingestion queue API router.  Provides REST endpoints for enqueuing vecto, Get (or create) the global BulkIngestionQueue singleton., Enqueue a single vector for async batch ingestion.      The vector will be sto (+13 more)

### Community 23 - "API Tests"
Cohesion: 0.07
Nodes (22): db_session(), Test getting a vector, Test getting all vectors, Test updating a vector, Test deleting a vector, Setup test database tables, Test getting statistics, Test creating vector with invalid data (+14 more)

### Community 24 - "OpenAI-Compat Tests"
Cohesion: 0.08
Nodes (10): Tests for OpenAI-compatible API endpoints.  Most tests use the TestClient from, encoding_format field is accepted but we always return float., Without collection_id, falls through to direct LLM call., With collection_id, RAG pipeline should run., Preserves full conversation history., Response matches OpenAI's chat completion format., Response shape matches OpenAI's API., TestChatCompletionsEndpoint (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.13
Nodes (18): contains(), eq(), exists(), Filter, FilterPredicate, from_dict(), gt(), gte() (+10 more)

### Community 26 - "Community 26"
Cohesion: 0.1
Nodes (10): Async SDK client for interacting with the Vector Database API. Built on httpx., Async Python SDK wrapper for the Vector DB API., Search via natural language (server-side embedding)., Perform a RAG pipeline query., Close the underlying HTTP client., Insert a single vector., Search for similar vectors., Create a new collection namespace. (+2 more)

### Community 27 - "Community 27"
Cohesion: 0.1
Nodes (19): create_tenant(), create_tenant_api_key(), delete_tenant(), get_key_manager(), get_tenant(), get_tenant_service(), list_tenant_api_keys(), list_tenants() (+11 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (13): RAGService, get_streaming_rag_service(), Streaming RAG endpoints with SSE., Stream a RAG query response token-by-token using SSE., Generic streaming LLM completion., stream_llm(), stream_rag_query(), Streaming RAG with Server-Sent Events support. (+5 more)

### Community 29 - "Community 29"
Cohesion: 0.16
Nodes (7): Log a vector insertion., Log a vector deletion., Log a metadata update., Clears the WAL. Typically called after a successful background          snapshot, Reads all operations for crash recovery playback., Append-only Write-Ahead Log (WAL) for durability.     Records all index mutation, WriteAheadLog

### Community 30 - "Community 30"
Cohesion: 0.17
Nodes (8): _cosine_batch_numba(), cosine_distance_batch(), estimate_optimal_m(), _jit_decorator(), numba_jit(), optimize_ef_search(), Conditional JIT decorator that applies Numba JIT compilation if available., suggest_index_parameters()

### Community 31 - "Community 31"
Cohesion: 0.17
Nodes (6): db_session(), OpenAPI structure check — no database required., Text-to-image search uses CLIP text encoder on image collections., setup_test_db(), test_openapi_lists_media_routes(), test_text_search_on_clip_image_collection()

### Community 32 - "Community 32"
Cohesion: 0.22
Nodes (3): load(), LSHIndex, Locality-Sensitive Hashing for cosine distance.

### Community 33 - "Community 33"
Cohesion: 0.17
Nodes (11): aggregate_series(), get_latest_per_series(), insert_timeseries_vector(), list_series(), Time-series vector support API router.  Provides endpoints for storing and que, List all distinct time-series in a collection., Get the most recent vectors in a time-series., Aggregate vector statistics over time windows for a series. (+3 more)

### Community 34 - "Community 34"
Cohesion: 0.18
Nodes (4): db_session(), _mock_embed_text(), Deterministic tiny embeddings for CI without loading transformers., setup_test_db()

### Community 35 - "Community 35"
Cohesion: 0.31
Nodes (2): AutoReindexService, IndexStatus

### Community 36 - "Community 36"
Cohesion: 0.33
Nodes (5): api(), drawIndexChart(), drawLatencyChart(), loadCollections(), loadOverview()

### Community 37 - "Community 37"
Cohesion: 0.25
Nodes (3): db_session(), API tests for collections and text ingest (requires PostgreSQL)., setup_test_db()

### Community 38 - "Community 38"
Cohesion: 0.22
Nodes (3): GPUDistanceKernel, GPU-accelerated distance computation using CUDA (optional).  Requires cupy to, GPU-accelerated distance functions using CuPy.

### Community 39 - "Community 39"
Cohesion: 0.29
Nodes (5): ingest_pdf(), rag_query(), RAG API router - PDF ingest and Q&A query endpoints., Upload a PDF, extract text, chunk, embed, and store vectors., Ask a question and get an answer grounded in the collection's documents.

### Community 41 - "Community 41"
Cohesion: 0.4
Nodes (3): gql_check(), Comprehensive test script for new endpoints: 1. GraphQL API 2. Time-series vec, Check GraphQL responses (use 'data' key instead of 'success').

### Community 42 - "Community 42"
Cohesion: 0.4
Nodes (4): Run migrations in 'offline' mode., Run migrations in 'online' mode., run_migrations_offline(), run_migrations_online()

### Community 43 - "Community 43"
Cohesion: 0.4
Nodes (2): Tests for WebSocket search (connection-level only)., TestWebSocketSearch

### Community 44 - "Community 44"
Cohesion: 0.4
Nodes (1): IndexSerializer

### Community 45 - "Community 45"
Cohesion: 0.67
Nodes (3): main(), _print_search(), End-to-end multimodal demo using the Vector DB Python SDK.  Requires the API r

### Community 46 - "Community 46"
Cohesion: 0.67
Nodes (2): Configure application logging          Args:         log_level: Logging level, setup_logging()

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Check database and index health.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): List all collections.

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Get a collection by ID.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Search for similar vectors using ANN.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Get index statistics.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Create a new collection namespace.

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Delete a vector by ID.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): Build an ANN index (HNSW or IVF).

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Embed and store text in a collection.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Create the store, embed the texts, and insert them.          Parameters

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Create the store from ``Document`` objects.          Parameters         -----

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Build SQLAlchemy filter conditions from a filter dict.          Simple format:

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Apply metadata filters as SQL WHERE clauses (pre-filtering).          This fil

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Post-filter results in memory (when DB-level filtering isn't available).

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Check if the cross-encoder model is loaded or can be loaded.

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): Search for similar vectors          Args:             query_vector: Query vec

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): Setup data with index for search tests

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): Deserialize an index from a JSON file.

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Metadata[key] == value

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Metadata[key] != value

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Metadata[key] > value

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Metadata[key] >= value

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): Metadata[key] < value

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): Metadata[key] <= value

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Metadata[key] is in the given list of values.

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Metadata[key] contains substring (for string values).

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Metadata has the given key (and it's not None).

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Metadata[key] is a list containing item.

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Build a filter from a simple {key: value} dictionary.         Each entry becomes

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (1): Compute cosine distances between query and all vectors on GPU.          Args:

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (1): Compute Euclidean distances between query and all vectors on GPU.          Arg

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (1): Compute inner product similarity on GPU.          Args:             query: 1-

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): Compute pairwise cosine distances between two sets on GPU.          Args:

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): Analyze HNSW index health: connectivity, neighbor distribution, etc.         Exp

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): Deserialize an index from a JSON file.

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (1): Calculate cosine distance for all vectors using vectorized NumPy ops.

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (1): Numba JIT path for cosine distance batch (parallel).

### Community 94 - "Community 94"
Cohesion: 1.0
Nodes (1): Calculate Euclidean distance for all vectors (parallelized)                  A

### Community 95 - "Community 95"
Cohesion: 1.0
Nodes (1): Find top k indices using partial sort (faster than full sort)

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (1): Quantize vectors to smaller data type                  Args:             vect

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (1): Normalize vectors for safe half-precision conversion                  Args:

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (1): Estimate memory usage for vector storage                  Args:             n

### Community 99 - "Community 99"
Cohesion: 1.0
Nodes (1): Calculate compression ratio                  Args:             original: Orig

### Community 100 - "Community 100"
Cohesion: 1.0
Nodes (1): Optimize ef_search parameter based on recall target                  Args:

### Community 101 - "Community 101"
Cohesion: 1.0
Nodes (1): Estimate optimal HNSW m parameter                  Args:             database

### Community 102 - "Community 102"
Cohesion: 1.0
Nodes (1): Suggest optimal index parameters                  Args:             database_

### Community 103 - "Community 103"
Cohesion: 1.0
Nodes (1): Deserialize an index from a JSON file.

### Community 104 - "Community 104"
Cohesion: 1.0
Nodes (1): Simple k-means for a single subspace.          Parameters         ----------

## Knowledge Gaps
- **722 isolated node(s):** `Admin Command Line Interface for the Vector Database.  Usage:   python cli.py st`, `Print the overall status of the vector database.`, `Print index and vector statistics.`, `Rebuild the specified vector index.`, `Run Alembic database migrations.` (+717 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 35`** (10 nodes): `AutoReindexService`, `.get_all_statuses()`, `.get_status()`, `.__init__()`, `._maybe_schedule_rebuild()`, `.record_change()`, `.register_collection()`, `._schedule_rebuild()`, `IndexStatus`, `auto_reindex.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (5 nodes): `test_ws_search.py`, `Tests for WebSocket search (connection-level only).`, `TestWebSocketSearch`, `.test_ws_route_details()`, `.test_ws_router_exists()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (5 nodes): `estimate_size()`, `IndexSerializer`, `load()`, `index_serializer.py`, `save()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (3 nodes): `logging.py`, `Configure application logging          Args:         log_level: Logging level`, `setup_logging()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Check database and index health.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `List all collections.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Get a collection by ID.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Search for similar vectors using ANN.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Get index statistics.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Create a new collection namespace.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Delete a vector by ID.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `Build an ANN index (HNSW or IVF).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Embed and store text in a collection.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Create the store, embed the texts, and insert them.          Parameters`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Create the store from ``Document`` objects.          Parameters         -----`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Build SQLAlchemy filter conditions from a filter dict.          Simple format:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Apply metadata filters as SQL WHERE clauses (pre-filtering).          This fil`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Post-filter results in memory (when DB-level filtering isn't available).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Check if the cross-encoder model is loaded or can be loaded.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `Search for similar vectors          Args:             query_vector: Query vec`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `Setup data with index for search tests`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `Deserialize an index from a JSON file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Metadata[key] == value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Metadata[key] != value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Metadata[key] > value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Metadata[key] >= value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `Metadata[key] < value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `Metadata[key] <= value`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Metadata[key] is in the given list of values.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Metadata[key] contains substring (for string values).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Metadata has the given key (and it's not None).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Metadata[key] is a list containing item.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Build a filter from a simple {key: value} dictionary.         Each entry becomes`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (1 nodes): `Compute cosine distances between query and all vectors on GPU.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `Compute Euclidean distances between query and all vectors on GPU.          Arg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (1 nodes): `Compute inner product similarity on GPU.          Args:             query: 1-`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `Compute pairwise cosine distances between two sets on GPU.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `Analyze HNSW index health: connectivity, neighbor distribution, etc.         Exp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `Deserialize an index from a JSON file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (1 nodes): `Calculate cosine distance for all vectors using vectorized NumPy ops.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (1 nodes): `Numba JIT path for cosine distance batch (parallel).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 94`** (1 nodes): `Calculate Euclidean distance for all vectors (parallelized)                  A`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 95`** (1 nodes): `Find top k indices using partial sort (faster than full sort)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 96`** (1 nodes): `Quantize vectors to smaller data type                  Args:             vect`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (1 nodes): `Normalize vectors for safe half-precision conversion                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 98`** (1 nodes): `Estimate memory usage for vector storage                  Args:             n`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 99`** (1 nodes): `Calculate compression ratio                  Args:             original: Orig`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 100`** (1 nodes): `Optimize ef_search parameter based on recall target                  Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 101`** (1 nodes): `Estimate optimal HNSW m parameter                  Args:             database`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 102`** (1 nodes): `Suggest optimal index parameters                  Args:             database_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 103`** (1 nodes): `Deserialize an index from a JSON file.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 104`** (1 nodes): `Simple k-means for a single subspace.          Parameters         ----------`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `VectorService` connect `Admin CLI & Dashboard` to `API Middleware & OpenAI Compat`, `Auth & API Keys`, `HNSW Search Service`, `gRPC Server & GNN Service`, `WebSocket Search & Embeddings`, `REST API Endpoints`, `Product Quantization`, `BM25 Hybrid Search`, `Vector Model & Repository`, `Cross-Encoder Reranker`, `Redis Cache Service`, `Async Ingestion Queue`, `Community 29`?**
  _High betweenness centrality (0.311) - this node is a cross-community bridge._
- **Why does `HNSWVectorDatabase` connect `HNSW Search Service` to `API Middleware & OpenAI Compat`, `Auth & API Keys`, `Admin CLI & Dashboard`, `HNSW Index & Scale Benchmarks`, `Vector Model & Repository`, `Benchmark Runner`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `HNSWIndex` connect `HNSW Index & Scale Benchmarks` to `API Middleware & OpenAI Compat`, `HNSW Search Service`, `Int8 Quantized Index`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Are the 50 inferred relationships involving `VectorService` (e.g. with `RequestSizeLimitMiddleware` and `CollectionType`) actually correct?**
  _`VectorService` has 50 INFERRED edges - model-reasoned connections that need verification._
- **Are the 40 inferred relationships involving `CollectionService` (e.g. with `RequestSizeLimitMiddleware` and `CollectionType`) actually correct?**
  _`CollectionService` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `HNSWIndex` (e.g. with `HNSWVectorDatabase` and `IndexMethod`) actually correct?**
  _`HNSWIndex` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 28 inferred relationships involving `PQIndex` (e.g. with `VectorService` and `TestTraining`) actually correct?**
  _`PQIndex` has 28 INFERRED edges - model-reasoned connections that need verification._