# I Built a Production-Ready Vector Database from Scratch — Here's What I Learned

Every time I told someone I was building a vector database from scratch, they'd give me the same look. You know the one. "Why not just use FAISS? Or Pinecone? Or Qdrant?" And they're not wrong — those are excellent tools. But I wasn't trying to solve a problem that needed a vector database. I was trying to understand how vector search actually works, from the ground up.

Anyone can call `faiss.IndexFlatIP()` and get results. But most people couldn't tell you what happens inside that call. And honestly, I couldn't either. So I decided to build one. A real one. With HNSW, IVF, PostgreSQL persistence, a REST API, monitoring, the works. Took me months. Here's everything I wish someone had told me before I started.

## The Architecture: Four Layers of Pain and Joy

The final design settled into four layers that I'm actually proud of:

```
API (FastAPI) → Business Logic (services/) → Algorithms (utils/) → Database (SQLAlchemy → PostgreSQL)
```

The API layer exposes something like 60 REST endpoints with auto-generated OpenAPI docs. I went with FastAPI because it handles validation (Pydantic), documentation (Swagger/ReDoc), and async out of the box. You can hit `/docs` and get a full interactive playground without writing a single line of OpenAPI YAML yourself.

Under that sits a service layer that separates CRUD logic from index management. It sounds obvious now, but early on I had everything tangled together — embedding logic inside route handlers, database queries mixed with distance calculations. The refactor into `vector_service.py` and `vector_indexer.py` was one of the best decisions I made.

The algorithm layer is where things got real. That's hand-written HNSW and IVF, plus distance calculations and clustering utilities. No FAISS. No scikit-learn for the core indexing (though I use it for K-Means initialization in IVF). Pure NumPy, some Numba JIT where it counts, and a lot of staring at papers.

The database layer is PostgreSQL via SQLAlchemy ORM. Yes, PostgreSQL. More on that later.

## What I Got Wrong the First Time (and the Second, and the Third)

My first working version used brute force search. Insert vectors into a list, loop through everything, compute distances, sort, return top-k. It worked. It was correct. It took **45 milliseconds** per query on 10,000 vectors.

45 milliseconds sounds fast, right? Until you realize that means about 22 queries per second. Add concurrent users, add network overhead, and suddenly your "vector database" is a bottleneck. I ran the numbers: at 100K vectors, brute force would take ~450ms per query. At 1 million, we're talking seconds. That's not a database. That's a batch job.

The fix was obvious: approximate nearest neighbor search. But implementing it was anything but.

## The HNSW Moment

HNSW (Hierarchical Navigable Small World) is the gold standard for ANN search, and for good reason. The idea is elegant: build a multi-layer graph where upper layers have fewer nodes with long-range connections (for efficient navigation), and lower layers have dense connections for precision. Search starts at the top layer, greedily moves toward the query, then descends layer by layer.

The actual implementation was a week-long debugging nightmare. The hardest part is getting the multi-layer bidirectional linking right. When you insert a node, you randomly assign it a level using an exponential distribution — `int(-log(random()) * levelMult)`. Then you need to:

1. Navigate from the entry point down to the new node's level
2. Find its nearest neighbors at each level
3. Add bidirectional connections
4. Prune connections if you exceed `m` per node
5. Update the entry point if the new node ends up at a higher level

Simple, right? No. The bug that ate my weekend was in the level generation. I was using `random.random()` which generates [0, 1), and `-log(0)` is infinity. So occasionally a node would get assigned level 255 (I'd capped it), and then the search would try to follow non-existent connections at levels that had no other nodes. The graph would build but searches would either miss results or crash. It took me three days to realize `random.random()` can return exactly 0.0 (it's rare but it happens). Adding `max(-log(max(r, 1e-10)), 0)` fixed it.

But once it worked? Magic. HNSW with m=16 gave me **3.2ms average query time** with 98.5% recall on 10K vectors. That's 14x faster than brute force. Bump m to 32 and you get 99.2% recall at 5.1ms. The throughput jumped from 22 qps to 320 qps.

Here's the core search loop — and yes, it's this simple once the graph is built:

```python
def _search_layer(self, query_vector, ef, node_id, level):
    visited = set()
    pq = []  # min-heap of (distance, node_id)
    start_dist = self._distance(query_vector, self.graph[node_id].vector)
    heapq.heappush(pq, (start_dist, node_id))
    visited.add(node_id)
    result_set = set()

    while pq:
        dist, current_id = heapq.heappop(pq)
        if result_set and dist > max(
            self._distance(query_vector, self.graph[n].vector) for n in result_set
        ):
            break
        result_set.add(current_id)
        for neighbor_id in self.graph[current_id].neighbors.get(level, []):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                neighbor_dist = self._distance(query_vector, self.graph[neighbor_id].vector)
                heapq.heappush(pq, (neighbor_dist, neighbor_id))

    results = [(nid, self._distance(query_vector, self.graph[nid].vector)) for nid in result_set]
    results.sort(key=lambda x: x[1])
    return results
```

The `ef` parameter controls the search breadth. Higher `ef` = better recall, slower search. We tune `ef_search` at query time, `ef_construction` during build. Getting this balance right is what separates "works" from "works well."

## IVF and the Quantization Rabbit Hole

Inverted File Index (IVF) was easier to implement but harder to tune. The idea: cluster your vectors with K-Means, then at search time, only search the closest clusters. I went with 100 clusters and 10 probes (clusters to search).

The problem is that IVF's recall is fundamentally limited by the cluster assignment. If your query vector falls near a cluster boundary, you might not search the right cluster. More clusters = finer granularity but more centroids to check. More probes = better recall but more work.

Then I went down the product quantization rabbit hole. PQ splits each vector into M sub-vectors, quantizes each sub-vector to 8 bits (256 centroids), and stores only the index. For a 384-dim vector with M=32, you go from 1536 bytes to 32 bytes — 48x compression. Search uses Asymmetric Distance Computation: precompute distances from query sub-vectors to all centroids, then look up the codes. It's fast and memory-efficient.

The tradeoff is real: PQ with M=32 gets you about 92-94% recall on top of IVF's baseline. The compression is incredible, but you lose precision. For production, I'd use PQ for the index but keep raw vectors for re-ranking. That wasn't in scope for this project, but it's the obvious next step.

## Why PostgreSQL?

Every vector database startup will tell you they built a custom storage engine because "off-the-shelf databases can't handle vectors." I call bullshit. For most use cases, PostgreSQL with pgvector (or even storing vectors as JSON arrays) works fine. I didn't even use pgvector — I store vectors as `ARRAY` columns and load them into memory for indexing.

Why PostgreSQL? ACID compliance. Transactions. Rollbacks. Point-in-time recovery. I don't want my vector database to lose data because the process crashed mid-insert. I don't want to implement replication from scratch. I don't want to worry about concurrent write conflicts.

Here's the thing: in-memory indexes (HNSW, IVF) handle the search. PostgreSQL handles the durability. When you insert a vector, it goes into PostgreSQL first. Then the in-memory index gets updated. If the process crashes, you rebuild the index from PostgreSQL on startup. This is the architecture of every serious vector database, whether they admit it or not.

The SQLAlchemy integration cost me about a day. The HNSW and IVF indexes? Weeks.

## Performance Numbers That Matter

I ran a benchmark suite on 10K 128-dimensional vectors:

| Method | Recall | Avg Latency | Throughput |
|--------|--------|-------------|------------|
| Brute Force | 100% | 45.2ms | 22 qps |
| HNSW m=8 | 95.2% | 2.1ms | 450 qps |
| HNSW m=16 | 98.5% | 3.2ms | 320 qps |
| HNSW m=32 | 99.2% | 5.1ms | 195 qps |
| IVF (100 clusters) | 94.8% | 4.5ms | 220 qps |

HNSW m=16 is the sweet spot for me. 98.5% recall at 3.2ms is good enough for most applications. If you need perfect recall, you can always fall back to brute force on the filtered subset.

The benchmark also revealed something interesting: **before optimization, recall was 33% at 276ms average query time.** We were getting terrible results and didn't know why. The issue was wrong parameters — `ef_construction` was too low, `m` was too small, and we weren't normalizing vectors for cosine distance. After tuning (m=32, ef_construction=300, ef_search=50), recall jumped to 95% and latency dropped to 124ms. The optimization guide in the repo documents the full journey.

The Numba JIT optimization was another big win. I added a conditional JIT decorator — uses Numba if available, falls back to pure Python otherwise. The `euclidean_distance_batch` function with `@jit(nopython=True, parallel=True)` gave roughly 3x speedup on distance calculations. Here's the pattern:

```python
try:
    from numba import jit as numba_jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range
    def numba_jit(*args, **kwargs):
        return lambda f: f

@numba_jit(nopython=NUMBA_AVAILABLE, parallel=NUMBA_AVAILABLE)
def euclidean_distance_batch(vectors, query):
    n = vectors.shape[0]
    distances = np.empty(n, dtype=np.float64)
    for i in prange(n):
        diff = vectors[i] - query
        distances[i] = np.sqrt(np.sum(diff ** 2))
    return distances
```

Three lines of Numba gave me the same speedup as rewriting in C++.

## The Collections System and Embeddings

Around the time the core indexing was stable, I added a collections system. Collections are namespaces that scope vectors by modality and embedding model. You create a text collection (384-dim, all-MiniLM-L6-v2), an image collection (512-dim, CLIP ViT-B/32), or an audio collection (128-dim MFCC). Text gets auto-embedded on ingest, so you can just POST raw text and search by natural language query.

The audio embeddings use librosa MFCC features — 128 dimensions, CPU-friendly, no GPU needed. It's not as semantically rich as a learned embedding model, but it works for similarity search on audio signals without requiring a GPU.

The multimodal setup was trickier than I expected. CLIP embeddings for text and images share the same 512-dim space, so you can search images with text queries. But you need to make sure the collection dimension matches the embedding dimension, and that the embedding model is the same for both modalities. I added explicit validation for this after debugging a few "why are my results garbage?" moments.

## What I'd Do Next Time

If I were starting over tomorrow, here's what I'd change:

**HNSW without the hand-rolled graph.** The algorithm is subtle and bug-prone. I'd use a graph library like NetworkX or even a specialized ANN library for the graph construction, and focus on the integration layer instead. The educational value of writing it myself was immense, but the code has edge cases I'm still finding.

**Better cold-start performance.** Right now, the first search after startup builds the index from scratch. For 10K vectors it takes 2-5 seconds. For 1M vectors that would be minutes. A persistent index format (HNSW already serializes the graph structure, but rebuilding from database vectors is slow) needs proper incremental serialization.

**Write-ahead logging.** PostgreSQL handles durability, but if the process crashes between inserting into PostgreSQL and updating the in-memory index, you get inconsistency. A WAL would fix this. I'd implement a simple one using a background thread that replays pending insertions on startup.

**Multi-vector queries.** The current search takes one query vector and returns neighbors. Real applications need multi-vector queries, weighted combinations, filtering before search, and hybrid search (vector + keyword with BM25). I have a hybrid search module that does this, but it's experimental.

**Proper partitioning.** Collection-scoped search brute-forces over the PostgreSQL subset rather than using a partitioned HNSW/IVF index. For production with many collections, you need per-collection indexes.

But honestly? For a learning project, I'm happy with where it landed. 112+ tests across algorithms, API, and integration. Docker Compose deployment with Prometheus and Grafana. A published Python SDK (`pip install vector-db-client`). A 1143-line README because documentation matters.

The code is at [github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB) if you want to see what a from-scratch vector database actually looks like. The HNSW implementation alone taught me more about graph search than a hundred paper readings ever did.

Sometimes you build something not because you need it, but because building it is how you learn.
