# How HNSW Actually Works: Building Hierarchical Navigable Small World from Scratch in Python

I read the HNSW paper (Malkov & Yashunin, 2016) and thought — how hard could it be? A graph with multiple layers, greedy search, insert with bidirectional connections. Maybe 200 lines of Python. Two weeks of late nights later, I had a working implementation and a much deeper appreciation for what the paper glosses over.

This is the post I wish existed when I started.

## The Core Idea: Skip Lists for Vectors

Here's the picture that made it click for me. Imagine you're in a massive city and need to find the closest coffee shop. You could walk every street (that's brute force — O(n)). Or you could pull up a highway map, find which highway gets you near the right neighborhood, take the exit, then navigate local streets. That's HNSW.

The "highways" are upper layers with sparse connections (long-range links). The "local streets" are layer 0 with dense connections (short-range links). You enter at the top layer, greedily descend toward your target, and by the time you hit layer 0 you're in the right neighborhood.

The level distribution is exponential:

```python
def _calculate_level(self) -> int:
    r = random.random()
    level = int(-np.log(r) * self.level_mult)
    level = min(level, 255)
    return level
```

There's nothing magical about `1/log(2)` as the `level_mult`. It comes from the paper's formula: `level = floor(-ln(unif(0,1)) * mL)` where `mL = 1/ln(M)`. For `M=2`, that's `1/ln(2)`. The goal is statistical — most nodes land at level 0, a few at level 1, exponentially fewer above that. With ~10K nodes, I rarely see anything above level 4-5. The cap at 255 is defensive; in practice you'll never hit it unless you have billions of vectors.

The hard part wasn't the math — it was getting the graph construction right.

## Building the Graph: Where Everything Goes Wrong

The insert algorithm sounds simple on paper:

1. Pick a random level for the new node
2. Start at the entry point
3. At each layer above the new node's level, greedily search toward it (ef=1)
4. At the node's level and layer 0, do a wider search and connect

Here's the actual insert flow:

```python
def insert(self, vector, vector_id, metadata=None, level=None):
    # ... create node, normalize vector ...
    
    ep = self.entry_point
    
    # Phase 1: Navigate down from top to node's level + 1
    for l in range(self.max_level, level, -1):
        if ep is not None:
            ep = self._search_layer(vector_array, 1, ep, l)[0][0]
    
    # Phase 2: Connect at the node's level
    if ep is not None:
        self._connect_node(vector_id, level, ep)
    else:
        self.entry_point = vector_id
    
    # Phase 3: Always connect at layer 0
    self._connect_node(vector_id, 0, self.entry_point)
```

Phase 1 uses `ef=1` — you only need the single closest node at each upper layer to guide your descent. Phase 2 uses `ef_construction` (default 200) for a proper search.

### The Bug That Cost Me Two Days

Here's a subtle bug that had me tearing my hair out. Look at the Phase 1 loop:

```python
ep = self._search_layer(vector_array, 1, ep, l)[0][0] if ... else ep
```

If the new node's level is *higher* than any node currently in the graph, you enter that loop with `l` from `self.max_level` down to `level + 1`. But `ep` might not have connections at that layer, or the search returns empty. The first time this happened, `_search_layer` returned `[]`, the ternary collapsed to `ep`, and I was fine — but barely.

The real nightmare was the opposite case: inserting a low-level node into a graph where the entry point was at level 5. The entry point has connections at levels 5, 4, 3, 2, 1, 0. The new node is at level 0. The loop runs from max_level (5) down to level+1 (1). At each layer, it searches from the current ep and returns the closest node. But here's the thing — if the entry point has good long-range connections, this works beautifully. If *somehow* the entry point got disconnected from its upper layers... well, we'll get to that.

The *real* bug was in `_connect_node`. If the entry point was at a higher level than the new node, and the greedy descent through upper layers landed on a node that *didn't* exist in the target layer's neighbor lists, the candidate filtering in `_connect_node` would return empty:

```python
candidates = [nid for nid, _ in search_results 
              if nid != node_id and level in self.graph[nid].neighbors]
```

That `level in self.graph[nid].neighbors` filter. I added it because I was getting KeyErrors trying to access neighbor lists that didn't exist at certain layers. But it also means if your greedy descent lands on a node that has no connections at layer 0 (yes, this happens), you get zero candidates and an isolated node. The entry point suddenly becomes wrong, and every subsequent insert compounds the error.

## The Search Algorithm: Greedy Best-First with a Beam

The heart of HNSW is `_search_layer`. It's a best-first search with a priority queue and a termination condition:

```python
def _search_layer(self, query_vector, ef, node_id, level):
    visited = set()
    pq = []
    
    start_node = self.graph[node_id]
    start_distance = self._distance(query_vector, start_node.vector)
    heapq.heappush(pq, (start_distance, node_id))
    visited.add(node_id)
    
    result_set = set()
    
    while pq:
        distance, current_id = heapq.heappop(pq)
        
        # Early termination
        if result_set and distance > max(... for n in result_set)):
            break
        
        result_set.add(current_id)
        
        if len(result_set) >= ef:
            if not pq:
                break
            worst_result_dist = max(...)
            if pq[0][0] >= worst_result_dist:
                break
        
        # Explore neighbors
        current_node = self.graph[current_id]
        if level in current_node.neighbors:
            for neighbor_id in current_node.neighbors[level]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    # ... push to queue
```

The priority queue always pops the closest node. You explore its neighbors. The ef parameter controls how wide your beam is — with `ef=1`, you're doing pure hill climbing (follow the closest neighbor, stop when they get worse). With `ef=200`, you maintain a buffer of 200 candidates and only stop when the furthest candidate is closer than anything left to explore.

The "aha" moment was realizing this is NOT Dijkstra's algorithm. You never need the shortest path through the graph — you just need to find the region closest to your query point. The graph is a navigable structure, not a shortest-path network.

## Bidirectional Connections Are Not Optional

When I first built this, I only connected new nodes to their neighbors. Here's what happened:

```
Node A connects to Node B.
B can reach A.
Never. Searches from C would find B, then B's neighbors — but not A.
```

The graph becomes directional. A search might wander into a cul-de-sac with no way out. The fix is simple but easy to forget:

```python
def _connect_node(self, node_id, level, ep_node_id=None):
    # ... search for neighbors ...
    
    for neighbor_id in neighbors:
        neighbor = self.graph[neighbor_id]
        if level not in neighbor.neighbors:
            neighbor.neighbors[level] = []
        
        # Forward connection
        node.neighbors[level].append(neighbor_id)
        
        # Backward connection — CRITICAL
        neighbor.neighbors[level].append(node_id)
```

Every "forward" edge must have a "backward" edge. Otherwise high-degree nodes create one-way streets and search quality collapses.

## The Parameters That Actually Matter

HNSW has three knobs. Here's what each does:

**m (default 16)** — Out-degree per node per layer. Higher m = denser graph = better recall, more memory, slower construction. Doubling m roughly doubles edges. The paper uses `m0 = 2*m` for layer 0 since that's where fine-grained navigation happens.

**ef_construction (default 200)** — Search breadth during construction. Higher = better graph quality (better connected, more accurate neighbor selection). Diminishing returns past 200-400. The tradeoff is *construction time* — each insert does a search with ef_construction at multiple layers.

**ef_search (runtime param)** — Search breadth during query. This is your production tuning knob. Start with `ef_search = k` and increase until recall saturates. For k=10, I typically use ef_search between 50-200. The cost is linear — ef=200 is about 2x slower than ef=100.

Here's the relationship I've observed empirically with 10K 128-dimensional vectors:

| ef_search | Recall@10 | Avg Latency |
|-----------|-----------|-------------|
| 10 | ~82% | ~5ms |
| 50 | ~93% | ~28ms |
| 100 | ~97% | ~65ms |
| 200 | ~98.5% | ~128ms |

But that latency is in Python with numpy distance calculations. In a compiled language with SIMD, you'd be looking at microseconds.

## What the Paper Didn't Tell Me

### Edge Case 1: The First Node

When the graph is empty, there's nothing to connect to:

```python
if ep_node_id is None:
    self.entry_point = node_id
    return node_id
```

The first node becomes the entry point with zero connections. Every subsequent node uses it as a starting point and adds edges. This means the first node's connectivity and position matter more than any other node's — it's the "seed" of the entire graph hierarchy.

### Edge Case 2: The Empty Graph Search

```python
def search(self, query_vector, k=5, ef=None, level=None):
    if len(self.graph) == 0:
        return []
```

Obvious in hindsight. Not mentioned in the paper.

### Edge Case 3: Level Distribution Clipping

The exponential distribution occasionally generates absurd levels for low node counts. With 1000 nodes, `-ln(random()) * level_mult` can produce level 7 or 8. That node becomes a "skycraper" with sparse connections at 7 layers above most nodes. It's correct algorithmically but wasteful. The cap at 255 is my safety valve, but in practice I'd clamp to `max_level = floor(log2(N))` or similar.

### Edge Case 4: The Entry Point Gets Deleted

```python
def delete(self, vector_id):
    # ... remove from graph ...
    
    if self.entry_point == vector_id:
        if self.graph:
            self.entry_point = next(iter(self.graph.keys()))
        else:
            self.entry_point = None
```

Not mentioned in the paper (the paper doesn't discuss deletion at all). If you delete the entry point, you need to pick another one. Any node will do — the graph is navigable from any starting point, though you might lose a few steps. I just grab the first key.

## Performance vs Brute Force

I benchmarked against brute-force cosine similarity on 10K vectors with 128 dimensions. Default params: m=16, ef_construction=200.

- **Brute force:** ~45ms per query (100% recall)
- **HNSW (ef=200):** ~3.2ms avg, 98.5% recall

That's roughly 14x faster with 98.5% recall. The "missing" 1.5% isn't a bug — it's the nature of approximate search. The graph construction has randomness (level assignment), so two runs with the same data may produce different graphs and slightly different recall. Some queries will always land at 100%, some at 95%.

The variance is real. Look at the benchmark data: some queries resolve in 0.5ms (query lands right next to the entry point), others take 700ms+ (query has to traverse half the graph). The median is around 39ms, which is far below the mean of 128ms. A few pathological queries drag up the average. I suspect this correlates with nodes at the periphery of the vector space with sparse graph connections.

## Production Lessons (Hard-Won)

### JSON Serialization Is Slow

```python
def save(self, filepath):
    graph_data = {}
    for node_id, node in self.graph.items():
        graph_data[node_id] = {
            "node_id": node.node_id,
            "vector": node.vector.tolist(),
            "neighbors": {str(k): v for k, v in node.neighbors.items()},
        }
    # ... dump to JSON ...
```

For 10K vectors, this produces a ~50MB JSON file and takes seconds to serialize. For 100K vectors, you're looking at 500MB+ and it becomes unusable. The `.tolist()` conversion alone is expensive — numpy to Python list to JSON string.

If I were shipping this to production, I'd write a binary format: flat array of floats for vectors, adjacency lists as packed uint32 arrays, and a header with metadata. mmap the whole thing at load time. My back-of-envelope estimate is 10-20x smaller and 100x faster to load.

### Index Rebuild on Schema Change

There's no incremental schema migration in HNSW. If you change dimensions or distance metric, you rebuild the entire graph. This is fundamental — the graph structure depends on distances between vectors. Every edge encodes "these two vectors are close in this metric space." Change the metric or dimensionality, and those edges become meaningless or misleading.

I learned this the hard way when I normalized vectors during insert (cosine distance requires unit vectors) and accidentally skipped normalization on a batch. The index wasn't corrupt — it was just wrong. Every search returned garbage and I spent a day debugging before I noticed the unnormalized vectors.

### The Entry Point Matters

The entry point is the gate to your entire graph. If it gets isolated or its neighbors are poorly chosen, every search suffers. I've considered strategies like picking the centroid of your data as the entry point, or periodically reassigning it. So far, the random assignment from the first node hasn't been a problem in practice, but I can see it becoming one at scale with adversarial data distributions.

## Should You Build Your Own HNSW?

If you're doing research, absolutely. Building this from scratch taught me more about approximate nearest neighbor search than any paper could. I now understand why `ef_construction` and `ef_search` exist as separate parameters, why the level distribution matters, and why bidirectional connections aren't optional.

If you're shipping product, use FAISS or hnswlib. Their implementations are battle-tested, use SIMD-optimized distance calculations, have proper concurrent insert, and handle edge cases I haven't encountered yet. My Python implementation is ~650 lines of surprisingly readable code — but it's a learning tool, not a production database.

That said, if you do build one, you'll learn things you can't get from reading. The first time you see your search return the correct results, watch the latency drop from O(n) to O(log n), and realize you built a graph that actually navigates toward the right answer — that feeling is worth every bug.
