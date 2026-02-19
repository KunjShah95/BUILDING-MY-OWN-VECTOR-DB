# IVF (Inverted File Index) Algorithm: A Comprehensive Technical Guide

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)

**A deep dive into the IVF algorithm with production-ready implementation**

[GitHub Repository](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB) · [Documentation](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/blob/main/docs/ivf_vector_search_guide.md) · [Report Bug](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues) · [Request Feature](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues)

</div>

---

## Table of Contents

1. [Introduction](#introduction)
2. [The Fundamental Challenge](#the-fundamental-challenge)
3. [Clustering-Based Indexing: The Foundation](#1-clustering-based-indexing-the-foundation)
4. [The Algorithm in Detail: K-Means Clustering](#2-the-algorithm-in-detail-k-means-clustering)
5. [The Search Process: Understanding n_probes](#3-the-search-process-understanding-n_probes)
6. [Product Quantization: Reducing Memory Further](#4-product-quantization-reducing-memory-further)
7. [Performance Characteristics and Trade-offs](#5-performance-characteristics-and-trade-offs)
8. [IVF vs HNSW: A Comparison](#6-ivf-vs-hnsw-a-comparison)
9. [Practical Implementation: The API Layer](#7-practical-implementation-the-api-layer)
10. [Practical Tuning Guide](#8-practical-tuning-guide)
11. [Implementation Architecture](#9-implementation-architecture)
12. [Advanced Topics](#10-advanced-topics)
13. [Conclusion](#conclusion)
14. [Contributing](#contributing)
15. [License](#license)
16. [References](#references)

---

## Introduction

In the realm of vector databases and approximate nearest neighbor (ANN) search, the **Inverted File Index (IVF)** stands as one of the most foundational and widely deployed indexing techniques. As organizations increasingly rely on vector embeddings for semantic search, recommendation systems, and AI-powered applications, understanding how IVF works becomes essential for building efficient, scalable systems.

This comprehensive guide delves deep into the IVF algorithm, exploring its core mechanisms, implementation details, and practical considerations for deployment—specifically looking at how we've implemented it in our own vector database project.

> **Project Note**: This IVF implementation is part of our production-ready vector database built from scratch with Python, featuring HNSW and IVF indexing algorithms backed by PostgreSQL. Check out the [GitHub repository](#) for the complete implementation.

---

## The Fundamental Challenge

The fundamental challenge in vector search is that as datasets grow to millions or billions of vectors, the naive approach of comparing a query vector against every vector in the collection becomes computationally prohibitive. A brute-force search with 100 million 128-dimensional vectors would require 100 million distance computations per query—a requirement that simply cannot meet the latency expectations of modern applications.

### The Scale Problem

Let's break down the mathematics:

| Dataset Size | Vector Dimension | Brute Force Operations/Query |
|--------------|-----------------|------------------------------|
| 10,000       | 128             | 1.28 million                |
| 100,000      | 128             | 12.8 million                |
| 1,000,000    | 128             | 128 million                 |
| 10,000,000   | 128             | 1.28 billion                |
| 100,000,000  | 128             | 12.8 billion                |

At scale, this becomes economically and technically infeasible.

### The IVF Solution

IVF addresses this through an elegant partitioning strategy that dramatically reduces the search space by exploiting a key observation: **vectors that are similar to each other tend to cluster together in the high-dimensional space**.

By partitioning the vector space into clusters and only searching the most relevant ones, IVF can achieve:
- **10-100x speedup** over brute-force search
- **Predictable performance** that scales linearly with parameters
- **Memory efficiency** suitable for billion-scale datasets

---

## 1. Clustering-Based Indexing: The Foundation

### The Core Concept

Clustering-based indexing represents a paradigm shift from exhaustive search to intelligent partitioning. Rather than searching through all vectors, the approach partitions the vector space into distinct regions, each represented by a centroid. During query execution, the system identifies the most promising regions and conducts searches only within those boundaries.

This approach trades a small amount of accuracy for substantial performance gains—often achieving 10-100x speedups compared to brute-force search.

### The Two Primary Components

In our implementation, the `IVFIndex` consists of two primary components:

1. **Coarse Quantizer**: A set of cluster centroids that partition the vector space into distinct regions
2. **Inverted Lists (Posting Lists)**: Each centroid maintains a list of all vectors assigned to its cluster

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Vector Space                                     │
│                                                                          │
│        ○ ○ ○                      ○ ○ ○                                 │
│        ○ ● ○          ●          ○ ● ○       ● = Centroid              │
│        ○ ○ ○                      ○ ○ ○       ○ = Vectors               │
│                                                                          │
│   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐         │
│   │   Cluster 0   │     │   Cluster 1   │     │   Cluster 2   │         │
│   │  Voronoi Cell │     │  Voronoi Cell │     │  Voronoi Cell │         │
│   └───────────────┘     └───────────────┘     └───────────────┘         │
│                                                                          │
│   Inverted File Structure:                                               │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │ Cluster 0 → [vec_id_1, vec_id_5, vec_id_9, vec_id_12, ...]    │   │
│   │ Cluster 1 → [vec_id_2, vec_id_3, vec_id_7, vec_id_15, ...]    │   │
│   │ Cluster 2 → [vec_id_4, vec_id_6, vec_id_8, vec_id_11, ...]    │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Training the Index

When building an IVF index, the system begins with a training phase. This involves running a clustering algorithm—typically k-means—to identify where representative points (centroids) should be placed to minimize the average distance between vectors and their assigned centroids.

In our codebase, the `IVFIndex.train()` method orchestrates this process:

```python
# from utils/ivf_index.py
def train(self, vectors: List[List[float]]) -> 'IVFIndex':
    vectors_array = np.array(vectors, dtype=np.float32)
    
    print(f"Training coarse quantizer with {self.n_clusters} clusters...")
    self.coarse_quantizer.train(vectors_array)
    
    # Compute residuals for fine quantization
    cluster_ids = self.coarse_quantizer.encode_batch(vectors_array)
    residuals = []
    
    for vector, cluster_id in zip(vectors_array, cluster_ids):
        centroid = self.coarse_quantizer.decode(cluster_id)
        residual = vector - centroid
        residuals.append(residual)
    
    residuals_array = np.array(residuals, dtype=np.float32)
    
    print("Training fine quantizer...")
    self.fine_quantizer.train(residuals_array)
    
    self.is_trained = True
    print("Index training complete!")
    
    return self
```

### Voronoi Partitioning

The result of k-means clustering creates a **Voronoi decomposition** of the vector space. Each Voronoi cell (or Voronoi region) contains all points that are closer to a particular centroid than to any other centroid.

This geometric interpretation is powerful because it guarantees that for any query vector, the nearest vectors must lie within the Voronoi cells closest to the query. This property enables the dramatic search space reduction that makes IVF efficient.

**Key Property**: If a query vector q is closest to centroid c, then the nearest neighbors of q are most likely in the Voronoi cell of c (or its nearest neighbors).

---

## 2. The Algorithm in Detail: K-Means Clustering

### K-Means Clustering for Centroids

K-means clustering forms the backbone of IVF index construction. Our project implements a custom `KMeans` class in `utils/clustering.py` which follows the standard Lloyd's algorithm:

```python
# from utils/clustering.py
class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
    
    def fit(self, data: np.ndarray) -> 'KMeans':
        # Initialize centroids
        self.centroids = self._initialize_centroids(data)
        
        for iteration in range(self.max_iters):
            # 1. Assignment Step: Assign each point to nearest centroid
            labels = self._assign_clusters(data, self.centroids)
            
            # 2. Update Step: Reposition centroids to the mean
            new_centroids = self._update_centroids(data, labels)
            
            # Check for convergence
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                print(f"Converged after {iteration + 1} iterations")
                break
            
            self.centroids = new_centroids
        
        self.labels = labels
        return self
```

The algorithm proceeds through an iterative refinement process that alternates between two steps:

1. **Assignment Step**: Each vector is assigned to the cluster whose centroid is closest to it
2. **Update Step**: Each centroid is repositioned to be the mean of all vectors assigned to its cluster

### The Mathematical Objective

K-means minimizes the **Within-Cluster Sum of Squares (WCSS)**:

$$\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - c_i\|^2$$

where:
- $k$ is the number of clusters
- $C_i$ is the set of vectors in cluster $i$
- $c_i$ is the centroid of cluster $i$
- $\|x - c_i\|^2$ is the squared Euclidean distance

### Implementation Details

Let's look at the helper methods in our implementation:

```python
# from utils/clustering.py
def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
    """Initialize centroids using the Forgy method (random data points)"""
    n_samples, n_features = data.shape
    centroids = np.zeros((self.k, n_features))
    
    # Randomly select k data points as initial centroids
    indices = np.random.choice(n_samples, self.k, replace=False)
    for i, idx in enumerate(indices):
        centroids[i] = data[idx]
    
    return centroids

def _assign_clusters(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to the nearest centroid"""
    labels = []
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        labels.append(np.argmin(distances))
    return np.array(labels)

def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Update centroids based on current cluster assignments"""
    centroids = np.zeros((self.k, data.shape[1]))
    for i in range(self.k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
        else:
            # Keep old centroid for empty clusters
            centroids[i] = self.centroids[i]
    return centroids
```

### Choosing the Number of Clusters

The number of clusters (often called `n_clusters` or `nlist`) is the most critical hyperparameter in IVF index construction.

**Rule of Thumb**: Set `nlist ≈ √N`, where N is the total number of vectors in the dataset.

| Dataset Size | Recommended Clusters | Use Case |
|--------------|---------------------|----------|
| 1,000        | ~32                | Testing  |
| 10,000       | ~100               | Small    |
| 100,000      | ~316               | Medium   |
| 1,000,000    | ~1,000             | Large    |
| 10,000,000   | ~3,162             | XL Scale |
| 100,000,000  | ~10,000            | Billion  |

**The Trade-off**:
- **More clusters**: Smaller inverted lists → faster search within a cluster, but more overhead computing distances to centroids
- **Fewer clusters**: Larger inverted lists → more vectors to scan per probed cluster, but more stable centroids

In practice, values between 100 and 10,000 clusters are common, depending on dataset size, vector dimension, and accuracy requirements.

---

## 3. The Search Process: Understanding n_probes

### How Search Works

The `n_probes` parameter controls how many clusters are searched during query execution. When a query arrives:

1. Compute distances from the query vector to all centroids
2. Select the top `n_probes` closest clusters
3. Search only within the inverted lists of those clusters
4. Return the top-k results

In our `IVFIndex.search()` implementation:

```python
# from utils/ivf_index.py
def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
    query_array = np.array(query_vector, dtype=np.float32)
    
    # Phase 1: Find closest clusters to query
    cluster_distances = []
    for cluster_id in range(self.n_clusters):
        centroid = self.coarse_quantizer.decode(cluster_id)
        distance = np.linalg.norm(query_array - centroid)
        cluster_distances.append((cluster_id, distance))
    
    # Phase 2: Sort and select top n_probes
    cluster_distances.sort(key=lambda x: x[1])
    probe_clusters = [c[0] for c in cluster_distances[:self.n_probes]]
    
    # Phase 3: Search in selected clusters
    candidate_results = []
    for cluster_id in probe_clusters:
        for vector_info in self.inverted_file[cluster_id]:
            vector_id = vector_info["vector_id"]
            
            # Compute distance using fine quantizer
            residual_codes = self.residuals[vector_id]
            residual = self.fine_quantizer.decode(residual_codes)
            centroid = self.coarse_quantizer.decode(cluster_id)
            
            # Asymmetric distance computation
            distance = np.linalg.norm(query_array - centroid - residual)
            
            candidate_results.append({
                "vector_id": vector_id,
                "distance": float(distance),
                "metadata": self.metadata.get(vector_id)
            })
    
    # Phase 4: Sort and return top k
    candidate_results.sort(key=lambda x: x["distance"])
    return candidate_results[:k]
```

### The Complete Search Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Query Search Pipeline                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. COARSE QUANTIZATION                                              │
│     ┌─────────────┐      ┌──────────────┐                            │
│     │ Query Vector │ ───→ │ Distance to │                            │
│     │    [128D]    │      │  Centroids  │                            │
│     └─────────────┘      └──────────────┘                            │
│                                   ↓                                   │
│                         ┌─────────────────┐                            │
│                         │ Sort by Distance│                           │
│                         │ Select Top N    │                           │
│                         └─────────────────┘                            │
│                                   ↓                                   │
│  2. INVERTED LIST SCANNING                                           │
│     ┌──────────────┐      ┌────────────────┐                        │
│     │ Probed Clusters│ ──→ │ Scan Inverted │                         │
│     │ [n_probes]    │      │ Lists          │                         │
│     └──────────────┘      └────────────────┘                            │
│                                   ↓                                   │
│                         ┌─────────────────┐                            │
│                         │ Compute Distances│                          │
│                         │ to Candidates    │                           │
│                         └─────────────────┘                            │
│                                   ↓                                   │
│  3. RANKING & RETURN                                                  │
│     ┌─────────────────┐      ┌────────────┐                          │
│     │ Sort by Distance│ ───→ │ Return Top │                          │
│     │                 │      │ K Results  │                          │
│     └─────────────────┘      └────────────┘                          │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Tuning n_probes

Increasing `n_probes` improves recall but increases latency:

| n_probes | Expected Recall | Latency Multiplier | When to Use |
|----------|-----------------|-------------------|-------------|
| 1        | ~60-70%        | 1x                | Fastest, lowest recall |
| 5        | ~80-90%        | 5x                | Speed priority |
| 10       | ~90-95%        | 10x               | Balanced |
| 20       | ~95-98%        | 20x               | High recall |
| 50       | ~98-99%        | 50x               | Near-perfect recall |

**Common Starting Point**: Set `n_probes ≈ nlist / 10` (1-5% of total clusters), then empirically test and adjust.

### Reranking for Higher Precision

For even higher accuracy, our implementation supports reranking using exact distances:

```python
# from utils/ivf_index.py
def search_with_rerank(self, query_vector: List[float], k: int = 5, 
                      initial_candidates: int = 50) -> List[Dict[str, Any]]:
    # Get more candidates than needed
    candidates = self.search(query_vector, k=initial_candidates)
    
    query_array = np.array(query_vector, dtype=np.float32)
    
    # Rerank using exact distance
    for candidate in candidates:
        vector_id = candidate["vector_id"]
        vector = self.vectors[vector_id]
        candidate["distance"] = float(np.linalg.norm(query_array - vector))
    
    # Sort by exact distance
    candidates.sort(key=lambda x: x["distance"])
    return candidates[:k]
```

---

## 4. Product Quantization: Reducing Memory Further

### What is Product Quantization?

Product Quantization (PQ) is a technique that compresses vectors by splitting them into sub-vectors and independently quantizing each sub-vector. This is especially powerful in IVF because we can quantize the **residuals** (the difference between vectors and their cluster centroids).

Our implementation includes PQ through the `FineQuantizer` class:

```python
# from utils/ivf_index.py
class FineQuantizer:
    def __init__(self, n_subquantizers: int = 8, bits_per_subquantizer: int = 8):
        self.n_subquantizers = n_subquantizers
        self.bits_per_subquantizer = bits_per_subquantizer
        self.codebooks = None
        self.n_codes = 2 ** bits_per_subquantizer  # 256 codes per sub-quantizer
    
    def train(self, residuals: np.ndarray) -> 'FineQuantizer':
        """Train on residual vectors (vector - centroid)"""
        n_samples, n_features = residuals.shape
        dim_per_subquantizer = n_features // self.n_subquantizers
        
        for i in range(self.n_subquantizers):
            sub_vectors = residuals[:, i*dim_per_subquantizer:(i+1)*dim_per_subquantizer]
            codebook = self._create_codebook(sub_vectors)  # K-means on sub-vectors
            self.codebooks.append(codebook)
        
        return self
    
    def _create_codebook(self, vectors: np.ndarray) -> np.ndarray:
        """Create codebook using K-means"""
        kmeans = KMeans(k=self.n_codes, max_iters=20)
        kmeans.fit(vectors)
        return kmeans.get_cluster_centers()
```

### How PQ Works

```
Original Vector: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  (8 dimensions)

Split into Sub-vectors:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ [0.1, 0.2] │ [0.3, 0.4] │ [0.5, 0.6] │ [0.7, 0.8] │
│  SubVec 0   │  SubVec 1   │  SubVec 2   │  SubVec 3   │
└─────────────┴─────────────┴─────────────┴─────────────┘
       ↓              ↓              ↓              ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Quantize    │ Quantize    │ Quantize    │ Quantize    │
│ to 256 codes│ to 256 codes│ to 256 codes│ to 256 codes│
└─────────────┴─────────────┴─────────────┴─────────────┘
       ↓              ↓              ↓              ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Code: 42    │ Code: 128   │ Code: 200   │ Code: 15    │
└─────────────┴─────────────┴─────────────┴─────────────┘

Storage: 4 bytes (instead of 8 × 4 = 32 bytes) → 8x compression!
```

### Compression Ratios

| Bits per Code | Codes per Subvector | Compression Ratio |
|---------------|-------------------|-------------------|
| 4             | 16                | 32x              |
| 8             | 256               | 16x              |
| 12            | 4096              | ~10.6x           |
| 16            | 65536             | 8x               |

---

## 5. Performance Characteristics and Trade-offs

### Memory Efficiency

Unlike graph-based methods like HNSW, IVF offers excellent memory efficiency:

| Index Type | Memory Usage | Notes |
|------------|-------------|-------|
| IVF-Flat   | ~N × D × 4 bytes | Raw data + centroids |
| IVF-PQ (8-bit) | ~N × D × 0.5 bytes | 8x compression |
| IVF-PQ (4-bit) | ~N × D × 0.25 bytes | 16x compression |
| HNSW       | ~N × D × 4 bytes + graph | 2-3x overhead |

By using Product Quantization (PQ) on the residuals, we can compress vectors significantly—making it possible to search billion-scale datasets on limited hardware.

### Query Latency Analysis

IVF scaling is predictable:

$$\text{Latency} = \underbrace{O(nlist \times D)}_{\text{Distance to centroids}} + \underbrace{O(n\_probes \times \frac{N}{nlist} \times D)}_{\text{Scanning inverted lists}}$$

**Breakdown**:
- **First term**: Distance to all centroids (usually small, ~0.1ms)
- **Second term**: Scanning inverted lists of probed clusters (main cost)

### Throughput Considerations

For batch query scenarios, IVF demonstrates excellent throughput because:
- Inverted list scans can be efficiently parallelized
- Memory access patterns are sequential (cache-friendly)
- SIMD instructions can accelerate distance calculations

### Build Time

Building an IVF index requires running k-means clustering:

$$\text{Build Time} = O(N \times nlist \times iterations)$$

| Dataset Size | Clusters | Estimated Build Time |
|--------------|----------|---------------------|
| 10,000       | 100      | ~30 seconds         |
| 100,000      | 316      | ~5 minutes          |
| 1,000,000    | 1,000    | ~15 minutes         |
| 10,000,000   | 3,162    | ~1 hour             |

---

## 6. IVF vs HNSW: A Comparison

| Aspect                | IVF                              | HNSW                      |
|-----------------------|----------------------------------|---------------------------|
| **Single-query latency** | ~1-10ms                       | ~0.1-1ms                 |
| **Memory overhead**   | Low                              | High (graph structure)   |
| **Scalability**       | Excellent (billion-scale)        | Limited by memory        |
| **Build time**        | Moderate                         | Moderate                 |
| **Recall tuning**    | n_probes                         | ef_search                |
| **Incremental update**| Difficult                        | Difficult                |
| **Batch throughput**  | Excellent                        | Good                     |
| **Implementation**    | Simple                           | Complex                  |

### When to Use Each

**Use IVF when**:
- Dataset is very large (millions to billions of vectors)
- Memory is constrained
- Predictable, consistent performance is required
- Need to scale to billion-scale with IVF-PQ
- Batch query throughput is important

**Use HNSW when**:
- Latency is critical (sub-millisecond)
- Dataset fits comfortably in memory
- Maximum recall is needed
- In-memory workloads
- Simpler query patterns

---

## 7. Practical Implementation: The API Layer

In our project, the IVF functionality is exposed through a production-ready FastAPI interface:

### Creating an IVF Index

```http
POST /index
{
    "method": "ivf",
    "n_clusters": 128,
    "n_probes": 10
}
```

**Response**:
```json
{
    "success": true,
    "message": "IVF Index created with 128 clusters",
    "stats": {
        "total_vectors": 50000,
        "n_clusters": 128,
        "n_probes": 10,
        "avg_cluster_size": 390.6,
        "is_trained": true
    }
}
```

### Searching with IVF

```http
POST /search
{
    "query_vector": [0.1, 0.2, 0.3, ...],
    "k": 5,
    "method": "ivf",
    "n_probes": 20,
    "use_rerank": true
}
```

### Complete Usage Example

```python
from utils.ivf_index import IVFIndex
import numpy as np

# Create and train index
index = IVFIndex(n_clusters=100, n_probes=10)

# Training data (representative sample)
training_vectors = np.random.randn(10000, 128).tolist()
index.train(training_vectors)

# Add vectors to index
for i in range(50000):
    vector = np.random.randn(128).tolist()
    index.add(vector, vector_id=f"vec_{i}", metadata={"category": "test"})

# Search
query = np.random.randn(128).tolist()
results = index.search(query, k=5)

print(f"Found {len(results)} results")
for r print(f"  in results:
    ID: {r['vector_id']}, Distance: {r['distance']:.4f}")

# Or with reranking for higher accuracy
results_reranked = index.search_with_rerank(query, k=5, initial_candidates=50)

# Save index for later use
index.save("my_ivf_index.json")
```

### Database Integration

```python
from database.ivf_database import IVFVectorDatabase

# Create index from database
db = IVFVectorDatabase(session)
result = db.create_ivf_index(n_clusters=200, n_probes=15)

# Search with automatic reranking
results = db.search(
    query_vector=[0.1] * 128,
    k=10,
    use_ivf=True,
    use_rerank=True,
    n_probes=15
)

# Compare search methods
comparison = db.compare_search_methods(query_vector, k=10)
for method, data in comparison["methods"].items():
    print(f"{method}: {data['time']*1000:.2f}ms")
```

The `VectorService` in our backend handles the logic of switching between brute force, HNSW, and IVF depending on the user's request.

---

## 8. Practical Tuning Guide

### Step 1: Choose n_clusters

```python
import numpy as np

# Start with √N heuristic
n_clusters = int(np.sqrt(num_vectors))

# Adjust based on accuracy requirements
# Higher accuracy needed → more clusters
n_clusters = int(n_clusters * 1.5)  # For 95%+ recall
```

**Recommendations by scale**:

| Scale | n_clusters | Notes |
|-------|------------|-------|
| 10K vectors | 50-100 | Small dataset |
| 100K vectors | 200-500 | Medium dataset |
| 1M vectors | 500-1000 | Large dataset |
| 10M+ vectors | 1000-5000 | Very large |

### Step 2: Set initial n_probes

```python
# Start with 1-5% of clusters
n_probes = max(1, n_clusters // 20)
```

### Step 3: Tune for your workload

```python
# Test with representative queries
test_queries = load_test_queries()

best_n_probes = 1
best_score = 0

for n_probes in [1, 5, 10, 20, 30, 50]:
    index.n_probes = n_probes
    
    recall = measure_recall(index, test_queries, ground_truth)
    latency = measure_p99_latency(index, test_queries)
    
    # F1 score combining recall and latency
    score = recall / (latency ** 0.3)  # Tunable weighting
    
    print(f"n_probes={n_probes}: recall={recall:.2%}, p99={latency:.1f}ms, score={score:.2f}")
    
    if score > best_score:
        best_score = score
        best_n_probes = n_probes
    
    if latency > target_latency:
        break

print(f"Optimal n_probes: {best_n_probes}")
```

### Step 4: Consider Product Quantization for Large Datasets

```python
# For billion-scale datasets, use IVF-PQ
index = IVFIndex(n_clusters=1000, n_probes=50)

# Fine quantizer with PQ compresses residuals
# from utils/ivf_index.py
fine_quantizer = FineQuantizer(
    n_subquantizers=8,      # Split into 8 sub-vectors
    bits_per_subquantizer=8 # 256 codes each (8 bits)
)
```

---

## 9. Implementation Architecture

Here's how the components fit together in our implementation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         IVFIndex                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐      ┌──────────────────────────────────┐   │
│  │  CoarseQuantizer   │      │       Inverted File              │   │
│  ├─────────────────────┤      ├──────────────────────────────────┤   │
│  │ n_clusters: 100    │      │ Cluster 0 → [vec_1, vec_5, ...] │   │
│  │ centroids: (100,D) │      │ Cluster 1 → [vec_2, vec_3, ...] │   │
│  │                    │      │ ...                              │   │
│  │ train(vectors)     │      │                                  │   │
│  │ encode(vector)     │      │                                  │   │
│  └─────────────────────┘      └──────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────┐      ┌──────────────────────────────────┐   │
│  │  FineQuantizer     │      │       Storage                    │   │
│  ├─────────────────────┤      ├──────────────────────────────────┤   │
│  │ codebooks: list    │      │ vectors: {id → array}           │   │
│  │ n_subquantizers: 8 │      │ residuals: {id → codes}         │   │
│  │ bits: 8            │      │ metadata: {id → dict}            │   │
│  └─────────────────────┘      └──────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Full Project Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Vector Database System                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Layer                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │   │
│  │  │ Vector   │  │  Search  │  │  Index   │  │  Health  │  │   │
│  │  │ CRUD     │  │  Endpoints│ │  Management│ │  Checks  │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│  ┌───────────────────────────▼────────────────────────────────┐   │
│  │                   Service Layer                            │   │
│  │  ┌──────────────────┐    ┌────────────────────────────┐   │   │
│  │  │  VectorService   │    │    VectorIndexer           │   │   │
│  │  │  (CRUD ops)      │    │    (Index management)      │   │   │
│  │  └──────────────────┘    └────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│  ┌───────────────────────────▼────────────────────────────────┐   │
│  │                 Indexing Algorithms                         │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │   │
│  │  │ HNSW Index    │  │ IVF Index      │  │  Distance    │ │   │
│  │  │ (hnsw_index) │  │ (ivf_index.py) │  │  Calculations│ │   │
│  │  └────────────────┘  └────────────────┘  └───────────────┘ │   │
│  │  ┌────────────────┐  ┌────────────────┐                    │   │
│  │  │ K-Means       │  │ Optimization   │                    │   │
│  │  │ (clustering)  │  │ (optimization) │                    │   │
│  │  └────────────────┘  └────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│  ┌───────────────────────────▼────────────────────────────────┐   │
│  │                   Database Layer                           │   │
│  │  ┌────────────────┐  ┌────────────────┐                    │   │
│  │  │ PostgreSQL     │  │  SQLAlchemy    │                    │   │
│  │  │ (Persistence)  │  │  (ORM)          │                    │   │
│  │  └────────────────┘  └────────────────┘                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Advanced Topics

### A. Multi-Probe Optimization

Instead of searching a fixed number of clusters, adaptive multi-probe dynamically adjusts based on query characteristics:

```python
def adaptive_search(self, query_vector, k, min_recall=0.95):
    """Adaptively determine n_probes based on cluster density"""
    # Start with closest cluster
    n_probes = 1
    best_clusters = [self.find_closest_cluster(query_vector)]
    
    while n_probes < self.n_clusters:
        # Estimate recall based on cluster density
        estimated_recall = self.estimate_recall(best_clusters, k)
        
        if estimated_recall >= min_recall:
            break
        
        # Add next closest cluster
        n_probes += 1
        best_clusters = self.get_closest_clusters(query_vector, n_probes)
    
    return self.search_in_clusters(query_vector, best_clusters, k)
```

### B. Cluster Balancing

Poor cluster balance can degrade performance. Techniques include:

1. **K-Means++ Initialization**: Better initial centroid selection
2. **Ball K-Means**: More balanced cluster sizes
3. **Dynamic Re-partitioning**: Periodically rebalance clusters

```python
def rebalance_clusters(self, vectors):
    """Rebalance clusters for better distribution"""
    # Move vectors from overloaded to underloaded clusters
    for cluster_id, vectors_in_cluster in self.inverted_file.items():
        if len(vectors_in_cluster) > self.max_cluster_size:
            # Move some vectors to nearest underloaded cluster
            self._redistribute_vectors(cluster_id, vectors_in_cluster)
```

### C. Hierarchical IVF

Multiple levels of clustering can provide better recall:

```
Level 0: 10 coarse clusters (nlist=10)
    ↓ Each coarse cluster contains
Level 1: 100 fine clusters per coarse (nlist=1000 total)

Search: First find best coarse cluster, then search within its fine clusters
```

### D. Disk-Based IVF for Massive Datasets

For datasets exceeding RAM:

```python
class DiskIVFIndex:
    def __init__(self, n_clusters, cache_size=1000):
        self.cache = LRUCache(cache_size)  # Keep hot clusters in memory
        self.disk_index = mmapio.open('clusters/')  # Memory-mapped files
    
    def search(self, query, k, n_probes):
        # Load relevant clusters from disk
        clusters = self._load_clusters(query, n_probes)
        
        # Search in memory
        results = self._search_clusters(clusters, query, k)
        
        # Update cache
        for cluster in clusters:
            self.cache.put(cluster.id, cluster)
        
        return results
```

---

## Conclusion

The Inverted File Index (IVF) is a powerhouse in the vector search ecosystem. Its strength lies in:

- **Conceptual simplicity**: Easy to understand, implement, and debug
- **Predictable performance**: Latency scales linearly with n_probes
- **Excellent scalability**: Combined with Product Quantization, scales to billion-scale datasets
- **Memory efficiency**: No graph overhead, supports compression
- **Versatility**: Works well with various distance metrics and data distributions

For engineers building vector search systems today, mastering the basics of IVF is essential. Whether you're building a vector database from scratch or optimizing an existing system, understanding the parameters that control IVF—particularly `n_clusters` and `n_probes`—is crucial for effectively deploying this technology.

The trade-off between search speed and recall can be tuned to meet specific application requirements, whether those requirements prioritize ultra-low latency, maximum recall, or optimal resource utilization.

---

## Contributing

We welcome contributions! This project is open source and we appreciate the community's involvement.

### Ways to Contribute

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: We want to hear your ideas for making this better
3. **Improve Documentation**: Help us make the guides clearer and more comprehensive
4. **Submit Pull Requests**: Fix bugs, add features, or improve performance

### Development Setup

```bash
# Clone the repository
git clone https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB.git
cd vector-database

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest test/ -v

# Run the API
python -m uvicorn api.main:app --reload
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public methods
- Add unit tests for new features

### Feature Requests

We're especially interested in:
- GPU acceleration support
- Additional distance metrics
- Integration with more databases
- Distributed/clustered deployment support
- WebSocket streaming search

> **Note**: This project was built as a comprehensive learning project demonstrating advanced vector indexing, database design, and performance optimization techniques. It's production-ready but we continue to improve it!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

1. Jégou, H., Douze, M., & Schmid, C. (2010). "Product quantization for nearest neighbor search"
2. Babenko, A., & Lempitsky, V. (2012). "The Inverted Multi-Index". CVPR 2012
3. Malkov, Y., & Yashunin, D. (2018). "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
4. FAISS: https://github.com/facebookresearch/faiss
5. pgvector: https://github.com/pgvector/pgvector
6. Milvus Vector Database: https://milvus.io/
7. Weaviate: https://github.com/weaviate/weaviate
8. Qdrant: https://github.com/qdrant/qdrant

---

## Additional Resources

For the full implementation and more examples, check out:

- `utils/ivf_index.py` - Core IVF index implementation
- `utils/clustering.py` - K-means clustering
- `database/ivf_database.py` - Database integration
- `test/test_ivf.py` - Unit tests and usage examples
- `examples/indexer_examples.py` - Comprehensive usage patterns
- `HNSW_OPTIMIZATION_GUIDE.md` - HNSW implementation guide
- `readme.md` - Full project documentation

---

<div align="center">

**Built with ❤️ as a comprehensive learning project**

[GitHub](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB) · [Report Bug](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues) · [Request Feature](https://github.com/KunjShah95/BUILDING-MY-OWN-VECTOR-DB/issues)

</div>
