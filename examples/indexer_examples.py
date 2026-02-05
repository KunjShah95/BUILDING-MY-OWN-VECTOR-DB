"""
Vector Indexer Integration Examples
Demonstrates usage of the unified VectorIndexer with HNSW and IVF
"""

from services.vector_indexer import VectorIndexer, VectorIndexerConfig, IndexMethod
from utils.optimization import QueryOptimizer, MemoryOptimizer, OptimizedDistanceCalculator
import numpy as np
from typing import List, Dict, Any


# ============================================================================
# Example 1: Basic HNSW Indexing
# ============================================================================

def example_hnsw_basic():
    """Basic HNSW indexing with default optimized parameters"""
    
    # Create sample vectors
    vectors = np.random.randn(10000, 128).tolist()
    queries = np.random.randn(100, 128).tolist()
    
    # Initialize indexer with HNSW
    config = VectorIndexerConfig(
        method=IndexMethod.HNSW,
        num_vectors=10000,
        vector_dim=128,
        recall_target=0.95
    )
    
    indexer = VectorIndexer(config)
    
    # Create index
    create_result = indexer.create_index(vectors)
    print(f"Index created: {create_result['success']}")
    print(f"Time: {create_result['creation_time']:.2f}s")
    print(f"HNSW Parameters: {create_result['config']['hnsw_params']}")
    
    # Search
    result = indexer.search(queries[0], k=10)
    print(f"\nSearch successful: {result['success']}")
    print(f"Results found: {result['count']}")
    print(f"Search time: {result['search_time']*1000:.2f}ms")


# ============================================================================
# Example 2: IVF Indexing with Speed Optimization
# ============================================================================

def example_ivf_speed_optimized():
    """IVF indexing optimized for speed"""
    
    vectors = np.random.randn(50000, 256).tolist()
    queries = np.random.randn(50, 256).tolist()
    
    # Speed-optimized configuration
    config = VectorIndexerConfig(
        method=IndexMethod.IVF,
        num_vectors=50000,
        vector_dim=256,
        speed_priority=True  # Optimize for speed
    )
    
    indexer = VectorIndexer(config)
    
    print("Creating IVF index with speed optimization...")
    result = indexer.create_index(vectors)
    print(f"IVF Parameters: {result['config']['ivf_params']}")
    
    # Batch search
    batch_result = indexer.batch_search(queries, k=5)
    print(f"\nBatch search:")
    print(f"  Total queries: {batch_result['total_queries']}")
    print(f"  Throughput: {batch_result['queries_per_second']:.2f} QPS")
    print(f"  Avg query time: {batch_result['avg_query_time']*1000:.2f}ms")


# ============================================================================
# Example 3: Hybrid Search (HNSW + IVF)
# ============================================================================

def example_hybrid_search():
    """Hybrid search combining HNSW and IVF"""
    
    vectors = np.random.randn(20000, 128).tolist()
    queries = np.random.randn(10, 128).tolist()
    
    # Create hybrid index
    config = VectorIndexerConfig(
        method=IndexMethod.HYBRID,
        num_vectors=20000,
        recall_target=0.95
    )
    
    indexer = VectorIndexer(config)
    
    print("Creating hybrid index (HNSW + IVF)...")
    result = indexer.create_index(vectors)
    print(f"HNSW created: {result['hnsw']['success']}")
    print(f"IVF created: {result['ivf']['success']}")
    
    # Hybrid search
    search_result = indexer.search(queries[0], k=10, method="hybrid")
    print(f"\nHybrid search:")
    print(f"  HNSW results: {search_result['hnsw_count']}")
    print(f"  IVF results: {search_result['ivf_count']}")
    print(f"  Combined results: {search_result['combined_count']}")
    print(f"  Final results: {search_result['count']}")


# ============================================================================
# Example 4: Auto Parameter Optimization
# ============================================================================

def example_auto_parameters():
    """Use QueryOptimizer to get suggested parameters"""
    
    database_sizes = [1000, 10000, 100000, 1000000]
    
    for size in database_sizes:
        suggestions = QueryOptimizer.suggest_index_parameters(
            database_size=size,
            recall_target=0.95
        )
        
        print(f"\nDatabase size: {size:,} vectors")
        print(f"  HNSW: m={suggestions['hnsw']['m']}, "
              f"ef_c={suggestions['hnsw']['ef_construction']}, "
              f"ef_s={suggestions['hnsw']['ef_search']}")
        print(f"  IVF: n_clusters={suggestions['ivf']['n_clusters']}, "
              f"n_probes={suggestions['ivf']['n_probes']}")


# ============================================================================
# Example 5: Memory Optimization
# ============================================================================

def example_memory_optimization():
    """Optimize memory usage with quantization"""
    
    vectors = np.random.randn(100000, 256).astype(np.float32)
    
    # Estimate memory usage
    print("Original vectors (float32):")
    mem_info = MemoryOptimizer.estimate_memory_usage(
        num_vectors=100000,
        dimension=256,
        dtype=np.float32
    )
    print(f"  Memory: {mem_info['mb']:.2f} MB")
    
    # Quantize to float16
    quantized = MemoryOptimizer.normalize_for_half_precision(vectors)
    print("\nQuantized vectors (float16):")
    mem_info = MemoryOptimizer.estimate_memory_usage(
        num_vectors=100000,
        dimension=256,
        dtype=np.float16
    )
    print(f"  Memory: {mem_info['mb']:.2f} MB")
    
    ratio = MemoryOptimizer.calculate_compression_ratio(vectors, quantized)
    print(f"  Compression ratio: {ratio:.2f}x")


# ============================================================================
# Example 6: Batch Processing with Optimization
# ============================================================================

def example_batch_processing():
    """Batch processing with performance metrics"""
    
    vectors = np.random.randn(30000, 128).tolist()
    queries = np.random.randn(200, 128).tolist()
    
    config = VectorIndexerConfig(
        method=IndexMethod.HNSW,
        num_vectors=30000
    )
    
    indexer = VectorIndexer(config)
    indexer.create_index(vectors)
    
    # Batch search
    print("Processing 200 queries...")
    result = indexer.batch_search(queries, k=10)
    
    print(f"Batch Results:")
    print(f"  Total time: {result['batch_time']:.2f}s")
    print(f"  Throughput: {result['queries_per_second']:.2f} QPS")
    print(f"  Avg query: {result['avg_query_time']*1000:.2f}ms")
    
    # Show indexer statistics
    stats = indexer.get_stats()
    print(f"\nIndexer Statistics:")
    print(f"  Total searches: {stats['statistics']['total_searched']}")
    print(f"  Avg search time: {stats['statistics']['avg_search_time']*1000:.2f}ms")


# ============================================================================
# Example 7: Save and Load Index
# ============================================================================

def example_persistence():
    """Save and load indexes"""
    
    vectors = np.random.randn(5000, 128).tolist()
    
    # Create and save
    config = VectorIndexerConfig(
        method=IndexMethod.HNSW,
        num_vectors=5000
    )
    
    indexer = VectorIndexer(config)
    indexer.create_index(vectors)
    
    # Save index
    save_result = indexer.save_index(path="./my_index")
    print(f"Index saved: {save_result['success']}")
    print(f"Path: {save_result['index_path']}")
    
    # Load index
    new_indexer = VectorIndexer()
    load_result = new_indexer.load_index(save_result['index_path'])
    print(f"\nIndex loaded: {load_result['success']}")
    print(f"Vectors: {load_result['vector_count']}")


# ============================================================================
# Example 8: Optimized Distance Calculations
# ============================================================================

def example_optimized_distances():
    """Use optimized distance calculations"""
    
    vectors = np.random.randn(10000, 256)
    queries = np.random.randn(100, 256)
    
    print("Computing distances with Numba JIT optimization...")
    
    for query in queries[:5]:
        # Batch distance calculation (parallelized)
        distances = OptimizedDistanceCalculator.cosine_distance_batch(vectors, query)
        print(f"Query shape: {query.shape}")
        print(f"Distances shape: {distances.shape}")
        
        # Find top k
        top_k = OptimizedDistanceCalculator.top_k_indices(distances, k=10)
        print(f"Top 10 indices: {top_k}")
        print()


# ============================================================================
# Example 9: Complete Workflow
# ============================================================================

def example_complete_workflow():
    """Complete workflow from data preparation to search"""
    
    print("=" * 70)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("=" * 70)
    
    # Step 1: Generate data
    print("\n1. Generating test data...")
    num_vectors = 50000
    dimension = 256
    vectors = np.random.randn(num_vectors, dimension).tolist()
    queries = np.random.randn(100, dimension).tolist()
    print(f"   Generated {num_vectors} vectors, dimension={dimension}")
    
    # Step 2: Configure indexer
    print("\n2. Configuring indexer...")
    config = VectorIndexerConfig(
        method=IndexMethod.HNSW,
        num_vectors=num_vectors,
        vector_dim=dimension,
        recall_target=0.95
    )
    print(f"   Method: {config.method.value}")
    print(f"   HNSW params: m={config.hnsw_params['m']}, "
          f"ef_c={config.hnsw_params['ef_construction']}")
    
    # Step 3: Create index
    print("\n3. Creating index...")
    indexer = VectorIndexer(config)
    result = indexer.create_index(vectors)
    print(f"   Time: {result['creation_time']:.2f}s")
    print(f"   Success: {result['success']}")
    
    # Step 4: Batch search
    print("\n4. Running batch search (100 queries)...")
    batch_result = indexer.batch_search(queries, k=10)
    print(f"   Throughput: {batch_result['queries_per_second']:.2f} QPS")
    print(f"   Total time: {batch_result['batch_time']:.2f}s")
    print(f"   Avg latency: {batch_result['avg_query_time']*1000:.2f}ms")
    
    # Step 5: Get statistics
    print("\n5. Index statistics...")
    stats = indexer.get_stats()
    print(f"   Vectors indexed: {stats['vector_count']}")
    print(f"   Total searches: {stats['statistics']['total_searched']}")
    print(f"   Avg search time: {stats['statistics']['avg_search_time']*1000:.2f}ms")
    
    # Step 6: Save index
    print("\n6. Saving index...")
    save_result = indexer.save_index(path="./complete_index")
    print(f"   Saved to: {save_result['index_path']}")
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        "1": ("Basic HNSW", example_hnsw_basic),
        "2": ("IVF Speed Optimized", example_ivf_speed_optimized),
        "3": ("Hybrid Search", example_hybrid_search),
        "4": ("Auto Parameters", example_auto_parameters),
        "5": ("Memory Optimization", example_memory_optimization),
        "6": ("Batch Processing", example_batch_processing),
        "7": ("Persistence", example_persistence),
        "8": ("Optimized Distances", example_optimized_distances),
        "9": ("Complete Workflow", example_complete_workflow),
        "all": ("All Examples", None),
    }
    
    print("\nVector Indexer Examples")
    print("=" * 50)
    for key, (name, _) in examples.items():
        print(f"{key}: {name}")
    print()
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Select example (1-9 or 'all'): ").strip()
    
    if choice == "all":
        for key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            print(f"\n{'='*70}")
            print(f"Running Example {key}: {examples[key][0]}")
            print(f"{'='*70}\n")
            try:
                examples[key][1]()
            except Exception as e:
                print(f"Error: {e}")
    elif choice in examples and choice != "all":
        print(f"\nRunning: {examples[choice][0]}\n")
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Invalid choice")
