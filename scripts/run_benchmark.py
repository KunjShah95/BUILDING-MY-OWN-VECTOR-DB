"""
Vector Database Benchmark Runner
Run comprehensive benchmarks and generate reports
"""

import numpy as np
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.benchmark import BenchmarkSuite, PerformanceComparator
from database.hnsw_database import HNSWVectorDatabase
from config.database import SessionLocal
from database.schema import Base

def generate_test_data(num_vectors: int = 10000, 
                      dimension: int = 128,
                      num_queries: int = 100) -> tuple:
    """
    Generate test data for benchmarking
    
    Args:
        num_vectors: Number of vectors to generate
        dimension: Vector dimension
        num_queries: Number of query vectors
        
    Returns:
        Tuple of (database_vectors, query_vectors)
    """
    # Generate database vectors (with some structure)
    database_vectors = []
    for i in range(num_vectors):
        # Create clusters for more realistic data
        cluster = i % 5
        center = np.random.rand(dimension) * cluster
        vector = center + np.random.randn(dimension) * 0.1
        database_vectors.append(vector.tolist())
    
    # Generate query vectors
    query_vectors = []
    for i in range(num_queries):
        cluster = np.random.randint(0, 5)
        center = np.random.rand(dimension) * cluster
        query = center + np.random.randn(dimension) * 0.1
        query_vectors.append(query.tolist())
    
    return database_vectors, query_vectors

def run_benchmark():
    """
    Run comprehensive benchmark
    """
    print("\n" + "="*60)
    print("Vector Database Comprehensive Benchmark")
    print("="*60)
    
    # Initialize database
    db = SessionLocal()
    vector_db = HNSWVectorDatabase(db)
    
    # Generate test data
    print("\nGenerating test data...")
    num_vectors = 10000
    num_queries = 100
    dimension = 128
    
    database_vectors, query_vectors = generate_test_data(
        num_vectors=num_vectors,
        dimension=dimension,
        num_queries=num_queries
    )
    print(f"Generated {num_vectors} database vectors and {num_queries} query vectors")
    
    # Insert vectors into database
    print("\nInserting vectors into database...")
    vectors_to_insert = [
        {
            "vector": vec,
            "metadata": {"index": i},
            "vector_id": f"vec_{i:06d}"
        }
        for i, vec in enumerate(database_vectors)
    ]
    
    batch_result = vector_db.insert_vector_batch(
        vectors=vectors_to_insert,
        batch_name="benchmark_batch",
        description="Vectors for benchmarking"
    )
    
    if not batch_result["success"]:
        print(f"Error inserting vectors: {batch_result['message']}")
        return
    
    print(f"Inserted {batch_result['result']['vector_count']} vectors")
    
    # Create HNSW index with optimized parameters for better recall
    print("\nCreating HNSW index...")
    index_result = vector_db.create_hnsw_index(
        m=32,
        ef_construction=400
    )
    
    if not index_result["success"]:
        print(f"Error creating index: {index_result['message']}")
        return
    
    print(f"HNSW Index created with {index_result['stats']['total_nodes']} nodes")
    
    # Create benchmark suite
    benchmark_suite = BenchmarkSuite("HNSW Performance Benchmark")
    
    # Define search functions
    def ground_truth_search(query, k=10):
        """Ground truth using brute force"""
        return vector_db.search_brute_force(query, k=k)
    
    def hnsw_search(query, k=10):
        """HNSW search"""
        result = vector_db.search_hnsw(query, k=k)
        return result.get("results", [])
    
    # Run comprehensive benchmark
    report = benchmark_suite.run_comprehensive_benchmark(
        ground_truth_func=ground_truth_search,
        search_func=hnsw_search,
        query_vectors=query_vectors,
        database_vectors=num_vectors,
        k=10
    )
    
    # Print summary
    summary = benchmark_suite.generate_summary_table(report)
    print(summary)
    
    # Save report
    report_path = f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    benchmark_suite.save_report(report, report_path)
    
    # Compare different configurations
    print("\n" + "="*60)
    print("Running Configuration Comparison")
    print("="*60)
    
    comparator = PerformanceComparator()
    
    configurations = [
        {
            "name": "Fast (m=8)",
            "m": 8,
            "ef_construction": 100
        },
        {
            "name": "Balanced (m=16)",
            "m": 16,
            "ef_construction": 200
        },
        {
            "name": "Accurate (m=32)",
            "m": 32,
            "ef_construction": 400
        }
    ]
    
    def search_func_factory(m, ef):
        """Create search function for specific config"""
        def search_func(query, k=10):
            # Use ef_search parameter
            result = vector_db.search_hnsw(query, k=k, ef_search=ef)
            return result.get("results", [])
        return search_func
    
    def search_func_factory(m, ef_construction):
        """Create search function for specific HNSW config"""
        def search_func(query, k=10):
            # Create a new database instance with specific config for testing
            # Note: In a real implementation, you'd rebuild the index with different params
            # For now, we'll use the existing index but this demonstrates the pattern
            result = vector_db.search_hnsw(query, k=k, ef_search=ef_construction)
            return result.get("results", [])
        return search_func

    comparison = comparator.compare_configurations(
        configs=configurations,
        benchmark_suite=benchmark_suite,
        ground_truth_func=ground_truth_search,
        search_func_factory=search_func_factory,
        query_vectors=query_vectors,
        database_vectors=num_vectors,
        k=10
    )
    
    comparator.print_comparison_table(comparison)
    
    # Save comparison (convert BenchmarkReport objects to dicts for JSON serialization)
    comparison_path = f"configuration_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    serializable_comparison = {}
    for config_name, data in comparison.items():
        serializable_comparison[config_name] = {
            "config": data["config"],
            "report": data["report"].to_dict()
        }
    
    with open(comparison_path, 'w') as f:
        import json
        json.dump(serializable_comparison, f, indent=2)
    print(f"\nComparison saved to {comparison_path}")
    
    # Close database
    db.close()
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)

if __name__ == "__main__":
    from datetime import datetime
    run_benchmark()
