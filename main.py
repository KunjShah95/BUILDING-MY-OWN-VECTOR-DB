import numpy as np
from database.ivf_database import IVFVectorDatabase
from config.database import engine, SessionLocal
from database.schema import Base
import time

# Create tables
Base.metadata.create_all(bind=engine)

def main():
    # Create database session
    db = SessionLocal()
    
    # Create IVF vector database instance
    vector_db = IVFVectorDatabase(db)
    
    print("IVF Vector Database initialized!")
    print("Available operations:")
    print("1. Insert vectors")
    print("2. Insert vector batch")
    print("3. Create IVF index")
    print("4. Load IVF index")
    print("5. Search vectors (IVF)")
    print("6. Compare search methods")
    print("7. Get database stats")
    
    # Example usage
    print("\n=== Example Usage ===")
    
    # Insert sample vectors
    print("Inserting sample vectors...")
    
    # Create sample vectors with metadata
    vectors_data = []
    for i in range(100):  # More vectors for better clustering
        vector = np.random.rand(128).tolist()
        metadata = {
            "id": i,
            "text": f"Sample text {i}",
            "category": "sample",
            "created_at": "2024-01-01",
            "tags": ["tag1", "tag2"] if i % 2 == 0 else ["tag3", "tag4"],
            "score": i * 0.1
        }
        
        vectors_data.append({
            "vector": vector,
            "metadata": metadata,
            "vector_id": f"sample_vector_{i:03d}"
        })
    
    # Insert batch
    batch_result = vector_db.insert_vector_batch(
        vectors=vectors_data,
        batch_name="sample_batch_1",
        description="Sample batch of vectors for testing"
    )
    
    if batch_result["success"]:
        print(f"Batch inserted successfully: {batch_result['result']['vector_count']} vectors")
    else:
        print(f"Batch insertion failed: {batch_result['message']}")
    
    # Create IVF index
    print("\nCreating IVF index...")
    index_result = vector_db.create_ivf_index(n_clusters=10, n_probes=5)
    
    if index_result["success"]:
        print(f"IVF Index created successfully!")
        print(f"Stats: {index_result['stats']}")
    else:
        print(f"IVF Index creation failed: {index_result['message']}")
    
    # Search for a query vector using IVF
    print("\nSearching for similar vectors (IVF)...")
    query_vector = np.random.rand(128).tolist()
    
    start_time = time.time()
    search_result = vector_db.search(query_vector, k=5, use_ivf=True, use_rerank=True)
    end_time = time.time()
    
    if search_result['success']:
        print(f"Found {search_result['total_results']} similar vectors using IVF search")
        print(f"Search time: {end_time - start_time:.4f} seconds")
        print(f"Method used: {search_result['method']}")
        for i, result in enumerate(search_result['results']):
            print(f"  {i+1}. Distance: {result['distance']:.4f}")
            print(f"     Vector ID: {result['vector_id']}")
            if 'metadata' in result and result['metadata']:
                print(f"     Category: {result['metadata'].get('category', 'N/A')}")
    else:
        print(f"Search failed: {search_result['message']}")
    
    # Compare search methods
    print("\nComparing search methods...")
    comparison_result = vector_db.compare_search_methods(query_vector, k=5)
    
    print("\nMethod Comparison:")
    for method, data in comparison_result["methods"].items():
        print(f"  {method}:")
        print(f"    Time: {data['time']:.4f} seconds")
        print(f"    Results: {data['count']}")
    
    # Get index statistics
    print("\nGetting index statistics...")
    index_stats = vector_db.get_index_stats()
    if index_stats['success']:
        print(f"Index Stats:")
        print(f"  Total vectors: {index_stats['stats']['total_vectors']}")
        print(f"  Number of clusters: {index_stats['stats']['n_clusters']}")
        print(f"  Average cluster size: {index_stats['stats']['avg_cluster_size']:.2f}")
        print(f"  Standard deviation: {index_stats['stats']['std_cluster_size']:.2f}")
        print(f"  Memory usage: {index_stats['stats']['memory_usage']}")
    
    # Get database statistics
    print("\nGetting database statistics...")
    db_stats = vector_db.get_database_stats()
    if db_stats['success']:
        print(f"Database Stats:")
        print(f"  Total vectors: {db_stats['stats']['total_vectors']}")
        print(f"  Average dimension: {db_stats['stats']['avg_dimension']}")
        print(f"  Database size: {db_stats['stats']['database_size']}")
    
    # Save index
    print("\nSaving IVF index...")
    save_result = vector_db.save_ivf_index()
    if save_result['success']:
        print(f"Index saved successfully!")
    else:
        print(f"Index save failed: {save_result['message']}")
    
    # Close database session
    db.close()

if __name__ == "__main__":
    main()
