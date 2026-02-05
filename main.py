import numpy as np
from database.hnsw_database import HNSWVectorDatabase
from config.database import engine, SessionLocal
from database.schema import Base
import time

# Create tables
Base.metadata.create_all(bind=engine)

def main():
    # Create database session
    db = SessionLocal()
    
    # Create HNSW vector database instance
    vector_db = HNSWVectorDatabase(db)
    
    print("HNSW Vector Database initialized!")
    print("Available operations:")
    print("1. Insert vectors")
    print("2. Insert vector batch")
    print("3. Create HNSW index")
    print("4. Load HNSW index")
    print("5. Search vectors (HNSW)")
    print("6. Compare search methods")
    print("7. Get database stats")
    
    # Example usage
    print("\n=== Example Usage ===")
    
    # Insert sample vectors
    print("Inserting sample vectors...")
    
    # Create sample vectors with metadata
    vectors_data = []
    for i in range(200):  # More vectors for better HNSW performance
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
    
    # Create HNSW index
    print("\nCreating HNSW index...")
    index_result = vector_db.create_hnsw_index(
        m=16,           # Number of neighbors
        m0=32,          # Neighbors in layer 0
        ef_construction=200  # Construction parameter
    )
    
    if index_result["success"]:
        print("HNSW Index created successfully!")
        print(f"Stats:")
        print(f"  Total nodes: {index_result['stats']['total_nodes']}")
        print(f"  Total edges: {index_result['stats']['total_edges']}")
        print(f"  Average connections: {index_result['stats']['avg_connections']:.2f}")
        print(f"  Max level: {index_result['stats']['max_level']}")
        print(f"  Level distribution: {index_result['stats']['level_distribution']}")
    else:
        print(f"HNSW Index creation failed: {index_result['message']}")
    
    # Search for a query vector using HNSW
    print("\nSearching for similar vectors (HNSW)...")
    query_vector = np.random.rand(128).tolist()
    
    # Test different ef_search values
    for ef_search in [10, 50, 100]:
        search_result = vector_db.search_hnsw(query_vector, k=5, ef_search=ef_search)
        
        if search_result['success']:
            print(f"\nHNSW Search (ef_search={ef_search}):")
            print(f"  Found {search_result['total_results']} similar vectors")
            print(f"  Search time: {search_result['search_time']:.4f} seconds")
            print(f"  Method: {search_result['method']}")
            
            for i, result in enumerate(search_result['results'][:3]):  # Show top 3
                print(f"  {i+1}. Distance: {result['distance']:.4f}")
                print(f"     Vector ID: {result['vector_id']}")
    
    # Compare with brute force
    print("\nComparing search methods...")
    comparison_result = vector_db.compare_search_methods(query_vector, k=5)
    
    print("\nMethod Comparison:")
    for method, data in comparison_result["methods"].items():
        print(f"  {method}:")
        print(f"    Time: {data['time']:.4f} seconds")
        print(f"    Results: {data['count']}")
    
    # Get index information
    print("\nGetting HNSW index information...")
    index_info = vector_db.get_hnsw_index_info()
    if index_info['success']:
        print(f"HNSW Index Info:")
        print(f"  Total nodes: {index_info['index_info']['total_nodes']}")
        print(f"  Max level: {index_info['index_info']['max_level']}")
        print(f"  Level distribution: {index_info['index_info']['level_distribution']}")
    
    # Get database statistics
    print("\nGetting database statistics...")
    db_stats = vector_db.get_database_stats()
    if db_stats['success']:
        print(f"Database Stats:")
        print(f"  Total vectors: {db_stats['stats']['total_vectors']}")
        print(f"  Average dimension: {db_stats['stats']['avg_dimension']}")
    
    # Save index
    print("\nSaving HNSW index...")
    save_result = vector_db.save_hnsw_index()
    if save_result['success']:
        print("Index saved successfully!")
    else:
        print(f"Index save failed: {save_result['message']}")
    
    # Test single vector insertion
    print("\nTesting single vector insertion...")
    single_vector = np.random.rand(128).tolist()
    single_metadata = {
        "id": 999,
        "text": "Single test vector",
        "category": "test"
    }
    
    insert_result = vector_db.insert_vector(single_vector, single_metadata, "single_test_vector")
    if insert_result['success']:
        print(f"Single vector inserted: {insert_result['vector_id']}")
    
    # Search after single insertion
    print("\nSearching after single vector insertion...")
    search_result = vector_db.search_hnsw(query_vector, k=3)
    if search_result['success']:
        print(f"Found {search_result['total_results']} vectors")
    
    # Close database session
    db.close()

if __name__ == "__main__":
    main()
