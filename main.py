import numpy as np
from database.vector_database import VectorDatabase
from config.database import engine, SessionLocal
from database.schema import Base
from utils.clustering import KMeans
import time

# Create tables
Base.metadata.create_all(bind=engine)

def main():
    # Create database session
    db = SessionLocal()
    
    # Create vector database instance
    vector_db = VectorDatabase(db)
    
    print("Vector Database initialized!")
    print("Available operations:")
    print("1. Insert vectors")
    print("2. Insert vector batch")
    print("3. Create index")
    print("4. Search vectors (indexed)")
    print("5. Get index information")
    
    # Example usage
    print("\n=== Example Usage ===")
    
    # Insert sample vectors
    print("Inserting sample vectors...")
    
    # Create sample vectors with metadata
    vectors_data = []
    for i in range(50):  # More vectors for better clustering
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
    
    # Create index
    print("\nCreating index...")
    index_result = vector_db.create_index(k=10)  # 10 clusters
    
    if index_result["success"]:
        print(f"Index created successfully!")
        print(f"Total clusters: {index_result['cluster_info']['total_clusters']}")
        print(f"Cluster sizes: {index_result['cluster_info']['cluster_sizes']}")
    else:
        print(f"Index creation failed: {index_result['message']}")
    
    # Search for a query vector using indexed approach
    print("\nSearching for similar vectors (indexed)...")
    query_vector = np.random.rand(128).tolist()
    
    start_time = time.time()
    search_result = vector_db.search(
        query_vector, 
        k=5, 
        use_index=True,
        n_probes=5
    )
    end_time = time.time()
    
    if search_result['success']:
        print(f"Found {search_result['total_results']} similar vectors using indexed search")
        print(f"Search time: {end_time - start_time:.4f} seconds")
        print(f"Method used: {search_result['method']}")
        for i, result in enumerate(search_result['results']):
            print(f"  {i+1}. Distance: {result['distance']:.4f}")
            print(f"     Vector ID: {result['vector_id']}")
            if 'metadata' in result:
                print(f"     Metadata: {result['metadata']}")
    else:
        print(f"Search failed: {search_result['message']}")
    
    # Search using brute force for comparison
    print("\nSearching for similar vectors (brute force)...")
    start_time = time.time()
    brute_force_result = vector_db.search(
        query_vector, 
        k=5, 
        use_index=False
    )
    end_time = time.time()
    
    if brute_force_result['success']:
        print(f"Found {brute_force_result['total_results']} similar vectors using brute force")
        print(f"Search time: {end_time - start_time:.4f} seconds")
        for i, result in enumerate(brute_force_result['results']):
            print(f"  {i+1}. Distance: {result['distance']:.4f}")
            print(f"     Vector ID: {result['vector_id']}")
    
    # Get index information
    print("\nGetting index information...")
    index_info = vector_db.get_index_info()
    if index_info['success']:
        print(f"Index Info:")
        print(f"  Is indexed: {index_info['index_info']['is_indexed']}")
        print(f"  Total clusters: {index_info['index_info']['total_clusters']}")
        print(f"  Total vectors: {index_info['index_info']['total_vectors']}")
    
    # Get cluster information
    print("\nGetting cluster information...")
    cluster_info = vector_db.get_cluster_vectors(0)
    if cluster_info['success']:
        print(f"Cluster 0 has {cluster_info['count']} vectors")
    
    # Close database session
    db.close()

if __name__ == "__main__":
    main()
