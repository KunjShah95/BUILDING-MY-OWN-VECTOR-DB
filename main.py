import numpy as np
from database.vector_database import VectorDatabase
from config.database import engine, SessionLocal
from database.schema import Base
import json

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
    print("3. Search vectors")
    print("4. Get database stats")
    print("5. Get vector metadata fields")
    
    # Example usage
    print("\n=== Example Usage ===")
    
    # Insert sample vectors
    print("Inserting sample vectors...")
    
    # Create sample vectors with metadata
    vectors_data = []
    for i in range(5):
        vector = np.random.rand(128).tolist()
        metadata = {
            "id": i,
            "text": f"Sample text {i}",
            "category": "sample",
            "created_at": "2024-01-01",
            "tags": ["tag1", "tag2"],
            "score": i * 0.1
        }
        
        vectors_data.append({
            "vector": vector,
            "metadata": metadata,
            "vector_id": f"sample_vector_{i}"
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
    
    # Insert single vector
    print("\nInserting single vector...")
    single_vector = np.random.rand(128).tolist()
    single_metadata = {
        "id": 100,
        "text": "Single sample vector",
        "category": "single",
        "created_at": "2024-01-02"
    }
    
    single_result = vector_db.insert_vector(single_vector, single_metadata, "single_vector_1")
    print(f"Single vector inserted: {single_result['message']}")
    
    # Search for a query vector
    print("\nSearching for similar vectors...")
    query_vector = np.random.rand(128).tolist()
    
    search_result = vector_db.search(query_vector, k=3, distance_metric='cosine')
    
    if search_result['success']:
        print(f"Found {search_result['total_results']} similar vectors:")
        for i, result in enumerate(search_result['results']):
            print(f"  {i+1}. Distance: {result['distance']:.4f}")
            print(f"     Vector ID: {result['vector_id']}")
            print(f"     Metadata: {result['metadata']}")
    else:
        print(f"Search failed: {search_result['message']}")
    
    # Get database stats
    stats = vector_db.get_database_stats()
    if stats['success']:
        print(f"\nDatabase Stats:")
        print(f"Total Vectors: {stats['stats']['total_vectors']}")
        print(f"Average Dimension: {stats['stats']['avg_dimension']}")
    
    # Get metadata fields
    metadata_fields = vector_db.get_vector_metadata_fields()
    if metadata_fields['success']:
        print(f"\nMetadata Fields:")
        print(f"Available fields: {metadata_fields['metadata_fields']}")
    
    # Get all vectors
    all_vectors = vector_db.get_all_vectors(limit=10)
    if all_vectors['success']:
        print(f"\nRetrieved {all_vectors['count']} vectors")
    
    # Close database session
    db.close()

if __name__ == "__main__":
    main()
