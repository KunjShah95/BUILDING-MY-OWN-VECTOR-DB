import numpy as np
from database.vector_database import VectorDatabase
from config.database import engine, SessionLocal, Base

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
    print("2. Search vectors")
    print("3. Get database stats")
    
    # Example usage
    print("\n=== Example Usage ===")
    
    # Insert sample vectors
    print("Inserting sample vectors...")
    
    # Create sample vectors (128 dimensions)
    for i in range(5):
        vector = np.random.rand(128).tolist()
        metadata = {
            "id": i,
            "text": f"Sample text {i}",
            "category": "sample",
            "created_at": "2024-01-01"
        }
        
        result = vector_db.insert_vector(vector, metadata, f"sample_vector_{i}")
        print(f"Inserted: {result['message']}")
    
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
        print(f"Total Vectors: {stats['total_vectors']}")
        print(f"Average Dimension: {stats['avg_dimension']}")
    
    # Close database session
    db.close()

if __name__ == "__main__":
    main()
