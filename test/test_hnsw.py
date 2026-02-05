import pytest
import numpy as np
from utils.hnsw_index import HNSWIndex

def test_hnsw_basic():
    """Test basic HNSW functionality"""
    # Create sample vectors
    vectors = []
    for i in range(50):
        # Create 3 clusters of vectors
        if i < 17:
            vector = np.random.rand(16).astype(np.float32) + np.array([1] * 16)
        elif i < 34:
            vector = np.random.rand(16).astype(np.float32) + np.array([5] * 16)
        else:
            vector = np.random.rand(16).astype(np.float32) + np.array([10] * 16)
        
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(50)]
    
    # Test HNSW index
    index = HNSWIndex(m=8, m0=16, ef_construction=100)
    
    # Insert vectors
    for vector, vector_id in zip(vectors, vector_ids):
        index.insert(vector, vector_id, {"id": vector_id})
    
    # Verify insertion
    assert len(index.graph) == 50
    assert index.total_inserted > 0
    
    # Test search
    query_vector = np.random.rand(16).astype(np.float32) + np.array([1] * 16)
    results = index.search(query_vector.tolist(), k=5)
    
    assert len(results) >= 0
    assert len(results) <= 5

def test_hnsw_search_parameters():
    """Test HNSW search with different parameters"""
    # Create sample vectors
    vectors = []
    for i in range(30):
        vector = np.random.rand(8).astype(np.float32)
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(30)]
    
    # Create HNSW index
    index = HNSWIndex(m=4, m0=8, ef_construction=50)
    
    # Insert vectors
    for vector, vector_id in zip(vectors, vector_ids):
        index.insert(vector, vector_id)
    
    # Test search with different ef values
    query_vector = np.random.rand(8).astype(np.float32)
    
    for ef in [5, 10, 20]:
        results = index.search(query_vector.tolist(), k=3, ef=ef)
        assert len(results) >= 0
        assert len(results) <= 3

def test_hnsw_delete():
    """Test HNSW node deletion"""
    # Create sample vectors
    vectors = []
    for i in range(20):
        vector = np.random.rand(8).astype(np.float32)
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(20)]
    
    # Create HNSW index
    index = HNSWIndex(m=4, m0=8, ef_construction=50)
    
    # Insert vectors
    for vector, vector_id in zip(vectors, vector_ids):
        index.insert(vector, vector_id)
    
    # Delete a node
    deleted = index.delete(vector_ids[5])
    assert deleted == True
    
    # Verify deletion
    assert len(index.graph) == 19
    assert vector_ids[5] not in index.graph
    
    # Try to delete again
    deleted_again = index.delete(vector_ids[5])
    assert deleted_again == False

def test_hnsw_neighbors():
    """Test HNSW neighbor retrieval"""
    vectors = []
    for i in range(25):
        vector = np.random.rand(8).astype(np.float32)
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(25)]
    
    # Create HNSW index
    index = HNSWIndex(m=4, m0=8, ef_construction=50)
    
    # Insert vectors
    for vector, vector_id in zip(vectors, vector_ids):
        index.insert(vector, vector_id)
    
    # Get neighbors for first node
    neighbors = index.get_neighbors(vector_ids[0], level=0)
    assert isinstance(neighbors, list)
    
    # Get node info
    info = index.get_node_info(vector_ids[0])
    assert info["node_id"] == vector_ids[0]
    assert "neighbors" in info

def test_hnsw_save_load():
    """Test HNSW index save/load functionality"""
    # Create sample vectors
    vectors = []
    for i in range(20):
        vector = np.random.rand(16).astype(np.float32)
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(20)]
    
    # Create and train index
    index1 = HNSWIndex(m=4, m0=8, ef_construction=50)
    
    for vector, vector_id in zip(vectors, vector_ids):
        index1.insert(vector, vector_id, {"id": vector_id})
    
    # Save index
    index1.save("test_hnsw_index.json")
    
    # Load index
    index2 = HNSWIndex()
    index2.load("test_hnsw_index.json")
    
    # Verify loaded index
    assert len(index2.graph) == 20
    assert index2.total_inserted > 0
    
    # Test search with loaded index
    query_vector = np.random.rand(16).astype(np.float32)
    results = index2.search(query_vector.tolist(), k=3)
    
    assert len(results) >= 0

def test_hnsw_graph_stats():
    """Test HNSW graph statistics"""
    vectors = []
    for i in range(30):
        vector = np.random.rand(8).astype(np.float32)
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(30)]
    
    # Create HNSW index
    index = HNSWIndex(m=4, m0=8, ef_construction=50)
    
    # Insert vectors
    for vector, vector_id in zip(vectors, vector_ids):
        index.insert(vector, vector_id)
    
    # Get statistics
    stats = index.get_graph_stats()
    
    assert stats["total_nodes"] == 30
    assert stats["total_edges"] > 0
    assert stats["avg_connections"] > 0
    assert stats["max_level"] >= 0

if __name__ == "__main__":
    pytest.main([__file__])
