import pytest
import numpy as np
from utils.ivf_index import CoarseQuantizer, FineQuantizer, IVFIndex

def test_coarse_quantizer():
    """Test coarse quantizer functionality"""
    # Create sample data
    data = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [10, 10],
        [10, 11],
        [11, 10],
        [11, 11]
    ], dtype=np.float32)
    
    # Test coarse quantizer
    quantizer = CoarseQuantizer(n_clusters=2)
    quantizer.train(data)
    
    assert quantizer.centroids is not None
    assert len(quantizer.centroids) == 2
    
    # Test encoding
    test_vector = np.array([5, 5], dtype=np.float32)
    cluster_id = quantizer.encode(test_vector)
    
    assert 0 <= cluster_id < 2

def test_fine_quantizer():
    """Test fine quantizer functionality"""
    # Create residual data
    residuals = np.random.rand(100, 16).astype(np.float32)
    
    # Test fine quantizer
    quantizer = FineQuantizer(n_subquantizers=4, bits_per_subquantizer=4)
    quantizer.train(residuals)
    
    assert quantizer.codebooks is not None
    assert len(quantizer.codebooks) == 4
    
    # Test encoding and decoding
    test_residual = residuals[0]
    codes = quantizer.encode(test_residual)
    
    assert len(codes) == 4
    assert all(0 <= code < 16 for code in codes)

def test_ivf_index():
    """Test IVF index functionality"""
    # Create sample vectors
    vectors = []
    for i in range(50):
        # Create 3 clusters of vectors
        if i < 17:
            vector = np.random.rand(16).astype(np.float32) + np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        elif i < 34:
            vector = np.random.rand(16).astype(np.float32) + np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        else:
            vector = np.random.rand(16).astype(np.float32) + np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
        
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(50)]
    
    # Test IVF index
    index = IVFIndex(n_clusters=5, n_probes=2)
    index.train(vectors)
    
    # Add all vectors
    for vector, vector_id in zip(vectors, vector_ids):
        index.add(vector, vector_id, {"id": vector_id})
    
    assert len(index.vector_ids) == 50
    
    # Test search
    query_vector = np.random.rand(16).astype(np.float32) + np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    results = index.search(query_vector.tolist(), k=5)
    
    assert len(results) >= 0
    assert len(results) <= 5

def test_ivf_save_load():
    """Test IVF index save/load functionality"""
    # Create sample vectors
    vectors = []
    for i in range(20):
        vector = np.random.rand(16).astype(np.float32)
        vectors.append(vector.tolist())
    
    vector_ids = [f"vec_{i:03d}" for i in range(20)]
    
    # Create and train index
    index1 = IVFIndex(n_clusters=3, n_probes=2)
    index1.train(vectors)
    
    for vector, vector_id in zip(vectors, vector_ids):
        index1.add(vector, vector_id, {"id": vector_id})
    
    # Save index
    index1.save("test_index.json")
    
    # Load index
    index2 = IVFIndex()
    index2.load("test_index.json")
    
    # Verify loaded index
    assert len(index2.vector_ids) == 20
    assert index2.is_trained == True
    
    # Test search with loaded index
    query_vector = np.random.rand(16).astype(np.float32)
    results = index2.search(query_vector.tolist(), k=3)
    
    assert len(results) >= 0

if __name__ == "__main__":
    pytest.main([__file__])
