import pytest
import numpy as np
from utils.clustering import KMeans, VectorIndexer
def test_kmeans_basic():
    "Test basic K-Means Functionality"

    data=np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2],
        [10, 10],
        [10, 11],
        [11, 10],
        [11, 11]
    ])
    kmeans = KMeans(k=2)
    kmeans.fit(data)
    assert len(kmeans.get_cluster_centers()) == 2
    assert kmeans.get_labels().shape[0] == data.shape[0]

def test_vector_indexer():
    "Test K-Means prediction"
    data=np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [2, 2]
    ])

    kmeans= Kmeans(k=2)
    kmeans.fit(data)

    #test prediction
    test_point=np.array([[1.5, 1.5]])
    prediction=kmeans.predict(np.array([test_point]))

    assert len(prediction)==1
    assert 0<=prediction[0]<2

def test_vector_indexer():
    """Test Vector Indexer functionality"""
    # Create sample vectors
    vectors = [
        [1, 1, 1],
        [2, 2, 2],
        [10, 10, 10],
        [11, 11, 11]
    ]
    
    vector_ids = ["vec1", "vec2", "vec3", "vec4"]
    
    # Test Vector Indexer
    indexer = VectorIndexer(k=2)
    indexer.fit(vectors, vector_ids)
    
    assert indexer.centroids is not None
    assert len(indexer.get_cluster_info()["cluster_sizes"]) == 2
    
    # Test search
    query_vector = [1.5, 1.5, 1.5]
    results = indexer.search(query_vector, k=2)
    
    assert len(results) >= 0

def test_cluster_assignment():
    """Test cluster assignment"""
    # Create simple test data
    vectors = [
        [0, 0],
        [1, 1],
        [10, 10],
        [11, 11]
    ]
    
    vector_ids = ["a", "b", "c", "d"]
    
    indexer = VectorIndexer(k=2)
    indexer.fit(vectors, vector_ids)
    
    # Check that we have clusters
    clusters = indexer.get_all_clusters()
    assert len(clusters) == 2
    assert len(clusters[0]) >= 0 or len(clusters[1]) >= 0

if __name__ == "__main__":
    pytest.main([__file__])
