import numpy as np
from typing import List, Tuple, Dict, Any
import random
from collections import defaultdict

class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tol: float = 1e-4):
        """
        Initialize K-Means clustering
        
        Args:
            k: Number of clusters
            max_iters: Maximum number of iterations
            tol: Tolerance for convergence
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(point1 - point2)
    
    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """Initialize centroids randomly"""
        n_samples, n_features = data.shape
        centroids = np.zeros((self.k, n_features))
        
        # Randomly select k data points as initial centroids
        for i in range(self.k):
            centroid_idx = random.randint(0, n_samples - 1)
            centroids[i] = data[centroid_idx]
            
        return centroids
    
    def _assign_clusters(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each data point to the nearest centroid"""
        labels = []
        for point in data:
            distances = [self._euclidean_distance(point, centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            labels.append(closest_centroid)
        return np.array(labels)
    
    def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids based on current cluster assignments"""
        centroids = np.zeros((self.k, data.shape[1]))
        for i in range(self.k):
            # Get all points assigned to cluster i
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                # Calculate mean of all points in cluster
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If no points assigned, keep the old centroid
                centroids[i] = self.centroids[i]
        return centroids
    
    def fit(self, data: np.ndarray) -> 'KMeans':
        """
        Fit the K-Means model to the data
        
        Args:
            data: Array of shape (n_samples, n_features)
            
        Returns:
            self: Fitted KMeans object
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(data)
        
        for iteration in range(self.max_iters):
            # Assign points to clusters
            labels = self._assign_clusters(data, self.centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(data, labels)
            
            # Check for convergence
            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
            self.labels = labels
            
        self.labels = labels
        return self
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data points
        
        Args:
            data: Array of shape (n_samples, n_features)
            
        Returns:
            Array of cluster labels
        """
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")
            
        return self._assign_clusters(data, self.centroids)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get the final cluster centroids"""
        return self.centroids
    
    def get_labels(self) -> np.ndarray:
        """Get the cluster labels for training data"""
        return self.labels
    
    def calculate_wcss(self, data: np.ndarray) -> float:
        """
        Calculate Within-Cluster Sum of Squares (WCSS)
        
        Args:
            data: Array of shape (n_samples, n_features)
            
        Returns:
            WCSS value
        """
        wcss = 0
        for i in range(self.k):
            cluster_points = data[self.labels == i]
            if len(cluster_points) > 0:
                centroid = self.centroids[i]
                wcss += np.sum((cluster_points - centroid) ** 2)
        return wcss

class VectorIndexer:
    def __init__(self, k: int = 100):
        """
        Initialize vector indexer with K-Means clustering
        
        Args:
            k: Number of clusters for K-Means
        """
        self.k = k
        self.kmeans = None
        self.clusters = defaultdict(list)  # Map cluster_id -> list of vector_ids
        self.centroids = None
        
    def fit(self, vectors: List[np.ndarray], vector_ids: List[str] = None) -> 'VectorIndexer':
        """
        Fit the indexer to vectors
        
        Args:
            vectors: List of vector arrays
            vector_ids: List of vector IDs (optional)
            
        Returns:
            self: Fitted VectorIndexer object
        """
        # Convert to numpy array
        if isinstance(vectors[0], list):
            vectors = np.array(vectors)
        
        # Fit K-Means
        self.kmeans = KMeans(k=self.k)
        self.kmeans.fit(vectors)
        
        # Store cluster assignments
        self.centroids = self.kmeans.get_cluster_centers()
        labels = self.kmeans.get_labels()
        
        # Group vectors by cluster
        for i, (vector, label) in enumerate(zip(vectors, labels)):
            vector_id = vector_ids[i] if vector_ids else f"vector_{i}"
            self.clusters[label].append({
                "vector_id": vector_id,
                "vector": vector,
                "index": i
            })
            
        return self
    
    def get_closest_cluster(self, query_vector: np.ndarray) -> int:
        """
        Find the closest cluster to the query vector
        
        Args:
            query_vector: Query vector array
            
        Returns:
            Cluster ID (int)
        """
        if self.centroids is None:
            raise ValueError("Indexer must be fitted before querying")
            
        distances = [np.linalg.norm(query_vector - centroid) for centroid in self.centroids]
        return np.argmin(distances)
    
    def search_in_cluster(self, query_vector: np.ndarray, cluster_id: int, 
                         k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors within a specific cluster
        
        Args:
            query_vector: Query vector
            cluster_id: Cluster to search in
            k: Number of results to return
            
        Returns:
            List of similar vectors with distances
        """
        if cluster_id not in self.clusters:
            return []
            
        cluster_vectors = self.clusters[cluster_id]
        results = []
        
        for vector_info in cluster_vectors:
            distance = np.linalg.norm(query_vector - vector_info["vector"])
            results.append({
                "vector_id": vector_info["vector_id"],
                "distance": float(distance),
                "vector": vector_info["vector"].tolist()
            })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x["distance"])
        return results[:k]
    
    def search(self, query_vector: np.ndarray, k: int = 5, 
              n_probes: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cluster-based approach
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            n_probes: Number of clusters to probe (for approximate search)
            
        Returns:
            List of similar vectors with distances
        """
        if self.centroids is None:
            raise ValueError("Indexer must be fitted before querying")
            
        # Find the closest cluster(s)
        closest_cluster = self.get_closest_cluster(query_vector)
        
        # For better results, search multiple nearby clusters
        # Calculate distances to all centroids
        distances = [np.linalg.norm(query_vector - centroid) for centroid in self.centroids]
        cluster_distances = list(enumerate(distances))
        cluster_distances.sort(key=lambda x: x[1])
        
        # Select top n_probes clusters
        selected_clusters = [cluster_id for cluster_id, _ in cluster_distances[:n_probes]]
        
        # Search in all selected clusters
        all_results = []
        for cluster_id in selected_clusters:
            cluster_results = self.search_in_cluster(query_vector, cluster_id, k)
            all_results.extend(cluster_results)
        
        # Sort all results by distance and return top k
        all_results.sort(key=lambda x: x["distance"])
        return all_results[:k]
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about all clusters
        
        Returns:
            Dictionary with cluster information
        """
        if self.centroids is None:
            return {}
            
        cluster_info = {
            "total_clusters": self.k,
            "cluster_sizes": [len(vectors) for vectors in self.clusters.values()],
            "centroids": self.centroids.tolist()
        }
        
        return cluster_info
    
    def get_cluster_vectors(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get all vectors in a specific cluster
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List of vectors in the cluster
        """
        return self.clusters.get(cluster_id, [])
    
    def get_all_clusters(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get all clusters with their vectors
        
        Returns:
            Dictionary mapping cluster_id to vectors
        """
        return dict(self.clusters)
