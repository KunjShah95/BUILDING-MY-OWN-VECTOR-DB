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
    
    def _kmeans_plus_plus_init(self, data: np.ndarray, k: int) -> np.ndarray:
        n = data.shape[0]
        centroids = [data[np.random.randint(n)]]
        for _ in range(1, k):
            distances = np.min(
                np.array([np.sum((data - c) ** 2, axis=1) for c in centroids]), axis=0
            )
            dist_sum = distances.sum()
            if dist_sum == 0:
                centroids.append(data[np.random.randint(n)])
            else:
                probs = distances / dist_sum
                centroids.append(data[np.random.choice(n, p=probs)])
        return np.array(centroids)

    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++"""
        return self._kmeans_plus_plus_init(data, self.k)
    
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
