import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import pickle
import os
import json

class CoarseQuantizer:
    """
    Coarse quantizer using K-Means clustering
    Assigns vectors to nearest cluster centroids
    """
    
    def __init__(self, n_clusters: int):
        """
        Initialize coarse quantizer
        
        Args:
            n_clusters: Number of clusters (coarse quantization levels)
        """
        self.n_clusters = n_clusters
        self.centroids = None
        self.kmeans = None
        
    def train(self, vectors: np.ndarray, max_iters: int = 100) -> 'CoarseQuantizer':
        """
        Train the coarse quantizer
        
        Args:
            vectors: Training vectors of shape (n_samples, n_features)
            max_iters: Maximum iterations for K-Means
            
        Returns:
            self
        """
        from utils.clustering import KMeans
        
        self.kmeans = KMeans(k=self.n_clusters, max_iters=max_iters)
        self.kmeans.fit(vectors)
        self.centroids = self.kmeans.get_cluster_centers()
        
        return self
    
    def encode(self, vector: np.ndarray) -> int:
        """
        Encode a single vector to cluster ID
        
        Args:
            vector: Input vector
            
        Returns:
            Cluster ID
        """
        if self.centroids is None:
            raise ValueError("Quantizer not trained")
            
        distances = [np.linalg.norm(vector - centroid) for centroid in self.centroids]
        return np.argmin(distances)
    
    def encode_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode multiple vectors to cluster IDs
        
        Args:
            vectors: Input vectors of shape (n_samples, n_features)
            
        Returns:
            Cluster IDs of shape (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Quantizer not trained")
            
        cluster_ids = []
        for vector in vectors:
            distances = [np.linalg.norm(vector - centroid) for centroid in self.centroids]
            cluster_ids.append(np.argmin(distances))
            
        return np.array(cluster_ids)
    
    def decode(self, cluster_id: int) -> np.ndarray:
        """
        Get centroid for a cluster ID
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Centroid vector
        """
        if self.centroids is None:
            raise ValueError("Quantizer not trained")
            
        return self.centroids[cluster_id]


class FineQuantizer:
    """
    Fine quantizer using residual vectors
    Refines the coarse quantization by quantizing residuals
    """
    
    def __init__(self, n_subquantizers: int = 8, bits_per_subquantizer: int = 8):
        """
        Initialize fine quantizer
        
        Args:
            n_subquantizers: Number of sub-quantizers
            bits_per_subquantizer: Bits per sub-quantizer
        """
        self.n_subquantizers = n_subquantizers
        self.bits_per_subquantizer = bits_per_subquantizer
        self.codebooks = None  # List of codebooks for each sub-quantizer
        self.n_codes = 2 ** bits_per_subquantizer
        
    def train(self, residuals: np.ndarray) -> 'FineQuantizer':
        """
        Train the fine quantizer on residual vectors
        
        Args:
            residuals: Residual vectors from coarse quantization
            
        Returns:
            self
        """
        n_samples, n_features = residuals.shape
        self.codebooks = []
        
        # Split dimensions into sub-vectors
        dim_per_subquantizer = n_features // self.n_subquantizers
        
        for i in range(self.n_subquantizers):
            start_idx = i * dim_per_subquantizer
            end_idx = start_idx + dim_per_subquantizer
            
            sub_vectors = residuals[:, start_idx:end_idx]
            
            # Create codebook for this sub-quantizer using K-Means
            codebook = self._create_codebook(sub_vectors)
            self.codebooks.append(codebook)
        
        return self
    
    def _create_codebook(self, vectors: np.ndarray) -> np.ndarray:
        """
        Create codebook for a sub-quantizer
        
        Args:
            vectors: Sub-vectors
            
        Returns:
            Codebook of shape (n_codes, dim_per_subquantizer)
        """
        from utils.clustering import KMeans
        
        kmeans = KMeans(k=self.n_codes, max_iters=20)
        kmeans.fit(vectors)
        
        return kmeans.get_cluster_centers()
    
    def encode(self, residual: np.ndarray) -> List[int]:
        """
        Encode a residual vector to codes
        
        Args:
            residual: Residual vector
            
        Returns:
            List of codes for each sub-quantizer
        """
        if self.codebooks is None:
            raise ValueError("Quantizer not trained")
            
        codes = []
        dim_per_subquantizer = len(residual) // self.n_subquantizers
        
        for i, codebook in enumerate(self.codebooks):
            start_idx = i * dim_per_subquantizer
            end_idx = start_idx + dim_per_subquantizer
            
            sub_vector = residual[start_idx:end_idx]
            
            # Find nearest code in codebook
            distances = [np.linalg.norm(sub_vector - code) for code in codebook]
            code = np.argmin(distances)
            codes.append(code)
            
        return codes
    
    def decode(self, codes: List[int]) -> np.ndarray:
        """
        Decode codes back to residual vector
        
        Args:
            codes: List of codes
            
        Returns:
            Decoded residual vector
        """
        if self.codebooks is None:
            raise ValueError("Quantizer not trained")
            
        reconstructed = []
        for code, codebook in zip(codes, self.codebooks):
            reconstructed.append(codebook[code])
            
        return np.concatenate(reconstructed)
    
    def compute_distance(self, query: np.ndarray, codes: List[int]) -> float:
        """
        Compute distance between query and encoded vector using asymmetric distance
        
        Args:
            query: Query vector
            codes: Encoded codes
            
        Returns:
            Approximate distance
        """
        if self.codebooks is None:
            raise ValueError("Quantizer not trained")
        
        reconstructed = self.decode(codes)
        return np.linalg.norm(query - reconstructed)


class IVFIndex:
    """
    Inverted File Index implementation
    Combines coarse and fine quantizers for efficient search
    """
    
    def __init__(self, n_clusters: int = 100, n_probes: int = 10):
        """
        Initialize IVF Index
        
        Args:
            n_clusters: Number of clusters for coarse quantization
            n_probes: Number of clusters to probe during search
        """
        self.n_clusters = n_clusters
        self.n_probes = n_probes
        
        # Quantizers
        self.coarse_quantizer = CoarseQuantizer(n_clusters)
        self.fine_quantizer = FineQuantizer()
        
        # Inverted file structure: cluster_id -> list of vector info
        self.inverted_file = defaultdict(list)
        
        # Vector storage
        self.vectors = {}  # vector_id -> vector
        self.residuals = {}  # vector_id -> residual codes
        
        # Metadata storage
        self.metadata = {}  # vector_id -> metadata
        self.vector_ids = []  # Ordered list of vector IDs
        
        # Training flag
        self.is_trained = False
        
    def train(self, vectors: List[List[float]]) -> 'IVFIndex':
        """
        Train the IVF index on training vectors
        
        Args:
            vectors: Training vectors
            
        Returns:
            self
        """
        vectors_array = np.array(vectors, dtype=np.float32)
        
        print(f"Training coarse quantizer with {self.n_clusters} clusters...")
        self.coarse_quantizer.train(vectors_array)
        
        # Compute residuals
        cluster_ids = self.coarse_quantizer.encode_batch(vectors_array)
        residuals = []
        
        for vector, cluster_id in zip(vectors_array, cluster_ids):
            centroid = self.coarse_quantizer.decode(cluster_id)
            residual = vector - centroid
            residuals.append(residual)
        
        residuals_array = np.array(residuals, dtype=np.float32)
        
        print("Training fine quantizer...")
        self.fine_quantizer.train(residuals_array)
        
        self.is_trained = True
        print("Index training complete!")
        
        return self
    
    def add(self, vector: List[float], vector_id: str, metadata: Dict[str, Any] = None):
        """
        Add a vector to the index
        
        Args:
            vector: Vector to add
            vector_id: Unique vector ID
            vector_id: Vector metadata
        """
        if not self.is_trained:
            raise ValueError("Index must be trained before adding vectors")
        
        vector_array = np.array(vector, dtype=np.float32)
        
        # Encode with coarse quantizer
        cluster_id = self.coarse_quantizer.encode(vector_array)
        
        # Compute residual
        centroid = self.coarse_quantizer.decode(cluster_id)
        residual = vector_array - centroid
        
        # Encode residual with fine quantizer
        residual_codes = self.fine_quantizer.encode(residual)
        
        # Store in inverted file
        self.inverted_file[cluster_id].append({
            "vector_id": vector_id,
            "distance_to_centroid": float(np.linalg.norm(residual))
        })
        
        # Store vector data
        self.vectors[vector_id] = vector_array
        self.residuals[vector_id] = residual_codes
        self.metadata[vector_id] = metadata
        self.vector_ids.append(vector_id)
    
    def add_batch(self, vectors: List[Dict[str, Any]]):
        """
        Add multiple vectors to the index
        
        Args:
            vectors: List of vector dictionaries with 'vector', 'vector_id', 'metadata'
        """
        for vector_data in vectors:
            self.add(
                vector=vector_data["vector"],
                vector_id=vector_data["vector_id"],
                metadata=vector_data.get("metadata")
            )
    
    def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of search results with distances and metadata
        """
        if not self.is_trained:
            raise ValueError("Index must be trained before searching")
        
        query_array = np.array(query_vector, dtype=np.float32)
        
        # Find closest clusters
        cluster_distances = []
        for cluster_id in range(self.n_clusters):
            centroid = self.coarse_quantizer.decode(cluster_id)
            distance = np.linalg.norm(query_array - centroid)
            cluster_distances.append((cluster_id, distance))
        
        # Sort by distance and select top n_probes
        cluster_distances.sort(key=lambda x: x[1])
        probe_clusters = [c[0] for c in cluster_distances[:self.n_probes]]
        
        # Search in selected clusters
        candidate_results = []
        
        for cluster_id in probe_clusters:
            for vector_info in self.inverted_file[cluster_id]:
                vector_id = vector_info["vector_id"]
                
                # Compute distance using fine quantizer
                residual_codes = self.residuals[vector_id]
                residual = self.fine_quantizer.decode(residual_codes)
                
                # Get original centroid
                centroid = self.coarse_quantizer.decode(cluster_id)
                
                # Asymmetric distance computation
                # Distance = ||query - centroid - residual||
                distance = np.linalg.norm(query_array - centroid - residual)
                
                candidate_results.append({
                    "vector_id": vector_id,
                    "distance": float(distance),
                    "metadata": self.metadata.get(vector_id)
                })
        
        # Sort all candidates and return top k
        candidate_results.sort(key=lambda x: x["distance"])
        return candidate_results[:k]
    
    def search_with_rerank(self, query_vector: List[float], k: int = 5, 
                          initial_candidates: int = 50) -> List[Dict[str, Any]]:
        """
        Search with reranking using exact distance
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            initial_candidates: Number of initial candidates to rerank
            
        Returns:
            List of reranked search results
        """
        # Get more candidates than needed
        candidates = self.search(query_vector, k=initial_candidates)
        
        query_array = np.array(query_vector, dtype=np.float32)
        
        # Rerank using exact distance
        for candidate in candidates:
            vector_id = candidate["vector_id"]
            vector = self.vectors[vector_id]
            candidate["distance"] = float(np.linalg.norm(query_array - vector))
        
        # Sort by exact distance
        candidates.sort(key=lambda x: x["distance"])
        return candidates[:k]
    
    def save(self, filepath: str):
        """
        Save the index to disk
        
        Args:
            filepath: Path to save the index
        """
        # Helper function to convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert numpy int64 keys and values to serializable types
        inverted_file_serializable = {str(k): convert_numpy_types(v) for k, v in self.inverted_file.items()}
        residuals_serializable = convert_numpy_types(self.residuals)
        
        index_data = {
            "n_clusters": self.n_clusters,
            "n_probes": self.n_probes,
            "coarse_centroids": self.coarse_quantizer.centroids.tolist() if self.coarse_quantizer.centroids is not None else None,
            "codebooks": [codebook.tolist() for codebook in self.fine_quantizer.codebooks] if self.fine_quantizer.codebooks else None,
            "inverted_file": inverted_file_serializable,
            "vectors": {k: v.tolist() for k, v in self.vectors.items()},
            "residuals": residuals_serializable,
            "metadata": self.metadata,
            "vector_ids": self.vector_ids,
            "is_trained": self.is_trained
        }
        
        with open(filepath, 'w') as f:
            json.dump(index_data, f)
        
        print(f"Index saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the index from disk
        
        Args:
            filepath: Path to load the index from
        """
        with open(filepath, 'r') as f:
            index_data = json.load(f)
        
        self.n_clusters = index_data["n_clusters"]
        self.n_probes = index_data["n_probes"]
        self.is_trained = index_data["is_trained"]
        
        # Rebuild coarse quantizer
        self.coarse_quantizer = CoarseQuantizer(self.n_clusters)
        self.coarse_quantizer.centroids = np.array(index_data["coarse_centroids"])
        
        # Rebuild fine quantizer
        self.fine_quantizer = FineQuantizer()
        self.fine_quantizer.codebooks = [np.array(codebook) for codebook in index_data["codebooks"]]
        
        # Load data
        self.inverted_file = defaultdict(list)
        for cluster_id, vectors in index_data["inverted_file"].items():
            self.inverted_file[int(cluster_id)] = vectors
        
        self.vectors = {k: np.array(v) for k, v in index_data["vectors"].items()}
        self.residuals = index_data["residuals"]
        self.metadata = index_data["metadata"]
        self.vector_ids = index_data["vector_ids"]
        
        print(f"Index loaded from {filepath}")
        
        return self
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics
        
        Returns:
            Dictionary with index statistics
        """
        total_vectors = len(self.vector_ids)
        
        # Calculate cluster distribution
        cluster_sizes = [len(vectors) for vectors in self.inverted_file.values()]
        
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        std_cluster_size = np.std(cluster_sizes) if cluster_sizes else 0
        
        return {
            "total_vectors": total_vectors,
            "n_clusters": self.n_clusters,
            "n_probes": self.n_probes,
            "avg_cluster_size": float(avg_cluster_size),
            "std_cluster_size": float(std_cluster_size),
            "is_trained": self.is_trained,
            "memory_usage": f"~{total_vectors * 128 * 4} bytes"  # Approximate
        }
    
    def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a vector from the index
        
        Args:
            vector_id: Vector ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if vector_id not in self.vectors:
            return False
        
        # Find and remove from inverted file
        for cluster_id, vectors in self.inverted_file.items():
            self.inverted_file[cluster_id] = [
                v for v in vectors if v["vector_id"] != vector_id
            ]
        
        # Remove from storage
        del self.vectors[vector_id]
        del self.residuals[vector_id]
        del self.metadata[vector_id]
        
        if vector_id in self.vector_ids:
            self.vector_ids.remove(vector_id)
        
        return True
    
    def clear(self):
        """
        Clear all vectors from the index
        """
        self.inverted_file = defaultdict(list)
        self.vectors = {}
        self.residuals = {}
        self.metadata = {}
        self.vector_ids = []
