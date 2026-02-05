import numpy as np
from typing import List, Dict, Any, Optional
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# Try to import numba for JIT compilation, fall back to standard Python if not available
try:
    import numba
    from numba import jit as numba_jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define prange as range if numba is not available
    prange = range
    # Define a no-op jit decorator
    def numba_jit(*args, **kwargs):
        return lambda f: f


def _jit_decorator(*args, **kwargs):
    """
    Conditional JIT decorator that applies Numba JIT compilation if available.
    Falls back to identity decorator if Numba is not installed.
    """
    if NUMBA_AVAILABLE:
        return numba_jit(*args, **kwargs)
    else:
        return lambda f: f


class OptimizedDistanceCalculator:
    """
    Optimized distance calculations using Numba JIT compilation (if available)
    Falls back to standard NumPy if Numba is not installed
    """
    
    @staticmethod
    def cosine_distance_batch(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Calculate cosine distance for all vectors (parallelized)
        
        Args:
            vectors: Array of shape (n, d)
            query: Query vector of shape (d,)
            
        Returns:
            Array of distances of shape (n,)
        """
        n = vectors.shape[0]
        distances = np.empty(n, dtype=np.float64)
        
        # Calculate query norm
        query_norm = np.sqrt(np.sum(query ** 2))
        if query_norm == 0:
            query_norm = 1.0
        
        for i in prange(n):
            vector = vectors[i]
            dot_product = np.dot(vector, query)
            vector_norm = np.sqrt(np.sum(vector ** 2))
            if vector_norm == 0:
                vector_norm = 1.0
            
            # Cosine distance = 1 - cosine similarity
            cosine_sim = dot_product / (vector_norm * query_norm)
            distances[i] = 1.0 - cosine_sim
        
        return distances
    
    @staticmethod
    @_jit_decorator(nopython=NUMBA_AVAILABLE, parallel=NUMBA_AVAILABLE)
    def euclidean_distance_batch(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance for all vectors (parallelized)
        
        Args:
            vectors: Array of shape (n, d)
            query: Query vector of shape (d,)
            
        Returns:
            Array of distances of shape (n,)
        """
        n = vectors.shape[0]
        distances = np.empty(n, dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            # Numba version with prange
            for i in prange(n):
                diff = vectors[i] - query
                distances[i] = np.sqrt(np.sum(diff ** 2))
        else:
            # Standard NumPy version
            for i in range(n):
                diff = vectors[i] - query
                distances[i] = np.sqrt(np.sum(diff ** 2))
        
        return distances
    
    @staticmethod
    @_jit_decorator(nopython=NUMBA_AVAILABLE)
    def top_k_indices(distances: np.ndarray, k: int) -> np.ndarray:
        """
        Find top k indices using partial sort (faster than full sort)
        
        Args:
            distances: Distance array
            k: Number of top results
            
        Returns:
            Indices of top k smallest distances
        """
        return np.argpartition(distances, k)[:k]


class VectorBatchProcessor:
    """
    Batch processing for efficient vector operations
    """
    
    def __init__(self, batch_size: int = 1000):
        """
        Initialize batch processor
        
        Args:
            batch_size: Batch size for processing
        """
        self.batch_size = batch_size
        
    def process_in_batches(self,
                          vectors: List[List[float]],
                          process_func: callable,
                          **kwargs) -> List[Any]:
        """
        Process vectors in batches using multiple threads
        
        Args:
            vectors: List of vectors
            process_func: Function to process each batch
            **kwargs: Additional arguments for process_func
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            batch_result = process_func(batch, **kwargs)
            results.extend(batch_result)
            
            # Progress indicator
            progress = min(i + self.batch_size, len(vectors))
            print(f"Processed {progress}/{len(vectors)} vectors ({100*progress/len(vectors):.1f}%)")
        
        return results
    
    def parallel_insert(self,
                       vectors: List[List[float]],
                       insert_func: callable,
                       max_workers: int = None) -> List[Any]:
        """
        Insert vectors in parallel using ThreadPoolExecutor
        
        Args:
            vectors: List of vectors
            insert_func: Function to insert each vector
            max_workers: Maximum number of threads
            
        Returns:
            List of results
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(insert_func, vectors))
        
        return results


class MemoryOptimizer:
    """
    Memory optimization utilities for large vector datasets
    """
    
    @staticmethod
    def quantize_vectors(vectors: np.ndarray, 
                        dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Quantize vectors to smaller data type
        
        Args:
            vectors: Input vectors
            dtype: Target data type
            
        Returns:
            Quantized vectors
        """
        return vectors.astype(dtype)
    
    @staticmethod
    def normalize_for_half_precision(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for safe half-precision conversion
        
        Args:
            vectors: Input vectors
            
        Returns:
            Normalized vectors
        """
        # Ensure values are within float16 range
        max_val = np.max(np.abs(vectors))
        if max_val > 65504:  # max float16
            scale = 65504 / max_val
            vectors = vectors * scale
        
        return vectors.astype(np.float16)
    
    @staticmethod
    def estimate_memory_usage(num_vectors: int, 
                            dimension: int,
                            dtype: np.dtype = np.float32) -> Dict[str, float]:
        """
        Estimate memory usage for vector storage
        
        Args:
            num_vectors: Number of vectors
            dimension: Vector dimension
            dtype: Data type
            
        Returns:
            Memory usage estimates
        }
        """
        bytes_per_element = np.dtype(dtype).itemsize
        total_bytes = num_vectors * dimension * bytes_per_element
        
        return {
            "bytes": total_bytes,
            "kb": total_bytes / 1024,
            "mb": total_bytes / (1024 * 1024),
            "gb": total_bytes / (1024 * 1024 * 1024)
        }
    
    @staticmethod
    def calculate_compression_ratio(original: np.ndarray,
                                   compressed: np.ndarray) -> float:
        """
        Calculate compression ratio
        
        Args:
            original: Original array
            compressed: Compressed array
            
        Returns:
            Compression ratio
        """
        original_size = original.nbytes
        compressed_size = compressed.nbytes
        
        return original_size / compressed_size if compressed_size > 0 else 0


class QueryOptimizer:
    """
    Query optimization utilities
    """
    
    @staticmethod
    def optimize_ef_search(recall_target: float = 0.95,
                          base_ef: int = 10) -> int:
        """
        Optimize ef_search parameter based on recall target
        
        Args:
            recall_target: Target recall (0-1)
            base_ef: Base ef value
            
        Returns:
            Optimized ef_search value
        """
        # Simple heuristic: increase ef for higher recall
        if recall_target >= 0.99:
            return base_ef * 10
        elif recall_target >= 0.95:
            return base_ef * 5
        elif recall_target >= 0.90:
            return base_ef * 3
        else:
            return base_ef
    
    @staticmethod
    def estimate_optimal_m(database_size: int) -> int:
        """
        Estimate optimal HNSW m parameter
        
        Args:
            database_size: Number of vectors
            
        Returns:
            Recommended m value
        """
        if database_size < 1000:
            return 4
        elif database_size < 10000:
            return 8
        elif database_size < 100000:
            return 16
        elif database_size < 1000000:
            return 32
        else:
            return 64
    
    @staticmethod
    def suggest_index_parameters(database_size: int,
                                recall_target: float = 0.95) -> Dict[str, Any]:
        """
        Suggest optimal index parameters
        
        Args:
            database_size: Number of vectors
            recall_target: Target recall
            
        Returns:
            Dictionary of recommended parameters
        """
        m = QueryOptimizer.estimate_optimal_m(database_size)
        ef_construction = min(200 + int(database_size / 1000), 400)
        ef_search = QueryOptimizer.optimize_ef_search(recall_target)
        
        return {
            "hnsw": {
                "m": m,
                "m0": 2 * m,
                "ef_construction": ef_construction,
                "ef_search": ef_search
            },
            "ivf": {
                "n_clusters": min(int(np.sqrt(database_size)), 10000),
                "n_probes": min(int(np.sqrt(database_size) / 10), 100)
            }
        }
