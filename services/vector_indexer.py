"""
Unified Vector Database Indexer
Supports HNSW and IVF indexing methods with automatic parameter optimization
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
import json
import os
from datetime import datetime
import time

from utils.hnsw_index import HNSWIndex
from utils.ivf_index import IVFIndex
from utils.optimization import (
    OptimizedDistanceCalculator, 
    VectorBatchProcessor,
    MemoryOptimizer,
    QueryOptimizer
)
from config.settings import get_settings


class IndexMethod(str, Enum):
    """Supported indexing methods"""
    HNSW = "hnsw"
    IVF = "ivf"
    HYBRID = "hybrid"  # Uses both HNSW and IVF


class VectorIndexerConfig:
    """Configuration for vector indexer"""
    
    def __init__(self,
                 method: Union[str, IndexMethod] = IndexMethod.HNSW,
                 num_vectors: int = 10000,
                 vector_dim: int = 128,
                 recall_target: float = 0.95,
                 speed_priority: bool = False):
        """
        Initialize indexer configuration
        
        Args:
            method: Indexing method (hnsw, ivf, hybrid)
            num_vectors: Expected number of vectors
            vector_dim: Vector dimensionality
            recall_target: Target recall for searches (0-1)
            speed_priority: If True, optimize for speed over accuracy
        """
        self.method = IndexMethod(method) if isinstance(method, str) else method
        self.num_vectors = num_vectors
        self.vector_dim = vector_dim
        self.recall_target = recall_target
        self.speed_priority = speed_priority
        
        # Get optimized parameters
        self.hnsw_params = self._get_hnsw_params()
        self.ivf_params = self._get_ivf_params()
    
    def _get_hnsw_params(self) -> Dict[str, int]:
        """Get optimized HNSW parameters"""
        settings = get_settings()
        
        if self.speed_priority:
            # Optimize for speed
            m = max(8, settings.DEFAULT_M // 2)
            ef_construction = max(100, settings.DEFAULT_EF_CONSTRUCTION // 2)
            ef_search = 20
        else:
            # Use default optimized values
            m = settings.DEFAULT_M
            ef_construction = settings.DEFAULT_EF_CONSTRUCTION
            ef_search = settings.DEFAULT_EF_SEARCH
        
        return {
            "m": m,
            "m0": 2 * m,
            "ef_construction": ef_construction,
            "ef_search": ef_search
        }
    
    def _get_ivf_params(self) -> Dict[str, int]:
        """Get optimized IVF parameters"""
        settings = get_settings()
        n_clusters = min(int(np.sqrt(self.num_vectors)), 10000)
        n_probes = min(int(np.sqrt(self.num_vectors) / 10), 100)
        
        if self.speed_priority:
            n_probes = max(1, n_probes // 2)
        
        return {
            "n_clusters": n_clusters,
            "n_probes": n_probes
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "method": self.method.value,
            "num_vectors": self.num_vectors,
            "vector_dim": self.vector_dim,
            "recall_target": self.recall_target,
            "speed_priority": self.speed_priority,
            "hnsw_params": self.hnsw_params,
            "ivf_params": self.ivf_params
        }


class VectorIndexer:
    """
    Unified vector indexer supporting multiple indexing methods
    """
    
    def __init__(self, config: VectorIndexerConfig = None):
        """
        Initialize vector indexer
        
        Args:
            config: Indexer configuration
        """
        self.config = config or VectorIndexerConfig()
        
        # Index storage
        self.hnsw_index: Optional[HNSWIndex] = None
        self.ivf_index: Optional[IVFIndex] = None
        
        # Metadata
        self.vector_count = 0
        self.created_at = datetime.now()
        self.last_updated = None
        
        # Statistics
        self.stats = {
            "total_indexed": 0,
            "total_searched": 0,
            "total_search_time": 0.0,
            "avg_search_time": 0.0
        }
        
        # Batch processor
        self.batch_processor = VectorBatchProcessor(batch_size=1000)
    
    def create_index(self, vectors: List[List[float]], 
                    metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create index from vectors
        
        Args:
            vectors: List of vectors to index
            metadata: Optional metadata for vectors
            
        Returns:
            Creation result with statistics
        """
        try:
            start_time = time.time()
            vectors_array = np.array(vectors)
            self.vector_count = len(vectors)
            
            results = {
                "success": True,
                "method": self.config.method.value,
                "vectors_indexed": self.vector_count,
                "config": self.config.to_dict()
            }
            
            # Create HNSW index
            if self.config.method in [IndexMethod.HNSW, IndexMethod.HYBRID]:
                hnsw_result = self._create_hnsw_index(vectors, metadata)
                results["hnsw"] = hnsw_result
            
            # Create IVF index
            if self.config.method in [IndexMethod.IVF, IndexMethod.HYBRID]:
                ivf_result = self._create_ivf_index(vectors_array, metadata)
                results["ivf"] = ivf_result
            
            elapsed = time.time() - start_time
            results["creation_time"] = elapsed
            
            self.last_updated = datetime.now()
            self.stats["total_indexed"] += self.vector_count
            
            return results
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_hnsw_index(self, vectors: List[List[float]], 
                          metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create HNSW index"""
        try:
            start_time = time.time()
            params = self.config.hnsw_params
            
            self.hnsw_index = HNSWIndex(
                m=params["m"],
                m0=params["m0"],
                ef_construction=params["ef_construction"]
            )
            
            # Insert vectors
            for i, vector in enumerate(vectors):
                vector_id = f"vec_{i}"
                meta = metadata[i] if metadata else None
                self.hnsw_index.insert(vector, vector_id, meta)
            
            elapsed = time.time() - start_time
            
            # Get statistics
            stats = self.hnsw_index.get_graph_stats()
            
            return {
                "success": True,
                "creation_time": elapsed,
                "parameters": params,
                "graph_stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_ivf_index(self, vectors: np.ndarray,
                         metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create IVF index"""
        try:
            start_time = time.time()
            params = self.config.ivf_params
            
            self.ivf_index = IVFIndex(n_clusters=params["n_clusters"])
            self.ivf_index.build(vectors)
            
            elapsed = time.time() - start_time
            
            return {
                "success": True,
                "creation_time": elapsed,
                "parameters": params,
                "num_clusters": params["n_clusters"]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search(self, query_vector: List[float], k: int = 5,
              method: Optional[str] = None,
              ef_search: Optional[int] = None) -> Dict[str, Any]:
        """
        Search using configured method
        
        Args:
            query_vector: Query vector
            k: Number of results
            method: Override search method (hnsw, ivf, hybrid)
            ef_search: HNSW search parameter
            
        Returns:
            Search results
        """
        try:
            search_method = method or self.config.method.value
            start_time = time.time()
            
            if search_method == "hnsw":
                results = self._search_hnsw(query_vector, k, ef_search)
            elif search_method == "ivf":
                results = self._search_ivf(query_vector, k)
            elif search_method == "hybrid":
                results = self._search_hybrid(query_vector, k, ef_search)
            else:
                return {"success": False, "error": f"Unknown method: {search_method}"}
            
            elapsed = time.time() - start_time
            
            # Update statistics
            self.stats["total_searched"] += 1
            self.stats["total_search_time"] += elapsed
            self.stats["avg_search_time"] = self.stats["total_search_time"] / max(
                self.stats["total_searched"], 1
            )
            
            results["search_time"] = elapsed
            results["method"] = search_method
            
            return results
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _search_hnsw(self, query_vector: List[float], k: int,
                    ef_search: Optional[int] = None) -> Dict[str, Any]:
        """Search using HNSW"""
        if not self.hnsw_index:
            return {"success": False, "error": "HNSW index not created"}
        
        if ef_search is None:
            ef_search = self.config.hnsw_params["ef_search"]
        
        results = self.hnsw_index.search(query_vector, k, ef=ef_search)
        
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "ef_search": ef_search
        }
    
    def _search_ivf(self, query_vector: List[float], k: int) -> Dict[str, Any]:
        """Search using IVF"""
        if not self.ivf_index:
            return {"success": False, "error": "IVF index not created"}
        
        query_array = np.array(query_vector)
        results = self.ivf_index.search(query_array, k, n_probes=self.config.ivf_params["n_probes"])
        
        return {
            "success": True,
            "results": results,
            "count": len(results),
            "n_probes": self.config.ivf_params["n_probes"]
        }
    
    def _search_hybrid(self, query_vector: List[float], k: int,
                      ef_search: Optional[int] = None) -> Dict[str, Any]:
        """Search using both HNSW and IVF, combine results"""
        hnsw_result = self._search_hnsw(query_vector, k, ef_search)
        ivf_result = self._search_ivf(query_vector, k)
        
        if not hnsw_result["success"] or not ivf_result["success"]:
            return {
                "success": False,
                "error": "One or both indexes not available"
            }
        
        # Combine and deduplicate results
        combined = {}
        for res in hnsw_result["results"] + ivf_result["results"]:
            vid = res.get("vector_id") or res.get("id")
            if vid not in combined:
                combined[vid] = res
        
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x.get("distance", float("inf"))
        )[:k]
        
        return {
            "success": True,
            "results": sorted_results,
            "count": len(sorted_results),
            "hnsw_count": len(hnsw_result["results"]),
            "ivf_count": len(ivf_result["results"]),
            "combined_count": len(combined)
        }
    
    def batch_search(self, query_vectors: List[List[float]], k: int = 5,
                    method: Optional[str] = None) -> Dict[str, Any]:
        """
        Batch search multiple queries
        
        Args:
            query_vectors: List of query vectors
            k: Number of results per query
            method: Search method
            
        Returns:
            Batch search results with performance metrics
        """
        try:
            start_time = time.time()
            batch_results = []
            
            for query_vec in query_vectors:
                result = self.search(query_vec, k, method)
                batch_results.append(result)
            
            elapsed = time.time() - start_time
            
            return {
                "success": True,
                "total_queries": len(query_vectors),
                "results": batch_results,
                "batch_time": elapsed,
                "avg_query_time": elapsed / len(query_vectors) if query_vectors else 0,
                "queries_per_second": len(query_vectors) / elapsed if elapsed > 0 else 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_index(self, path: str = None) -> Dict[str, Any]:
        """
        Save index to disk
        
        Args:
            path: Base path for saving
            
        Returns:
            Save result
        """
        try:
            if path is None:
                path = f"indexes/index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            os.makedirs(path, exist_ok=True)
            
            results = {}
            
            # Save HNSW
            if self.hnsw_index:
                hnsw_path = os.path.join(path, "hnsw.pkl")
                self.hnsw_index.save(hnsw_path)
                results["hnsw"] = {"saved": True, "path": hnsw_path}
            
            # Save IVF
            if self.ivf_index:
                ivf_path = os.path.join(path, "ivf.pkl")
                self.ivf_index.save(ivf_path)
                results["ivf"] = {"saved": True, "path": ivf_path}
            
            # Save metadata
            metadata = {
                "config": self.config.to_dict(),
                "stats": self.stats,
                "created_at": self.created_at.isoformat(),
                "last_updated": self.last_updated.isoformat() if self.last_updated else None,
                "vector_count": self.vector_count
            }
            
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            results["metadata"] = {"saved": True, "path": metadata_path}
            
            return {
                "success": True,
                "index_path": path,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_index(self, path: str) -> Dict[str, Any]:
        """
        Load index from disk
        
        Args:
            path: Path to index directory
            
        Returns:
            Load result
        """
        try:
            results = {}
            
            # Load metadata first
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.config = VectorIndexerConfig(**metadata["config"])
                    self.stats = metadata["stats"]
                    self.vector_count = metadata["vector_count"]
            
            # Load HNSW
            hnsw_path = os.path.join(path, "hnsw.pkl")
            if os.path.exists(hnsw_path):
                self.hnsw_index = HNSWIndex()
                self.hnsw_index.load(hnsw_path)
                results["hnsw"] = {"loaded": True}
            
            # Load IVF
            ivf_path = os.path.join(path, "ivf.pkl")
            if os.path.exists(ivf_path):
                self.ivf_index = IVFIndex()
                self.ivf_index.load(ivf_path)
                results["ivf"] = {"loaded": True}
            
            return {
                "success": True,
                "results": results,
                "vector_count": self.vector_count
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics"""
        return {
            "config": self.config.to_dict(),
            "vector_count": self.vector_count,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "statistics": self.stats,
            "hnsw_available": self.hnsw_index is not None,
            "ivf_available": self.ivf_index is not None
        }
    
    def optimize_parameters(self, eval_vectors: List[List[float]],
                           eval_queries: List[List[float]],
                           ground_truth: List[List[int]]) -> Dict[str, Any]:
        """
        Optimize index parameters based on evaluation dataset
        
        Args:
            eval_vectors: Evaluation vectors (for ground truth)
            eval_queries: Evaluation queries
            ground_truth: Ground truth results (k nearest neighbors per query)
            
        Returns:
            Optimization results with recommendations
        """
        try:
            results = {
                "current_config": self.config.to_dict()
            }
            
            # Test different parameters
            test_configs = []
            for m in [16, 24, 32]:
                for ef_c in [200, 300, 400]:
                    test_configs.append((m, ef_c))
            
            best_config = None
            best_score = 0
            
            for m, ef_c in test_configs:
                # Create test indexer
                test_config = VectorIndexerConfig(
                    method=IndexMethod.HNSW,
                    num_vectors=len(eval_vectors),
                    recall_target=self.config.recall_target
                )
                test_config.hnsw_params = {
                    "m": m,
                    "m0": 2 * m,
                    "ef_construction": ef_c,
                    "ef_search": 50
                }
                
                test_indexer = VectorIndexer(test_config)
                test_indexer.create_index(eval_vectors)
                
                # Evaluate
                recall_sum = 0
                for i, query in enumerate(eval_queries):
                    result = test_indexer.search(query, k=10)
                    if result["success"]:
                        result_ids = [r["vector_id"] for r in result["results"]]
                        gt = ground_truth[i]
                        recall = len(set(result_ids) & set(gt)) / len(gt)
                        recall_sum += recall
                
                avg_recall = recall_sum / len(eval_queries)
                score = avg_recall
                
                if score > best_score:
                    best_score = score
                    best_config = (m, ef_c)
            
            results["best_config"] = {
                "m": best_config[0],
                "ef_construction": best_config[1],
                "score": best_score
            }
            
            return {
                "success": True,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
