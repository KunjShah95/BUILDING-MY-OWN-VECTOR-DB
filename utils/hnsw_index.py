import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
import random
import heapq
from collections import defaultdict
import json
import os

@dataclass
class Node:
    """
    Node in the HNSW graph
    """
    node_id: str
    vector: np.ndarray
    neighbors: Dict[int, List[str]] = field(default_factory=dict)
    # neighbors[layer] = list of neighbor node IDs
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False

class HNSWIndex:
    """
    Hierarchical Navigable Small World Index Implementation
    
    This implementation provides:
    - Hierarchical graph structure
    - Greedy best-first search
    - Dynamic index construction
    - Adjustable parameters for recall/speed tradeoff
    """
    
    def __init__(self, m: int = 16, m0: int = None, ef_construction: int = 200,
                 level_mult: float = 1.0 / np.log(2.0),
                 distance_metric: str = "cosine"):
        """
        Initialize HNSW index
        
        Args:
            m: Number of neighbors per node in each layer
            m0: Number of neighbors in layer 0 (default: 2*m)
            ef_construction: Search breadth during construction (higher = better quality, slower)
            level_mult: Multiplier for level generation (controls height distribution)
            distance_metric: Distance metric to use (cosine or euclidean)
        """
        self.m = m
        self.m0 = m0 if m0 is not None else 2 * m
        self.ef_construction = ef_construction
        self.level_mult = level_mult
        self.distance_metric = distance_metric
        
        # Graph structure
        self.graph: Dict[str, Node] = {}
        self.entry_point: Optional[str] = None
        self.max_level: int = 0
        
        # Vector storage
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.total_inserted = 0
        self.total_connections_made = 0
    
    def _calculate_level(self) -> int:
        """
        Randomly determine the level for a new node using exponential distribution
        
        Returns:
            Level for the new node
        """
        # Generate random number between 0 and 1
        r = random.random()
        
        # Calculate level based on exponential distribution
        level = int(-np.log(r) * self.level_mult)
        
        # Limit level to prevent very high levels
        level = min(level, 255)
        
        return level
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        if self.distance_metric != "cosine":
            return vector

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate distance between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Distance between vectors
        """
        if self.distance_metric == "cosine":
            return 1.0 - float(np.dot(vec1, vec2))

        return float(np.linalg.norm(vec1 - vec2))
    
    def _search_layer(self, query_vector: np.ndarray, ef: int, 
                     node_id: str, level: int) -> List[Tuple[str, float]]:
        """
        Best-first search in a single layer
        
        Args:
            query_vector: Query vector
            ef: Number of nearest neighbors to maintain
            node_id: Starting node ID
            level: Layer to search in
            
        Returns:
            List of (node_id, distance) tuples
        """
        visited: Set[str] = set()
        
        # Min-heap for candidates (exploration frontier, closest first)
        candidates: List[Tuple[float, str]] = []
        # Max-heap for results (negative distance for max-heap behavior)
        results: List[Tuple[float, str]] = []
        
        if node_id not in self.graph:
            return []
        
        start_node = self.graph[node_id]
        start_distance = self._distance(query_vector, start_node.vector)
        heapq.heappush(candidates, (start_distance, node_id))
        visited.add(node_id)
        
        while candidates:
            distance, current_id = heapq.heappop(candidates)
            
            # Termination: closest remaining candidate is >= farthest result
            if results:
                farthest_distance = -results[0][0]
                if distance >= farthest_distance:
                    break
            
            # Add to results
            heapq.heappush(results, (-distance, current_id))
            
            # If we have more than ef results, pop farthest
            if len(results) > ef:
                heapq.heappop(results)
            
            # Explore neighbors
            current_node = self.graph[current_id]
            if level in current_node.neighbors:
                for neighbor_id in current_node.neighbors[level]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor = self.graph[neighbor_id]
                        neighbor_dist = self._distance(query_vector, neighbor.vector)
                        heapq.heappush(candidates, (neighbor_dist, neighbor_id))
        
        # Convert results to sorted list
        sorted_results = [(nid, -neg_dist) for neg_dist, nid in results]
        sorted_results.sort(key=lambda x: x[1])
        
        return sorted_results
    
    def _select_neighbors(self, query_vector: np.ndarray, candidates: List[str], 
                         m: int, level: int) -> List[str]:
        """
        Select the best m neighbors from candidates
        
        Args:
            query_vector: Query vector (used for distance calculation)
            candidates: List of candidate node IDs
            m: Number of neighbors to select
            level: Layer level
            
        Returns:
            List of selected neighbor IDs
        """
        if len(candidates) <= m:
            return candidates
        
        # Calculate distances for all candidates
        distances = [(cid, self._distance(query_vector, self.graph[cid].vector)) 
                    for cid in candidates]
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Return closest m
        return [cid for cid, _ in distances[:m]]
    
    def _connect_node(self, node_id: str, level: int, 
                     ep_node_id: str = None) -> str:
        """
        Connect a node at a specific level
        
        Args:
            node_id: Node to connect
            level: Level to connect at
            ep_node_id: Entry point node (default: current entry point)
            
        Returns:
            The entry point used
        """
        if ep_node_id is None:
            ep_node_id = self.entry_point
        
        if ep_node_id is None:
            # This is the first node
            self.entry_point = node_id
            return node_id
        
        # Search for nearest neighbors in this layer
        search_results = self._search_layer(
            self.graph[node_id].vector,
            self.ef_construction,
            ep_node_id,
            level
        )
        
        # Get candidate neighbors (excluding the node itself)
        candidates = [nid for nid, _ in search_results 
                     if nid != node_id and level in self.graph[nid].neighbors]
        
        # Select neighbors
        neighbors = self._select_neighbors(
            self.graph[node_id].vector,
            candidates,
            self.m,
            level
        )
        
        # Add bidirectional connections
        node = self.graph[node_id]
        if level not in node.neighbors:
            node.neighbors[level] = []
        
        for neighbor_id in neighbors:
            neighbor = self.graph[neighbor_id]
            if level not in neighbor.neighbors:
                neighbor.neighbors[level] = []
            
            # Add to current node
            if neighbor_id not in node.neighbors[level]:
                node.neighbors[level].append(neighbor_id)
                self.total_inserted += 1
            
            # Add bidirectional connection
            if node_id not in neighbor.neighbors[level]:
                neighbor.neighbors[level].append(node_id)
                self.total_inserted += 1
        
        return search_results[0][0] if search_results else ep_node_id
    
    def insert(self, vector: List[float], vector_id: str, 
              metadata: Dict[str, Any] = None, level: int = None):
        """
        Insert a vector into the HNSW index
        
        Args:
            vector: Vector to insert
            vector_id: Unique vector ID
            metadata: Optional metadata
            level: Optional level (randomly determined if not provided)
        """
        # Create node
        vector_array = np.array(vector, dtype=np.float32)
        vector_array = self._normalize_vector(vector_array)
        
        # Determine level if not provided
        if level is None:
            level = self._calculate_level()
        
        node = Node(node_id=vector_id, vector=vector_array)
        self.graph[vector_id] = node
        self.vectors[vector_id] = vector_array
        self.metadata[vector_id] = metadata
        self.total_inserted += 1
        
        # Update max level
        if level > self.max_level:
            self.max_level = level
        
        # Start from entry point
        ep = self.entry_point
        
        # Navigate down from max_level to level+1 (ef=1, no connections)
        for l in range(self.max_level, level, -1):
            if ep is not None:
                search_results = self._search_layer(vector_array, 1, ep, l)
                if search_results:
                    ep = search_results[0][0]

        # Connect at each level from level down to 0
        if ep is not None:
            for l in range(level, -1, -1):
                ep = self._connect_node(vector_id, l, ep)
        else:
            self.entry_point = vector_id
            # Initialize neighbor structure for the first node
            self._connect_node(vector_id, 0, self.entry_point)
            if level > 0:
                self._connect_node(vector_id, level, self.entry_point)
    
    def insert_batch(self, vectors: List[Dict[str, Any]]):
        """
        Insert multiple vectors into the index
        
        Args:
            vectors: List of vector dictionaries with 'vector', 'vector_id', 'metadata'
        """
        for vector_data in vectors:
            self.insert(
                vector=vector_data["vector"],
                vector_id=vector_data["vector_id"],
                metadata=vector_data.get("metadata")
            )
    
    def search(self, query_vector: List[float], k: int = 5, 
              ef: int = None, level: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            ef: Search breadth (higher = better recall, slower)
            level: Level to start search from (default: max_level)
            
        Returns:
            List of search results with distances and metadata
        """
        query_array = np.array(query_vector, dtype=np.float32)
        query_array = self._normalize_vector(query_array)
        
        if len(self.graph) == 0:
            return []
        
        if ef is None:
            ef = max(k, 10)  # Default ef
        
        if level is None:
            level = self.max_level
        
        # Start from entry point
        ep = self.entry_point
        
        if ep is None:
            return []
        
        # Navigate down from top level
        for l in range(self.max_level, 0, -1):
            search_results = self._search_layer(query_array, 1, ep, l)
            if search_results:
                ep = search_results[0][0]
        
        # Search at level 0 with ef parameter
        results = self._search_layer(query_array, ef, ep, 0)
        
        # Get top k results
        top_k = results[:k]
        
        # Build response
        response = []
        for node_id, distance in top_k:
            if node_id in self.graph:
                response.append({
                    "vector_id": node_id,
                    "distance": distance,
                    "metadata": self.metadata.get(node_id)
                })
        
        return response
    
    def search_optimized(self, query_vector: List[float], k: int = 5, 
                        ef_search: int = None) -> List[Dict[str, Any]]:
        """
        Optimized search with early termination
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            ef_search: Search breadth (default: max(k, 10))
            
        Returns:
            List of search results
        """
        return self.search(query_vector, k, ef=ef_search)
    
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector from the index
        
        Args:
            vector_id: Vector ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if vector_id not in self.graph:
            return False
        
        node = self.graph[vector_id]
        
        # Remove from all neighbor lists
        for level, neighbors in node.neighbors.items():
            for neighbor_id in neighbors:
                if neighbor_id in self.graph:
                    neighbor = self.graph[neighbor_id]
                    if level in neighbor.neighbors:
                        neighbor.neighbors[level] = [n for n in neighbor.neighbors[level] 
                                                    if n != vector_id]
        
        # Remove from graph
        del self.graph[vector_id]
        del self.vectors[vector_id]
        
        if vector_id in self.metadata:
            del self.metadata[vector_id]
        
        # Update entry point if needed
        if self.entry_point == vector_id:
            if self.graph:
                self.entry_point = next(iter(self.graph.keys()))
            else:
                self.entry_point = None
        
        return True
    
    def get_neighbors(self, vector_id: str, level: int = 0) -> List[str]:
        """
        Get neighbors of a node at a specific level
        
        Args:
            vector_id: Node ID
            level: Level
            
        Returns:
            List of neighbor IDs
        """
        if vector_id in self.graph:
            node = self.graph[vector_id]
            return node.neighbors.get(level, [])
        return []
    
    def get_node_info(self, vector_id: str) -> Dict[str, Any]:
        """
        Get information about a node
        
        Args:
            vector_id: Node ID
            
        Returns:
            Node information dictionary
        """
        if vector_id not in self.graph:
            return {}
        
        node = self.graph[vector_id]
        
        return {
            "node_id": node.node_id,
            "vector_shape": node.vector.shape,
            "neighbors": dict(node.neighbors),
            "total_neighbors": sum(len(neighbors) for neighbors in node.neighbors.values()),
            "metadata": self.metadata.get(vector_id)
        }
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        if len(self.graph) == 0:
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "avg_connections": 0,
                "max_level": 0
            }
        
        # Calculate statistics
        total_edges = sum(
            sum(len(neighbors) for neighbors in node.neighbors.values())
            for node in self.graph.values()
        )
        
        # Count nodes per level
        level_counts = defaultdict(int)
        for node in self.graph.values():
            max_level = max(node.neighbors.keys()) if node.neighbors else 0
            level_counts[max_level] += 1
        
        # Calculate average connections per node
        avg_connections = total_edges / len(self.graph) if self.graph else 0
        
        return {
            "total_nodes": len(self.graph),
            "total_edges": total_edges,
            "avg_connections": avg_connections,
            "max_level": self.max_level,
            "level_distribution": dict(level_counts),
            "total_inserted": self.total_inserted,
            "entry_point": self.entry_point
        }
    
    def save(self, filepath: str, format: str = "json"):
        """
        Save the index to disk
        
        Args:
            filepath: Path to save the index
            format: Serialization format ("json" or "binary")
        """
        if format == "binary":
            import pickle
            graph_data = {}
            for node_id, node in self.graph.items():
                graph_data[node_id] = {
                    "node_id": node.node_id,
                    "vector": node.vector,
                    "neighbors": node.neighbors,
                    "metadata": self.metadata.get(node_id)
                }
            data = {
                "m": self.m,
                "m0": self.m0,
                "ef_construction": self.ef_construction,
                "level_mult": self.level_mult,
                "distance_metric": self.distance_metric,
                "entry_point": self.entry_point,
                "max_level": self.max_level,
                "total_inserted": self.total_inserted,
                "graph": graph_data
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"HNSW Index saved to {filepath} (binary)")
            return
        
        # Default JSON serialization
        def make_serializable(obj):
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            return str(obj)
        
        graph_data = {}
        for node_id, node in self.graph.items():
            metadata = self.metadata.get(node_id)
            serializable_metadata = make_serializable(metadata) if metadata else None
            graph_data[node_id] = {
                "node_id": node.node_id,
                "vector": node.vector.tolist(),
                "neighbors": {str(k): v for k, v in node.neighbors.items()},
                "metadata": serializable_metadata
            }
        
        index_data = {
            "m": self.m,
            "m0": self.m0,
            "ef_construction": self.ef_construction,
            "level_mult": self.level_mult,
            "distance_metric": self.distance_metric,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "total_inserted": self.total_inserted,
            "graph": graph_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"HNSW Index saved to {filepath}")
    
    def load(self, filepath: str) -> 'HNSWIndex':
        """
        Load the index from disk
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            self
        """
        # Detect format from file extension or content
        is_binary = filepath.endswith('.pkl') or filepath.endswith('.pickle')
        
        if is_binary:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        self.m = data["m"]
        self.m0 = data["m0"]
        self.ef_construction = data["ef_construction"]
        self.level_mult = data["level_mult"]
        self.distance_metric = data.get("distance_metric", self.distance_metric)
        self.entry_point = data["entry_point"]
        self.max_level = data["max_level"]
        self.total_inserted = data["total_inserted"]
        
        if is_binary:
            for node_id, node_data in data["graph"].items():
                vector_array = self._normalize_vector(node_data["vector"])
                node = Node(
                    node_id=node_data["node_id"],
                    vector=vector_array
                )
                node.neighbors = node_data["neighbors"]
                self.graph[node_id] = node
                self.vectors[node_id] = node.vector
                self.metadata[node_id] = node_data.get("metadata")
        else:
            for node_id, node_data in data["graph"].items():
                vector_array = np.array(node_data["vector"], dtype=np.float32)
                vector_array = self._normalize_vector(vector_array)
                node = Node(
                    node_id=node_data["node_id"],
                    vector=vector_array
                )
                node.neighbors = {int(k): v for k, v in node_data["neighbors"].items()}
                self.graph[node_id] = node
                self.vectors[node_id] = node.vector
                self.metadata[node_id] = node_data.get("metadata")
        
        print(f"HNSW Index loaded from {filepath}")
        
        return self
    
    def clear(self):
        """
        Clear all data from the index
        """
        self.graph.clear()
        self.vectors.clear()
        self.metadata.clear()
        self.entry_point = None
        self.max_level = 0
        self.total_inserted = 0
        self.total_connections_made = 0
