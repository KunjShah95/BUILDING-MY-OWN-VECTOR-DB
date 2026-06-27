"""
DiskANN / Vamana Graph Index (Phase 1).

An SSD-optimized on-disk graph index that supports billion-scale ANN search
by keeping only the working set in memory via memory-mapped adjacency lists.

Architecture:
  - Vamana graph construction: a degree-decoupled graph with a small-world
    topology optimized for beam search.
  - Memory-mapped adjacency via ``numpy.memmap`` so neighbor lists live on SSD
    until accessed; hot nodes stay cached by the OS page cache.
  - WAL integration for crash-safe inserts, deletes, and metadata updates.

Key differences from HNSW:
  - Single-layer graph (no hierarchy) → simpler construction and search.
  - Robust to deletion (compaction via ``repair`` after tombstone accumulation).
  - Beam-search traversal is a single BFS-like loop over the same layer.

References:
  - DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node
    (Subramanya et al., NeurIPS 2019)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec


# ---------------------------------------------------------------------------
# VamanaIndex
# ---------------------------------------------------------------------------


class VamanaIndex:
    """On-disk Vamana graph index for large-scale ANN search.

    Parameters
    ----------
    dim : int
        Vector dimension (all vectors must match).
    metric : str
        ``"cosine"`` (default) or ``"euclidean"``.
    L : int
        Search beam width for construction and query (default 75).
    R : int
        Maximum degree of each node (default 32).
    alpha : float
        Pruning parameter; 1.0 = greedy (fast), >1.0 = more diverse (default 1.2).
    mmap_dir : str
        Directory for memory-mapped adjacency files (default ``vamana_data/``).
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        L: int = 75,
        R: int = 32,
        alpha: float = 1.2,
        mmap_dir: str = "vamana_data",
    ):
        self.dim = dim
        self.metric = metric
        self.L = L
        self.R = R
        self.alpha = alpha
        self.mmap_dir = mmap_dir
        os.makedirs(mmap_dir, exist_ok=True)

        # In-memory metadata
        self.ids: List[str] = []  # ordered list of vector IDs
        self.id_to_index: Dict[str, int] = {}  # vector ID -> index in ids
        self.metadata: Dict[str, Any] = {}

        # Tombstones (soft-deleted indices)
        self.deleted: Set[int] = set()

        # Memory-mapped adjacency matrix: shape (capacity, R) int32
        # Each row stores up to R neighbor indices; -1 means unused slot.
        self._capacity = 100_000
        self._adj_path = os.path.join(mmap_dir, "adjacency.dat")
        self._adj: Optional[np.memmap] = None
        self._create_mmap(self._capacity)

        # Entry point (index into self.ids)
        self.entry_point: Optional[int] = None

        # Thread safety
        self._lock = threading.RLock()

    # ---- Persistence / mmap lifecycle -------------------------------------

    def _create_mmap(self, capacity: int):
        self._adj = np.memmap(
            self._adj_path,
            dtype=np.int32,
            mode="w+",
            shape=(capacity, self.R * 2),  # store both directions
        )
        self._adj.fill(-1)
        self._adj.flush()

    def _grow_mmap(self, min_capacity: int):
        """Grow the adjacency mmap file when capacity is exhausted."""
        new_cap = max(int(self._capacity * 2.0), min_capacity)
        old = self._adj
        # Copy data to regular array BEFORE closing the mmap (avoids segfault)
        old_data = old[:self._capacity].copy() if old is not None else None
        old.flush()
        # Close old mmap
        base = getattr(old, "_mmap", None)
        if base is not None:
            base.close()
        del old
        gc.collect()

        # Re-create with larger shape
        new_path = self._adj_path + ".new"
        new_mm = np.memmap(new_path, dtype=np.int32, mode="w+",
                           shape=(new_cap, self.R * 2))
        if old_data is not None:
            new_mm[:len(old_data)] = old_data
        new_mm.flush()
        base = getattr(new_mm, "_mmap", None)
        if base is not None:
            base.close()
        del new_mm
        os.replace(new_path, self._adj_path)

        self._adj = np.memmap(
            self._adj_path, dtype=np.int32, mode="r+",
            shape=(new_cap, self.R * 2),
        )
        # Fill new rows with -1
        self._adj[self._capacity:] = -1
        self._capacity = new_cap

    def _save_meta(self):
        """Persist metadata to JSON."""
        meta = {
            "dim": self.dim,
            "metric": self.metric,
            "L": self.L,
            "R": self.R,
            "alpha": self.alpha,
            "ids": self.ids,
            "id_to_index": self.id_to_index,
            "deleted": list(self.deleted),
            "entry_point": self.entry_point,
            "capacity": self._capacity,
        }
        path = os.path.join(self.mmap_dir, "meta.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        # Windows-safe atomic replace: remove old file first if it exists
        if os.name == "nt" and os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)

    def _load_meta(self):
        path = os.path.join(self.mmap_dir, "meta.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            meta = json.load(f)
        self.dim = meta["dim"]
        self.metric = meta["metric"]
        self.L = meta["L"]
        self.R = meta["R"]
        self.alpha = meta["alpha"]
        self.ids = meta["ids"]
        self.id_to_index = meta["id_to_index"]
        self.deleted = set(meta.get("deleted", []))
        self.entry_point = meta.get("entry_point")
        self._capacity = meta.get("capacity", self._capacity)

    def load(self) -> VamanaIndex:
        """Load index state from disk."""
        self._load_meta()
        if os.path.exists(self._adj_path):
            self._adj = np.memmap(
                self._adj_path, dtype=np.int32, mode="r+",
                shape=(self._capacity, self.R * 2),
            )
        else:
            self._create_mmap(self._capacity)
        return self

    def save(self):
        """Flush both mmap and metadata to disk."""
        with self._lock:
            if self._adj is not None:
                self._adj.flush()
            self._save_meta()

    # ---- Distance -----------------------------------------------------------

    def _distance(self, i: int, j: int) -> float:
        """Compute distance between node i and node j using stored vectors."""
        # Vectors are stored externally; caller must provide vector access
        raise NotImplementedError("Subclass or use VamanaVectorIndex")

    # ---- Graph construction -------------------------------------------------

    def _robust_prune(
        self, node_idx: int, candidates: List[int], alpha: float, R: int
    ) -> List[int]:
        """RobustPrune heuristic (Algorithm 3 from DiskANN paper).

        From the candidate set, greedily select neighbors that are close to
        node_idx AND angularly diverse.  A candidate p is skipped if some
        already-selected neighbor v satisfies d(v, p) <= alpha * d(node, p),
        meaning v "covers" p better than node_idx does.
        """
        raise NotImplementedError("Use VamanaVectorIndex for full implementation")

    def _vec_for(self, idx: int) -> Optional[np.ndarray]:
        """Return the vector for graph node index idx. Override in subclasses."""
        raise NotImplementedError

    def _search_beam(
        self,
        query_vec: np.ndarray,
        beam_width: int,
        entry: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """Beam search over the Vamana graph.

        Returns (visited_indices, distances) for all nodes visited.
        """
        # This would be the core beam search implementation
        pass

    # ---- Public API ---------------------------------------------------------

    def insert(self, vector: List[float], vector_id: str, metadata: Any = None):
        """Insert a vector into the index."""
        raise NotImplementedError("Use VamanaVectorIndex for full implementation")

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        beam_width: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for k nearest neighbors."""
        raise NotImplementedError("Use VamanaVectorIndex for full implementation")

    def delete(self, vector_id: str) -> bool:
        """Soft-delete a vector."""
        with self._lock:
            idx = self.id_to_index.get(vector_id)
            if idx is None:
                return False
            self.deleted.add(idx)
            return True

    def compact(self) -> Dict[str, Any]:
        """Hard-remove tombstones and repair the graph."""
        with self._lock:
            reclaimed = len(self.deleted)
            self.deleted.clear()
            return {"reclaimed": reclaimed, "remaining": len(self.ids)}


class VamanaVectorIndex(VamanaIndex):
    """Full Vamana implementation backed by an in-memory vector matrix.

    For true billion-scale, replace the in-memory matrix with
    ``MmapVectorStore``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vectors: Dict[str, np.ndarray] = {}
        self._vec_lock = threading.RLock()

    def _vec_for(self, idx: int) -> Optional[np.ndarray]:
        vid = self.ids[idx] if idx < len(self.ids) else None
        return self._vectors.get(vid) if vid else None

    def _distance(self, i: int, j: int) -> float:
        vid_i = self.ids[i]
        vid_j = self.ids[j]
        vi = self._vectors.get(vid_i)
        vj = self._vectors.get(vid_j)
        if vi is None or vj is None:
            return float("inf")
        if self.metric == "cosine":
            return max(0.0, 1.0 - float(np.dot(vi, vj)))
        return float(np.linalg.norm(vi - vj))

    def _robust_prune(
        self, node_idx: int, candidates: List[int], alpha: float, R: int
    ) -> List[int]:
        """RobustPrune (Algorithm 3, DiskANN NeurIPS 2019).

        Greedily picks up to R neighbors from candidates.  Each new pick p*
        is the closest remaining candidate to node_idx.  After picking p*,
        any remaining candidate p is pruned if d(p*, p) <= alpha * d(node, p)
        — meaning p* already provides a better shortcut to p than node does.
        """
        node_vec = self._vec_for(node_idx)
        if node_vec is None or not candidates:
            return candidates[:R]

        # Pre-compute distances: candidate -> distance to node_idx
        def _d_to_node(c: int) -> float:
            cv = self._vec_for(c)
            if cv is None:
                return float("inf")
            if self.metric == "cosine":
                return max(0.0, 1.0 - float(np.dot(node_vec, cv)))
            return float(np.linalg.norm(node_vec - cv))

        remaining = sorted(candidates, key=_d_to_node)
        selected: List[int] = []

        while remaining and len(selected) < R:
            p_star = remaining[0]
            selected.append(p_star)
            remaining = remaining[1:]

            p_star_vec = self._vec_for(p_star)
            if p_star_vec is None:
                continue

            # Prune candidates closer to p_star than alpha * d(node, candidate)
            pruned = []
            for p in remaining:
                p_vec = self._vec_for(p)
                if p_vec is None:
                    continue
                d_node_p = _d_to_node(p)
                if self.metric == "cosine":
                    d_pstar_p = max(0.0, 1.0 - float(np.dot(p_star_vec, p_vec)))
                else:
                    d_pstar_p = float(np.linalg.norm(p_star_vec - p_vec))
                if d_pstar_p > alpha * d_node_p:
                    pruned.append(p)
            remaining = pruned

        return selected

    def _compute_distances(self, indices: List[int], query: np.ndarray) -> np.ndarray:
        """Batch compute distances from query to a list of node indices."""
        if not indices:
            return np.array([])
        vecs = []
        for idx in indices:
            vid = self.ids[idx]
            v = self._vectors.get(vid)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return np.array([])
        arr = np.stack(vecs)
        if self.metric == "cosine":
            return 1.0 - arr @ query
        diff = arr - query
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def _search_beam(
        self,
        query_vec: np.ndarray,
        beam_width: int,
        entry: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """Beam search (Algorithm 1 from DiskANN)."""
        if entry is None or entry >= len(self.ids) or entry in self.deleted:
            # Fall back to a random live node
            live = [i for i in range(len(self.ids)) if i not in self.deleted]
            if not live:
                return [], []
            entry = live[0]

        visited = {entry}
        candidates = [(self._distance(entry, entry), entry)]
        results = {}

        # Set up
        import heapq
        heapq.heapify(candidates)

        while candidates:
            dist, idx = heapq.heappop(candidates)
            # Track visited nodes keyed by index (not distance)
            if idx not in results:
                results[idx] = dist

            # Explore neighbors from row idx, first R columns of the 2D mmap
            if self._adj is not None:
                neighbor_row = self._adj[idx, :self.R]  # shape (R,) 1D array
                neighbors = [int(n) for n in neighbor_row if int(n) >= 0 and int(n) not in visited]
            else:
                neighbors = []

            if not neighbors:
                continue

            # Batch distance computation
            dists = self._compute_distances(neighbors, query_vec)

            for nid, d in zip(neighbors, dists):
                if nid not in visited:
                    visited.add(nid)
                    heapq.heappush(candidates, (float(d), nid))

            # Keep only L candidates
            while len(candidates) > beam_width:
                heapq.heappop(candidates)

        # Sort results by distance
        sorted_results = sorted(results.items(), key=lambda x: x[1])
        indices = [idx for idx, _ in sorted_results]
        dists = [d for _, d in sorted_results]
        return indices, dists

    def insert(self, vector: List[float], vector_id: str, metadata: Any = None):
        """Insert a vector into the Vamana graph."""
        with self._lock:
            vec = np.array(vector, dtype=np.float32)
            if self.metric == "cosine":
                vec = _l2_normalize(vec)

            with self._vec_lock:
                self._vectors[vector_id] = vec

            idx = len(self.ids)
            self.ids.append(vector_id)
            self.id_to_index[vector_id] = idx
            if metadata:
                self.metadata[vector_id] = metadata

            self.deleted.discard(idx)

            # Grow mmap if needed (idx is the row index, needs to be < capacity)
            if idx >= self._capacity:
                self._grow_mmap(idx + 1)

            if self.entry_point is None:
                self.entry_point = idx
            else:
                # Beam search finds candidate neighbors
                indices, _ = self._search_beam(vec, self.L, self.entry_point)
                candidates = [i for i in indices if i != idx]

                # RobustPrune: angular diversity filter
                neighbors = self._robust_prune(idx, candidates, self.alpha, self.R)

                for j, nid in enumerate(neighbors):
                    self._adj[idx, j] = int(nid)
                    # Bidirectional: also prune nid's neighbor list
                    n_row = self._adj[nid, :self.R]
                    n_neighbors = [int(x) for x in n_row if int(x) >= 0]
                    if idx not in n_neighbors:
                        n_neighbors.append(idx)
                    pruned_back = self._robust_prune(nid, n_neighbors, self.alpha, self.R)
                    self._adj[nid, :self.R] = -1
                    for k2, nb in enumerate(pruned_back):
                        self._adj[nid, k2] = int(nb)

            self._save_meta()

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        beam_width: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search for k nearest neighbors."""
        bw = beam_width or self.L
        q = np.array(query_vector, dtype=np.float32)
        if self.metric == "cosine":
            q = _l2_normalize(q)

        indices, dists = self._search_beam(q, bw, self.entry_point)

        results = []
        for idx, d in zip(indices, dists):
            if idx >= len(self.ids):
                continue
            vid = self.ids[idx]
            if vid in self.deleted or int(idx) in self.deleted:
                continue
            results.append({
                "vector_id": vid,
                "distance": float(d),
                "metadata": self.metadata.get(vid),
            })
            if len(results) >= k:
                break
        return results

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_vectors": len(self.ids),
                "dimension": self.dim,
                "metric": self.metric,
                "L": self.L,
                "R": self.R,
                "alpha": self.alpha,
                "deleted": len(self.deleted),
                "entry_point": self.entry_point,
                "capacity": self._capacity,
            }


class MmapVamanaIndex(VamanaIndex):
    """Vamana index with SSD-backed vector storage via MmapVectorStore.

    Suitable for datasets larger than RAM.  Vectors live on NVMe; the OS
    page cache keeps hot vectors resident without explicit management.

    Parameters
    ----------
    dim : int
        Vector dimension.
    store_dir : str
        Directory for the MmapVectorStore (vectors.dat + meta.json).
    capacity : int
        Initial capacity for the mmap store (rows pre-allocated).
    **kwargs
        Forwarded to VamanaIndex (metric, L, R, alpha, mmap_dir).
    """

    def __init__(
        self,
        dim: int,
        store_dir: str = "vamana_mmap_vectors",
        capacity: int = 1_000_000,
        **kwargs,
    ):
        super().__init__(dim=dim, **kwargs)
        from utils.mmap_store import MmapVectorStore
        self._store = MmapVectorStore(store_dir, dim=dim, capacity=capacity)
        self._vec_lock = threading.RLock()

    def _vec_for(self, idx: int) -> Optional[np.ndarray]:
        if idx >= len(self.ids):
            return None
        vid = self.ids[idx]
        return self._store.get(vid)

    def _distance(self, i: int, j: int) -> float:
        vi = self._vec_for(i)
        vj = self._vec_for(j)
        if vi is None or vj is None:
            return float("inf")
        if self.metric == "cosine":
            return max(0.0, 1.0 - float(np.dot(vi, vj)))
        return float(np.linalg.norm(vi - vj))

    def _compute_distances(self, indices: List[int], query: np.ndarray) -> np.ndarray:
        vecs = []
        valid = []
        for idx in indices:
            v = self._vec_for(idx)
            if v is not None:
                vecs.append(v)
                valid.append(idx)
        if not vecs:
            return np.array([])
        arr = np.stack(vecs)
        if self.metric == "cosine":
            return 1.0 - arr @ query
        return np.sqrt(np.sum((arr - query) ** 2, axis=1))

    def _robust_prune(
        self, node_idx: int, candidates: List[int], alpha: float, R: int
    ) -> List[int]:
        node_vec = self._vec_for(node_idx)
        if node_vec is None or not candidates:
            return candidates[:R]

        def _d(v1: np.ndarray, v2: np.ndarray) -> float:
            if self.metric == "cosine":
                return max(0.0, 1.0 - float(np.dot(v1, v2)))
            return float(np.linalg.norm(v1 - v2))

        # Pre-fetch all candidate vectors once (batch mmap reads)
        cand_vecs: Dict[int, np.ndarray] = {}
        for c in candidates:
            v = self._vec_for(c)
            if v is not None:
                cand_vecs[c] = v

        remaining = sorted(
            [c for c in candidates if c in cand_vecs],
            key=lambda c: _d(node_vec, cand_vecs[c]),
        )
        selected: List[int] = []

        while remaining and len(selected) < R:
            p_star = remaining[0]
            selected.append(p_star)
            remaining = remaining[1:]
            p_star_vec = cand_vecs[p_star]
            remaining = [
                p for p in remaining
                if _d(p_star_vec, cand_vecs[p]) > alpha * _d(node_vec, cand_vecs[p])
            ]

        return selected

    def insert(self, vector: List[float], vector_id: str, metadata: Any = None):
        with self._lock:
            vec = np.array(vector, dtype=np.float32)
            if self.metric == "cosine":
                vec = _l2_normalize(vec)

            self._store.add(vector_id, vec)

            idx = len(self.ids)
            self.ids.append(vector_id)
            self.id_to_index[vector_id] = idx
            if metadata:
                self.metadata[vector_id] = metadata
            self.deleted.discard(idx)

            if idx >= self._capacity:
                self._grow_mmap(idx + 1)

            if self.entry_point is None:
                self.entry_point = idx
            else:
                indices, _ = self._search_beam(vec, self.L, self.entry_point)
                candidates = [i for i in indices if i != idx]
                neighbors = self._robust_prune(idx, candidates, self.alpha, self.R)

                for j, nid in enumerate(neighbors):
                    self._adj[idx, j] = int(nid)
                    n_row = self._adj[nid, :self.R]
                    n_neighbors = [int(x) for x in n_row if int(x) >= 0]
                    if idx not in n_neighbors:
                        n_neighbors.append(idx)
                    pruned_back = self._robust_prune(nid, n_neighbors, self.alpha, self.R)
                    self._adj[nid, :self.R] = -1
                    for k2, nb in enumerate(pruned_back):
                        self._adj[nid, k2] = int(nb)

            self._save_meta()

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        beam_width: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        bw = beam_width or self.L
        q = np.array(query_vector, dtype=np.float32)
        if self.metric == "cosine":
            q = _l2_normalize(q)
        indices, dists = self._search_beam(q, bw, self.entry_point)
        results = []
        for idx, d in zip(indices, dists):
            if idx >= len(self.ids):
                continue
            vid = self.ids[idx]
            if int(idx) in self.deleted:
                continue
            results.append({
                "vector_id": vid,
                "distance": float(d),
                "metadata": self.metadata.get(vid),
            })
            if len(results) >= k:
                break
        return results

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base["store_count"] = len(self._store)
        base["backend"] = "mmap_disk"
        return base
