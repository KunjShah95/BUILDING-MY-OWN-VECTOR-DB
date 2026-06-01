import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import heapq
import json
from utils.distance import cosine_distance, euclidean_distance


class VPNode:
    __slots__ = ("vantage_point", "vantage_id", "radius", "left", "right", "metadata", "points")

    def __init__(self, vector: np.ndarray, vector_id: str, metadata: Optional[Dict] = None):
        self.vantage_point = vector
        self.vantage_id = vector_id
        self.radius = 0.0
        self.left: Optional["VPNode"] = None
        self.right: Optional["VPNode"] = None
        self.metadata = metadata or {}
        self.points: List[Tuple[np.ndarray, str, dict]] = []


class VPTreeIndex:
    def __init__(self, distance_metric: str = "cosine", leaf_size: int = 20):
        self.root: Optional[VPNode] = None
        self.distance_metric = distance_metric
        self.leaf_size = leaf_size
        self.size = 0
        self._all_vectors: Dict[str, np.ndarray] = {}
        self._all_metadata: Dict[str, dict] = {}

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.distance_metric == "euclidean":
            return euclidean_distance(v1.tolist(), v2.tolist())
        return cosine_distance(v1.tolist(), v2.tolist())

    def build(self, vectors: np.ndarray, ids: List[str], metadata_list: Optional[List[dict]] = None):
        points = [(vectors[i], ids[i], (metadata_list[i] if metadata_list else {})) for i in range(len(ids))]
        self._all_vectors = {ids[i]: vectors[i] for i in range(len(ids))}
        self._all_metadata = {ids[i]: (metadata_list[i] if metadata_list else {}) for i in range(len(ids))}
        if points:
            self.root = self._build(points)
        self.size = len(ids)

    def _build(self, points: List[Tuple[np.ndarray, str, dict]]) -> Optional[VPNode]:
        if not points:
            return None

        if len(points) <= self.leaf_size:
            node = VPNode(points[0][0], points[0][1], points[0][2])
            node.points = points
            return node

        idx = np.random.randint(len(points))
        vp_vec, vp_id, vp_meta = points.pop(idx)
        node = VPNode(vp_vec, vp_id, vp_meta)

        dists = [self._distance(vp_vec, p[0]) for p in points]
        median = np.median(dists)
        node.radius = float(median)

        left_points = [p for i, p in enumerate(points) if dists[i] <= median]
        right_points = [p for i, p in enumerate(points) if dists[i] > median]

        node.left = self._build(left_points)
        node.right = self._build(right_points)
        return node

    def insert(self, vector: List[float], vector_id: str, metadata: Optional[Dict] = None):
        vec = np.array(vector, dtype=np.float64)
        self._all_vectors[vector_id] = vec
        self._all_metadata[vector_id] = metadata or {}
        all_vecs = np.array(list(self._all_vectors.values()))
        all_ids = list(self._all_vectors.keys())
        all_meta = [self._all_metadata[i] for i in all_ids]
        self.build(all_vecs, all_ids, all_meta)

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._all_vectors:
            return False
        del self._all_vectors[vector_id]
        self._all_metadata.pop(vector_id, None)
        if not self._all_vectors:
            self.root = None
            self.size = 0
            return True
        all_vecs = np.array(list(self._all_vectors.values()))
        all_ids = list(self._all_vectors.keys())
        all_meta = [self._all_metadata[i] for i in all_ids]
        self.build(all_vecs, all_ids, all_meta)
        return True

    def search(self, query: List[float], k: int = 10) -> List[Dict[str, Any]]:
        if self.root is None or self.size == 0:
            return []
        q = np.array(query, dtype=np.float64)
        heap: List[Tuple[float, str]] = []
        self._search(self.root, q, k, heap)
        results = []
        while heap:
            neg_dist, vid = heapq.heappop(heap)
            results.append({"id": vid, "distance": -neg_dist, "metadata": self._all_metadata.get(vid)})
        results.reverse()
        return results

    def _search(self, node: Optional[VPNode], query: np.ndarray, k: int,
                heap: List[Tuple[float, str]]):
        if node is None:
            return

        dist = self._distance(query, node.vantage_point)

        if len(heap) < k:
            heapq.heappush(heap, (-dist, node.vantage_id))
        elif dist < -heap[0][0]:
            heapq.heapreplace(heap, (-dist, node.vantage_id))

        if hasattr(node, "points") and node.points:
            for pvec, pid, pmeta in node.points:
                if pid == node.vantage_id:
                    continue
                d = self._distance(query, pvec)
                if len(heap) < k:
                    heapq.heappush(heap, (-d, pid))
                elif d < -heap[0][0]:
                    heapq.heapreplace(heap, (-d, pid))
            return

        tau = -heap[0][0] if len(heap) == k else float("inf")

        if dist <= node.radius:
            self._search(node.left, query, k, heap)
            if dist + tau >= node.radius:
                self._search(node.right, query, k, heap)
        else:
            self._search(node.right, query, k, heap)
            if dist - tau <= node.radius:
                self._search(node.left, query, k, heap)

    @property
    def is_trained(self) -> bool:
        return self.root is not None

    @property
    def dim(self) -> Optional[int]:
        if self._all_vectors:
            return len(next(iter(self._all_vectors.values())))
        return None

    def get_stats(self) -> Dict:
        return {
            "type": "VP-Tree",
            "distance_metric": self.distance_metric,
            "leaf_size": self.leaf_size,
            "size": self.size,
            "is_trained": self.is_trained,
        }

    def save(self, path: str):
        data = {
            "distance_metric": self.distance_metric,
            "leaf_size": self.leaf_size,
            "vectors": {vid: v.tolist() for vid, v in self._all_vectors.items()},
            "metadata": self._all_metadata,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def load(path: str) -> "VPTreeIndex":
        with open(path, "r") as f:
            data = json.load(f)
        idx = VPTreeIndex(
            distance_metric=data["distance_metric"],
            leaf_size=data.get("leaf_size", 20),
        )
        ids = list(data["vectors"].keys())
        vecs = np.array(list(data["vectors"].values()))
        meta = [data["metadata"].get(i, {}) for i in ids]
        idx.build(vecs, ids, meta)
        return idx
