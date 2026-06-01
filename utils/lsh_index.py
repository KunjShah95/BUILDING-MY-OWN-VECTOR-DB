import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json


class LSHIndex:
    """Locality-Sensitive Hashing for cosine distance."""

    def __init__(self, n_hash_tables: int = 10, n_hash_bits: int = 8, dim: int = 128):
        self.n_hash_tables = n_hash_tables
        self.n_hash_bits = n_hash_bits
        self.dim = dim
        self.hash_tables: List[Dict[str, List[Tuple[str, np.ndarray]]]] = [{} for _ in range(n_hash_tables)]
        self.random_planes: List[np.ndarray] = []
        self.vector_ids: List[str] = []
        self._all_vectors: Dict[str, np.ndarray] = {}
        self._all_metadata: Dict[str, dict] = {}
        self.is_trained = False

    def _generate_random_planes(self):
        self.random_planes = [
            np.random.randn(self.dim, self.n_hash_bits)
            for _ in range(self.n_hash_tables)
        ]

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        projections = vector @ self.random_planes[table_idx]
        bits = (projections > 0).astype(int)
        return ''.join(bits.astype(str))

    def train(self, vectors: List[np.ndarray]):
        if not vectors:
            raise ValueError("Need at least one vector to train")
        self.dim = vectors[0].shape[0]
        self._generate_random_planes()
        self.is_trained = True

    def add(self, vector: np.ndarray, vector_id: str, metadata: Optional[Dict] = None):
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        self._all_vectors[vector_id] = vector
        self._all_metadata[vector_id] = metadata or {}
        for i in range(self.n_hash_tables):
            h = self._hash_vector(vector, i)
            if h not in self.hash_tables[i]:
                self.hash_tables[i][h] = []
            self.hash_tables[i][h].append((vector_id, vector))
        self.vector_ids.append(vector_id)

    def search(self, query: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        if isinstance(query, list):
            query = np.array(query, dtype=np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        candidates = set()
        for i in range(self.n_hash_tables):
            h = self._hash_vector(query, i)
            if h in self.hash_tables[i]:
                for vid, _ in self.hash_tables[i][h]:
                    candidates.add(vid)

        if not candidates:
            return []

        results = []
        for table in self.hash_tables:
            for h, entries in table.items():
                for vid, vec in entries:
                    if vid in candidates:
                        dist = 1 - float(np.dot(vec, query))
                        results.append((vid, dist))

        seen = set()
        unique = []
        for vid, dist in results:
            if vid not in seen:
                seen.add(vid)
                unique.append((vid, dist))

        unique.sort(key=lambda x: x[1])

        top_k = unique[:k]
        return [
            {"id": vid, "distance": dist, "metadata": self._all_metadata.get(vid)}
            for vid, dist in top_k
        ]

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._all_vectors:
            return False
        del self._all_vectors[vector_id]
        self._all_metadata.pop(vector_id, None)
        for table in self.hash_tables:
            for h in list(table.keys()):
                table[h] = [(vid, v) for vid, v in table[h] if vid != vector_id]
                if not table[h]:
                    del table[h]
        self.vector_ids = [vid for vid in self.vector_ids if vid != vector_id]
        return True

    def get_stats(self) -> Dict:
        return {
            "type": "LSH",
            "n_hash_tables": self.n_hash_tables,
            "n_hash_bits": self.n_hash_bits,
            "dim": self.dim,
            "n_vectors": len(self.vector_ids),
            "is_trained": self.is_trained,
        }

    def save(self, filepath: str):
        hash_tables_serializable = []
        for table in self.hash_tables:
            table_ser = {}
            for h, entries in table.items():
                table_ser[h] = [(vid, v.tolist()) for vid, v in entries]
            hash_tables_serializable.append(table_ser)
        data = {
            "n_hash_tables": self.n_hash_tables,
            "n_hash_bits": self.n_hash_bits,
            "dim": self.dim,
            "random_planes": [p.tolist() for p in self.random_planes],
            "hash_tables": hash_tables_serializable,
            "vector_ids": self.vector_ids,
            "vectors": {vid: v.tolist() for vid, v in self._all_vectors.items()},
            "metadata": {vid: m for vid, m in self._all_metadata.items()},
            "is_trained": self.is_trained,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load(filepath: str) -> "LSHIndex":
        with open(filepath, 'r') as f:
            data = json.load(f)
        idx = LSHIndex(
            n_hash_tables=data["n_hash_tables"],
            n_hash_bits=data["n_hash_bits"],
            dim=data["dim"],
        )
        idx.random_planes = [np.array(p) for p in data["random_planes"]]
        idx.hash_tables = []
        for table_ser in data["hash_tables"]:
            table = {}
            for h, entries in table_ser.items():
                table[h] = [(vid, np.array(v)) for vid, v in entries]
            idx.hash_tables.append(table)
        idx.vector_ids = data["vector_ids"]
        idx._all_vectors = {vid: np.array(v) for vid, v in data.get("vectors", {}).items()}
        idx._all_metadata = {vid: dict(m) for vid, m in data.get("metadata", {}).items()}
        idx.is_trained = data.get("is_trained", True)
        return idx
