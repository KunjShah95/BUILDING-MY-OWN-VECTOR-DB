"""
IVF-PQ Index — Inverted File with Product Quantization residuals.

Two-level quantization for billion-scale ANN:
  1. Coarse IVF: k-means partitions space into `nlist` Voronoi cells.
     Each vector is assigned to its nearest centroid.
  2. Fine PQ: the residual (vector − cell_centroid) is encoded with
     Product Quantization for compact in-memory representation.

Search: probe the `nprobe` nearest cells, ADC over PQ-encoded residuals,
return top-k globally.

Memory: IVF centroids (nlist × D × 4 bytes) + PQ codebooks (M × 256 × sub_dim × 4)
        + codes (N × M bytes).  For 1B × 128-dim, M=8: ~1 GB codes vs 512 GB raw.

Persistence: numpy .npy binary (not JSON) for fast save/load at scale.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _kmeans(data: np.ndarray, k: int, max_iter: int = 25, seed: int = 0) -> np.ndarray:
    """Mini k-means with Forgy init and early stopping. Always returns shape (k, d)."""
    rng = np.random.RandomState(seed)
    n, d = data.shape
    if n <= k:
        # Pad with zeros so shape is always (k, d)
        out = np.zeros((k, d), dtype=np.float32)
        out[:n] = data.astype(np.float32)
        return out
    centroids = data[rng.choice(n, k, replace=False)].copy().astype(np.float32)
    for _ in range(max_iter):
        # (n, k) distance matrix via broadcasting
        diffs = data[:, None, :] - centroids[None, :, :]          # (n, k, d)
        dists = np.einsum("nkd,nkd->nk", diffs, diffs)            # (n, k)
        labels = np.argmin(dists, axis=1)                          # (n,)
        new_c = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=k)
        for j in range(k):
            mask = labels == j
            new_c[j] = data[mask].mean(axis=0) if mask.any() else centroids[j]
        shift = float(np.linalg.norm(new_c - centroids))
        centroids = new_c
        if shift < 1e-6:
            break
    return centroids


class IVFPQIndex:
    """Inverted File + Product Quantization index.

    Parameters
    ----------
    nlist : int
        Number of Voronoi cells (coarse quantizer centroids). Typical: sqrt(N).
    M : int
        PQ sub-quantizers. Must divide the vector dimension. Default 8.
    k_sub : int
        Centroids per PQ sub-quantizer (max 256 for uint8). Default 256.
    nprobe : int
        Cells to probe at query time. Higher = better recall, slower. Default 8.
    metric : str
        "cosine" or "euclidean". Default "cosine".
    """

    def __init__(
        self,
        nlist: int = 256,
        M: int = 8,
        k_sub: int = 256,
        nprobe: int = 8,
        metric: str = "cosine",
    ):
        if k_sub > 256:
            raise ValueError("k_sub must be ≤ 256")
        self.nlist = nlist
        self.M = M
        self.k_sub = k_sub
        self.nprobe = nprobe
        self.metric = metric

        # Set during train()
        self.dim: int = 0
        self.sub_dim: int = 0
        self.coarse_centroids: Optional[np.ndarray] = None   # (nlist, D)
        self.pq_codebooks: Optional[np.ndarray] = None       # (M, k_sub, sub_dim)
        self.is_trained: bool = False

        # Per-cell inverted lists: cell_id -> list of (vector_id, code)
        # codes: uint8 array of length M
        self._cells: Dict[int, List[Tuple[str, np.ndarray]]] = {
            i: [] for i in range(nlist)
        }
        self._id_to_cell: Dict[str, int] = {}
        self._metadata: Dict[str, Any] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ train

    def train(self, vectors: List[List[float]]) -> "IVFPQIndex":
        """Learn coarse centroids and PQ codebooks from training vectors."""
        arr = np.array(vectors, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms > 0, norms, 1.0)

        n, D = arr.shape
        self.dim = D

        # Validate / infer M
        if D % self.M != 0:
            # Find largest divisor of D ≤ M
            self.M = max(m for m in range(1, self.M + 1) if D % m == 0)
        self.sub_dim = D // self.M

        # 1. Coarse IVF centroids
        logger.info("IVF-PQ train: coarse k-means nlist=%d on %d vectors", self.nlist, n)
        self.coarse_centroids = _kmeans(arr, self.nlist)

        # 2. Assign to cells, compute residuals
        cell_assignments = self._assign_coarse(arr)          # (n,)
        residuals = arr - self.coarse_centroids[cell_assignments]

        # 3. PQ codebooks on residuals
        logger.info("IVF-PQ train: PQ M=%d k_sub=%d", self.M, self.k_sub)
        self.pq_codebooks = np.zeros((self.M, self.k_sub, self.sub_dim), dtype=np.float32)
        for m in range(self.M):
            sub = residuals[:, m * self.sub_dim:(m + 1) * self.sub_dim]
            self.pq_codebooks[m] = _kmeans(sub, self.k_sub)

        self.is_trained = True
        logger.info("IVF-PQ training complete D=%d nlist=%d M=%d", D, self.nlist, self.M)
        return self

    # ------------------------------------------------------------------- add

    def add(self, vector: List[float], vector_id: str, metadata: Any = None) -> None:
        if not self.is_trained:
            raise RuntimeError("Must call train() before add()")
        vec = np.array(vector, dtype=np.float32)
        if self.metric == "cosine":
            vec = _l2_normalize(vec)
        cell = int(self._assign_coarse(vec[None])[0])
        residual = vec - self.coarse_centroids[cell]
        code = self._pq_encode(residual)
        with self._lock:
            if vector_id in self._id_to_cell:
                old_cell = self._id_to_cell[vector_id]
                self._cells[old_cell] = [
                    (vid, c) for vid, c in self._cells[old_cell] if vid != vector_id
                ]
            self._cells[cell].append((vector_id, code))
            self._id_to_cell[vector_id] = cell
            if metadata is not None:
                self._metadata[vector_id] = metadata

    def add_batch(self, entries: List[Dict[str, Any]]) -> int:
        """Bulk add. Each entry needs 'vector', 'vector_id', optional 'metadata'."""
        vectors = np.array([e["vector"] for e in entries], dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms > 0, norms, 1.0)
        cells = self._assign_coarse(vectors)
        residuals = vectors - self.coarse_centroids[cells]
        codes = self._pq_encode_batch(residuals)
        with self._lock:
            for i, entry in enumerate(entries):
                vid = entry["vector_id"]
                cell = int(cells[i])
                if vid in self._id_to_cell:
                    old = self._id_to_cell[vid]
                    self._cells[old] = [(v, c) for v, c in self._cells[old] if v != vid]
                self._cells[cell].append((vid, codes[i]))
                self._id_to_cell[vid] = cell
                if entry.get("metadata") is not None:
                    self._metadata[vid] = entry["metadata"]
        return len(entries)

    # ----------------------------------------------------------------- search

    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        nprobe: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_trained:
            raise RuntimeError("Must call train() before search()")

        nprobe = nprobe or self.nprobe
        q = np.array(query_vector, dtype=np.float32)
        if self.metric == "cosine":
            q = _l2_normalize(q)

        # 1. Find nprobe nearest coarse centroids
        if self.metric == "cosine":
            coarse_dists = 1.0 - self.coarse_centroids @ q
        else:
            diff = self.coarse_centroids - q
            coarse_dists = np.einsum("nd,nd->n", diff, diff)

        probe_cells = np.argpartition(coarse_dists, min(nprobe, self.nlist) - 1)[:nprobe]

        # 2. For each probed cell, compute residual query and ADC
        candidates: List[Tuple[float, str]] = []
        with self._lock:
            for cell in probe_cells:
                cell_entries = self._cells[int(cell)]
                if not cell_entries:
                    continue
                q_residual = q - self.coarse_centroids[cell]
                dist_tables = self._compute_dist_tables(q_residual)
                for vid, code in cell_entries:
                    approx = sum(float(dist_tables[m][int(code[m])]) for m in range(self.M))
                    candidates.append((approx, vid))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0])
        return [
            {
                "vector_id": vid,
                "distance": float(d),
                "metadata": self._metadata.get(vid),
            }
            for d, vid in candidates[:k]
        ]

    def delete(self, vector_id: str) -> bool:
        with self._lock:
            cell = self._id_to_cell.pop(vector_id, None)
            if cell is None:
                return False
            self._cells[cell] = [(v, c) for v, c in self._cells[cell] if v != vector_id]
            self._metadata.pop(vector_id, None)
            return True

    # ------------------------------------------------------------ persistence

    def save(self, directory: str) -> None:
        """Save index to directory using numpy binary format (fast at scale)."""
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "coarse.npy"), self.coarse_centroids)
        np.save(os.path.join(directory, "pq_codebooks.npy"), self.pq_codebooks)

        # Encode inverted lists as parallel arrays for numpy storage
        all_vids: List[str] = []
        all_cells: List[int] = []
        all_codes: List[np.ndarray] = []
        for cell, entries in self._cells.items():
            for vid, code in entries:
                all_vids.append(vid)
                all_cells.append(cell)
                all_codes.append(code)

        codes_arr = np.array(all_codes, dtype=np.uint8) if all_codes else np.empty((0, self.M), dtype=np.uint8)
        cells_arr = np.array(all_cells, dtype=np.int32)
        np.save(os.path.join(directory, "codes.npy"), codes_arr)
        np.save(os.path.join(directory, "cells.npy"), cells_arr)

        import json
        meta = {
            "nlist": self.nlist, "M": self.M, "k_sub": self.k_sub,
            "nprobe": self.nprobe, "metric": self.metric, "dim": self.dim,
            "sub_dim": self.sub_dim, "is_trained": self.is_trained,
            "vector_ids": all_vids,
            "metadata": {k: v for k, v in self._metadata.items()},
        }
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump(meta, f)
        logger.info("IVF-PQ saved to %s (%d vectors)", directory, len(all_vids))

    @classmethod
    def load(cls, directory: str) -> "IVFPQIndex":
        """Load from directory saved by save()."""
        import json
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)

        obj = cls(
            nlist=meta["nlist"], M=meta["M"], k_sub=meta["k_sub"],
            nprobe=meta["nprobe"], metric=meta["metric"],
        )
        obj.dim = meta["dim"]
        obj.sub_dim = meta["sub_dim"]
        obj.is_trained = meta["is_trained"]
        obj.coarse_centroids = np.load(os.path.join(directory, "coarse.npy"))
        obj.pq_codebooks = np.load(os.path.join(directory, "pq_codebooks.npy"))

        codes_arr = np.load(os.path.join(directory, "codes.npy"))
        cells_arr = np.load(os.path.join(directory, "cells.npy"))
        vector_ids: List[str] = meta["vector_ids"]
        obj._metadata = meta.get("metadata", {})
        obj._cells = {i: [] for i in range(obj.nlist)}
        for i, vid in enumerate(vector_ids):
            cell = int(cells_arr[i])
            obj._cells[cell].append((vid, codes_arr[i]))
            obj._id_to_cell[vid] = cell

        logger.info("IVF-PQ loaded from %s (%d vectors)", directory, len(vector_ids))
        return obj

    # ------------------------------------------------------------------ stats

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = sum(len(v) for v in self._cells.values())
            sizes = [len(v) for v in self._cells.values()]
        return {
            "total_vectors": total,
            "nlist": self.nlist,
            "M": self.M,
            "k_sub": self.k_sub,
            "nprobe": self.nprobe,
            "metric": self.metric,
            "dim": self.dim,
            "is_trained": self.is_trained,
            "bytes_per_vector": self.M,
            "compression_ratio": round((self.dim * 4) / self.M, 1) if self.M else 0,
            "avg_cell_size": round(total / self.nlist, 1) if self.nlist else 0,
            "max_cell_size": max(sizes) if sizes else 0,
        }

    def __len__(self) -> int:
        return sum(len(v) for v in self._cells.values())

    # --------------------------------------------------------- internal helpers

    def _assign_coarse(self, vectors: np.ndarray) -> np.ndarray:
        """Return nearest centroid index for each row in vectors."""
        if self.metric == "cosine":
            # centroids assumed unit-normed after training
            scores = vectors @ self.coarse_centroids.T       # (n, nlist)
            return np.argmax(scores, axis=1)
        diff = vectors[:, None, :] - self.coarse_centroids[None, :, :]
        dists = np.einsum("nkd,nkd->nk", diff, diff)
        return np.argmin(dists, axis=1)

    def _pq_encode(self, residual: np.ndarray) -> np.ndarray:
        """Encode a single residual into M uint8 codes."""
        code = np.empty(self.M, dtype=np.uint8)
        for m in range(self.M):
            sub = residual[m * self.sub_dim:(m + 1) * self.sub_dim]
            cb = self.pq_codebooks[m]                        # (k_sub, sub_dim)
            dists = np.einsum("kd,d->k", cb, sub) if self.metric == "cosine" \
                else np.sum((cb - sub) ** 2, axis=1)
            code[m] = int(np.argmax(dists) if self.metric == "cosine" else np.argmin(dists))
        return code

    def _pq_encode_batch(self, residuals: np.ndarray) -> np.ndarray:
        """Encode a batch of residuals. Shape (n, D) -> (n, M) uint8."""
        n = residuals.shape[0]
        codes = np.empty((n, self.M), dtype=np.uint8)
        for m in range(self.M):
            sub = residuals[:, m * self.sub_dim:(m + 1) * self.sub_dim]  # (n, sub_dim)
            cb = self.pq_codebooks[m]                                      # (k_sub, sub_dim)
            if self.metric == "cosine":
                scores = sub @ cb.T                                        # (n, k_sub)
                codes[:, m] = np.argmax(scores, axis=1)
            else:
                dists = np.sum((sub[:, None, :] - cb[None, :, :]) ** 2, axis=2)
                codes[:, m] = np.argmin(dists, axis=1)
        return codes

    def _compute_dist_tables(self, q_residual: np.ndarray) -> List[np.ndarray]:
        """ADC distance tables: list of (k_sub,) arrays, one per subspace."""
        tables = []
        for m in range(self.M):
            q_sub = q_residual[m * self.sub_dim:(m + 1) * self.sub_dim]
            cb = self.pq_codebooks[m]
            if self.metric == "cosine":
                tables.append(1.0 - cb @ q_sub)
            else:
                tables.append(np.sum((cb - q_sub) ** 2, axis=1))
        return tables
