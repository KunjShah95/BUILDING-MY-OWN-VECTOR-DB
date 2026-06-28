"""Federated Embedding Sync — Phase 15: Vector-Native AI Features.

Simulates on-device federated learning: clients register, submit local
gradients computed via federated averaging, and receive a global model delta.
No torch required — pure numpy.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClientRecord:
    client_id: str
    model_version: str
    local_stats: Dict[str, Any]
    registered_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FederatedEmbeddingService:
    """Simulates a federated embedding coordinator."""

    def __init__(self) -> None:
        self._clients: Dict[str, ClientRecord] = {}
        self._global_version: str = "v1.0.0"
        self._global_centroid: Optional[np.ndarray] = None
        self._rounds_completed: int = 0
        self._pending_updates: List[Dict[str, Any]] = []

    # ---- Client registration ----------------------------------------------

    def register_client(
        self,
        client_id: str,
        model_version: str,
        local_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        record = ClientRecord(
            client_id=client_id,
            model_version=model_version,
            local_stats=local_stats,
        )
        if client_id in self._clients:
            self._clients[client_id].last_seen = time.time()
            self._clients[client_id].model_version = model_version
        else:
            self._clients[client_id] = record

        return {
            "success": True,
            "client_id": client_id,
            "global_model_version": self._global_version,
            "rounds_completed": self._rounds_completed,
        }

    # ---- Gradient computation ---------------------------------------------

    def compute_gradient_update(
        self,
        client_id: str,
        local_embeddings: List[List[float]],
        labels: List[int],
    ) -> Dict[str, Any]:
        """Federated averaging: compute per-class centroids, diff from global."""
        if not local_embeddings or not labels:
            return {"success": False, "message": "Empty embeddings or labels"}
        if len(local_embeddings) != len(labels):
            return {"success": False, "message": "Length mismatch"}

        emb = np.array(local_embeddings, dtype=np.float64)
        lbl = np.array(labels)
        dim = emb.shape[1]

        # Per-class mean (pseudo-gradient target)
        class_means: Dict[int, np.ndarray] = {}
        for cls in np.unique(lbl):
            mask = lbl == cls
            class_means[int(cls)] = emb[mask].mean(axis=0)

        local_centroid = emb.mean(axis=0)

        if self._global_centroid is None:
            self._global_centroid = local_centroid.copy()

        delta = local_centroid - self._global_centroid

        update = {
            "client_id": client_id,
            "n_samples": len(local_embeddings),
            "local_centroid": local_centroid.tolist(),
            "delta": delta.tolist(),
            "class_means": {str(k): v.tolist() for k, v in class_means.items()},
            "dim": dim,
        }
        self._pending_updates.append(update)

        return {
            "success": True,
            "delta_weights": delta.tolist(),
            "global_model_version": self._global_version,
            "message": "Update accepted; call /aggregate to apply",
        }

    # ---- Aggregation ------------------------------------------------------

    def aggregate_updates(
        self, client_updates: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """FedAvg: weighted average of gradients by sample count."""
        updates = client_updates if client_updates is not None else self._pending_updates
        if not updates:
            return {"success": False, "message": "No updates to aggregate"}

        n_clients = len(updates)
        total_samples = sum(u.get("n_samples", 1) for u in updates)
        dim = len(updates[0]["delta"])
        agg_delta = np.zeros(dim)

        for u in updates:
            w = u.get("n_samples", 1) / total_samples
            agg_delta += w * np.array(u["delta"])

        if self._global_centroid is None:
            self._global_centroid = agg_delta.copy()
        else:
            self._global_centroid += agg_delta

        self._rounds_completed += 1
        # Bump minor version
        parts = self._global_version.lstrip("v").split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        self._global_version = "v" + ".".join(parts)

        self._pending_updates.clear()

        return {
            "success": True,
            "rounds_completed": self._rounds_completed,
            "global_model_version": self._global_version,
            "aggregated_delta": agg_delta.tolist(),
            "n_clients": n_clients,
        }

    # ---- Accessors --------------------------------------------------------

    def get_global_model_version(self) -> str:
        return self._global_version

    def status(self) -> Dict[str, Any]:
        return {
            "global_model_version": self._global_version,
            "rounds_completed": self._rounds_completed,
            "connected_clients": len(self._clients),
            "pending_updates": len(self._pending_updates),
            "clients": [
                {
                    "client_id": c.client_id,
                    "model_version": c.model_version,
                    "last_seen": c.last_seen,
                }
                for c in self._clients.values()
            ],
        }
