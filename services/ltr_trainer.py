"""LTR Trainer — Phase 15: Vector-Native AI Features.

Collects labeled query-document pairs, trains pointwise (linear/MLP) and
pairwise (RankNet-style) ranking models using numpy only, and exposes
NDCG/MRR evaluation. All weights are serialised to JSON.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TRAINING_DATA_FILE = "ltr_training_data.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrainingPair:
    query_id: str
    doc_id: str
    label: float          # relevance grade 0-3
    features: Dict[str, float]
    created_at: float = field(default_factory=time.time)


@dataclass
class LTRModel:
    model_type: str                   # 'linear' | 'mlp' | 'pairwise'
    weights: Any                      # numpy array(s) serialised as list(s)
    feature_names: List[str]
    trained_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def score(self, feature_vec: np.ndarray) -> float:
        if self.model_type in ("linear", "pairwise"):
            w = np.array(self.weights, dtype=np.float64)
            return float(w @ feature_vec)
        if self.model_type == "mlp":
            W1 = np.array(self.weights["W1"], dtype=np.float64)
            b1 = np.array(self.weights["b1"], dtype=np.float64)
            W2 = np.array(self.weights["W2"], dtype=np.float64)
            b2 = float(self.weights["b2"])
            h = np.tanh(W1 @ feature_vec + b1)
            return float(W2 @ h + b2)
        raise ValueError(f"Unknown model_type: {self.model_type}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LTRTrainer:
    """Collect training pairs, train ranking models, evaluate, export."""

    def __init__(self, data_file: str = _TRAINING_DATA_FILE):
        self.data_file = data_file
        self._model: Optional[LTRModel] = None

    # ---- Data collection ---------------------------------------------------

    def collect_training_pair(
        self,
        query_id: str,
        doc_id: str,
        label: float,
        features: Dict[str, float],
    ) -> TrainingPair:
        """Append a labelled query-document pair to the JSONL store."""
        pair = TrainingPair(
            query_id=query_id,
            doc_id=doc_id,
            label=float(label),
            features=features,
        )
        with open(self.data_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(pair)) + "\n")
        return pair

    def load_training_data(self) -> List[TrainingPair]:
        if not os.path.exists(self.data_file):
            return []
        pairs: List[TrainingPair] = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                pairs.append(TrainingPair(**d))
        return pairs

    # ---- Feature helpers ---------------------------------------------------

    def _to_matrix(
        self, pairs: List[TrainingPair]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        feature_names: List[str] = sorted(
            {k for p in pairs for k in p.features}
        )
        X = np.zeros((len(pairs), len(feature_names)), dtype=np.float64)
        y = np.zeros(len(pairs), dtype=np.float64)
        for i, p in enumerate(pairs):
            for j, fn in enumerate(feature_names):
                X[i, j] = p.features.get(fn, 0.0)
            y[i] = p.label
        return X, y, feature_names

    # ---- Pointwise training ------------------------------------------------

    def train_pointwise(self, model_type: str = "linear") -> LTRModel:
        """Train linear regression or simple MLP on stored pairs."""
        pairs = self.load_training_data()
        if not pairs:
            raise ValueError("No training data available")

        X, y, feature_names = self._to_matrix(pairs)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        Xn = (X - X_mean) / X_std

        if model_type == "linear":
            weights = self._fit_linear(Xn, y)
            model = LTRModel(
                model_type="linear",
                weights=weights.tolist(),
                feature_names=feature_names,
                meta={"X_mean": X_mean.tolist(), "X_std": X_std.tolist()},
            )
        elif model_type == "mlp":
            W1, b1, W2, b2 = self._fit_mlp(Xn, y)
            model = LTRModel(
                model_type="mlp",
                weights={
                    "W1": W1.tolist(),
                    "b1": b1.tolist(),
                    "W2": W2.tolist(),
                    "b2": float(b2),
                },
                feature_names=feature_names,
                meta={"X_mean": X_mean.tolist(), "X_std": X_std.tolist()},
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._model = model
        return model

    def _fit_linear(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        lam = 1e-3
        A = X.T @ X + lam * np.eye(X.shape[1])
        b = X.T @ y
        return np.linalg.solve(A, b)

    def _fit_mlp(
        self, X: np.ndarray, y: np.ndarray,
        hidden: int = 16, epochs: int = 200, lr: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        n_feat = X.shape[1]
        rng = np.random.default_rng(42)
        W1 = rng.normal(0, 0.1, (hidden, n_feat))
        b1 = np.zeros(hidden)
        W2 = rng.normal(0, 0.1, hidden)
        b2 = 0.0

        for _ in range(epochs):
            h = np.tanh(X @ W1.T + b1)
            pred = h @ W2 + b2
            err = pred - y
            dW2 = err @ h / len(y)
            db2 = err.mean()
            dh = np.outer(err, W2) * (1 - h**2)
            dW1 = dh.T @ X / len(y)
            db1 = dh.mean(axis=0)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        return W1, b1, W2, b2

    # ---- Pairwise training -------------------------------------------------

    def train_pairwise(self) -> LTRModel:
        """RankNet-style pairwise loss; numpy only."""
        pairs = self.load_training_data()
        if not pairs:
            raise ValueError("No training data available")

        X, y, feature_names = self._to_matrix(pairs)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        Xn = (X - X_mean) / X_std

        pref_i, pref_j = [], []
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] > y[j]:
                    pref_i.append(i)
                    pref_j.append(j)
                elif y[j] > y[i]:
                    pref_i.append(j)
                    pref_j.append(i)

        if not pref_i:
            return self.train_pointwise("linear")

        pi = np.array(pref_i)
        pj = np.array(pref_j)
        w = np.zeros(Xn.shape[1])
        lr = 0.01

        for _ in range(200):
            si = Xn[pi] @ w
            sj = Xn[pj] @ w
            sigma = 1.0 / (1.0 + np.exp(-(si - sj)))
            diff = Xn[pi] - Xn[pj]
            grad = ((sigma - 1.0)[:, None] * diff).mean(axis=0)
            w -= lr * grad

        model = LTRModel(
            model_type="pairwise",
            weights=w.tolist(),
            feature_names=feature_names,
            meta={"X_mean": X_mean.tolist(), "X_std": X_std.tolist(),
                  "n_pref_pairs": len(pref_i)},
        )
        self._model = model
        return model

    # ---- Evaluation --------------------------------------------------------

    def evaluate(self, test_pairs: List[TrainingPair]) -> Dict[str, Any]:
        if self._model is None:
            raise ValueError("No model trained yet")

        model = self._model
        feature_names = model.feature_names
        X_mean = np.array(model.meta.get("X_mean", [0.0] * len(feature_names)))
        X_std = np.array(model.meta.get("X_std", [1.0] * len(feature_names)))

        from collections import defaultdict
        groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for p in test_pairs:
            fvec = np.array([p.features.get(fn, 0.0) for fn in feature_names])
            fvec = (fvec - X_mean) / (X_std + 1e-8)
            score = model.score(fvec)
            groups[p.query_id].append((score, p.label))

        ndcg_scores: List[float] = []
        mrr_scores: List[float] = []

        for qid, scored in groups.items():
            scored.sort(key=lambda x: x[0], reverse=True)
            top10 = scored[:10]
            dcg = sum(
                (2 ** rel - 1) / math.log2(rank + 2)
                for rank, (_, rel) in enumerate(top10)
            )
            ideal = sorted([rel for _, rel in scored], reverse=True)[:10]
            idcg = sum(
                (2 ** rel - 1) / math.log2(rank + 2)
                for rank, rel in enumerate(ideal)
            )
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

            for rank, (_, rel) in enumerate(scored[:10]):
                if rel > 0:
                    mrr_scores.append(1.0 / (rank + 1))
                    break
            else:
                mrr_scores.append(0.0)

        return {
            "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "num_queries": len(groups),
            "num_pairs": len(test_pairs),
        }

    # ---- Serialisation -----------------------------------------------------

    def export_model(self, path: str) -> None:
        if self._model is None:
            raise ValueError("No model to export")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._model), f, indent=2)

    def load_model(self, path: str) -> LTRModel:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self._model = LTRModel(**d)
        return self._model

    @property
    def model(self) -> Optional[LTRModel]:
        return self._model

    def stats(self) -> Dict[str, Any]:
        pairs = self.load_training_data()
        return {
            "training_data_size": len(pairs),
            "model_trained": self._model is not None,
            "model_type": self._model.model_type if self._model else None,
            "feature_names": self._model.feature_names if self._model else [],
            "weights_preview": (
                self._model.weights[:5]
                if self._model and isinstance(self._model.weights, list)
                else None
            ),
        }
