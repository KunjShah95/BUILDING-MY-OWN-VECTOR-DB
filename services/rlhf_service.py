"""RLHF Feedback Loop — Phase 15: Vector-Native AI Features.

Records click/skip/dwell signals, trains a Bradley-Terry reward model via
gradient descent (numpy only), and predicts relevance for (query, doc) pairs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SIGNAL_FILE = "rlhf_signals.jsonl"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UserSignal:
    query_id: str
    result_id: str
    signal_type: str        # 'click' | 'skip' | 'dwell'
    dwell_ms: Optional[float]
    implicit_relevance: float
    created_at: float = field(default_factory=time.time)


@dataclass
class RewardModel:
    weights: List[float]       # Bradley-Terry score weights (per feature dim)
    trained_at: float = field(default_factory=time.time)
    accuracy: float = 0.0
    n_pairs: int = 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class RLHFService:
    """Collects user feedback signals and trains a reward model."""

    def __init__(self, signal_file: str = _SIGNAL_FILE):
        self.signal_file = signal_file
        self._reward_model: Optional[RewardModel] = None

    # ---- Signal collection -------------------------------------------------

    def record_signal(
        self,
        query_id: str,
        result_id: str,
        signal_type: str,
        dwell_ms: Optional[float] = None,
    ) -> UserSignal:
        """Store a user interaction signal with implicit relevance."""
        if signal_type == "click":
            relevance = 1.0
        elif signal_type == "skip":
            relevance = 0.0
        elif signal_type == "dwell" and dwell_ms is not None:
            relevance = math.log(dwell_ms / 1000.0 + 1.0) / math.log(61.0)
            relevance = min(max(relevance, 0.0), 1.0)
        else:
            relevance = 0.0

        sig = UserSignal(
            query_id=query_id,
            result_id=result_id,
            signal_type=signal_type,
            dwell_ms=dwell_ms,
            implicit_relevance=relevance,
        )
        with open(self.signal_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(sig)) + "\n")
        return sig

    def _load_signals(self) -> List[UserSignal]:
        if not os.path.exists(self.signal_file):
            return []
        signals: List[UserSignal] = []
        with open(self.signal_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                signals.append(UserSignal(**json.loads(line)))
        return signals

    # ---- Reward model training ---------------------------------------------

    def train_reward_model(self) -> RewardModel:
        """Bradley-Terry model: P(i>j) = sigmoid(score_i - score_j).

        We treat implicit_relevance as score; learn a scalar correction per
        result via gradient descent on all (click, skip) pairwise comparisons
        within the same query group.
        """
        signals = self._load_signals()
        if not signals:
            raise ValueError("No signals recorded yet")

        from collections import defaultdict
        groups: Dict[str, List[UserSignal]] = defaultdict(list)
        for s in signals:
            groups[s.query_id].append(s)

        # Build pairwise dataset: (rel_i, rel_j) where i is more relevant
        pairs: List[tuple] = []
        for qid, sigs in groups.items():
            for i, si in enumerate(sigs):
                for j, sj in enumerate(sigs):
                    if i != j and si.implicit_relevance > sj.implicit_relevance:
                        pairs.append((si.implicit_relevance, sj.implicit_relevance))

        if not pairs:
            raise ValueError("No preference pairs from signals")

        # Simple scalar Bradley-Terry: learn a single bias w per signal score
        # P(i>j) = sigmoid(w * rel_i - w * rel_j)
        # gradient descent on binary cross-entropy loss
        w = np.array([1.0])   # single scale parameter
        lr = 0.05

        rel_i = np.array([p[0] for p in pairs])
        rel_j = np.array([p[1] for p in pairs])

        for _ in range(100):
            delta = w[0] * (rel_i - rel_j)
            sigma = 1.0 / (1.0 + np.exp(-delta))
            # -log(sigma) loss; grad = (sigma - 1) * (rel_i - rel_j)
            grad = ((sigma - 1.0) * (rel_i - rel_j)).mean()
            w[0] -= lr * grad

        # Accuracy: fraction of pairs correctly ordered
        delta_test = w[0] * (rel_i - rel_j)
        correct = (delta_test > 0).mean()

        model = RewardModel(
            weights=w.tolist(),
            accuracy=float(correct),
            n_pairs=len(pairs),
        )
        self._reward_model = model
        return model

    def predict_relevance(
        self,
        query_vector: List[float],
        doc_vector: List[float],
    ) -> float:
        """Predict relevance 0-1 from cosine similarity scaled by reward weight."""
        qv = np.array(query_vector, dtype=np.float64)
        dv = np.array(doc_vector, dtype=np.float64)
        qn = np.linalg.norm(qv)
        dn = np.linalg.norm(dv)
        if qn == 0 or dn == 0:
            return 0.0
        cosine = float(qv @ dv / (qn * dn))
        raw = (cosine + 1.0) / 2.0       # map [-1,1] -> [0,1]

        if self._reward_model:
            w = self._reward_model.weights[0]
        else:
            w = 1.0

        # sigmoid(w * raw) to stay in (0,1)
        return float(1.0 / (1.0 + math.exp(-w * raw * 4)))

    # ---- Stats -------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        signals = self._load_signals()
        counts: Dict[str, int] = {}
        for s in signals:
            counts[s.signal_type] = counts.get(s.signal_type, 0) + 1
        return {
            "total_signals": len(signals),
            "signal_counts": counts,
            "model_trained": self._reward_model is not None,
            "model_accuracy": (
                self._reward_model.accuracy if self._reward_model else None
            ),
            "model_n_pairs": (
                self._reward_model.n_pairs if self._reward_model else None
            ),
        }
