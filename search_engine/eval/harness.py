"""Offline retrieval metrics + a tiny evaluation harness.

Feed a list of queries with their ranked retrieved doc_ids and the set/graded
relevance of the gold docs; get back averaged NDCG@k, MRR, Recall@k, Precision@k.
Pure stdlib/math — no numpy needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Union

Relevance = Union[Sequence[str], Mapping[str, float]]


def _rel_map(relevant: Relevance) -> Dict[str, float]:
    if isinstance(relevant, Mapping):
        return {str(k): float(v) for k, v in relevant.items()}
    return {str(d): 1.0 for d in relevant}


def recall_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    rel = _rel_map(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in rel)
    return hits / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    if k <= 0:
        return 0.0
    rel = _rel_map(relevant)
    hits = sum(1 for d in retrieved[:k] if d in rel)
    return hits / k


def mrr(retrieved: Sequence[str], relevant: Relevance) -> float:
    rel = _rel_map(relevant)
    for i, d in enumerate(retrieved):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    rel = _rel_map(relevant)
    if not rel:
        return 0.0
    dcg = 0.0
    for i, d in enumerate(retrieved[:k]):
        gain = rel.get(d, 0.0)
        if gain:
            dcg += gain / math.log2(i + 2)
    ideal_gains = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class EvalResult:
    k: int
    n_queries: int
    ndcg: float
    mrr: float
    recall: float
    precision: float
    per_query: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "k": self.k, "n_queries": self.n_queries,
            f"ndcg@{self.k}": round(self.ndcg, 4),
            "mrr": round(self.mrr, 4),
            f"recall@{self.k}": round(self.recall, 4),
            f"precision@{self.k}": round(self.precision, 4),
        }


def evaluate(
    runs: Sequence[Mapping[str, object]],
    k: int = 10,
) -> EvalResult:
    """Average metrics over a set of query runs.

    Each run: {"retrieved": [doc_id, ...], "relevant": [...] | {doc_id: grade}}.
    """
    if not runs:
        return EvalResult(k, 0, 0.0, 0.0, 0.0, 0.0)
    n = len(runs)
    s_ndcg = s_mrr = s_rec = s_prec = 0.0
    per: List[Dict[str, float]] = []
    for run in runs:
        retrieved = list(run.get("retrieved", []))            # type: ignore[arg-type]
        relevant = run.get("relevant", [])                    # type: ignore[assignment]
        q_ndcg = ndcg_at_k(retrieved, relevant, k)            # type: ignore[arg-type]
        q_mrr = mrr(retrieved, relevant)                      # type: ignore[arg-type]
        q_rec = recall_at_k(retrieved, relevant, k)           # type: ignore[arg-type]
        q_prec = precision_at_k(retrieved, relevant, k)       # type: ignore[arg-type]
        s_ndcg += q_ndcg; s_mrr += q_mrr; s_rec += q_rec; s_prec += q_prec
        per.append({"ndcg": q_ndcg, "mrr": q_mrr, "recall": q_rec, "precision": q_prec})
    return EvalResult(k, n, s_ndcg / n, s_mrr / n, s_rec / n, s_prec / n, per)
