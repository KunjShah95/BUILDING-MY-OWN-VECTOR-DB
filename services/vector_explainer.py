"""Vector Explainer — Phase 15: Vector-Native AI Features.

Explains why a result matched a query by decomposing the dot product into
per-dimension contributions and generating a natural language summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    query_id: Optional[str]
    result_id: Optional[str]
    semantic_similarity: float
    matched_dimensions: List[Dict[str, Any]]    # [{dim, label, contribution}, ...]
    dimension_labels: List[str]
    natural_language_summary: str
    metadata_highlights: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class VectorExplainer:
    """Explain vector search matches via dimension-level contribution analysis."""

    def explain(
        self,
        query_vector: List[float],
        result_vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        top_k_dims: int = 10,
        query_id: Optional[str] = None,
        result_id: Optional[str] = None,
    ) -> Explanation:
        """Explain a single match.

        contribution[i] = query[i] * result[i]  (element-wise dot product).
        Top positive contributions indicate which dimensions drove the match.
        """
        qv = np.array(query_vector, dtype=np.float64)
        rv = np.array(result_vector, dtype=np.float64)

        qn = np.linalg.norm(qv)
        rn = np.linalg.norm(rv)
        cosine = float(qv @ rv / (qn * rn)) if qn > 0 and rn > 0 else 0.0

        contributions = qv * rv   # element-wise
        top_indices = np.argsort(np.abs(contributions))[::-1][:top_k_dims]

        dim_labels = [f"dim_{i}" for i in range(len(qv))]
        matched_dims = []
        for idx in top_indices:
            c = float(contributions[idx])
            label = dim_labels[idx]
            direction = "positive" if c > 0 else "negative"
            matched_dims.append({
                "dim": int(idx),
                "label": label,
                "contribution": round(c, 6),
                "direction": direction,
                "query_value": round(float(qv[idx]), 6),
                "result_value": round(float(rv[idx]), 6),
            })

        # Metadata highlights
        meta_highlights: Dict[str, Any] = {}
        if metadata:
            for key, val in metadata.items():
                if isinstance(val, str) and len(val) > 2:
                    meta_highlights[key] = val[:200]

        # Natural language summary
        positive_dims = [d for d in matched_dims if d["direction"] == "positive"]
        negative_dims = [d for d in matched_dims if d["direction"] == "negative"]
        sim_pct = round(cosine * 100, 1)
        summary_parts = [
            f"The result matches the query with {sim_pct}% cosine similarity."
        ]
        if positive_dims:
            top3 = [d["label"] for d in positive_dims[:3]]
            summary_parts.append(
                f"The strongest positive contributions came from {', '.join(top3)}."
            )
        if negative_dims:
            neg3 = [d["label"] for d in negative_dims[:2]]
            summary_parts.append(
                f"Slightly opposing dimensions include {', '.join(neg3)}."
            )
        if meta_highlights:
            first_key = next(iter(meta_highlights))
            summary_parts.append(
                f'Metadata field "{first_key}" may be relevant: '
                f'"{str(meta_highlights[first_key])[:80]}".'
            )
        natural_summary = " ".join(summary_parts)

        return Explanation(
            query_id=query_id,
            result_id=result_id,
            semantic_similarity=round(cosine, 6),
            matched_dimensions=matched_dims,
            dimension_labels=dim_labels[:top_k_dims],
            natural_language_summary=natural_summary,
            metadata_highlights=meta_highlights,
        )

    def explain_batch(
        self,
        query_vector: List[float],
        results: List[Dict[str, Any]],
        top_k_dims: int = 10,
    ) -> List[Explanation]:
        """Explain multiple results for a single query vector."""
        explanations: List[Explanation] = []
        for r in results:
            rv = r.get("vector") or r.get("result_vector")
            if rv is None:
                continue
            meta = r.get("metadata") or {}
            rid = r.get("vector_id") or r.get("result_id")
            exp = self.explain(
                query_vector=query_vector,
                result_vector=rv,
                metadata=meta,
                top_k_dims=top_k_dims,
                result_id=rid,
            )
            explanations.append(exp)
        return explanations
