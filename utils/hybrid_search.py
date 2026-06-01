"""Hybrid search engine: fuses dense vector results with sparse BM25 results.

Uses Reciprocal Rank Fusion (RRF) to combine ranked lists from different
retrieval methods into a single consensus ranking.

RRF formula:
    RRF(d) = Σ 1/(k + rank_i(d))   for each result set i

Where rank_i(d) is the rank of document d in result set i, and k is a
regularisation constant (default 60).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fuse dense and sparse result lists via Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_results : list of dict
        Dense search results, each containing at least ``vector_id``.
        Typically from HNSW/IVF/brute-force search.
    sparse_results : list of dict
        Sparse (BM25) results, each containing at least ``vector_id``.
    k : int
        RRF ranking constant (default 60).  Larger values smooth the
        contribution of high ranks.
    top_n : int or None
        Number of fused results to return.  Defaults to ``len(dense_results)``.

    Returns
    -------
    list of dict
        Fused results sorted by RRF score descending.  Each dict contains
        ``vector_id``, ``distance`` (inverted normalised RRF score), and
        ``rrf_score``.
    """
    if not dense_results and not sparse_results:
        return []

    # Build RRF score accumulator
    rrf_scores: Dict[str, float] = defaultdict(float)
    result_map: Dict[str, Dict[str, Any]] = {}

    def _add_set(results: List[Dict[str, Any]], source: str) -> None:
        for rank, r in enumerate(results):
            vid = r.get("vector_id") or r.get("id")
            if not vid:
                continue
            rrf_scores[vid] += 1.0 / (k + rank + 1)
            if vid not in result_map:
                result_map[vid] = {
                    "vector_id": vid,
                    "distance": 1.0,
                    "metadata": r.get("metadata"),
                    "sources": [],
                }
            result_map[vid]["sources"].append(source)
            # Store the best (lowest) dense distance as the combined distance
            dist = r.get("distance", 1.0)
            if isinstance(dist, (int, float)) and dist < result_map[vid].get("distance", 1.0):
                result_map[vid]["distance"] = dist

    _add_set(dense_results, "dense")
    _add_set(sparse_results, "sparse")

    # Normalise RRF scores to [0, 1] and invert for distance-like metric
    if rrf_scores:
        max_score = max(rrf_scores.values())
        for vid in rrf_scores:
            rrf_scores[vid] /= max_score  # normalise to [0, 1]

    # Build output sorted by RRF score descending
    fused = sorted(rrf_scores.keys(), key=lambda v: -rrf_scores[v])
    if top_n is not None:
        fused = fused[:top_n]

    results = []
    for vid in fused:
        entry = result_map[vid]
        entry["rrf_score"] = round(rrf_scores[vid], 6)
        entry["distance"] = round(1.0 - rrf_scores[vid], 6)  # invert so lower = better
        results.append(entry)

    return results


class HybridSearchEngine:
    """Orchestrates dense + sparse search and fusion.

    Simplifies calling hybrid search from service layers that maintain their
    own dense index and BM25 index.
    """

    def __init__(self, bm25_index=None):
        self.bm25_index = bm25_index

    def search(
        self,
        dense_search_fn,
        query_vector: List[float],
        query_text: str,
        k: int = 10,
        dense_k_multiplier: int = 2,
        **dense_kwargs,
    ) -> Dict[str, Any]:
        """Run hybrid search.

        Parameters
        ----------
        dense_search_fn : callable
            Function that takes ``query_vector``, ``k`` and kwargs and returns
            ``{"results": [...], ...}``.
        query_vector : list of float
            Dense query vector.
        query_text : str
            Text query for BM25 sparse retrieval.
        k : int
            Number of final fused results.
        dense_k_multiplier : int
            Multiplier for dense candidate pool (``k * dense_k_multiplier``).

        Returns
        -------
        dict with keys ``results``, ``total_results``, and metadata.
        """
        candidate_count = k * dense_k_multiplier

        # 1. Dense search
        dense_result = dense_search_fn(query_vector=query_vector, k=candidate_count, **dense_kwargs)
        dense_results = dense_result.get("results", [])

        # 2. Sparse BM25 search
        sparse_results: List[Dict[str, Any]] = []
        if query_text and self.bm25_index is not None:
            try:
                sparse_raw = self.bm25_index.search(query_text, k=candidate_count)
                for doc_id, bm25_score in sparse_raw:
                    # Convert BM25 score to a distance-like metric (lower = better)
                    norm_score = min(bm25_score / 10.0, 1.0) if bm25_score > 0 else 1.0
                    sparse_results.append({
                        "vector_id": doc_id,
                        "distance": round(1.0 - norm_score, 6),
                    })
            except Exception:
                pass

        # 3. Fuse via RRF
        fused = reciprocal_rank_fusion(dense_results, sparse_results, top_n=k)

        return {
            "success": True,
            "results": fused,
            "total_results": len(fused),
            "method": "hybrid",
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
        }
