"""Cross-encoder re-ranking for improved search accuracy.

Uses a cross-encoder model (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2) to
score (query, document) pairs directly, yielding much more accurate relevance
judgments than cosine similarity on dense embeddings.

Falls back to Cohere's re-rank API if the local model is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _extract_text(result: Dict[str, Any]) -> str:
    """Extract the text content from a search result dict, checking common keys."""
    # Check top-level + nested metadata keys
    meta = result.get("metadata") or result.get("meta_data") or {}
    if isinstance(meta, dict):
        for key in ("text", "content", "chunk_text", "description", "title"):
            val = meta.get(key)
            if val and isinstance(val, str) and val.strip():
                return val.strip()
    # Check top-level text keys
    for key in ("text", "content", "chunk_text", "description"):
        val = result.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    # Last resort: vector_id
    return result.get("vector_id", "")


class RerankerService:
    """Cross-encoder re-ranker with optional Cohere API fallback.

    Parameters
    ----------
    model_name : str
        Cross-encoder model name (default: cross-encoder/ms-marco-MiniLM-L-6-v2).
    cohere_api_key : str, optional
        Cohere API key for cloud re-ranking fallback. If None, Cohere is skipped.
    cohere_model : str
        Cohere re-rank model name (default: rerank-english-v3.0).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cohere_api_key: Optional[str] = None,
        cohere_model: str = "rerank-english-v3.0",
    ):
        self.model_name = model_name
        self.cohere_api_key = cohere_api_key
        self.cohere_model = cohere_model
        self._model = None
        self._model_loaded = False
        self._load_error: Optional[str] = None

    def _load_model(self) -> bool:
        """Lazy-load the cross-encoder model. Returns True if available."""
        if self._model_loaded:
            return True
        if self._load_error:
            return False
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            self._model_loaded = True
            logger.info("Cross-encoder model %s loaded", self.model_name)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "Cross-encoder model '%s' not available: %s. "
                "Falling back to vector distance ordering.",
                self.model_name, exc,
            )
            return False

    @property
    def is_available(self) -> bool:
        """Check if the cross-encoder model is loaded or can be loaded."""
        return self._load_model()

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        keep_rerank_score: bool = True,
    ) -> List[Dict[str, Any]]:
        """Re-rank search results using a cross-encoder.

        Parameters
        ----------
        query : str
            The original search query text.
        results : list of dict
            Search results, each containing at least ``vector_id`` and
            optionally ``metadata`` with ``text`` or ``content``.
        top_k : int, optional
            Number of top results to return after re-ranking.
            Defaults to ``len(results)``.
        keep_rerank_score : bool
            If True, attach ``rerank_score`` to each result.

        Returns
        -------
        list of dict
            Results sorted by cross-encoder relevance score (descending).
        """
        if not results:
            return []

        if top_k is None:
            top_k = len(results)

        # Try local cross-encoder model first
        if self._load_model():
            return self._rerank_local(query, results, top_k, keep_rerank_score)

        # Try Cohere API fallback
        if self.cohere_api_key:
            try:
                return self._rerank_cohere(query, results, top_k, keep_rerank_score)
            except Exception as exc:
                logger.warning("Cohere re-rank failed: %s", exc)

        # Fallback: sort by existing distance (no-op, results already sorted)
        logger.info("Cross-encoder unavailable — returning original ordering")
        if top_k and top_k < len(results):
            results = results[:top_k]
        return results

    def _rerank_local(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        keep_rerank_score: bool,
    ) -> List[Dict[str, Any]]:
        """Re-rank using local cross-encoder model."""
        pairs: List[Tuple[str, str]] = []
        valid_indices: List[int] = []
        no_text_indices: List[int] = []

        for i, r in enumerate(results):
            text = _extract_text(r)
            if text:
                pairs.append((query, text))
                valid_indices.append(i)
            else:
                no_text_indices.append(i)

        if not pairs:
            # No text found in any result — return original ordering
            return results[:top_k]

        try:
            scores = self._model.predict(pairs)  # type: ignore
        except Exception as exc:
            logger.error("Cross-encoder predict failed: %s", exc)
            return results[:top_k]

        # Build scored list preserving original result dicts
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores, valid_indices):
            r = results[idx]
            if keep_rerank_score:
                r["rerank_score"] = float(score)
            scored.append((float(score), r))

        # Sort by score descending (higher = more relevant)
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [r for _, r in scored]

        # Append results without text at the end (preserving original order among themselves)
        for idx in no_text_indices:
            reranked.append(results[idx])

        return reranked[:top_k]

    def _rerank_cohere(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        keep_rerank_score: bool,
    ) -> List[Dict[str, Any]]:
        """Re-rank using Cohere's re-rank API."""
        import cohere

        co = cohere.Client(self.cohere_api_key)

        # Gather document texts
        documents: List[str] = []
        valid_indices: List[int] = []

        for i, r in enumerate(results):
            text = _extract_text(r)
            if text:
                documents.append(text)
                valid_indices.append(i)

        if not documents:
            return results[:top_k]

        response = co.rerank(
            model=self.cohere_model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )

        reranked: List[Dict[str, Any]] = []
        for hit in response.results:
            idx = valid_indices[hit.index]
            r = dict(results[idx])  # shallow copy
            if keep_rerank_score:
                r["rerank_score"] = float(hit.relevance_score)
            reranked.append(r)

        return reranked

    def batch_rerank(
        self,
        queries: List[str],
        results_batch: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Re-rank multiple query/result pairs.

        Parameters
        ----------
        queries : list of str
            One query per result list.
        results_batch : list of list of dict
            Corresponding search result lists.
        top_k : int, optional
            Number of top results per query.

        Returns
        -------
        list of list of dict
            Re-ranked results for each query.
        """
        return [
            self.rerank(q, rs, top_k=top_k)
            for q, rs in zip(queries, results_batch)
        ]
