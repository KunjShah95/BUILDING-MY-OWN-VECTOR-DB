"""Query understanding — dependency-free rule layer with optional LLM hooks.

Goals (cheap, deterministic, no network):
  - normalize: whitespace/case cleanup, strip junk
  - expand_query: append domain synonyms/abbreviations to widen recall
  - multi_query: produce N query variants for multi-query retrieval
  - rewrite: lightweight rewrite (abbrev expansion + de-noising)

LLM-backed HyDE / rewrite can be layered on later via `rewrite(..., llm=fn)`.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional

_WS_RE = re.compile(r"\s+")

# Common technical abbreviations -> expansions (bidirectional recall boost).
_ABBREV = {
    "db": "database",
    "ann": "approximate nearest neighbor",
    "knn": "k nearest neighbor",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "pq": "product quantization",
    "ivf": "inverted file index",
    "hnsw": "hierarchical navigable small world",
}

# Light synonym groups for query expansion.
_SYNONYMS = {
    "fast": ["quick", "low latency"],
    "search": ["retrieval", "lookup"],
    "build": ["create", "implement"],
    "vector": ["embedding"],
    "similar": ["nearest", "related"],
}

_STOP_NOISE = {"please", "kindly", "could", "would", "tell", "me", "show"}


def normalize(query: str) -> str:
    """Trim, collapse whitespace, drop leading/trailing punctuation noise."""
    q = _WS_RE.sub(" ", query.strip())
    return q


def _tokens(query: str) -> List[str]:
    return re.findall(r"\b\w+\b", query.lower())


def expand_query(query: str, max_extra: int = 6) -> str:
    """Append abbreviation expansions + a few synonyms to the original query.

    Returns a single string (original first, so exact terms keep their weight).
    Conservative: bounded by *max_extra* added terms to avoid topic drift.
    """
    q = normalize(query)
    extra: List[str] = []
    seen = set(_tokens(q))
    for tok in _tokens(q):
        if tok in _ABBREV:
            for w in _ABBREV[tok].split():
                if w not in seen:
                    extra.append(w)
                    seen.add(w)
        if tok in _SYNONYMS:
            for syn in _SYNONYMS[tok]:
                if syn not in seen:
                    extra.append(syn)
                    seen.add(syn)
        if len(extra) >= max_extra:
            break
    if not extra:
        return q
    return q + " " + " ".join(extra[:max_extra])


def rewrite(query: str, llm: Optional[Callable[[str], str]] = None) -> str:
    """De-noise the query; if an LLM fn is supplied, defer to it."""
    if llm is not None:
        try:
            out = llm(query)
            if out and out.strip():
                return normalize(out)
        except Exception:
            pass
    toks = [t for t in _tokens(query) if t not in _STOP_NOISE]
    return normalize(" ".join(toks)) if toks else normalize(query)


_KEYWORD_SIGNALS = {"site:", "filetype:", "intitle:", '"', "AND", "OR", "NOT"}
_SEMANTIC_SIGNALS = {"how", "why", "explain", "difference", "meaning",
                     "understand", "concept", "overview", "compare", "describe"}


def route(query: str) -> str:
    """Return 'keyword', 'neural', or 'hybrid' for a query.

    Cheap, dependency-free heuristic. Used by the web search router to choose
    whether to enable query expansion and reranking.
    """
    raw = query.strip()
    # Operator syntax → pure keyword
    if any(sig in raw for sig in _KEYWORD_SIGNALS):
        return "keyword"
    toks = _tokens(raw)
    # Very short queries with no question words → keyword (exact lookup)
    if len(toks) <= 2:
        return "keyword"
    # Conceptual / question → neural
    if any(t in _SEMANTIC_SIGNALS for t in toks):
        return "neural"
    return "hybrid"


def multi_query(query: str, n: int = 3) -> List[str]:
    """Produce up to *n* query variants for multi-query retrieval (recall++)."""
    variants = [normalize(query)]
    rw = rewrite(query)
    if rw and rw not in variants:
        variants.append(rw)
    exp = expand_query(query)
    if exp not in variants:
        variants.append(exp)
    return variants[:n]
