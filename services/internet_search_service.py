"""Internet-scale search service.

Free/open providers (no paid API required by default):
  1. DuckDuckGo     — no key, uses duckduckgo-search library
  2. SearXNG        — self-hosted, set SEARXNG_URL (e.g. http://searxng:8080)
  3. Serper.dev     — free tier 2500/mo, set SERPER_API_KEY
  4. Tavily         — free tier 1000/mo, set TAVILY_API_KEY
  5. Exa            — free tier 1000/mo, set EXA_API_KEY (neural/semantic)
  6. Brave Search   — free tier 2000/mo, set BRAVE_SEARCH_API_KEY
  7. SerpAPI        — free tier 100/mo,  set SERPAPI_KEY
  8. Local index    — self-crawled corpus (always available)

All results fused via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 8.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(items: List[Dict], source: str) -> List[Dict[str, Any]]:
    """Ensure every result has url/title/snippet/score/source."""
    out = []
    for it in items:
        url = it.get("url") or it.get("href") or it.get("link") or ""
        if not url:
            continue
        out.append({
            "url": url,
            "title": it.get("title", ""),
            "snippet": it.get("snippet") or it.get("body") or it.get("description") or it.get("text", ""),
            "score": float(it.get("score", 1.0)),
            "source": source,
        })
    return out


# ── Provider: DuckDuckGo (no key) ────────────────────────────────────────────

def _ddg_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=count))
        return _norm(raw, "duckduckgo")
    except ImportError:
        logger.debug("duckduckgo-search not installed: pip install duckduckgo-search")
        return []
    except Exception as exc:
        logger.warning("DuckDuckGo error: %s", exc)
        return []


# ── Provider: SearXNG (self-hosted, no key) ───────────────────────────────────

def _searxng_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    url = os.getenv("SEARXNG_URL", "")
    if not url:
        return []
    try:
        import httpx
        r = httpx.get(
            f"{url.rstrip('/')}/search",
            params={"q": query, "format": "json", "engines": "google,bing,duckduckgo", "results": count},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return _norm(r.json().get("results", []), "searxng")
    except Exception as exc:
        logger.warning("SearXNG error: %s", exc)
        return []


# ── Provider: Serper.dev (free tier, Google results) ─────────────────────────

def _serper_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    key = os.getenv("SERPER_API_KEY", "")
    if not key:
        return []
    try:
        import httpx
        r = httpx.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": min(count, 10)},
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        raw = [{"url": i.get("link"), "title": i.get("title"), "snippet": i.get("snippet")}
               for i in r.json().get("organic", [])]
        return _norm(raw, "serper")
    except Exception as exc:
        logger.warning("Serper error: %s", exc)
        return []


# ── Provider: Tavily (free tier, AI-optimised results) ────────────────────────

def _tavily_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    key = os.getenv("TAVILY_API_KEY", "")
    if not key:
        return []
    try:
        import httpx
        r = httpx.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": min(count, 10), "api_key": key,
                  "include_raw_content": False},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        raw = [{"url": i.get("url"), "title": i.get("title"), "snippet": i.get("content"),
                "score": i.get("score", 1.0)} for i in r.json().get("results", [])]
        return _norm(raw, "tavily")
    except Exception as exc:
        logger.warning("Tavily error: %s", exc)
        return []


# ── Provider: Exa (neural/semantic, free tier) ────────────────────────────────

def _exa_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    key = os.getenv("EXA_API_KEY", "")
    if not key:
        return []
    try:
        import httpx
        r = httpx.post(
            "https://api.exa.ai/search",
            json={"query": query, "numResults": min(count, 10), "useAutoprompt": True, "type": "neural"},
            headers={"x-api-key": key, "Content-Type": "application/json"},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        raw = [{"url": i.get("url"), "title": i.get("title"),
                "snippet": i.get("text") or (i.get("highlights") or [""])[0],
                "score": i.get("score", 1.0)} for i in r.json().get("results", [])]
        return _norm(raw, "exa")
    except Exception as exc:
        logger.warning("Exa error: %s", exc)
        return []


# ── Provider: Brave Search (free tier) ────────────────────────────────────────

def _brave_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    if not key:
        return []
    try:
        import httpx
        r = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(count, 20), "safesearch": "moderate"},
            headers={"Accept": "application/json", "X-Subscription-Token": key},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        raw = [{"url": i.get("url"), "title": i.get("title"), "snippet": i.get("description")}
               for i in r.json().get("web", {}).get("results", [])]
        return _norm(raw, "brave")
    except Exception as exc:
        logger.warning("Brave error: %s", exc)
        return []


# ── Provider: SerpAPI (free tier) ────────────────────────────────────────────

def _serpapi_search(query: str, count: int = 10) -> List[Dict[str, Any]]:
    key = os.getenv("SERPAPI_KEY", "")
    if not key:
        return []
    try:
        import httpx
        r = httpx.get(
            "https://serpapi.com/search",
            params={"q": query, "num": min(count, 10), "api_key": key, "engine": "google"},
            timeout=_HTTP_TIMEOUT,
        )
        r.raise_for_status()
        raw = [{"url": i.get("link"), "title": i.get("title"), "snippet": i.get("snippet")}
               for i in r.json().get("organic_results", [])]
        return _norm(raw, "serpapi")
    except Exception as exc:
        logger.warning("SerpAPI error: %s", exc)
        return []


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf_fuse(lists: List[List[Dict[str, Any]]], k: int = 60, top_n: int = 20) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    by_url: Dict[str, Dict[str, Any]] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            url = item.get("url", "")
            if not url:
                continue
            scores[url] = scores.get(url, 0.0) + 1.0 / (k + rank + 1)
            by_url[url] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{**by_url[url], "rrf_score": round(score, 6)} for url, score in ranked]


# ── Registry ──────────────────────────────────────────────────────────────────

_PROVIDERS = {
    "duckduckgo": _ddg_search,
    "searxng":    _searxng_search,
    "serper":     _serper_search,
    "tavily":     _tavily_search,
    "exa":        _exa_search,
    "brave":      _brave_search,
    "serpapi":    _serpapi_search,
}

# Default priority order (free-first)
_DEFAULT_ORDER = ["duckduckgo", "searxng", "serper", "tavily", "exa", "brave", "serpapi"]


def available_providers() -> List[str]:
    """Return providers that are usable right now (key present or no key needed)."""
    avail = []
    if _ddg_search("test", 1) is not None:  # always available if lib installed
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            avail.append("duckduckgo")
        except ImportError:
            pass
    if os.getenv("SEARXNG_URL"):
        avail.append("searxng")
    if os.getenv("SERPER_API_KEY"):
        avail.append("serper")
    if os.getenv("TAVILY_API_KEY"):
        avail.append("tavily")
    if os.getenv("EXA_API_KEY"):
        avail.append("exa")
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        avail.append("brave")
    if os.getenv("SERPAPI_KEY"):
        avail.append("serpapi")
    return avail


def internet_search(
    query: str,
    count: int = 10,
    providers: Optional[List[str]] = None,
    local_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Search via all configured providers and fuse with local index via RRF.

    Returns {success, query, results, providers_used, local_fused, latency_ms}.
    Falls back gracefully: if no provider configured, returns local-only results.
    """
    t0 = time.perf_counter()
    want = list(providers) if providers else _DEFAULT_ORDER
    all_lists: List[List[Dict[str, Any]]] = []
    used: List[str] = []

    for name in want:
        fn = _PROVIDERS.get(name)
        if fn is None:
            continue
        res = fn(query, count)
        if res:
            all_lists.append(res)
            used.append(name)

    if local_results:
        normalised = [
            {
                "url": r.get("url", r.get("metadata", {}).get("url", "")),
                "title": r.get("title", r.get("metadata", {}).get("title", "")),
                "snippet": r.get("snippet") or r.get("text", ""),
                "score": r.get("score", 0.0),
                "source": "local",
            }
            for r in local_results
        ]
        all_lists.append(normalised)

    fused = _rrf_fuse(all_lists, top_n=count) if all_lists else []
    latency = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "success": True,
        "query": query,
        "results": fused,
        "providers_used": used,
        "local_fused": bool(local_results),
        "total": len(fused),
        "latency_ms": latency,
    }
