"""Self-hosted web search layer on top of the Vector DB.

Provides Exa-style (neural) + SerpAPI-style (keyword) web search without any
external search APIs: own crawler -> own dual index -> existing SearchEngineService.

Subpackages:
    crawler/  async fetch, robots, frontier, dedup, HTML extract
    ingest/   crawled page -> dense vector + BM25 sparse index
    query/    query understanding (rewrite / expansion / multi-query)
    eval/     NDCG / MRR / Recall@k / Precision@k harness
    recrawl   freshness tracker + adaptive recrawl scheduling

Status: MVP complete through Polish (crawler, dual-index ingest, query intel,
eval, freshness). Search + crawl served at /api/web/search and /api/web/crawl
via SearchEngineService. 18 tests in search_engine/tests/.
"""

__all__ = ["crawler", "ingest", "query", "eval"]
