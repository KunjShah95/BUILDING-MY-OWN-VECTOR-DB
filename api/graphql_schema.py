"""GraphQL schema for the Vector DB — Phase 18.

Exposes core operations as a typed GraphQL API alongside the existing REST API.
Uses Strawberry (https://strawberry.rocks) mounted at /graphql.

Supported operations
--------------------
Queries
  search(query, collectionId, topK, mode)  → [SearchResult]
  collections()                             → [Collection]
  vectorInfo(vectorId, collectionId)        → VectorInfo | null
  indexStats(collectionId)                  → IndexStats
  webSearch(query, k, collectionId)         → WebSearchResponse

Mutations
  insertVector(collectionId, vectorId, vector, metadata)  → MutationResult
  deleteVector(vectorId, collectionId)                    → MutationResult
  crawlUrl(seeds, collectionId, maxPages)                 → CrawlJob
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.scalars import JSON

logger = logging.getLogger(__name__)


# ── Types ─────────────────────────────────────────────────────────────────────

@strawberry.type
class SearchResult:
    vector_id: str
    score: float
    metadata: Optional[str] = None  # JSON-serialised metadata blob

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SearchResult":
        meta = d.get("metadata")
        return SearchResult(
            vector_id=str(d.get("vector_id", d.get("id", ""))),
            score=float(d.get("score", d.get("distance", 0.0))),
            metadata=json.dumps(meta) if meta is not None else None,
        )


@strawberry.type
class WebResult:
    url: str
    title: str
    snippet: str
    score: float

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WebResult":
        return WebResult(
            url=d.get("url", ""),
            title=d.get("title", ""),
            snippet=d.get("snippet") or d.get("text", ""),
            score=float(d.get("score", 0.0)),
        )


@strawberry.type
class WebSearchResponse:
    query: str
    route: str
    results: List[WebResult]
    total: int


@strawberry.type
class Collection:
    collection_id: str
    vector_count: int
    has_sparse: bool = False


@strawberry.type
class VectorInfo:
    vector_id: str
    collection_id: str
    dimension: int
    metadata: Optional[str] = None


@strawberry.type
class IndexStats:
    collection_id: str
    total_docs: int
    index_types: List[str]
    cache_entries: int


@strawberry.type
class MutationResult:
    success: bool
    message: str
    vector_id: Optional[str] = None


@strawberry.type
class CrawlJob:
    job_id: str
    status: str
    seeds: List[str]
    collection_id: str


# Patchable in tests
_INDEX_ROOT_FOR_COLLECTIONS = "indexes"


# ── DB helpers ────────────────────────────────────────────────────────────────

def _get_db():
    from config.database import SessionLocal
    return SessionLocal()


def _embed(text: str) -> List[float]:
    from services.embedding_service import embed_text
    return embed_text(text)


# ── Query ─────────────────────────────────────────────────────────────────────

@strawberry.type
class Query:
    @strawberry.field(description="Hybrid vector+keyword search")
    def search(
        self,
        query: str,
        collection_id: str = "default",
        top_k: int = 10,
        mode: str = "hybrid",  # hybrid | neural | keyword
    ) -> List[SearchResult]:
        db = _get_db()
        try:
            from services.vector_service import VectorService
            from services.search_engine_service import SearchEngineService

            vs = VectorService(db)
            engine = SearchEngineService(vs)

            try:
                qvec = _embed(query)
            except Exception:
                qvec = None

            if mode == "keyword" or qvec is None:
                raw = engine.keyword_search(query=query, collection_id=collection_id, top_k=top_k)
            else:
                raw = engine.search(query=query, query_vector=qvec,
                                    collection_id=collection_id, top_k=top_k)

            return [SearchResult.from_dict(r) for r in raw.get("results", [])]
        except Exception as exc:
            logger.error("GraphQL search error: %s", exc)
            return []
        finally:
            db.close()

    @strawberry.field(description="List all collections")
    def collections(self) -> List[Collection]:
        import os
        import api.graphql_schema as _self_mod
        index_root = _self_mod._INDEX_ROOT_FOR_COLLECTIONS
        result: List[Collection] = []
        if not os.path.isdir(index_root):
            return result
        for name in os.listdir(index_root):
            col_dir = os.path.join(index_root, name)
            if not os.path.isdir(col_dir):
                continue
            sparse = os.path.isfile(os.path.join(col_dir, "sparse.json"))
            # Approximate vector count from sparse index if available
            count = 0
            sparse_path = os.path.join(col_dir, "sparse.json")
            if sparse and os.path.isfile(sparse_path):
                try:
                    data = json.loads(open(sparse_path).read())
                    count = data.get("doc_count", len(data.get("docs", {})))
                except Exception:
                    pass
            result.append(Collection(collection_id=name, vector_count=count, has_sparse=sparse))
        return result

    @strawberry.field(description="Get metadata for a specific vector")
    def vector_info(
        self,
        vector_id: str,
        collection_id: str = "default",
    ) -> Optional[VectorInfo]:
        db = _get_db()
        try:
            from services.vector_service import VectorService
            vs = VectorService(db)
            resp = vs.get_vector(vector_id)
            if not resp.get("success"):
                return None
            vec = resp.get("vector", {})
            vdata = vec.get("vector_data", [])
            return VectorInfo(
                vector_id=vector_id,
                collection_id=collection_id,
                dimension=len(vdata),
                metadata=json.dumps(vec.get("metadata")) if vec.get("metadata") else None,
            )
        except Exception as exc:
            logger.error("vector_info error: %s", exc)
            return None
        finally:
            db.close()

    @strawberry.field(description="Index and cache statistics for a collection")
    def index_stats(self, collection_id: str = "web") -> IndexStats:
        import os
        import api.graphql_schema as _self_mod
        index_root = _self_mod._INDEX_ROOT_FOR_COLLECTIONS
        col_dir = os.path.join(index_root, collection_id)
        index_types: List[str] = []
        if os.path.isdir(col_dir):
            for f in os.listdir(col_dir):
                if f.endswith(".json"):
                    index_types.append(f.replace(".json", ""))
                elif os.path.isdir(os.path.join(col_dir, f)):
                    index_types.append(f)

        # sparse doc count as proxy for total_docs
        total = 0
        sp = os.path.join(col_dir, "sparse.json")
        if os.path.isfile(sp):
            try:
                data = json.loads(open(sp).read())
                total = data.get("doc_count", len(data.get("docs", {})))
            except Exception:
                pass

        from api.routers.web_search import _search_cache
        return IndexStats(
            collection_id=collection_id,
            total_docs=total,
            index_types=index_types,
            cache_entries=len(_search_cache),
        )

    @strawberry.field(description="Web search (neural+keyword hybrid over crawled corpus)")
    def web_search(
        self,
        query: str,
        k: int = 10,
        collection_id: str = "web",
    ) -> WebSearchResponse:
        db = _get_db()
        try:
            from services.vector_service import VectorService
            from services.search_engine_service import SearchEngineService
            from search_engine.query import expand_query, route

            detected = route(query)
            try:
                qvec = _embed(expand_query(query))
            except Exception:
                qvec = [0.0] * 384

            engine = SearchEngineService(VectorService(db))
            raw = engine.search(query=query, query_vector=qvec,
                                collection_id=collection_id, top_k=k)
            results = [WebResult.from_dict(r) for r in raw.get("results", [])]
            return WebSearchResponse(
                query=query, route=detected,
                results=results, total=len(results),
            )
        except Exception as exc:
            logger.error("web_search error: %s", exc)
            return WebSearchResponse(query=query, route="error", results=[], total=0)
        finally:
            db.close()


# ── Mutation ──────────────────────────────────────────────────────────────────

@strawberry.type
class Mutation:
    @strawberry.mutation(description="Insert a vector into a collection")
    def insert_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: List[float],
        metadata: Optional[str] = None,
    ) -> MutationResult:
        db = _get_db()
        try:
            from services.vector_service import VectorService
            vs = VectorService(db)
            meta = json.loads(metadata) if metadata else None
            resp = vs.store_vector(
                vector_id=vector_id,
                vector_data=vector,
                metadata=meta,
                collection_id=collection_id,
            )
            return MutationResult(
                success=bool(resp.get("success")),
                message=resp.get("message", ""),
                vector_id=vector_id,
            )
        except Exception as exc:
            return MutationResult(success=False, message=str(exc))
        finally:
            db.close()

    @strawberry.mutation(description="Delete a vector from a collection")
    def delete_vector(
        self,
        vector_id: str,
        collection_id: str = "default",
    ) -> MutationResult:
        db = _get_db()
        try:
            from services.vector_service import VectorService
            vs = VectorService(db)
            resp = vs.delete_vector(vector_id)
            return MutationResult(
                success=bool(resp.get("success")),
                message=resp.get("message", ""),
                vector_id=vector_id,
            )
        except Exception as exc:
            return MutationResult(success=False, message=str(exc))
        finally:
            db.close()

    @strawberry.mutation(description="Trigger a web crawl job")
    def crawl_url(
        self,
        seeds: List[str],
        collection_id: str = "web",
        max_pages: int = 50,
    ) -> CrawlJob:
        import uuid
        import time
        from api.routers.web_search import _crawl_jobs
        job_id = uuid.uuid4().hex[:8]
        _crawl_jobs[job_id] = {
            "status": "queued", "pages": 0, "ingested": 0,
            "seeds": seeds, "collection_id": collection_id,
            "started_at": time.time(),
        }
        # Fire and forget in a thread to avoid blocking GraphQL response
        import threading
        from api.routers.web_search import _run_crawl
        t = threading.Thread(
            target=_run_crawl,
            args=(seeds, collection_id, max_pages, 3, True, job_id),
            daemon=True,
        )
        t.start()
        return CrawlJob(job_id=job_id, status="queued",
                        seeds=seeds, collection_id=collection_id)


# ── Schema + Router ───────────────────────────────────────────────────────────

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_router = GraphQLRouter(schema, path="/graphql")
