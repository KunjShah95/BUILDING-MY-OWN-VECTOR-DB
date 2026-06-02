"""GraphQL API schema for the Vector Database.

Provides a GraphQL interface to query and mutate vector collections,
search vectors, and manage indexes.

Requires: pip install strawberry-graphql
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import strawberry
from strawberry.types import Info
from sqlalchemy.orm import Session as SASession
import numpy as np


# ---- GraphQL Types ---------------------------------------------------------


@strawberry.type
class CollectionType:
    collection_id: str
    name: str
    modality: str
    dimension: int
    embedding_model: str
    distance_metric: str
    description: Optional[str] = None
    tenant_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CollectionType":
        return cls(
            collection_id=data["collection_id"],
            name=data["name"],
            modality=data["modality"],
            dimension=data["dimension"],
            embedding_model=data["embedding_model"],
            distance_metric=data.get("distance_metric", "cosine"),
            description=data.get("description"),
            tenant_id=data.get("tenant_id"),
        )


@strawberry.type
class VectorType:
    vector_id: str
    vector: List[float]
    metadata: Optional[str] = None  # JSON string
    collection_id: Optional[str] = None
    created_at: Optional[str] = None


@strawberry.type
class SearchHitType:
    vector_id: str
    distance: float
    metadata: Optional[str] = None
    collection_id: Optional[str] = None


@strawberry.type
class SearchResultType:
    success: bool
    results: List[SearchHitType]
    total_results: int
    search_time: float
    method: str


@strawberry.type
class IndexStatsType:
    total_vectors: int
    dimension: int
    index_type: str
    compression_ratio: Optional[float] = None
    is_trained: Optional[bool] = None


@strawberry.type
class HealthType:
    status: str
    database: str
    index_available: bool
    total_vectors: int


@strawberry.type
class MutationResult:
    success: bool
    message: str
    vector_id: Optional[str] = None
    collection_id: Optional[str] = None


# ---- Helper to get DB session from context ---------------------------------


def _get_db(info: Info) -> SASession:
    """Get a SQLAlchemy session for this request.

    Tries to get from GraphQL context first; falls back to creating
    a new session (for standalone ASGI mounts without custom context).
    """
    if info.context and "db" in info.context:
        return info.context["db"]
    from config.database import SessionLocal
    return SessionLocal()


# ---- Queries ---------------------------------------------------------------


@strawberry.type
class Query:
    @strawberry.field
    def health(self, info: Info) -> HealthType:
        """Check database and index health."""
        from services.vector_service import VectorService
        svc = VectorService(_get_db(info))
        h = svc.get_health_status()
        return HealthType(
            status=h.get("status", "unknown"),
            database=h.get("database", "unknown"),
            index_available=h.get("index_available", False),
            total_vectors=h.get("total_vectors", 0),
        )

    @strawberry.field
    def collections(
        self,
        info: Info,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CollectionType]:
        """List all collections."""
        from services.collection_service import CollectionService
        svc = CollectionService(_get_db(info))
        result = svc.list_collections(limit=limit, offset=offset)
        if not result.get("success"):
            return []
        return [
            CollectionType.from_dict(c) for c in result.get("collections", [])
        ]

    @strawberry.field
    def collection(
        self, info: Info, collection_id: str
    ) -> Optional[CollectionType]:
        """Get a collection by ID."""
        from services.collection_service import CollectionService
        svc = CollectionService(_get_db(info))
        result = svc.get_collection(collection_id)
        if not result.get("success"):
            return None
        return CollectionType.from_dict(result["collection"])

    @strawberry.field
    def search_vectors(
        self,
        info: Info,
        query_vector: List[float],
        k: int = 10,
        method: str = "hnsw",
        collection_id: Optional[str] = None,
        filters: Optional[str] = None,
    ) -> SearchResultType:
        """Search for similar vectors using ANN."""
        import json, time
        from services.vector_service import VectorService

        svc = VectorService(_get_db(info))
        parsed_filters = json.loads(filters) if filters else None

        start = time.time()
        result = svc.search_vectors(
            query_vector=query_vector,
            k=k,
            method=method,
            collection_id=collection_id,
            filters=parsed_filters,
        )
        elapsed = time.time() - start

        hits = [
            SearchHitType(
                vector_id=r.get("vector_id", ""),
                distance=r.get("distance", 0.0),
                metadata=json.dumps(r.get("metadata", {})),
                collection_id=r.get("collection_id"),
            )
            for r in result.get("results", [])
        ]
        return SearchResultType(
            success=result.get("success", False),
            results=hits,
            total_results=result.get("total_results", 0),
            search_time=elapsed,
            method=result.get("method", method),
        )

    @strawberry.field
    def index_stats(
        self,
        info: Info,
        method: str = "hnsw",
        collection_id: Optional[str] = None,
    ) -> Optional[IndexStatsType]:
        """Get index statistics."""
        from services.vector_service import VectorService
        svc = VectorService(_get_db(info))
        result = svc.get_index_info(method)
        if not result.get("success"):
            return None
        stats = result.get("stats", result.get("index_info", {}))
        return IndexStatsType(
            total_vectors=stats.get("total_vectors", stats.get("total_nodes", 0)),
            dimension=stats.get("dimension", 0),
            index_type=method,
            compression_ratio=stats.get("compression_ratio"),
            is_trained=stats.get("is_trained"),
        )


# ---- Mutations -------------------------------------------------------------


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_collection(
        self,
        info: Info,
        name: str,
        collection_id: Optional[str] = None,
        modality: str = "text",
        dimension: Optional[int] = None,
        distance_metric: str = "cosine",
        description: Optional[str] = None,
    ) -> MutationResult:
        """Create a new collection namespace."""
        from services.collection_service import CollectionService
        svc = CollectionService(_get_db(info))
        result = svc.create_collection(
            name=name,
            collection_id=collection_id,
            modality=modality,
            dimension=dimension,
            distance_metric=distance_metric,
            description=description,
        )
        return MutationResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            collection_id=result.get("collection", {}).get("collection_id"),
        )

    @strawberry.mutation
    def create_vector(
        self,
        info: Info,
        vector: List[float],
        metadata: Optional[str] = None,
        vector_id: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> MutationResult:
        """Insert a new vector."""
        import json
        from services.vector_service import VectorService
        svc = VectorService(_get_db(info))
        parsed_meta = json.loads(metadata) if metadata else None
        result = svc.create_vector(
            vector_data=vector,
            metadata=parsed_meta,
            vector_id=vector_id,
            collection_id=collection_id,
        )
        return MutationResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            vector_id=result.get("vector_id"),
        )

    @strawberry.mutation
    def delete_vector(
        self,
        info: Info,
        vector_id: str,
    ) -> MutationResult:
        """Delete a vector by ID."""
        from services.vector_service import VectorService
        svc = VectorService(_get_db(info))
        result = svc.delete_vector(vector_id)
        return MutationResult(
            success=result.get("success", False),
            message=result.get("message", ""),
        )

    @strawberry.mutation
    def build_index(
        self,
        info: Info,
        method: str = "hnsw",
        collection_id: Optional[str] = None,
        m: int = 16,
        ef_construction: int = 200,
        n_clusters: int = 100,
        n_probes: int = 10,
    ) -> MutationResult:
        """Build an ANN index (HNSW or IVF)."""
        from services.vector_service import VectorService
        svc = VectorService(_get_db(info))
        result = svc.create_index(
            method=method,
            m=m,
            ef_construction=ef_construction,
            n_clusters=n_clusters,
            n_probes=n_probes,
            collection_id=collection_id,
        )
        return MutationResult(
            success=result.get("success", False),
            message=result.get("message", ""),
        )

    @strawberry.mutation
    def ingest_text(
        self,
        info: Info,
        collection_id: str,
        text: str,
        metadata: Optional[str] = None,
    ) -> MutationResult:
        """Embed and store text in a collection."""
        import json
        from services.multimodal_service import MultimodalService
        svc = MultimodalService(_get_db(info))
        parsed_meta = json.loads(metadata) if metadata else None
        result = svc.ingest_text(
            collection_id=collection_id,
            text=text,
            metadata=parsed_meta,
        )
        return MutationResult(
            success=result.get("success", False),
            message=result.get("message", ""),
            vector_id=result.get("vector_id"),
        )


# ---- Schema -----------------------------------------------------------------

schema = strawberry.Schema(query=Query, mutation=Mutation)
