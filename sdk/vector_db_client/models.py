from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Collection:
    collection_id: str
    name: str
    modality: str
    dimension: int
    embedding_model: str
    distance_metric: str = "cosine"
    description: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "Collection":
        return cls(
            collection_id=data["collection_id"],
            name=data["name"],
            modality=data["modality"],
            dimension=data["dimension"],
            embedding_model=data["embedding_model"],
            distance_metric=data.get("distance_metric", "cosine"),
            description=data.get("description"),
            raw=data,
        )


@dataclass
class SearchHit:
    vector_id: str
    distance: float
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_api(cls, row: Dict[str, Any]) -> "SearchHit":
        return cls(
            vector_id=row["vector_id"],
            distance=row["distance"],
            metadata=row.get("metadata") or row.get("meta_data"),
        )


@dataclass
class SearchResult:
    success: bool
    results: List[SearchHit]
    total_results: int
    search_time: float
    method: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "SearchResult":
        hits = [SearchHit.from_api(r) for r in data.get("results", [])]
        return cls(
            success=data.get("success", False),
            results=hits,
            total_results=data.get("total_results", len(hits)),
            search_time=data.get("search_time", 0.0),
            method=data.get("method", ""),
            raw=data,
        )
