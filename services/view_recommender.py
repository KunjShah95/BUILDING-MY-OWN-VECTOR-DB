"""Materialized view auto-recommender based on repeated query patterns."""
import math
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

from services.materialized_views import MaterializedViewService, mat_view_service

logger = logging.getLogger(__name__)


@dataclass
class ViewRecommendation:
    recommendation_id: str
    collection_id: str
    filter_pattern: Optional[Dict]
    representative_vector: List[float]
    estimated_speedup: float
    query_count: int
    applied: bool = False


@dataclass
class _QueryRecord:
    query_vector: List[float]
    k: int
    filters: Optional[Dict]
    collection_id: str


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class ViewRecommender:
    """Logs queries and identifies patterns that benefit from materialized views."""

    SIMILARITY_THRESHOLD = 0.95

    def __init__(self, mv_service: Optional[MaterializedViewService] = None):
        self._lock = threading.Lock()
        self._records: List[_QueryRecord] = []
        self._recommendations: Dict[str, ViewRecommendation] = {}
        self._mv_service = mv_service or mat_view_service

    def record_query(
        self,
        query_vector: List[float],
        k: int,
        filters: Optional[Dict],
        collection_id: str,
    ) -> None:
        with self._lock:
            self._records.append(_QueryRecord(query_vector, k, filters, collection_id))
            if len(self._records) > 100_000:
                self._records = self._records[-50_000:]

    def analyze(self, min_frequency: int = 10) -> List[ViewRecommendation]:
        """Cluster similar queries and return recommendations for frequent clusters."""
        with self._lock:
            records = list(self._records)

        if not records:
            return []

        # Simple greedy clustering by cosine similarity
        clusters: List[List[_QueryRecord]] = []
        for rec in records:
            placed = False
            for cluster in clusters:
                centroid = cluster[0].query_vector
                if (
                    cluster[0].collection_id == rec.collection_id
                    and _cosine_similarity(centroid, rec.query_vector) >= self.SIMILARITY_THRESHOLD
                ):
                    cluster.append(rec)
                    placed = True
                    break
            if not placed:
                clusters.append([rec])

        recommendations: List[ViewRecommendation] = []
        for cluster in clusters:
            if len(cluster) < min_frequency:
                continue
            # Use first record as representative
            rep = cluster[0]
            # Estimate speedup: more queries = higher speedup (capped at 10x)
            speedup = min(10.0, 1.0 + len(cluster) / 20.0)
            rec_id = str(uuid.uuid4())
            recommendation = ViewRecommendation(
                recommendation_id=rec_id,
                collection_id=rep.collection_id,
                filter_pattern=rep.filters,
                representative_vector=rep.query_vector,
                estimated_speedup=round(speedup, 2),
                query_count=len(cluster),
            )
            recommendations.append(recommendation)

        # Store for later retrieval
        with self._lock:
            for r in recommendations:
                self._recommendations[r.recommendation_id] = r

        return recommendations

    def get_recommendations(self) -> List[ViewRecommendation]:
        with self._lock:
            return list(self._recommendations.values())

    def get_recommendation(self, recommendation_id: str) -> Optional[ViewRecommendation]:
        with self._lock:
            return self._recommendations.get(recommendation_id)

    def auto_create(self, recommendation: ViewRecommendation) -> str:
        """Call existing MaterializedViewService to create the view."""
        view_id = self._mv_service.create_view(
            name=f"auto_{recommendation.collection_id}_{recommendation.recommendation_id[:8]}",
            collection_id=recommendation.collection_id,
            query_embedding=recommendation.representative_vector,
            k=100,
            refresh_interval=300,
        )
        with self._lock:
            if recommendation.recommendation_id in self._recommendations:
                self._recommendations[recommendation.recommendation_id].applied = True
        logger.info("Auto-created view %s from recommendation %s", view_id, recommendation.recommendation_id)
        return view_id


view_recommender = ViewRecommender()
