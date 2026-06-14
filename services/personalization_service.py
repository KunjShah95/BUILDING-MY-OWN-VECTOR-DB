"""
Personalization Engine (Phase 7: Production Search Orchestration).

Injects user-profile embeddings into query formulation to personalize search
results. Maintains per-user interaction history and learns preference vectors.

Architecture:
  - User profiles: stored embeddings + behavior history
  - Query personalization: blend user embedding with query embedding
  - Result personalization: boost results matching user's preferences
  - Session-level personalization: recent interaction context
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UserProfile:
    """In-memory user profile with preference vector and interaction history."""

    user_id: str
    preference_vector: Optional[List[float]] = None
    dim: int = 384

    # Interaction history
    query_history: List[str] = field(default_factory=lambda: deque(maxlen=100))
    clicked_ids: Set[str] = field(default_factory=set)
    positive_ids: Set[str] = field(default_factory=set)
    negative_ids: Set[str] = field(default_factory=set)
    total_queries: int = 0

    # Session context
    session_queries: List[str] = field(default_factory=lambda: deque(maxlen=10))
    session_clicked: Set[str] = field(default_factory=set)

    def update_preference(self, vector: List[float], alpha: float = 0.1):
        """Update the user's preference vector using exponential moving average."""
        arr = np.array(vector, dtype=np.float32)
        if self.preference_vector is None:
            self.preference_vector = arr.tolist()
        else:
            pref = np.array(self.preference_vector, dtype=np.float32)
            updated = (1 - alpha) * pref + alpha * arr
            norm = np.linalg.norm(updated)
            if norm > 0:
                updated = updated / norm
            self.preference_vector = updated.tolist()

    def record_click(self, vector_id: str, metadata: Optional[Dict] = None):
        self.clicked_ids.add(vector_id)
        self.positive_ids.add(vector_id)
        self.session_clicked.add(vector_id)

    def record_negative(self, vector_id: str):
        self.negative_ids.add(vector_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total_queries": self.total_queries,
            "clicked_count": len(self.clicked_ids),
            "positive_count": len(self.positive_ids),
            "negative_count": len(self.negative_ids),
            "has_preference_vector": self.preference_vector is not None,
            "dim": self.dim,
        }


class PersonalizationService:
    """Per-user search personalization engine.

    Usage::

        ps = PersonalizationService()
        ps.record_interaction("user1", query_vec, clicked_ids=[...])
        personalized = ps.personalize_query("user1", query_vector)
        boosted = ps.boost_personalized("user1", candidates)
    """

    def __init__(self, storage_path: str = "user_profiles.json"):
        self.storage_path = storage_path
        self._profiles: Dict[str, UserProfile] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                for uid, profile_data in data.items():
                    profile = UserProfile(user_id=uid)
                    profile.preference_vector = profile_data.get("preference_vector")
                    profile.clicked_ids = set(profile_data.get("clicked_ids", []))
                    profile.positive_ids = set(profile_data.get("positive_ids", []))
                    profile.negative_ids = set(profile_data.get("negative_ids", []))
                    profile.total_queries = profile_data.get("total_queries", 0)
                    self._profiles[uid] = profile
                logger.info("Loaded %d user profiles", len(self._profiles))
            except Exception as exc:
                logger.warning("Could not load profiles: %s", exc)

    def _save(self):
        data = {}
        for uid, profile in self._profiles.items():
            d = profile.to_dict()
            # Persist the actual preference vector, not just the flag
            if profile.preference_vector is not None:
                d["preference_vector"] = profile.preference_vector
            data[uid] = d
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        with self._lock:
            if user_id not in self._profiles:
                self._profiles[user_id] = UserProfile(user_id=user_id)
            return self._profiles[user_id]

    def record_interaction(
        self,
        user_id: str,
        query_text: str,
        query_vector: List[float],
        clicked_ids: Optional[List[str]] = None,
        negative_ids: Optional[List[str]] = None,
    ):
        """Record a user search interaction and update preferences."""
        profile = self.get_or_create_profile(user_id)
        with self._lock:
            profile.total_queries += 1
            profile.query_history.append(query_text)
            profile.session_queries.append(query_text)

            # Update preference from clicked items (positive signal)
            if clicked_ids:
                profile.update_preference(query_vector, alpha=0.05)
                for vid in clicked_ids:
                    profile.record_click(vid)

            # Negative feedback
            if negative_ids:
                for vid in negative_ids:
                    profile.record_negative(vid)

            self._save()

    def personalize_query(
        self,
        user_id: str,
        query_vector: List[float],
        weight: float = 0.3,
    ) -> List[float]:
        """Blend the user preference vector with the query vector.

        Args:
            user_id: User identifier.
            query_vector: Original query embedding.
            weight: How much to bias toward user preferences (0-1).

        Returns:
            Personalized query vector.
        """
        profile = self.get_or_create_profile(user_id)
        if profile.preference_vector is None:
            return query_vector

        q = np.array(query_vector, dtype=np.float32)
        p = np.array(profile.preference_vector, dtype=np.float32)

        blended = (1 - weight) * q + weight * p
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm
        return blended.tolist()

    def boost_personalized(
        self,
        user_id: str,
        candidates: List[Dict[str, Any]],
        boost_factor: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Boost results matching the user's positive interaction history.

        Args:
            user_id: User identifier.
            candidates: Search result candidates.
            boost_factor: How much to boost matched items.

        Returns:
            Re-ranked candidates with boosted scores.
        """
        profile = self.get_or_create_profile(user_id)
        if not profile.positive_ids:
            return candidates

        for cand in candidates:
            vid = cand.get("vector_id", "")
            if vid in profile.positive_ids:
                # Boost the score
                current = cand.get("score", cand.get("distance", 0))
                if "distance" in cand or "score" in cand:
                    boost = current * (1 - boost_factor)
                    cand["personalized_score"] = boost
                    cand["personalized"] = True

        # Sort by personalized score if available
        candidates.sort(
            key=lambda x: (
                x.get("personalized_score", 0)
                if "personalized_score" in x
                else -(x.get("distance", 0) if "distance" in x else 0)
            ),
            reverse=True,
        )
        return candidates

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        profile = self._profiles.get(user_id)
        return profile.to_dict() if profile else None

    def get_top_queries(self, user_id: str, n: int = 10) -> List[str]:
        profile = self._profiles.get(user_id)
        if not profile:
            return []
        return list(profile.query_history)[-n:]

    def get_user_vector(self, user_id: str) -> Optional[List[float]]:
        profile = self._profiles.get(user_id)
        return profile.preference_vector if profile else None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_users": len(self._profiles),
            "total_interactions": sum(p.total_queries for p in self._profiles.values()),
        }
