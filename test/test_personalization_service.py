"""Tests for Personalization Engine (Phase 7)."""

import os
import json
import numpy as np
import pytest

from services.personalization_service import PersonalizationService, UserProfile


class TestUserProfile:
    def test_init(self):
        p = UserProfile(user_id="user1")
        assert p.user_id == "user1"
        assert p.total_queries == 0
        assert p.preference_vector is None

    def test_update_preference_first_time(self):
        p = UserProfile(user_id="u1", dim=4)
        p.update_preference([1.0, 0.0, 0.0, 0.0])
        assert p.preference_vector is not None
        assert np.allclose(p.preference_vector, [1.0, 0.0, 0.0, 0.0])

    def test_update_preference_moving_average(self):
        p = UserProfile(user_id="u1", dim=4)
        p.update_preference([1.0, 0.0, 0.0, 0.0], alpha=0.5)
        p.update_preference([0.0, 1.0, 0.0, 0.0], alpha=0.5)
        # Should be blend of both
        assert p.preference_vector is not None
        blended = np.array(p.preference_vector)
        assert not np.allclose(blended, [1.0, 0.0, 0.0, 0.0])
        assert not np.allclose(blended, [0.0, 1.0, 0.0, 0.0])

    def test_update_preference_normalization(self):
        p = UserProfile(user_id="u1", dim=4)
        # First update with normalized vector (UserProfile stores as-is on first call)
        p.update_preference([1.0, 1.0, 0.0, 0.0], alpha=0.5)
        norm = np.linalg.norm(p.preference_vector)
        assert np.allclose(norm, np.sqrt(2.0), atol=1e-5)  # [1,1,0,0] has norm sqrt(2)

        # Second update: blending normalizes automatically
        p.update_preference([0.0, 0.0, 1.0, 0.0], alpha=0.5)
        # After blend + normalize, should be unit vector
        norm2 = np.linalg.norm(p.preference_vector)
        assert np.allclose(norm2, 1.0, atol=1e-5)

    def test_record_click(self):
        p = UserProfile(user_id="u1")
        p.record_click("v1")
        assert "v1" in p.clicked_ids
        assert "v1" in p.positive_ids
        assert "v1" in p.session_clicked

    def test_record_negative(self):
        p = UserProfile(user_id="u1")
        p.record_negative("v1")
        assert "v1" in p.negative_ids

    def test_to_dict(self):
        p = UserProfile(user_id="u1", dim=8)
        p.total_queries = 5
        d = p.to_dict()
        assert d["user_id"] == "u1"
        assert d["total_queries"] == 5
        assert d["has_preference_vector"] is False


class TestPersonalizationService:
    @pytest.fixture
    def svc(self, tmp_path):
        return PersonalizationService(storage_path=str(tmp_path / "profiles.json"))

    def test_get_or_create_profile_new(self, svc):
        p = svc.get_or_create_profile("new_user")
        assert p.user_id == "new_user"
        assert "new_user" in svc._profiles

    def test_get_or_create_profile_existing(self, svc):
        p1 = svc.get_or_create_profile("u1")
        p2 = svc.get_or_create_profile("u1")
        assert p1 is p2

    def test_record_interaction(self, svc):
        svc.record_interaction(
            user_id="u1",
            query_text="hello",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            clicked_ids=["v1", "v2"],
        )
        p = svc.get_or_create_profile("u1")
        assert p.total_queries == 1
        assert "v1" in p.clicked_ids
        assert p.preference_vector is not None

    def test_record_interaction_negative(self, svc):
        svc.record_interaction(
            user_id="u1",
            query_text="bad",
            query_vector=[0.0, 1.0, 0.0, 0.0],
            negative_ids=["v3"],
        )
        p = svc.get_or_create_profile("u1")
        assert "v3" in p.negative_ids

    def test_personalize_query_no_preference(self, svc):
        q = [1.0, 0.0, 0.0, 0.0]
        result = svc.personalize_query("unknown", q)
        assert result == q

    def test_personalize_query_with_preference(self, svc):
        svc.record_interaction("u1", "test", [1.0, 0.0, 0.0, 0.0], clicked_ids=["v1"])
        original = [0.0, 1.0, 0.0, 0.0]
        personalized = svc.personalize_query("u1", original, weight=0.5)
        assert personalized != original
        norm = np.linalg.norm(personalized)
        assert np.allclose(norm, 1.0, atol=1e-5)

    def test_boost_personalized_no_history(self, svc):
        candidates = [{"vector_id": "v1", "score": 0.9}]
        result = svc.boost_personalized("unknown", candidates)
        assert result == candidates

    def test_boost_personalized_with_history(self, svc):
        svc.record_interaction("u1", "test", [1.0, 0.0, 0.0, 0.0], clicked_ids=["v1"])
        candidates = [
            {"vector_id": "v1", "distance": 0.5},
            {"vector_id": "v2", "distance": 0.3},
        ]
        result = svc.boost_personalized("u1", candidates, boost_factor=0.2)
        # v1 should be boosted and sorted first
        assert result[0]["vector_id"] == "v1"
        assert "personalized_score" in result[0]
        assert result[0]["personalized"] is True

    def test_boost_personalized_distance_metric(self, svc):
        svc.record_interaction("u1", "test", [1.0, 0.0, 0.0, 0.0], clicked_ids=["v1"])
        candidates = [
            {"vector_id": "v1", "distance": 0.5},
            {"vector_id": "v2", "distance": 0.1},
        ]
        result = svc.boost_personalized("u1", candidates)
        # v1 (clicked) should be boosted, v2 has lower distance
        assert result[0]["vector_id"] == "v1"

    def test_get_user_profile(self, svc):
        svc.record_interaction("u1", "x", [1.0, 0.0, 0.0, 0.0])
        profile = svc.get_user_profile("u1")
        assert profile is not None
        assert profile["total_queries"] == 1

    def test_get_user_profile_not_found(self, svc):
        assert svc.get_user_profile("nonexistent") is None

    def test_get_top_queries(self, svc):
        for i in range(5):
            svc.record_interaction("u1", f"query_{i}", [1.0, 0.0, 0.0, 0.0])
        queries = svc.get_top_queries("u1", n=3)
        assert len(queries) == 3

    def test_get_user_vector(self, svc):
        svc.record_interaction("u1", "x", [0.5, 0.5, 0.0, 0.0], clicked_ids=["v1"])
        vec = svc.get_user_vector("u1")
        assert vec is not None
        assert len(vec) > 0

    def test_get_user_vector_no_preference(self, svc):
        assert svc.get_user_vector("unknown") is None

    def test_get_stats(self, svc):
        stats = svc.get_stats()
        assert stats["total_users"] == 0
        svc.record_interaction("u1", "x", [1.0, 0.0, 0.0, 0.0])
        stats = svc.get_stats()
        assert stats["total_users"] == 1
        assert stats["total_interactions"] == 1

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "profiles.json")
        svc1 = PersonalizationService(storage_path=path)
        svc1.record_interaction("u1", "hello", [1.0, 0.0, 0.0, 0.0], clicked_ids=["v1"])

        svc2 = PersonalizationService(storage_path=path)
        profile = svc2.get_user_profile("u1")
        assert profile is not None
        assert profile["total_queries"] == 1
        assert profile["has_preference_vector"] is True
        # Note: preference_vector is not serialized in to_dict() in current implementation
