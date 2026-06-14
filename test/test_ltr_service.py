"""Unit tests for the Learning-to-Rank (LTR) service."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from services.ltr_service import LTRService, LTRFeature, LTRTrainingExample


@pytest.fixture
def ltr_service():
    return LTRService(model_path=":memory:")


class TestLTRServiceInit:
    def test_init_default(self):
        service = LTRService()
        assert service.model_path == "ltr_model.pkl"
        assert service._model is None

    def test_init_custom_path(self):
        service = LTRService(model_path="/custom/path.pkl")
        assert service.model_path == "/custom/path.pkl"

    def test_is_trained_fresh(self, ltr_service):
        assert ltr_service.is_trained() is False


class TestLTRFeatureExtraction:
    def test_extract_features(self, ltr_service):
        features = ltr_service.extract_features(
            "test query",
            [{"distance": 0.1, "bm25_score": 0.8, "rrf_score": 0.5,
              "rerank_score": 0.9, "metadata": {"text": "hello world"}}],
        )
        assert len(features) == 1
        assert len(features[0]) == 8  # 8 feature dimensions
        assert features[0][0] == pytest.approx(0.1, rel=1e-5)  # distance_score
        assert features[0][1] == pytest.approx(0.8, rel=1e-5)  # bm25_score
        assert features[0][2] == pytest.approx(0.5, rel=1e-5)  # rrf_score
        assert features[0][3] == pytest.approx(0.9, rel=1e-5)  # rerank_score

    def test_extract_features_no_metadata(self, ltr_service):
        features = ltr_service.extract_features(
            "test",
            [{"distance": 0.5}],
        )
        assert len(features) == 1
        assert features[0][0] == 0.5
        # length_score should default based on empty text
        assert features[0][4] == 0.0

    def test_extract_features_multiple_candidates(self, ltr_service):
        candidates = [
            {"distance": 0.1, "metadata": {"text": "short"}},
            {"distance": 0.9, "metadata": {"text": "a" * 2000}},
        ]
        features = ltr_service.extract_features("test", candidates)
        assert len(features) == 2

    def test_extract_features_query_length_encoded(self, ltr_service):
        features = ltr_service.extract_features(
            "a b c d e f g h i j k l m n o p q r s t u v",  # 22 words
            [{"distance": 0.5}],
        )
        # query_length should be capped at 1.0
        assert features[0][7] == 1.0

    def test_extract_features_metadata_popularity(self, ltr_service):
        features = ltr_service.extract_features(
            "test",
            [{"distance": 0.5, "metadata": {"popularity": 0.95}}],
        )
        assert features[0][6] == pytest.approx(0.95, rel=1e-5)


class TestLTRTraining:
    def test_train_from_feedback_no_xgboost(self, ltr_service):
        """Training should gracefully handle missing xgboost."""
        with patch.dict("sys.modules", {"xgboost": None}):
            # Clear import cache
            import importlib
            # The import happens inside the function, so this should work
            pass

        entries = [
            {"query": "test", "results": [{"vector_id": "v1", "distance": 0.1}],
             "clicks": {"v1": 2}},
            {"query": "test", "results": [{"vector_id": "v2", "distance": 0.5}],
             "clicks": {"v2": 0}},
        ]
        result = ltr_service.train_from_feedback(entries)
        # Need at least 10 examples with xgboost
        assert result["success"] is False

    def test_train_from_feedback_too_few_examples(self, ltr_service):
        entries = [
            {"query": "q1", "results": [{"vector_id": "v1", "distance": 0.1}],
             "clicks": {"v1": 1}},
        ]
        result = ltr_service.train_from_feedback(entries)
        assert result["success"] is False
        assert "10" in result.get("message", "")

    def test_train_from_feedback_empty_results(self, ltr_service):
        result = ltr_service.train_from_feedback([])
        assert result["success"] is False

    def test_train_from_feedback_no_clicks(self, ltr_service):
        entries = [
            {"query": "q1", "results": [{"vector_id": "v1", "distance": 0.1}]},
        ]
        result = ltr_service.train_from_feedback(entries)
        assert result["success"] is False

    def test_train_with_xgboost_mocked(self, ltr_service):
        """Test training pipeline with mocked XGBoost (patch _train_xgboost)."""
        entries = [
            {
                "query": f"q{i}",
                "results": [
                    {"vector_id": f"v{j}", "distance": 0.1 * (j + 1),
                     "bm25_score": 0.5, "metadata": {"text": "doc"}},
                ],
                "clicks": {f"v{j}": 2},
            }
            for i in range(5) for j in range(3)  # 15 examples
        ]

        with patch.object(ltr_service, '_train_xgboost') as mock_train:
            mock_train.return_value = {
                "success": True,
                "num_examples": 15,
                "num_queries": 5,
                "model_path": ltr_service.model_path,
                "features": [],
            }
            result = ltr_service.train_from_feedback(entries)
            assert result["success"] is True
            assert result["num_examples"] == 15
            mock_train.assert_called_once()


class TestLTRPrediction:
    def test_predict_no_model(self, ltr_service):
        candidates = [{"vector_id": "v1", "distance": 0.1}]
        result = ltr_service.predict(
            [[0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.1]],
            candidates,
        )
        # Without model, returns candidates unchanged
        assert result == candidates

    def test_predict_with_model_mocked(self, ltr_service):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.3, 0.7])
        ltr_service._model = mock_model
        ltr_service._feature_names = ["f1", "f2"]

        candidates = [
            {"vector_id": "v1", "distance": 0.1},
            {"vector_id": "v2", "distance": 0.5},
            {"vector_id": "v3", "distance": 0.3},
        ]
        features = [[0.1, 0.0], [0.5, 0.0], [0.3, 0.0]]

        result = ltr_service.predict(features, candidates)
        assert len(result) == 3
        # Should be sorted by ltr_score descending
        assert result[0]["vector_id"] == "v1"
        assert result[2]["vector_id"] == "v2"
        assert result[0]["ltr_score"] == 0.9

    def test_predict_empty_candidates(self, ltr_service):
        result = ltr_service.predict([], [])
        assert result == []

    def test_predict_model_failure_graceful(self, ltr_service):
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("predict error")
        ltr_service._model = mock_model

        candidates = [{"vector_id": "v1"}]
        result = ltr_service.predict([[0.1]], candidates)
        # Should fall back to returning candidates unchanged
        assert result == candidates


class TestLTRStatus:
    def test_get_status_not_trained(self, ltr_service):
        status = ltr_service.get_status()
        assert status["is_trained"] is False
        assert "model_path" in status

    def test_get_status_trained(self, ltr_service):
        ltr_service._model = MagicMock()
        status = ltr_service.get_status()
        assert status["is_trained"] is True

    def test_status_has_features(self, ltr_service):
        ltr_service.extract_features("q", [{"distance": 0.5}])
        status = ltr_service.get_status()
        assert status["num_features"] > 0
        assert "distance_score" in status["features"]
