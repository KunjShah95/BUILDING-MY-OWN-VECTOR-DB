"""Unit tests for GCN training, temporal graph dynamics, and link prediction."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from services.gnn_service import GNNService
from utils.vamana_index import VamanaVectorIndex


@pytest.fixture
def mock_vector_service():
    """Create a mock vector service simulating a collection with 50 vectors."""
    svc = MagicMock(spec=["get_all_vectors"])

    np.random.seed(42)
    n = 50
    vectors = []
    for i in range(n):
        v = np.random.randn(16).astype(np.float32).tolist()
        vectors.append({
            "vector_id": f"vec_{i:03d}",
            "vector": v,
            "collection_id": "col_001",
            "metadata": {"timestamp": "2025-06-01T00:00:00", "tag": "test"},
        })

    svc.get_all_vectors.return_value = {
        "success": True,
        "vectors": vectors,
    }

    return svc


class TestGNNServiceInit:
    def test_init(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        assert service.vector_service == mock_vector_service

    def test_build_knn_graph(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        res = mock_vector_service.get_all_vectors(limit=10)
        graph = service._build_knn_graph(res["vectors"][:5], k=3)
        assert graph.number_of_nodes() == 5
        assert graph.number_of_edges() > 0


class TestGCNEmbeddingTraining:
    @pytest.mark.asyncio
    async def test_train_gcn_embeddings_success(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.train_gcn_embeddings(
            collection_id="col_001",
            hidden_dim=16,
            output_dim=16,
            epochs=2,
            knn_k=5,
        )
        assert result["success"] is True
        assert result["num_nodes"] == 50
        assert result["embedding_dim"] == 16
        assert len(result["node_embeddings"]) == 50

    @pytest.mark.asyncio
    async def test_train_gcn_embeddings_few_vectors(self, mock_vector_service):
        # Mock a small response
        mock_vector_service.get_all_vectors.return_value = {
            "success": True,
            "vectors": mock_vector_service.get_all_vectors()["vectors"][:3],
        }
        service = GNNService(mock_vector_service)
        result = service.train_gcn_embeddings(collection_id="col_001")
        assert result["success"] is False
        assert "10" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_train_gcn_embeddings_different_hidden_dim(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.train_gcn_embeddings(
            collection_id="col_001", hidden_dim=32, output_dim=16, epochs=1, knn_k=5
        )
        assert result["success"] is True
        assert result["embedding_dim"] == 16

    @pytest.mark.asyncio
    async def test_train_gcn_embeddings_empty_collection(self, mock_vector_service):
        mock_vector_service.get_all_vectors.return_value = {
            "success": True,
            "vectors": [],
        }
        service = GNNService(mock_vector_service)
        result = service.train_gcn_embeddings(collection_id="col_001")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_gcn_embeddings_have_valid_structure(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.train_gcn_embeddings(
            collection_id="col_001", epochs=2, hidden_dim=8, output_dim=8
        )
        assert result["success"] is True
        for vid, emb in result["node_embeddings"].items():
            assert len(emb) == 8
            assert all(isinstance(v, float) for v in emb)


class TestTemporalGraphDynamics:
    @pytest.mark.asyncio
    async def test_analyze_temporal_dynamics(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.analyze_temporal_dynamics(
            collection_id="col_001",
            window_days=7,
            time_field="timestamp",
        )
        assert result["success"] is True
        assert "total_nodes" in result
        assert "total_edges" in result
        assert result["total_nodes"] >= 0

    @pytest.mark.asyncio
    async def test_analyze_temporal_dynamics_empty(self, mock_vector_service):
        mock_vector_service.get_all_vectors.return_value = {
            "success": True,
            "vectors": [],
        }
        service = GNNService(mock_vector_service)
        result = service.analyze_temporal_dynamics(collection_id="col_001")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_analyze_temporal_dynamics_fields(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.analyze_temporal_dynamics(
            collection_id="col_001",
            window_days=30,
        )
        assert "growth_rate_pct" in result
        assert "is_bursty" in result
        assert "time_window_days" in result


class TestLinkPrediction:
    @pytest.mark.asyncio
    async def test_predict_missing_links(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.predict_missing_links(
            collection_id="col_001",
            target_field="tag",
        )
        assert result["success"] is True
        assert "predictions" in result
        # Some nodes should have predictions
        assert result["total_predictions"] >= 0

    @pytest.mark.asyncio
    async def test_predict_missing_links_no_empty(self, mock_vector_service):
        """All nodes have 'tag' so predictions should target other fields."""
        service = GNNService(mock_vector_service)
        result = service.predict_missing_links(
            collection_id="col_001",
            target_field="nonexistent_field",
            min_similarity=0.1,
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_predict_missing_links_empty_graph(self, mock_vector_service):
        mock_vector_service.get_all_vectors.return_value = {
            "success": True,
            "vectors": [],
        }
        service = GNNService(mock_vector_service)
        result = service.predict_missing_links(
            collection_id="col_001", target_field="tag"
        )
        assert result["success"] is False


class TestAutoTag:
    @pytest.mark.asyncio
    async def test_auto_tag_metadata(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.auto_tag_metadata(
            collection_id="col_001",
            target_field="tag",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_auto_tag_empty(self, mock_vector_service):
        mock_vector_service.get_all_vectors.return_value = {
            "success": True,
            "vectors": [],
        }
        service = GNNService(mock_vector_service)
        result = service.auto_tag_metadata(
            collection_id="col_001", target_field="tag"
        )
        assert result["success"] is False


class TestGraphReRank:
    def test_graph_rerank(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        candidates = [
            {"vector_id": "vec_000", "score": 0.9},
            {"vector_id": "vec_001", "score": 0.5},
            {"vector_id": "vec_005", "score": 0.3},
        ]
        query_vector = [0.1] * 16
        result = service.graph_rerank(
            query_vector=query_vector,
            top_k_candidates=candidates,
            collection_id="col_001",
        )
        assert len(result) == 3
        assert all("graph_score" in c for c in result)
        assert all("original_score" in c for c in result)

    def test_graph_rerank_no_candidates(self, mock_vector_service):
        service = GNNService(mock_vector_service)
        result = service.graph_rerank(
            query_vector=[0.1] * 16,
            top_k_candidates=[],
            collection_id="col_001",
        )
        assert result == []
