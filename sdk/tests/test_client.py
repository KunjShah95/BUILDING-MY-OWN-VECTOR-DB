"""Tests for the vector-db SDK client."""
import json
import pytest
from vector_db_client import VectorDBClient, VectorDBError, VectorDBHTTPError
from vector_db_client.models import Collection, SearchHit, SearchResult
from vector_db_client._http import raise_for_status
from vector_db_client.collections import CollectionsAPI
from vector_db_client.vectors import VectorsAPI
from vector_db_client.multimodal import MultimodalAPI


class TestVectorDBClientInit:
    def test_init_defaults(self):
        client = VectorDBClient()
        assert client.base_url == "http://localhost:8000"
        assert client._client is not None
        client.close()

    def test_init_custom_url(self):
        client = VectorDBClient("http://127.0.0.1:9000")
        assert client.base_url == "http://127.0.0.1:9000"
        client.close()

    def test_init_strips_trailing_slash(self):
        client = VectorDBClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"
        client.close()

    def test_init_custom_timeout(self, mock_base_url):
        client = VectorDBClient(mock_base_url, timeout=120.0)
        assert client._client.timeout.connect == 120.0
        client.close()

    def test_sub_apis_attached(self, client):
        assert isinstance(client.collections, CollectionsAPI)
        assert isinstance(client.vectors, VectorsAPI)
        assert isinstance(client.multimodal, MultimodalAPI)

    def test_context_manager(self):
        with VectorDBClient() as c:
            assert c.base_url == "http://localhost:8000"

    def test_request_methods(self, client):
        assert callable(client.request)
        assert callable(client.get)
        assert callable(client.post)
        assert callable(client.delete)


class TestCollectionsAPI:
    def test_create_builds_correct_body(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=201,
            json=lambda: {"collection": {
                "collection_id": "abc", "name": "t", "modality": "text",
                "dimension": 384, "embedding_model": "miniLM",
            }},
            is_success=True,
        )
        api = CollectionsAPI(mock_client)
        col = api.create("test", collection_id="abc", modality="text", dimension=384)
        assert isinstance(col, Collection)
        assert col.collection_id == "abc"
        mock_client.post.assert_called_once()

    def test_create_with_all_params(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=201,
            json=lambda: {"collection": {
                "collection_id": "x", "name": "full", "modality": "image",
                "dimension": 512, "embedding_model": "clip",
                "distance_metric": "cosine", "description": "test",
            }},
            is_success=True,
        )
        api = CollectionsAPI(mock_client)
        col = api.create("full", collection_id="x",
                         modality="image", dimension=512,
                         embedding_model="clip", description="test")
        assert col.description == "test"
        assert col.dimension == 512

    def test_list(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"collections": [
                {"collection_id": "a", "name": "A", "modality": "text",
                 "dimension": 384, "embedding_model": "m"},
            ]},
            is_success=True,
        )
        api = CollectionsAPI(mock_client)
        result = api.list(limit=10)
        assert len(result) == 1
        assert result[0].name == "A"

    def test_get(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"collection": {
                "collection_id": "c1", "name": "C1", "modality": "audio",
                "dimension": 128, "embedding_model": "mfcc",
            }},
            is_success=True,
        )
        api = CollectionsAPI(mock_client)
        col = api.get("c1")
        assert col.modality == "audio"

    def test_delete(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.delete.return_value = mocker.Mock(
            status_code=200, json=lambda: {"success": True}, is_success=True,
        )
        api = CollectionsAPI(mock_client)
        result = api.delete("c1")
        assert result["success"] is True

    def test_build_index(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200, json=lambda: {"success": True}, is_success=True,
        )
        api = CollectionsAPI(mock_client)
        result = api.build_index("c1", method="hnsw", m=32, ef_construction=300)
        assert result["success"] is True

    def test_index_stats(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True, "node_count": 100},
            is_success=True,
        )
        api = CollectionsAPI(mock_client)
        result = api.index_stats("c1")
        assert result["node_count"] == 100


class TestVectorsAPI:
    def test_create(self, mocker, sample_vector, sample_metadata):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=201, json=lambda: {"success": True, "vector_id": "v1"},
            is_success=True,
        )
        api = VectorsAPI(mock_client)
        result = api.create(sample_vector, metadata=sample_metadata)
        assert result["vector_id"] == "v1"

    def test_create_with_vector_id(self, mocker, sample_vector):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=201, json=lambda: {"success": True, "vector_id": "my-id"},
            is_success=True,
        )
        api = VectorsAPI(mock_client)
        result = api.create(sample_vector, vector_id="my-id")
        assert result["vector_id"] == "my-id"

    def test_get(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True, "vector": {"vector_id": "v1", "vector": [0.1]}},
            is_success=True,
        )
        api = VectorsAPI(mock_client)
        result = api.get("v1")
        assert result["vector"]["vector_id"] == "v1"

    def test_delete(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.delete.return_value = mocker.Mock(
            status_code=200, json=lambda: {"success": True}, is_success=True,
        )
        api = VectorsAPI(mock_client)
        result = api.delete("v1")
        assert result["success"] is True

    def test_search(self, mocker, sample_vector):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {
                "success": True, "results": [
                    {"vector_id": "r1", "distance": 0.1, "metadata": {"text": "a"}},
                ],
                "total_results": 1, "search_time": 0.005, "method": "hnsw",
            },
            is_success=True,
        )
        api = VectorsAPI(mock_client)
        result = api.search(sample_vector, k=5, method="hnsw")
        assert result["success"] is True
        assert len(result["results"]) == 1

    def test_search_with_filters(self, mocker, sample_vector):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True, "results": [], "total_results": 0},
            is_success=True,
        )
        api = VectorsAPI(mock_client)
        result = api.search(sample_vector, filters={"source": "web"})
        assert result["total_results"] == 0


class TestModels:
    def test_collection_from_api(self):
        data = {
            "collection_id": "c1", "name": "C1", "modality": "text",
            "dimension": 384, "embedding_model": "m", "distance_metric": "cosine",
            "description": "test",
        }
        col = Collection.from_api(data)
        assert col.collection_id == "c1"
        assert col.name == "C1"
        assert col.dimension == 384
        assert col.description == "test"

    def test_collection_default_distance(self):
        data = {
            "collection_id": "c1", "name": "C1", "modality": "text",
            "dimension": 384, "embedding_model": "m",
        }
        col = Collection.from_api(data)
        assert col.distance_metric == "cosine"

    def test_search_hit_from_api(self):
        data = {"vector_id": "v1", "distance": 0.25, "metadata": {"key": "val"}}
        hit = SearchHit.from_api(data)
        assert hit.vector_id == "v1"
        assert hit.distance == 0.25
        assert hit.metadata["key"] == "val"

    def test_search_hit_fallback_metadata(self):
        data = {"vector_id": "v1", "distance": 0.5, "meta_data": {"fallback": True}}
        hit = SearchHit.from_api(data)
        assert hit.metadata["fallback"] is True

    def test_search_result_from_api(self):
        data = {
            "success": True,
            "results": [
                {"vector_id": "r1", "distance": 0.1},
                {"vector_id": "r2", "distance": 0.2},
            ],
            "search_time": 0.01,
            "method": "hnsw",
        }
        result = SearchResult.from_api(data)
        assert result.success is True
        assert len(result.results) == 2
        assert result.search_time == 0.01
        assert result.method == "hnsw"

    def test_search_result_empty(self):
        data = {"success": True, "results": [], "search_time": 0.0, "method": "brute"}
        result = SearchResult.from_api(data)
        assert result.total_results == 0
        assert result.results == []


class TestHTTPHelpers:
    def test_raise_for_status_success(self, mocker):
        resp = mocker.Mock()
        resp.is_success = True
        resp.json.return_value = {"success": True}
        result = raise_for_status(resp)
        assert result["success"] is True

    def test_raise_for_status_error(self, mocker):
        resp = mocker.Mock()
        resp.is_success = False
        resp.status_code = 404
        resp.json.return_value = {"detail": "not found"}
        with pytest.raises(VectorDBHTTPError) as exc:
            raise_for_status(resp)
        assert exc.value.status_code == 404

    def test_raise_for_status_bad_json(self, mocker):
        resp = mocker.Mock()
        resp.is_success = True
        resp.json.side_effect = json.JSONDecodeError("err", "", 0)
        result = raise_for_status(resp)
        assert "message" in result

    def test_vector_db_http_error_str(self):
        err = VectorDBHTTPError(500, "server error")
        assert "HTTP 500" in str(err)
        assert err.status_code == 500


class TestMultimodalAPI:
    def test_ingest_text(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=201, json=lambda: {"success": True, "vector_id": "v1"},
            is_success=True,
        )
        api = MultimodalAPI(mock_client)
        result = api.ingest_text("coll1", "hello world", metadata={"source": "test"})
        assert result["success"] is True

    def test_search_text(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {
                "success": True, "results": [
                    {"vector_id": "r1", "distance": 0.1},
                ],
                "total_results": 1, "search_time": 0.01, "method": "brute",
            },
            is_success=True,
        )
        api = MultimodalAPI(mock_client)
        result = api.search_text("coll1", "test query", k=3)
        assert isinstance(result, SearchResult)
        assert len(result.results) == 1

    def test_ingest_image_missing_source(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        api = MultimodalAPI(mock_client)
        with pytest.raises(ValueError, match="Provide path or data"):
            api.ingest_image("coll1")

    def test_search_image_missing_source(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        api = MultimodalAPI(mock_client)
        with pytest.raises(ValueError, match="Provide path or data"):
            api.search_image("coll1")

    def test_ingest_audio_missing_source(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        api = MultimodalAPI(mock_client)
        with pytest.raises(ValueError, match="Provide path or data"):
            api.ingest_audio("coll1")

    def test_search_audio_missing_source(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        api = MultimodalAPI(mock_client)
        with pytest.raises(ValueError, match="Provide path or data"):
            api.search_audio("coll1")

    def test_ingest_image_with_bytes(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=201, json=lambda: {"success": True}, is_success=True,
        )
        api = MultimodalAPI(mock_client)
        result = api.ingest_image("coll1", data=b"fake-image-bytes",
                                  filename="test.jpg", metadata={"sku": "A1"})
        assert result["success"] is True

    def test_ingest_audio_with_bytes(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=201, json=lambda: {"success": True}, is_success=True,
        )
        api = MultimodalAPI(mock_client)
        result = api.ingest_audio("coll1", data=b"fake-audio-bytes")
        assert result["success"] is True

    def test_search_image_with_bytes(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {
                "success": True, "results": [{"vector_id": "r1", "distance": 0.1}],
                "total_results": 1, "search_time": 0.01, "method": "brute",
            },
            is_success=True,
        )
        api = MultimodalAPI(mock_client)
        result = api.search_image("coll1", data=b"fake-query-img")
        assert isinstance(result, SearchResult)

    def test_search_audio_with_bytes(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {
                "success": True, "results": [{"vector_id": "r1", "distance": 0.1}],
                "total_results": 1, "search_time": 0.01, "method": "brute",
            },
            is_success=True,
        )
        api = MultimodalAPI(mock_client)
        result = api.search_audio("coll1", data=b"fake-query-audio")
        assert isinstance(result, SearchResult)

    def test_async_client_not_implemented(self):
        from vector_db_client.multimodal import AsyncVectorDBClient
        with pytest.raises(NotImplementedError):
            AsyncVectorDBClient()
