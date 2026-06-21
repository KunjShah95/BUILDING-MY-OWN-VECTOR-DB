import pytest
from vector_db_client import VectorDBClient
from vector_db_client.ann import AnnAPI


class TestAnnAPI:
    def test_create_index(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True, "message": "Created hnsw index"},
            is_success=True,
        )
        api = AnnAPI(mock_client)
        res = api.create_index("hnsw", m=16, ef_construction=200)
        assert res["success"] is True
        mock_client.post.assert_called_once_with(
            "/api/v1/ann/index",
            json={"index_type": "hnsw", "m": 16, "ef_construction": 200}
        )

    def test_get_index_info(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True, "index_type": "hnsw"},
            is_success=True,
        )
        api = AnnAPI(mock_client)
        res = api.get_index_info("hnsw")
        assert res["success"] is True
        mock_client.get.assert_called_once_with(
            "/api/v1/ann/index",
            params={"index_type": "hnsw"}
        )

    def test_get_index_info_all(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True},
            is_success=True,
        )
        api = AnnAPI(mock_client)
        res = api.get_index_info()
        assert res["success"] is True
        mock_client.get.assert_called_once_with(
            "/api/v1/ann/index",
            params={}
        )

    def test_save_index(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True},
            is_success=True,
        )
        api = AnnAPI(mock_client)
        res = api.save_index("hnsw")
        assert res["success"] is True
        mock_client.post.assert_called_once_with(
            "/api/v1/ann/index/save",
            params={"index_type": "hnsw"}
        )

    def test_load_index(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.post.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True},
            is_success=True,
        )
        api = AnnAPI(mock_client)
        res = api.load_index("hnsw")
        assert res["success"] is True
        mock_client.post.assert_called_once_with(
            "/api/v1/ann/index/load",
            params={"index_type": "hnsw"}
        )

    def test_compare_search(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True},
            is_success=True,
        )
        api = AnnAPI(mock_client)
        res = api.compare_search([0.1, 0.2, 0.3], k=5)
        assert res["success"] is True
        mock_client.get.assert_called_once_with(
            "/api/v1/ann/search/compare",
            params={"query_vector": "0.1,0.2,0.3", "k": 5}
        )

    def test_get_statistics(self, mocker):
        mock_client = mocker.Mock(spec=VectorDBClient)
        mock_client.get.return_value = mocker.Mock(
            status_code=200,
            json=lambda: {"success": True},
            is_success=True,
        )
        api = AnnAPI(mock_client)
        res = api.get_statistics()
        assert res["success"] is True
        mock_client.get.assert_called_once_with(
            "/api/v1/ann/stats"
        )
