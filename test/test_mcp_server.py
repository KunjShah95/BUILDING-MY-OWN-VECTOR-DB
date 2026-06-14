"""Tests for MCP Server (Phase 3)."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from services.mcp_server import MCPServer


class TestMCPServerInit:
    def test_defaults(self):
        svr = MCPServer()
        assert svr.api_url == "http://localhost:8000"
        assert svr.api_key is None

    def test_custom_url(self):
        svr = MCPServer(api_url="http://vectordb:8080", api_key="sk-123")
        assert svr.api_url == "http://vectordb:8080"
        assert svr.api_key == "sk-123"

    def test_trailing_slash_stripped(self):
        svr = MCPServer(api_url="http://localhost:8000/")
        assert svr.api_url == "http://localhost:8000"

    def test_set_services(self):
        svr = MCPServer()
        vs = MagicMock()
        cs = MagicMock()
        svr.set_services(vs, cs)
        assert svr._vector_service is vs
        assert svr._collection_service is cs


class TestMCPServerToolDefinitions:
    def test_get_tool_definitions_returns_list(self):
        svr = MCPServer()
        tools = svr.get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) == 4

    def test_tool_names(self):
        svr = MCPServer()
        names = [t["name"] for t in svr.get_tool_definitions()]
        assert "search_vectors" in names
        assert "ingest_text" in names
        assert "get_collection_stats" in names
        assert "hybrid_search" in names

    def test_tool_has_input_schema(self):
        svr = MCPServer()
        for tool in svr.get_tool_definitions():
            assert "inputSchema" in tool
            assert "properties" in tool["inputSchema"]

    def test_search_vectors_required_params(self):
        svr = MCPServer()
        tools = {t["name"]: t for t in svr.get_tool_definitions()}
        required = tools["search_vectors"]["inputSchema"]["required"]
        assert "query_text" in required
        assert "collection_id" in required


class TestMCPServerExecuteTool:
    """Tests for execute_tool, which imports httpx internally.

    The method does:
        import httpx
        async with httpx.AsyncClient(...) as client:
            resp = await client.post(...)
            result = resp.json()

    We patch httpx.AsyncClient globally: the module is already cached in
    sys.modules when the function executes 'import httpx', so the
    patched AsyncClient is what gets used.
    """

    @pytest.mark.asyncio
    async def test_search_vectors(self):
        svr = MCPServer(api_url="http://test:8000", api_key="key")

        # Create a mock response object
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"vector_id": "v1", "distance": 0.1, "metadata": {"text": "doc1"}}],
            "total": 1,
        }

        # Create the mock client that the async context manager returns
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Patch httpx.AsyncClient globally and services.embedding_service.embed_text
        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch("services.embedding_service.embed_text", return_value=[0.1, 0.2, 0.3]):

            result = await svr.execute_tool("search_vectors", {
                "query_text": "test query",
                "collection_id": "docs",
                "k": 5,
            })
            assert "results" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["vector_id"] == "v1"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        svr = MCPServer()
        result = await svr.execute_tool("nonexistent", {})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_ingest_text(self):
        svr = MCPServer(api_url="http://test:8000")

        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await svr.execute_tool("ingest_text", {
                "text": "hello",
                "collection_id": "docs",
                "metadata": {"source": "test"},
            })
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_collection_stats(self):
        svr = MCPServer(api_url="http://test:8000")

        resp1 = MagicMock(json=lambda: {"collection": {"id": "docs", "vector_count": 100}})
        resp2 = MagicMock(json=lambda: {"stats": {"index_type": "hnsw"}})

        mock_client = MagicMock()
        # get() is called twice with different responses — use side_effect
        mock_client.get = AsyncMock()
        mock_client.get.side_effect = [resp1, resp2]
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await svr.execute_tool("get_collection_stats", {
                "collection_id": "docs",
            })
            assert "collection" in result
            assert result["collection"]["id"] == "docs"
