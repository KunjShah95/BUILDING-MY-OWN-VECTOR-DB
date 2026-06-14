"""
MCP (Model Context Protocol) Server (Phase 3: Agentic Connectors).

Exposes the vector database as MCP tools that can be used by AI agents
(Claude, Copilot, etc.) for semantic search and retrieval-augmented tasks.

Tools exposed:
  - search_vectors(query_text, collection_id, k) → search results
  - ingest_text(text, collection_id, metadata) → store a text as vector
  - get_collection_stats(collection_id) → collection information
  - hybrid_search(hybrid_query, collection_id, k) → cost-based search
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP server for vector database tools.

    In a production deployment, this would run as a standalone FastMCP
    application. For now, we provide the tool definitions and execution
    logic that can be mounted in the API server.

    Reference: https://modelcontextprotocol.io/
    """

    def __init__(self, api_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._vector_service = None
        self._collection_service = None

    def set_services(self, vector_service, collection_service):
        """Inject service instances for direct in-process access."""
        self._vector_service = vector_service
        self._collection_service = collection_service

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible tool definitions."""
        return [
            {
                "name": "search_vectors",
                "description": "Search for semantically similar vectors in a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string", "description": "Search query text"},
                        "collection_id": {"type": "string", "description": "Collection to search in"},
                        "k": {"type": "integer", "description": "Number of results (1-100)", "default": 10},
                        "method": {"type": "string", "enum": ["hnsw", "ivf", "brute"], "default": "hnsw"},
                    },
                    "required": ["query_text", "collection_id"],
                },
            },
            {
                "name": "ingest_text",
                "description": "Embed text and store it as a vector in a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text content to ingest"},
                        "collection_id": {"type": "string", "description": "Target collection"},
                        "metadata": {"type": "object", "description": "Optional metadata", "default": {}},
                    },
                    "required": ["text", "collection_id"],
                },
            },
            {
                "name": "get_collection_stats",
                "description": "Get statistics and info about a collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_id": {"type": "string", "description": "Collection ID"},
                    },
                    "required": ["collection_id"],
                },
            },
            {
                "name": "hybrid_search",
                "description": "Cost-based hybrid search with metadata filters and semantic matching",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hybrid_query": {
                            "type": "string",
                            "description": "Hybrid query DSL, e.g. (category = 'tech' AND price < 100) OR semantic_match(\"laptops\")",
                        },
                        "collection_id": {"type": "string", "description": "Collection to search in"},
                        "k": {"type": "integer", "description": "Number of results", "default": 10},
                    },
                    "required": ["hybrid_query", "collection_id"],
                },
            },
        ]

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool and return the result.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        async with httpx.AsyncClient(base_url=self.api_url, headers=headers) as client:
            if tool_name == "search_vectors":
                text = arguments["query_text"]
                collection_id = arguments["collection_id"]
                k = arguments.get("k", 10)
                method = arguments.get("method", "hnsw")

                # Embed the text
                from services.embedding_service import embed_text
                vector = embed_text(text)

                resp = await client.post(
                    "/search",
                    json={
                        "query_vector": vector,
                        "k": k,
                        "method": method,
                    },
                    params={"collection_id": collection_id},
                )
                result = resp.json()

                # Format for MCP readability
                formatted = []
                for r in result.get("results", []):
                    meta = r.get("metadata", {}) or {}
                    formatted.append({
                        "vector_id": r.get("vector_id"),
                        "distance": r.get("distance"),
                        "text_snippet": (meta.get("text", "") or meta.get("content", ""))[:200],
                        "source": meta.get("source"),
                    })
                return {"results": formatted, "total": len(formatted)}

            elif tool_name == "ingest_text":
                text = arguments["text"]
                collection_id = arguments["collection_id"]
                metadata = arguments.get("metadata", {})

                resp = await client.post(
                    f"/collections/{collection_id}/ingest/text",
                    json={"text": text, "metadata": metadata},
                )
                return resp.json()

            elif tool_name == "get_collection_stats":
                collection_id = arguments["collection_id"]

                resp = await client.get(f"/collections/{collection_id}")
                col_data = resp.json()

                stats_resp = await client.get(
                    f"/collections/{collection_id}/index/stats"
                )
                index_stats = stats_resp.json().get("stats", {})

                return {
                    "collection": col_data.get("collection", {}),
                    "index": index_stats,
                }

            elif tool_name == "hybrid_search":
                hybrid_query = arguments["hybrid_query"]
                collection_id = arguments["collection_id"]
                k = arguments.get("k", 10)

                resp = await client.post(
                    "/search-engine/hybrid-query",
                    params={
                        "collection_id": collection_id,
                        "hybrid_query": hybrid_query,
                        "top_k": k,
                    },
                )
                return resp.json()

            else:
                return {"error": f"Unknown tool: {tool_name}"}
