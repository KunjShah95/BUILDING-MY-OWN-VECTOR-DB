"""Tests for OpenAI-compatible API endpoints.

Most tests use the TestClient from FastAPI to call the endpoint logic
directly without needing a live PostgreSQL instance.
"""

import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def app():
    from fastapi import FastAPI
    from api.routers.openai_compat import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ---- Embeddings endpoint tests --------------------------------------------


class TestEmbeddingsEndpoint:
    def test_embed_single_text(self, client):
        with patch("api.routers.openai_compat.embed_text") as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]

            response = client.post(
                "/v1/embeddings",
                json={"model": "text-embedding-ada-002", "input": "hello world"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert data["data"][0]["index"] == 0
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] > 0

    def test_embed_multiple_texts(self, client):
        with patch("api.routers.openai_compat.embed_text") as mock_embed:
            mock_embed.side_effect = [[0.1], [0.2], [0.3]]

            response = client.post(
                "/v1/embeddings",
                json={
                    "model": "text-embedding-ada-002",
                    "input": ["first", "second", "third"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 3
        assert data["data"][0]["embedding"] == [0.1]
        assert data["data"][2]["embedding"] == [0.3]

    def test_embed_empty_input(self, client):
        response = client.post(
            "/v1/embeddings",
            json={"model": "test", "input": ""},
        )
        # Empty string should be treated as a list with one empty string
        # which embed_text should reject, or return 400/500
        assert response.status_code in (400, 500)

    def test_embed_invalid_input_type(self, client):
        response = client.post(
            "/v1/embeddings",
            json={"model": "test", "input": 42},
        )
        assert response.status_code == 422  # Pydantic validation

    def test_embed_response_shape(self, client):
        """Response shape matches OpenAI's API."""
        with patch("api.routers.openai_compat.embed_text") as mock_embed:
            mock_embed.return_value = [0.5] * 384

            response = client.post(
                "/v1/embeddings",
                json={"model": "all-MiniLM-L6-v2", "input": "test"},
            )

        data = response.json()
        assert data["object"] == "list"
        assert data["model"] == "all-MiniLM-L6-v2"
        assert isinstance(data["data"], list)
        assert data["data"][0]["object"] == "embedding"
        assert isinstance(data["data"][0]["embedding"], list)
        assert all(isinstance(x, float) for x in data["data"][0]["embedding"])

    def test_embed_encoding_format(self, client):
        """encoding_format field is accepted but we always return float."""
        with patch("api.routers.openai_compat.embed_text") as mock_embed:
            mock_embed.return_value = [0.1, 0.2]

            response = client.post(
                "/v1/embeddings",
                json={
                    "model": "test",
                    "input": "hello",
                    "encoding_format": "base64",
                },
            )
        assert response.status_code == 200


# ---- Chat completions endpoint tests --------------------------------------


class TestChatCompletionsEndpoint:
    def test_chat_basic_no_rag(self, client):
        """Without collection_id, falls through to direct LLM call."""
        with patch("api.routers.openai_compat.openai_chat_completion") as mock_llm:
            mock_llm.return_value = "Hello! How can I help you today?"

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": "Say hello"},
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "Hello" in data["choices"][0]["message"]["content"]
        assert "usage" in data

    def test_chat_with_rag(self, client):
        """With collection_id, RAG pipeline should run."""
        with patch("api.routers.openai_compat._rag_service") as mock_rag_svc:
            mock_rag = MagicMock()
            mock_rag.query.return_value = {
                "success": True,
                "context": [
                    {"text": "The sky is blue.", "source": "doc1.pdf"},
                ],
                "answer": "The sky appears blue due to Rayleigh scattering.",
            }
            mock_rag_svc.return_value = mock_rag

            with patch("api.routers.openai_compat.openai_chat_completion") as mock_llm:
                mock_llm.return_value = "The sky is blue due to Rayleigh scattering."

                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "user", "content": "Why is the sky blue?"},
                        ],
                        "collection_id": "science-docs",
                        "k": 3,
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) == 1
        assert data["choices"][0]["finish_reason"] == "stop"

        # Verify RAG was called with correct params
        mock_rag.query.assert_called_once_with(
            collection_id="science-docs",
            query="Why is the sky blue?",
            k=3,
            llm_model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.3,
        )

    def test_chat_multiple_messages(self, client):
        """Preserves full conversation history."""
        with patch("api.routers.openai_compat.openai_chat_completion") as mock_llm:
            mock_llm.return_value = "Based on our conversation, I'd recommend Python."

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a helpful coder."},
                        {"role": "user", "content": "What language should I learn?"},
                        {"role": "assistant", "content": "Python is great for beginners."},
                        {"role": "user", "content": "Tell me more about why."},
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data["choices"]) == 1

    def test_chat_empty_messages(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": []},
        )
        assert response.status_code == 400

    def test_chat_response_shape(self, client):
        """Response matches OpenAI's chat completion format."""
        with patch("api.routers.openai_compat.openai_chat_completion") as mock_llm:
            mock_llm.return_value = "Sure, I can help with that."

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Help me"}],
                },
            )

        data = response.json()
        # Must match OpenAI's documented response shape
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert "choices" in data
        assert isinstance(data["choices"], list)
        assert "message" in data["choices"][0]
        assert "role" in data["choices"][0]["message"]
        assert "content" in data["choices"][0]["message"]
        assert "finish_reason" in data["choices"][0]
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]


# ---- Models endpoint tests -------------------------------------------------


class TestModelsEndpoint:
    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1
        for model in data["data"]:
            assert "id" in model
            assert model["object"] == "model"
