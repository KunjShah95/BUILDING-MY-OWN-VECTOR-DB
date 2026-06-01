"""OpenAI-compatible API router.

Provides endpoints that mirror OpenAI's REST API so existing OpenAI clients
(LangChain, LlamaIndex, custom code) can use this vector database as a
drop-in replacement.

Endpoints
---------
- GET  /v1/models                   -> list available models
- POST /v1/embeddings               -> text -> dense vector (mirrors OpenAI)
- POST /v1/chat/completions         -> RAG-powered chat (mirrors OpenAI)
"""

from __future__ import annotations

import json
import logging
import time
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import (
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIEmbeddingData,
    OpenAIEmbeddingUsage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChatChoice,
    OpenAIChatMessage,
    OpenAIChatUsage,
    OpenAIModel,
    OpenAIModelListResponse,
)
from services.embedding_service import embed_text
from services.rag_service import RAGService, openai_chat_completion

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OpenAI-Compatible"])

# ---- Internal helpers -----------------------------------------------------

def _estimate_tokens(texts: List[str]) -> int:
    """Rough token estimate (4 chars per token)."""
    return sum(len(t) // 4 + 1 for t in texts)


def _rag_service(db: Session) -> RAGService:
    return RAGService(db_session=db)


# ---- Models endpoint -------------------------------------------------------

@router.get("/v1/models", response_model=OpenAIModelListResponse)
async def list_models():
    """Mirrors ``GET /v1/models`` — lists embedding and chat models."""
    models = [
        OpenAIModel(id="sentence-transformers/all-MiniLM-L6-v2", owned_by="vector-db"),
        OpenAIModel(id="text-embedding-ada-002", owned_by="vector-db"),
        OpenAIModel(id="gpt-4o-mini", owned_by="vector-db"),
        OpenAIModel(id="gpt-4o", owned_by="vector-db"),
    ]
    return OpenAIModelListResponse(data=models)


# ---- Embeddings endpoint ---------------------------------------------------

@router.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def create_embeddings(
    body: OpenAIEmbeddingRequest,
):
    """Mirrors ``POST /v1/embeddings``.

    Accepts either a single string or a list of strings in ``input``.
    Uses the configured Sentence-Transformer model server-side.
    """
    texts: List[str] = []
    if isinstance(body.input, str):
        texts = [body.input]
    elif isinstance(body.input, list):
        texts = body.input
    else:
        raise HTTPException(status_code=400, detail="input must be a string or list of strings")

    if not texts:
        raise HTTPException(status_code=400, detail="input cannot be empty")

    try:
        embedding_list = [embed_text(t) for t in texts]

        data = [
            OpenAIEmbeddingData(index=i, embedding=vec)
            for i, vec in enumerate(embedding_list)
        ]

        prompt_tokens = _estimate_tokens(texts)
        return OpenAIEmbeddingResponse(
            data=data,
            model=body.model,
            usage=OpenAIEmbeddingUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {exc}")


# ---- Chat completions endpoint ---------------------------------------------

@router.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def create_chat_completion(
    body: OpenAIChatRequest,
    db: Session = Depends(get_db),
):
    """Mirrors ``POST /v1/chat/completions``.

    If ``collection_id`` is provided, the endpoint performs **RAG**:
    1. Extracts the last user message as the query
    2. Retrieves relevant chunks from the collection
    3. Injects context into the system prompt
    4. Sends to the LLM

    Without ``collection_id`` it falls through to direct LLM completion
    (no retrieval).
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    try:
        last_user_msg = ""
        for msg in reversed(body.messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        context_text = ""
        if body.collection_id and last_user_msg:
            rag = _rag_service(db)
            rag_result = rag.query(
                collection_id=body.collection_id,
                query=last_user_msg,
                k=body.k,
                llm_model=body.model,
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            )
            if rag_result.get("success") and rag_result.get("context"):
                context_parts = []
                for ctx in rag_result["context"]:
                    t = ctx.get("text", "")
                    s = ctx.get("source", "unknown")
                    if t:
                        context_parts.append(f"[Source: {s}] {t}")
                if context_parts:
                    context_text = "\n\n".join(context_parts)

        # Build messages for the LLM
        llm_messages = []
        for msg in body.messages:
            llm_messages.append({"role": msg.role, "content": msg.content})

        # Inject RAG context into system message if we have it
        if context_text:
            system_content = (
                "You are a helpful assistant. Answer the user's question based "
                "solely on the provided context. Cite sources when possible.\n\n"
                f"Context:\n{context_text}"
            )
            # Prepend a system message with context; keep original messages
            llm_messages.insert(0, {"role": "system", "content": system_content})

        # Call the LLM
        answer = openai_chat_completion(
            messages=llm_messages,
            model=body.model,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )

        now = int(time.time())
        return OpenAIChatResponse(
            id=f"chatcmpl-{now}",
            created=now,
            model=body.model,
            choices=[
                OpenAIChatChoice(
                    message=OpenAIChatMessage(role="assistant", content=answer or "")
                )
            ],
            usage=OpenAIChatUsage(
                prompt_tokens=_estimate_tokens([m.content for m in body.messages]),
                completion_tokens=_estimate_tokens([answer]) if answer else 0,
                total_tokens=_estimate_tokens([m.content for m in body.messages] + [answer]) if answer else _estimate_tokens([m.content for m in body.messages]),
            ),
        )
    except Exception as exc:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=500, detail=f"Chat completion error: {exc}")
