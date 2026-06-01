"""Streaming RAG with Server-Sent Events support."""
from typing import AsyncGenerator, Optional, Dict, Any, List
import json
import os
import logging
import asyncio

from services.rag_service import RAGService
from services.embedding_service import embed_text

logger = logging.getLogger(__name__)


async def stream_llm_response(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> AsyncGenerator[str, None]:
    """Stream LLM response as SSE chunks."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield f"data: {json.dumps({'error': 'openai not installed'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"

    yield "data: [DONE]\n\n"


class StreamingRAGService(RAGService):
    """RAG service with streaming LLM responses."""

    async def query_stream(
        self,
        collection_id: str,
        query: str,
        k: int = 5,
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream RAG answer token by token."""
        try:
            loop = asyncio.get_event_loop()
            query_vector = await loop.run_in_executor(None, embed_text, query)
            search_result = self._search_vectors(
                collection_id=collection_id,
                query_vector=query_vector,
                k=k,
                filters={"content_type": "rag_chunk"},
            )

            if not search_result.get("success"):
                yield f"data: {json.dumps({'error': 'Search failed'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            results = search_result.get("results", [])
            if not results:
                yield f"data: {json.dumps({'answer': 'No relevant documents found.'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            context_parts = []
            for r in results:
                meta = r.get("metadata", {})
                chunk_text = meta.get("text", "")
                source = meta.get("source", "unknown")
                if chunk_text:
                    context_parts.append(f"[Source: {source}]\n{chunk_text}")

            context = "\n\n---\n\n".join(context_parts)
            system = (
                "You are a helpful assistant. Answer based solely on the provided context. "
                "If the context lacks information, say so. Cite sources."
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]

            yield f"data: {json.dumps({'context': [{'text': r['metadata'].get('text', ''), 'source': r['metadata'].get('source', '')} for r in results]})}\n\n"

            async for chunk in stream_llm_response(messages, model=llm_model, api_key=api_key):
                yield chunk

        except Exception as e:
            logger.exception("Streaming RAG failed")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
