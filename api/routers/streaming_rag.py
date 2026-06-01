"""Streaming RAG endpoints with SSE."""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional

from config.database import get_db
from services.streaming_rag_service import StreamingRAGService, stream_llm_response

router = APIRouter(tags=["RAG Streaming"])


def get_streaming_rag_service(db: Session = Depends(get_db)) -> StreamingRAGService:
    return StreamingRAGService(db_session=db)


@router.get("/collections/{collection_id}/query/stream")
async def stream_rag_query(
    collection_id: str,
    query: str,
    k: int = 5,
    model: str = "gpt-4o-mini",
    db: Session = Depends(get_db),
):
    """Stream a RAG query response token-by-token using SSE."""
    svc = StreamingRAGService(db_session=db)

    return StreamingResponse(
        svc.query_stream(
            collection_id=collection_id,
            query=query,
            k=k,
            llm_model=model,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/llm/stream")
async def stream_llm(
    messages: list,
    model: str = "gpt-4o-mini",
    max_tokens: int = 500,
    temperature: float = 0.3,
):
    """Generic streaming LLM completion."""
    return StreamingResponse(
        stream_llm_response(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        media_type="text/event-stream",
    )
