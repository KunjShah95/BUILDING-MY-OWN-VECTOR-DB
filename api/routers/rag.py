"""RAG API router - PDF ingest and Q&A query endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
import tempfile
import os

from config.database import get_db
from services.rag_service import RAGService

router = APIRouter(tags=["RAG"])


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    return RAGService(db_session=db)


@router.post("/collections/{collection_id}/ingest/pdf", tags=["RAG"])
async def ingest_pdf(
    collection_id: str,
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    metadata: Optional[str] = Form(None),
    service: RAGService = Depends(get_rag_service),
):
    """Upload a PDF, extract text, chunk, embed, and store vectors."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail={"success": False, "message": "Empty file"})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        import json
        parsed_meta = json.loads(metadata) if metadata else None
    except (json.JSONDecodeError, TypeError):
        parsed_meta = None
    result = service.ingest_pdf(
        collection_id=collection_id, pdf_path=tmp_path,
        chunk_strategy=chunk_strategy, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, metadata=parsed_meta,
    )
    os.unlink(tmp_path)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    return result


@router.post("/collections/{collection_id}/query", tags=["RAG"])
async def rag_query(
    collection_id: str,
    query: str = Form(...),
    k: int = Form(5),
    model: str = Form("gpt-4o-mini"),
    max_tokens: int = Form(500),
    temperature: float = Form(0.3),
    service: RAGService = Depends(get_rag_service),
):
    """Ask a question and get an answer grounded in the collection's documents."""
    result = service.query(
        collection_id=collection_id, query=query, k=k,
        llm_model=model, max_tokens=max_tokens, temperature=temperature,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)
    return result
