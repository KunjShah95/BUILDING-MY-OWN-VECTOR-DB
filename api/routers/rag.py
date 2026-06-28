"""RAG API router — ingest anything, query with any LLM.

Ingest sources:
  POST /collections/{id}/ingest/pdf      — PDF file upload
  POST /collections/{id}/ingest/url      — Web page URL
  POST /collections/{id}/ingest/text     — Raw text/markdown
  POST /collections/{id}/ingest/file     — TXT, DOCX, MD, CSV file upload

Query endpoints:
  POST /collections/{id}/query           — RAG Q&A (OpenAI / Anthropic / Ollama)
  POST /collections/{id}/query/stream    — Server-Sent Events streaming answer
  GET  /collections/{id}/documents       — List ingested document sources
  DELETE /collections/{id}/documents/{doc_id} — Remove a document's chunks
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from config.database import get_db
from services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["RAG"])


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    return RAGService(db_session=db)


# ── LLM provider registry ─────────────────────────────────────────────────────
#
# All OpenAI-compatible providers share the same client; only base_url + key differ.
# Model string format:  "<prefix>:<model-name>"  e.g. "groq:llama3-8b-8192"
#
# Provider       Prefix        Env var             Free tier
# ──────────────────────────────────────────────────────────────────────────────
# Groq           groq:         GROQ_API_KEY        ~14 400 req/day (Llama3/Mixtral)
# OpenRouter     openrouter:   OPENROUTER_API_KEY  many free models (mistral-7b etc)
# Google Gemini  gemini:       GOOGLE_API_KEY      1 500 req/day (gemini-1.5-flash)
# NVIDIA NIM     nvidia:       NVIDIA_API_KEY      free credits (Llama3/Mistral)
# Together AI    together:     TOGETHER_API_KEY    $25 free credits
# Mistral AI     mistral:      MISTRAL_API_KEY     free tier (mistral-7b-instruct)
# DeepSeek       deepseek:     DEEPSEEK_API_KEY    very cheap / generous free tier
# HuggingFace    hf:           HF_API_KEY          free Inference API
# Ollama         ollama:       (none, local)        fully local, unlimited
# Anthropic      claude-*      ANTHROPIC_API_KEY   paid (kept for completeness)
# OpenAI         gpt-*         OPENAI_API_KEY      paid (kept for completeness)

_OPENAI_COMPAT: Dict[str, Dict[str, str]] = {
    "groq":       {"base": "https://api.groq.com/openai/v1",              "env": "GROQ_API_KEY"},
    "openrouter": {"base": "https://openrouter.ai/api/v1",                "env": "OPENROUTER_API_KEY"},
    "gemini":     {"base": "https://generativelanguage.googleapis.com/v1beta/openai/", "env": "GOOGLE_API_KEY"},
    "nvidia":     {"base": "https://integrate.api.nvidia.com/v1",         "env": "NVIDIA_API_KEY"},
    "together":   {"base": "https://api.together.xyz/v1",                 "env": "TOGETHER_API_KEY"},
    "mistral":    {"base": "https://api.mistral.ai/v1",                   "env": "MISTRAL_API_KEY"},
    "deepseek":   {"base": "https://api.deepseek.com/v1",                 "env": "DEEPSEEK_API_KEY"},
    "hf":         {"base": "https://api-inference.huggingface.co/v1",     "env": "HF_API_KEY"},
    "openai":     {"base": "https://api.openai.com/v1",                   "env": "OPENAI_API_KEY"},
}


def _parse_model(model: str):
    """Return (provider_key, bare_model_name).

    Examples:
      "groq:llama3-8b-8192"  → ("groq", "llama3-8b-8192")
      "gpt-4o-mini"          → ("openai", "gpt-4o-mini")
      "claude-haiku-4-5"     → ("anthropic", "claude-haiku-4-5")
      "ollama:llama3"        → ("ollama", "llama3")
    """
    if ":" in model:
        prefix, name = model.split(":", 1)
        if prefix in _OPENAI_COMPAT or prefix == "ollama":
            return prefix, name
    if model.startswith("claude"):
        return "anthropic", model
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai", model
    # Default: try openai-compat with the raw name
    return "openai", model


def _openai_compat_complete(provider: str, model: str, messages, max_tokens, temperature) -> str:
    cfg = _OPENAI_COMPAT[provider]
    key = os.getenv(cfg["env"], "")
    if not key:
        return f"[{provider}] No {cfg['env']} set."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=cfg["base"])
        r = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return r.choices[0].message.content or ""
    except Exception as exc:
        logger.error("%s error: %s", provider, exc)
        return f"[{provider} error] {exc}"


def _llm_complete(messages: List[Dict[str, str]], model: str,
                  max_tokens: int, temperature: float) -> str:
    provider, bare = _parse_model(model)

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            return "[Anthropic] No ANTHROPIC_API_KEY set."
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = client.messages.create(
                model=bare, max_tokens=max_tokens,
                system=sys_msg or "You are a helpful assistant.",
                messages=user_msgs,
            )
            return resp.content[0].text
        except Exception as exc:
            return f"[Anthropic error] {exc}"

    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            import httpx
            r = httpx.post(
                f"{base}/api/chat",
                json={"model": bare, "messages": messages, "stream": False,
                      "options": {"num_predict": max_tokens, "temperature": temperature}},
                timeout=60.0,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as exc:
            return f"[Ollama error] {exc}"

    return _openai_compat_complete(provider, bare, messages, max_tokens, temperature)


async def _stream_llm(messages, model, max_tokens, temperature):
    """Async SSE generator — works for all providers."""
    provider, bare = _parse_model(model)

    # ── Anthropic native streaming ────────────────────────────────────────────
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            yield f'data: {json.dumps({"token": "[No ANTHROPIC_API_KEY]"})}\n\n'
            yield "data: [DONE]\n\n"
            return
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_msgs = [m for m in messages if m["role"] != "system"]
            with client.messages.stream(
                model=bare, max_tokens=max_tokens,
                system=sys_msg or "You are a helpful assistant.",
                messages=user_msgs,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'token': text})}\n\n"
        except Exception as exc:
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'
        yield "data: [DONE]\n\n"
        return

    # ── Ollama native streaming ───────────────────────────────────────────────
    if provider == "ollama":
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{base}/api/chat",
                    json={"model": bare, "messages": messages, "stream": True,
                          "options": {"num_predict": max_tokens}},
                ) as r:
                    async for line in r.aiter_lines():
                        if line:
                            try:
                                tok = json.loads(line).get("message", {}).get("content", "")
                                if tok:
                                    yield f"data: {json.dumps({'token': tok})}\n\n"
                            except json.JSONDecodeError:
                                pass
        except Exception as exc:
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'
        yield "data: [DONE]\n\n"
        return

    # ── OpenAI-compatible streaming (Groq / OpenRouter / Gemini / NVIDIA / etc.)
    cfg = _OPENAI_COMPAT.get(provider, _OPENAI_COMPAT["openai"])
    key = os.getenv(cfg["env"], "")
    if not key:
        env_name = cfg["env"]
        yield f'data: {json.dumps({"token": f"[No {env_name} set]"})}\n\n'
        yield "data: [DONE]\n\n"
        return
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=key, base_url=cfg["base"])
        stream = await client.chat.completions.create(
            model=bare, messages=messages,
            max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        async for chunk in stream:
            tok = chunk.choices[0].delta.content
            if tok:
                yield f"data: {json.dumps({'token': tok})}\n\n"
    except Exception as exc:
        yield f'data: {json.dumps({"error": str(exc)})}\n\n'
    yield "data: [DONE]\n\n"


# ── Ingest helpers ────────────────────────────────────────────────────────────

def _ingest_text_chunks(service: RAGService, collection_id: str, text: str,
                         source: str, metadata: Optional[Dict],
                         chunk_strategy: str, chunk_size: int, chunk_overlap: int) -> Dict:
    """Chunk + embed raw text and store into the collection."""
    from utils.text_chunker import chunk_text_recursive, chunk_tokens, chunk_by_sentences
    from services.embedding_service import embed_texts

    if chunk_strategy == "tokens":
        from utils.text_chunker import chunk_tokens
        chunks = chunk_tokens(text, chunk_size=chunk_size, overlap=chunk_overlap)
    elif chunk_strategy == "sentences":
        from utils.text_chunker import chunk_by_sentences
        chunks = chunk_by_sentences(text, chunk_size=chunk_size)
    else:
        from utils.text_chunker import chunk_text_recursive
        chunks = chunk_text_recursive(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        return {"success": False, "message": "No chunks generated from content"}

    embeddings = embed_texts(chunks)
    vs = service._get_vector_service()
    stored = 0
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        meta = {
            **(metadata or {}),
            "source": source,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "content_type": "rag_chunk",
            "text": chunk[:500],
        }
        r = vs.create_vector(vector_data=emb, metadata=meta, collection_id=collection_id)
        if r.get("success"):
            stored += 1

    return {
        "success": True,
        "message": f"Stored {stored}/{len(chunks)} chunks",
        "source": source,
        "total_chunks": len(chunks),
        "stored": stored,
    }


# ── Endpoints: Ingest ─────────────────────────────────────────────────────────

@router.post("/collections/{collection_id}/ingest/pdf")
async def ingest_pdf(
    collection_id: str,
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    metadata: Optional[str] = Form(None),
    service: RAGService = Depends(get_rag_service),
):
    """Upload a PDF → extract text → chunk → embed → store."""
    content = await file.read()
    if not content:
        raise HTTPException(400, {"success": False, "message": "Empty file"})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
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
        raise HTTPException(400, result)
    return result


@router.post("/collections/{collection_id}/ingest/url")
async def ingest_url(
    collection_id: str,
    url: str = Body(..., embed=True),
    chunk_strategy: str = Body("recursive"),
    chunk_size: int = Body(500),
    chunk_overlap: int = Body(50),
    metadata: Optional[Dict[str, Any]] = Body(None),
    service: RAGService = Depends(get_rag_service),
):
    """Fetch a web page, extract clean text, chunk, embed, and store."""
    try:
        import httpx
        from search_engine.crawler.parser import extract_content
        resp = httpx.get(url, follow_redirects=True, timeout=15.0,
                         headers={"User-Agent": "VectorDB-RAG/1.0"})
        resp.raise_for_status()
        parsed = extract_content(resp.text, url)
        text = parsed.get("text", "") or ""
        title = parsed.get("title", url)
    except Exception as exc:
        raise HTTPException(400, {"success": False, "message": f"Failed to fetch {url}: {exc}"})

    if not text.strip():
        raise HTTPException(400, {"success": False, "message": "No text extracted from URL"})

    meta = {**(metadata or {}), "url": url, "title": title}
    result = _ingest_text_chunks(service, collection_id, text, source=url,
                                  metadata=meta, chunk_strategy=chunk_strategy,
                                  chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not result["success"]:
        raise HTTPException(400, result)
    return result


@router.post("/collections/{collection_id}/ingest/text", operation_id="rag_ingest_text")
async def ingest_text(
    collection_id: str,
    text: str = Body(...),
    source: str = Body("manual"),
    chunk_strategy: str = Body("recursive"),
    chunk_size: int = Body(500),
    chunk_overlap: int = Body(50),
    metadata: Optional[Dict[str, Any]] = Body(None),
    service: RAGService = Depends(get_rag_service),
):
    """Ingest raw text or Markdown directly."""
    if not text.strip():
        raise HTTPException(400, {"success": False, "message": "Empty text"})
    result = _ingest_text_chunks(service, collection_id, text, source=source,
                                  metadata=metadata, chunk_strategy=chunk_strategy,
                                  chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not result["success"]:
        raise HTTPException(400, result)
    return result


@router.post("/collections/{collection_id}/ingest/file")
async def ingest_file(
    collection_id: str,
    file: UploadFile = File(...),
    chunk_strategy: str = Form("recursive"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    metadata: Optional[str] = Form(None),
    service: RAGService = Depends(get_rag_service),
):
    """Upload TXT, MD, DOCX, or CSV file → chunk → embed → store."""
    content = await file.read()
    if not content:
        raise HTTPException(400, {"success": False, "message": "Empty file"})

    fname = (file.filename or "upload").lower()
    text = ""

    if fname.endswith(".txt") or fname.endswith(".md"):
        text = content.decode("utf-8", errors="replace")

    elif fname.endswith(".docx"):
        try:
            import docx, io
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(400, {"success": False, "message": "python-docx not installed: pip install python-docx"})

    elif fname.endswith(".csv"):
        import csv, io
        reader = csv.reader(io.StringIO(content.decode("utf-8", errors="replace")))
        rows = list(reader)
        text = "\n".join(",".join(row) for row in rows)

    elif fname.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            parsed_meta = json.loads(metadata) if metadata else None
        except Exception:
            parsed_meta = None
        result = service.ingest_pdf(
            collection_id=collection_id, pdf_path=tmp_path,
            chunk_strategy=chunk_strategy, chunk_size=chunk_size,
            chunk_overlap=chunk_overlap, metadata=parsed_meta,
        )
        os.unlink(tmp_path)
        if not result["success"]:
            raise HTTPException(400, result)
        return result

    else:
        raise HTTPException(400, {"success": False,
                                  "message": f"Unsupported file type '{fname}'. Supported: pdf, txt, md, docx, csv"})

    if not text.strip():
        raise HTTPException(400, {"success": False, "message": "No text extracted from file"})

    try:
        parsed_meta = json.loads(metadata) if metadata else None
    except Exception:
        parsed_meta = None

    result = _ingest_text_chunks(
        service, collection_id, text, source=file.filename or "upload",
        metadata=parsed_meta, chunk_strategy=chunk_strategy,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )
    if not result["success"]:
        raise HTTPException(400, result)
    return result


# ── Endpoints: Query ──────────────────────────────────────────────────────────

@router.post("/collections/{collection_id}/query")
async def rag_query(
    collection_id: str,
    query: str = Body(...),
    k: int = Body(5, ge=1, le=50),
    model: str = Body("gpt-4o-mini"),
    max_tokens: int = Body(800, ge=50, le=4096),
    temperature: float = Body(0.3, ge=0.0, le=2.0),
    system_prompt: Optional[str] = Body(None),
    service: RAGService = Depends(get_rag_service),
):
    """RAG Q&A — retrieve relevant chunks then synthesise with any LLM.

    model accepts:
      - OpenAI:    "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"
      - Anthropic: "claude-3-5-haiku-20241022", "claude-opus-4-8", etc.
      - Ollama:    "ollama:llama3", "ollama:mistral", "ollama:phi3"
    """
    from services.embedding_service import embed_text

    try:
        qvec = embed_text(query)
    except Exception as exc:
        raise HTTPException(503, {"success": False, "message": f"Embedding failed: {exc}"})

    search_result = service._search_vectors(collection_id, qvec, k=k)
    hits = search_result.get("results", [])
    if not hits:
        return {
            "success": True,
            "query": query,
            "answer": "No relevant documents found in this collection.",
            "sources": [],
            "model": model,
        }

    context_parts = []
    sources = []
    for i, hit in enumerate(hits):
        meta = hit.get("metadata") or {}
        chunk_text = meta.get("text", "")
        src = meta.get("source", meta.get("url", f"chunk_{i}"))
        if chunk_text:
            context_parts.append(f"[{i+1}] {chunk_text}")
        if src not in sources:
            sources.append(src)

    context = "\n\n".join(context_parts)
    sys_prompt = system_prompt or (
        "You are a helpful assistant. Answer the user's question based only on the "
        "provided context. If the answer is not in the context, say so clearly. "
        "Cite sources using [N] notation."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    answer = _llm_complete(messages, model, max_tokens, temperature)
    return {
        "success": True,
        "query": query,
        "answer": answer,
        "sources": sources,
        "context_chunks": len(hits),
        "model": model,
    }


@router.post("/collections/{collection_id}/query/stream")
async def rag_query_stream(
    collection_id: str,
    query: str = Body(...),
    k: int = Body(5),
    model: str = Body("claude-3-5-haiku-20241022"),
    max_tokens: int = Body(800),
    temperature: float = Body(0.3),
    service: RAGService = Depends(get_rag_service),
):
    """Streaming RAG — returns Server-Sent Events (text/event-stream)."""
    from services.embedding_service import embed_text

    try:
        qvec = embed_text(query)
    except Exception as exc:
        raise HTTPException(503, {"success": False, "message": f"Embedding failed: {exc}"})

    search_result = service._search_vectors(collection_id, qvec, k=k)
    hits = search_result.get("results", [])

    context = "\n\n".join(
        f"[{i+1}] {(h.get('metadata') or {}).get('text', '')}"
        for i, h in enumerate(hits)
        if (h.get('metadata') or {}).get('text')
    ) or "No context found."

    messages = [
        {"role": "system", "content": "Answer based on the context. Cite [N] sources."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    return StreamingResponse(
        _stream_llm(messages, model, max_tokens, temperature),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/collections/{collection_id}/documents")
async def list_documents(
    collection_id: str,
    limit: int = Query(50, ge=1, le=500),
    service: RAGService = Depends(get_rag_service),
):
    """List unique document sources ingested into a collection."""
    vs = service._get_vector_service()
    if vs is None:
        raise HTTPException(503, {"success": False, "message": "No database session"})
    result = vs.get_all_vectors(limit=10000, collection_id=collection_id)
    vectors = result.get("vectors", [])
    seen: Dict[str, Any] = {}
    for v in vectors:
        meta = v.get("metadata") or {}
        src = meta.get("source") or meta.get("url", "")
        if src and src not in seen:
            seen[src] = {
                "source": src,
                "title": meta.get("title", src),
                "content_type": meta.get("content_type", "unknown"),
                "chunk_count": 0,
            }
        if src in seen:
            seen[src]["chunk_count"] += 1

    return {
        "success": True,
        "collection_id": collection_id,
        "document_count": len(seen),
        "documents": list(seen.values())[:limit],
    }


@router.delete("/collections/{collection_id}/documents")
async def delete_document(
    collection_id: str,
    source: str = Query(..., description="Source URL or filename to remove"),
    service: RAGService = Depends(get_rag_service),
):
    """Remove all chunks belonging to a document source."""
    vs = service._get_vector_service()
    if vs is None:
        raise HTTPException(503, {"success": False, "message": "No database session"})
    result = vs.get_all_vectors(limit=10000, collection_id=collection_id)
    vectors = result.get("vectors", [])
    deleted = 0
    for v in vectors:
        meta = v.get("metadata") or {}
        if meta.get("source") == source or meta.get("url") == source:
            vs.delete_vector(v["vector_id"])
            deleted += 1
    return {"success": True, "source": source, "chunks_deleted": deleted}


@router.get("/rag/providers")
async def list_providers():
    """List available LLM providers and their configuration status."""
    return {
        "success": True,
        "providers": {
            "openai": {
                "available": bool(os.getenv("OPENAI_API_KEY")),
                "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            },
            "anthropic": {
                "available": bool(os.getenv("ANTHROPIC_API_KEY")),
                "models": ["claude-3-5-haiku-20241022", "claude-sonnet-4-6", "claude-opus-4-8"],
            },
            "ollama": {
                "available": bool(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
                "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "models": ["ollama:llama3", "ollama:mistral", "ollama:phi3", "ollama:gemma2"],
                "note": "Requires Ollama running locally — free, no API key needed",
            },
        },
    }
