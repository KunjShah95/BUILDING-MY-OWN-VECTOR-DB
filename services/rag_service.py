"""RAG pipeline: PDF -> Chunk -> Embed -> Vector DB -> Retrieve -> LLM completion."""
from typing import List, Dict, Any, Optional
import logging
import os

from services.embedding_service import embed_text, embed_texts, embed_image
from services.media_store import save_media
from utils.pdf_processor import extract_text_from_pdf
from utils.text_chunker import chunk_text_recursive, chunk_tokens, chunk_by_sentences

logger = logging.getLogger(__name__)


def openai_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """Call OpenAI chat completion."""
    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI SDK not installed. Install: pip install openai"
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"OpenAI API error: {exc}"


class RAGService:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, db_session=None):
        self.db = db_session
        self._vector_service = None
        self._collection_service = None
        self._index_service = None

    def _get_collection_service(self):
        if self._collection_service is None and self.db is not None:
            from services.collection_service import CollectionService
            self._collection_service = CollectionService(self.db)
        return self._collection_service

    def _get_vector_service(self):
        if self._vector_service is None and self.db is not None:
            from services.vector_service import VectorService
            self._vector_service = VectorService(self.db)
        return self._vector_service

    def _search_vectors(self, collection_id: str, query_vector: List[float],
                        k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        from services.collection_index_service import CollectionIndexService
        svc = CollectionIndexService(self.db)
        return svc.search_collection_indexed(
            collection_id=collection_id, query_vector=query_vector,
            k=k, method="brute", filters=filters, distance_metric="cosine",
        )

    def ingest_pdf(
        self, collection_id: str, pdf_path: str,
        chunk_strategy: str = "recursive", chunk_size: int = 500, chunk_overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract, chunk, embed, and store a PDF's contents."""
        try:
            coll_svc = self._get_collection_service()
            if coll_svc:
                coll = coll_svc.get_collection(collection_id)
                if not coll.get("success"):
                    return coll

            full_text = extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                return {"success": False, "message": "No text extracted from PDF"}

            chunks = self._chunk_text(full_text, chunk_strategy, chunk_size, chunk_overlap)

            if not chunks:
                return {"success": False, "message": "No chunks generated"}

            embeddings = embed_texts(chunks)
            vec_svc = self._get_vector_service()
            stored = 0
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_meta = {
                    **(metadata or {}),
                    "source": os.path.basename(pdf_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": "rag_chunk",
                    "text": chunk[:500],
                }
                result = vec_svc.create_vector(vector_data=emb, metadata=chunk_meta, collection_id=collection_id)
                if result.get("success"):
                    stored += 1

            return {
                "success": True,
                "message": f"Ingested {stored}/{len(chunks)} chunks from PDF",
                "total_chunks": len(chunks),
                "stored": stored,
            }
        except Exception as exc:
            logger.exception("PDF ingestion failed")
            return {"success": False, "message": f"PDF ingestion error: {exc}"}

    def _chunk_text(self, text: str, strategy: str, chunk_size: int, overlap: int) -> List[str]:
        if strategy == "tokens":
            return chunk_tokens(text, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "sentences":
            return chunk_by_sentences(text, max_sentences=chunk_size, overlap_sentences=overlap)
        return chunk_text_recursive(text, chunk_size=chunk_size, overlap=overlap)

    def ingest_document(
        self,
        collection_id: str,
        file_path: str,
        chunk_strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        extract_images: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest any supported document, optionally extracting embedded images."""
        from utils.document_processor import extract_text, extract_images_from_pdf

        try:
            text = extract_text(file_path)
            if not text.strip():
                return {"success": False, "message": "No text extracted"}

            chunks = self._chunk_text(text, chunk_strategy, chunk_size, chunk_overlap)
            if not chunks:
                return {"success": False, "message": "No chunks generated"}

            embeddings = embed_texts(chunks)
            vec_svc = self._get_vector_service()
            stored = 0

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_meta = {
                    **(metadata or {}),
                    "source": os.path.basename(file_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": "rag_chunk",
                    "text": chunk[:500],
                }
                result = vec_svc.create_vector(
                    vector_data=emb, metadata=chunk_meta, collection_id=collection_id
                )
                if result.get("success"):
                    stored += 1

            ext = os.path.splitext(file_path)[1].lower()
            image_count = 0
            if extract_images and ext == ".pdf":
                images = extract_images_from_pdf(file_path)
                for img_bytes in images:
                    if len(img_bytes) < 1024:
                        continue
                    content_uri = save_media(collection_id, f"pdf_img_{stored}_{image_count}.png", img_bytes)
                    try:
                        img_emb = embed_image(img_bytes)
                        img_meta = {
                            **(metadata or {}),
                            "source": os.path.basename(file_path),
                            "content_type": "rag_image",
                            "content_uri": content_uri,
                        }
                        vec_svc.create_vector(
                            vector_data=img_emb, metadata=img_meta, collection_id=collection_id
                        )
                        image_count += 1
                    except Exception:
                        pass

            return {
                "success": True,
                "message": f"Ingested {stored} text chunks + {image_count} images",
                "total_chunks": len(chunks),
                "stored": stored,
                "images_extracted": image_count,
            }
        except Exception as exc:
            logger.exception("Document ingestion failed")
            return {"success": False, "message": str(exc)}

    def query(
        self, collection_id: str, query: str, k: int = 5,
        llm_model: str = "gpt-4o-mini", api_key: Optional[str] = None,
        system_prompt: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """RAG query: embed -> retrieve -> build context -> LLM answer."""
        try:
            query_vector = embed_text(query)
            search_result = self._search_vectors(
                collection_id=collection_id, query_vector=query_vector, k=k,
                filters={"content_type": "rag_chunk"},
            )
            if not search_result.get("success"):
                return {"success": False, "message": "Search failed", "error": search_result.get("message")}

            results = search_result.get("results", [])
            if not results:
                return {"success": True, "answer": "No relevant documents found.", "context": [], "query": query}

            context_parts = []
            for r in results:
                meta = r.get("metadata", {})
                chunk_text = meta.get("text", "")
                source = meta.get("source", "unknown")
                if chunk_text:
                    context_parts.append(f"[Source: {source}]\n{chunk_text}")

            context = "\n\n---\n\n".join(context_parts)
            default_system = (
                "You are a helpful assistant. Answer the user's question based solely on the provided context. "
                "If the context doesn't contain enough information, say so. Cite the source document when possible."
            )
            messages = [
                {"role": "system", "content": system_prompt or default_system},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            answer = openai_chat_completion(
                messages=messages, model=llm_model, api_key=api_key,
                max_tokens=max_tokens, temperature=temperature,
            )
            return {
                "success": True, "answer": answer, "query": query,
                "context": [
                    {"text": r["metadata"].get("text", ""), "source": r["metadata"].get("source", ""),
                     "distance": r.get("distance", 0)}
                    for r in results
                ],
                "total_results": len(results),
            }
        except Exception as exc:
            logger.exception("RAG query failed")
            return {"success": False, "message": f"RAG query error: {exc}"}
