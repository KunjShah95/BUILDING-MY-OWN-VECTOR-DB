"""Text chunking strategies for RAG."""
from typing import List, Optional


def chunk_text_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Recursive character text splitting with overlap."""
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps or len(text) <= chunk_size:
            return [text]
        sep = seps[0]
        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        result = []
        for c in chunks:
            if len(c) > chunk_size:
                result.extend(_split(c, seps[1:]))
            else:
                result.append(c)
        return result

    chunks = _split(text, separators)

    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_end = chunks[i - 1][-overlap:]
                chunk = prev_end + chunk
            overlapped.append(chunk)
        chunks = overlapped

    return chunks


def chunk_by_sentences(text: str, max_sentences: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Split text into chunks by sentence count."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return []
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + max_sentences
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += max_sentences - overlap_sentences
    return chunks


def chunk_tokens(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Token-aware chunking using tiktoken with fallback."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append(chunk_text)
            start += chunk_size - overlap
        return chunks
    except ImportError:
        return chunk_text_recursive(text, chunk_size=chunk_size * 4, overlap=overlap * 4)
