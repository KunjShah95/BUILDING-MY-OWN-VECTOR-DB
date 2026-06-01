# I Built a RAG Pipeline Using My Own Vector Database (And You Can Too)

I've been building a vector database from scratch for a while now. HNSW graphs, IVF clusters, the whole thing. But a vector DB is just a fancy index if you can't actually *do* anything useful with it. So I built a RAG pipeline on top of it.

The goal was simple: point it at a PDF, ask questions, get answers grounded in the document. What I didn't expect was how many "easy" problems would turn out to be anything but.

## The Pipeline at 30,000 Feet

The flow is boringly standard on paper:

```
PDF → extract text → chunk → embed → store vectors → query → retrieve → build context → LLM
```

But every arrow hides a decision that can wreck your whole pipeline if you get it wrong.

Let me walk through what I built and, more importantly, why I made the choices I did.

## Chunking Is Surprisingly Hard

I thought chunking would be the easy part. Split text into pieces, feed them to an embedding model, done. First version was literally `text[i:i+500]`. Predictably terrible — it cut sentences in half, lost meaning, and the LLM answers were garbage.

I ended up with three strategies:

**Recursive character splitting** — my default. It tries to break on paragraph boundaries first (`\n\n`), falls back to sentences, then punctuation, then spaces. 500 characters per chunk with 50 characters of overlap:

```python
def chunk_text_recursive(text, chunk_size=500, overlap=50):
    separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

    def _split(text, seps):
        if not seps or len(text) <= chunk_size:
            return [text]
        sep = seps[0]
        parts = text.split(sep)
        # ...build chunks up to chunk_size, recurse with smaller separators
```

The overlap is crucial. The first time I ran without it, a sentence like *"The answer to your question is clearly explained in the following section"* got split right after "section" — and the LLM never saw the actual explanation. Overlap means the boundary context bleeds into adjacent chunks.

**Sentence-based** — 5 sentences per chunk, 1 sentence overlap. Cleaner for prose-heavy docs like legal agreements or narrative text. The split happens on actual sentence boundaries using a regex lookbehind:

```python
sentences = re.split(r'(?<=[.!?])\s+', text)
```

**Token-aware** — uses tiktoken (OpenAI's tokenizer) to split on actual token boundaries at 512 tokens. This is the most accurate if you're sending to an LLM, since it matches how the model actually sees text. Falls back to recursive splitting if tiktoken isn't available:

```python
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(text)
# ...split on token boundaries, decode each chunk
```

500 chars with 50 overlap is the default for good reason: it works well for most technical documents, which is what I process most. The overlap-to-chunk ratio is 10%, which is enough to preserve context without duplicating so much that you waste embedding capacity.

## Why Use Your Own Vector Database?

This is the part that might surprise you: I'm not using brute force search for the whole system. The vector DB has HNSW and IVF indexes, so general similarity searches run in about 3.2ms. Brute force against 100K vectors takes 45ms.

But here's the thing — for the RAG pipeline specifically, I default to brute force search within a single collection. Why? Because a RAG collection is typically one document or a handful of documents. At that scale, brute force is fast enough and it guarantees 100% recall. The approximate indexes (HNSW/IVF) trade a tiny bit of accuracy for speed, and when you're searching 500 chunks for a single answer, you don't need that trade-off.

The real win of owning the database is control. I control the schema, so I can store whatever metadata I want alongside each vector — source filename, chunk index, content type, the actual text snippet. When the LLM asks "where did you get that information?", I can trace it back to the exact chunk and source file. You don't get that level of introspection with Pinecone or Weaviate out of the box.

Storage is PostgreSQL with pgvector-style columns (a `vector` column and a flexible `metadata` JSONB column). Every vector I insert looks like this:

```python
chunk_meta = {
    "source": "report.pdf",
    "chunk_index": i,
    "total_chunks": 42,
    "content_type": "rag_chunk",
    "text": chunk[:500],  # stored for context building
}
```

That metadata is what lets me filter queries to only `rag_chunk` content type, or include images alongside text chunks, or pinpoint which source a piece of information came from.

## The Query Flow

The query endpoint at `POST /collections/{id}/query` does four things:

1. **Embed the question** — same model used during ingestion (all-MiniLM-L6-v2, 384 dims)
2. **Search** — brute force cosine similarity, filtered to `rag_chunk` content type, top-k results
3. **Build context** — stitch retrieved chunks together with source annotations
4. **Call the LLM** — inject context into the system prompt and let GPT answer

```python
def query(self, collection_id, query, k=5, llm_model="gpt-4o-mini"):
    query_vector = embed_text(query)
    search_result = self._search_vectors(
        collection_id=collection_id, query_vector=query_vector, k=k,
        filters={"content_type": "rag_chunk"},
    )
    context = build_context_from_results(search_result["results"])
    messages = [
        {"role": "system", "content": system_prompt_with_context(context)},
        {"role": "user", "content": query},
    ]
    answer = openai_chat_completion(messages, model=llm_model)
    return {"answer": answer, "context": results_with_sources}
```

The first time I saw a RAG answer come back hallucinating, I realized the system prompt was everything. Without a strong grounding instruction, the LLM would happily invent facts that sounded plausible. The prompt I settled on:

> "Answer based solely on the provided context. If the context lacks information, say so. Cite the source document when possible."

That last sentence — "cite the source" — was a game changer. Suddenly the LLM started saying "According to report.pdf..." instead of just making up numbers.

## Streaming Makes It Feel Alive

Waiting 5 seconds for a full LLM response feels broken in 2025. So I added an SSE streaming endpoint at `GET /collections/{id}/query/stream`.

The streaming RAG service inherits from the base RAG service — same ingestion, same search — but uses `AsyncOpenAI` to stream tokens back via Server-Sent Events:

```python
async def query_stream(self, collection_id, query, k=5, llm_model="gpt-4o-mini"):
    query_vector = embed_text(query)
    search_result = self._search_vectors(...)

    # Yield context first so the frontend can show sources
    yield f"data: {json.dumps({'context': context_data})}\n\n"

    # Then stream the answer token by token
    async for chunk in stream_llm_response(messages, model=llm_model):
        yield chunk
```

The frontend gets the retrieved context first (sources, distances, text snippets), then the answer appears word by word. It makes the UX feel responsive even for long answers.

The headers matter here — `X-Accel-Buffering: no` is critical if you're behind nginx:

```python
return StreamingResponse(
    svc.query_stream(...),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    },
)
```

## Multi-Modal Bonus

PDFs don't just have text — they have diagrams, charts, screenshots. A pure text RAG pipeline would miss all of that.

The `ingest_document()` function handles this. After extracting and chunking the text, it rips every embedded image out of the PDF using PyMuPDF's `page.get_images()`. Each image is saved to a media store, embedded via CLIP (clip-ViT-B-32), and stored as a separate vector with `content_type: "rag_image"`:

```python
if extract_images and ext == ".pdf":
    images = extract_images_from_pdf(file_path)
    for img_bytes in images:
        if len(img_bytes) < 1024:
            continue  # skip tiny images (icons, bullets)
        content_uri = save_media(collection_id, f"pdf_img_{i}.png", img_bytes)
        img_emb = embed_image(img_bytes)
        vec_svc.create_vector(
            vector_data=img_emb,
            metadata={"content_type": "rag_image", "content_uri": content_uri, ...}
        )
```

CLIP (which stands for Contrastive Language-Image Pre-training) maps both text and images into the same embedding space. So you can search "architecture diagram" and find the actual architecture diagram from the PDF, even though visually it looks nothing like text. The query flow handles this seamlessly because text and image vectors live in the same collection.

I skip images smaller than 1KB because those are usually bullet points or decorative icons that add noise.

## What I'd Improve

The pipeline works, but there are things I'd change:

**Better re-ranking.** Right now it's just cosine similarity → top-k. A cross-encoder re-ranker (like Cohere's or a small BERT model) would significantly improve the relevance of retrieved chunks. The initial retrieval can be generous — grab 20 chunks — then let the re-ranker pick the best 5.

**Hybrid search integration.** The vector DB already has a BM25 sparse index and a hybrid search service that fuses dense and sparse results. I need to plug that into the RAG query flow. Keyword matching catches things that semantic search misses (proper nouns, code snippets, IDs).

**More file types.** The document processor already handles PDF, DOCX, HTML, Markdown, and TXT. I'd add EPUB and CSV/Excel next. Spreadsheets are a huge blind spot — so much business knowledge lives in cells, not paragraphs.

**Cost control on long docs.** A 500-page PDF generates hundreds of chunks, each getting embedded. Embedding costs are low (running locally with sentence-transformers), but storing all those vectors adds up. Smart pruning — filtering chunks that are too similar to their neighbors — would reduce storage without sacrificing coverage.

## Closing Thoughts

Building a RAG pipeline on top of your own vector database is satisfying in a way that plugging into a managed service isn't. Every component is visible, every decision is yours. The chunker, the embedding model, the search strategy, the prompt — they all interact. Change one, and the quality of answers shifts.

The full pipeline is about 250 lines of Python (excluding the vector DB itself). The tests — 10 for the RAG service, 8 for the document processor — cover the critical paths: ingest, query, chunking edge cases, no-results handling.

If you're building something similar, my advice is: nail your chunking strategy early. Bad chunks mean bad embeddings, which mean bad retrieval, which means the best LLM in the world can't save you. Everything downstream depends on that first step.
