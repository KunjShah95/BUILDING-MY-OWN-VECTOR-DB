# SDK Feature Compatibility

| Legend | Meaning |
|--------|---------|
| ✅     | Fully supported |
| ➖     | Partial / limited support |
| ❌     | Not supported |

## Client Init

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| Default `base_url` (`http://localhost:8000`) | ✅ | ✅ | ✅ |
| Custom `base_url` | ✅ | ✅ | ✅ |
| Trailing-slash removal | ✅ | ✅ | ✅ |
| Custom timeout | ✅ | ✅ | ✅ |
| Client-level default headers | ✅ | ✅ | ✅ |
| Custom HTTP client/transport | ❌ | ✅ (`HTTPClient`) | ✅ (via `httpx.Client`) |
| Context manager (`with` / context) | ❌ | ✅ (Go `context`) | ✅ (`__enter__`/`__exit__`) |
| Context cancellation | ✅ (`AbortSignal`) | ✅ (`context.Context`) | ❌ |
| Per-request option overrides | ✅ | ✅ | ❌ |
| URL-encoded ID escaping | ✅ (`encodeURIComponent`) | ✅ (`url.PathEscape`) | ❌ (raw string) |

## Client Methods

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| `health()` | ✅ | ✅ | ❌ |
| `close()` / cleanup | ❌ | ❌ | ✅ |

## Collections API

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| `create()` | ✅ | ✅ | ✅ |
| `list(limit, offset)` | ✅ | ✅ | ✅ |
| `get(collectionId)` | ✅ | ✅ | ✅ |
| `delete(collectionId)` | ✅ | ✅ | ✅ |
| `buildIndex(params)` | ✅ | ✅ | ✅ |
| `IndexStats()` | ✅ | ✅ | ✅ |
| Typed `Collection` return | ✅ | ✅ | ✅ |

### Collection `create()` params

| Param | TypeScript | Go | Python |
|-------|:----------:|:--:|:------:|
| `name` | ✅ | ✅ | ✅ |
| `collection_id` | ✅ | ✅ | ✅ |
| `modality` | ✅ (default `"text"`) | ✅ (default `"text"`) | ✅ (default `"text"`) |
| `dimension` | ✅ | ✅ | ✅ |
| `embedding_model` | ✅ | ✅ | ✅ |
| `description` | ✅ | ✅ | ✅ |
| `distance_metric` | ✅ (default `"cosine"`) | ✅ (default `"cosine"`) | ✅ (default `"cosine"`) |

### `buildIndex()` params

| Param | TypeScript | Go | Python |
|-------|:----------:|:--:|:------:|
| `method` (default `"hnsw"`) | ✅ | ✅ | ✅ |
| `m` (default `16`) | ✅ | ✅ | ✅ |
| `m0` | ✅ | ✅ | ✅ |
| `ef_construction` (default `200`) | ✅ | ✅ | ✅ |
| `n_clusters` (default `100`) | ✅ | ✅ | ✅ |
| `n_probes` (default `10`) | ✅ | ✅ | ✅ |

## Vectors API

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| `create(vector, metadata, vectorId)` | ✅ | ✅ | ✅ |
| `get(vectorId)` | ✅ | ✅ | ✅ |
| `delete(vectorId)` | ✅ | ✅ | ✅ |
| `search(params)` | ✅ | ✅ | ✅ |

### Vector `search()` params

| Param | TypeScript | Go | Python |
|-------|:----------:|:--:|:------:|
| `query_vector` | ✅ | ✅ | ✅ |
| `k` (default `5`) | ✅ | ✅ | ✅ |
| `method` (default `"hnsw"`) | ✅ | ✅ | ✅ |
| `filters` | ✅ | ✅ | ✅ |

## Multimodal API

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| `ingestText(collectionId, text, ...)` | ✅ | ✅ | ✅ |
| `searchText(collectionId, query, ...)` | ✅ | ✅ | ✅ |
| `ingestImage(collectionId, blob/io.Reader, ...)` | ✅ (`Blob`/`File`) | ✅ (`io.Reader`) | ✅ (`path`/`bytes`) |
| `searchImage(collectionId, blob/io.Reader, ...)` | ✅ (`Blob`/`File`) | ✅ (`io.Reader`) | ✅ (`path`/`bytes`) |
| `ingestAudio(collectionId, blob/io.Reader, ...)` | ✅ (`Blob`/`File`) | ✅ (`io.Reader`) | ✅ (`path`/`bytes`) |
| `searchAudio(collectionId, blob/io.Reader, ...)` | ✅ (`Blob`/`File`) | ✅ (`io.Reader`) | ✅ (`path`/`bytes`) |
| `query(collectionId, query, k)` — RAG | ✅ | ✅ | ❌ |
| Typed `SearchResult` return | ✅ | ✅ | ✅ |

### SearchText params

| Param | TypeScript | Go | Python |
|-------|:----------:|:--:|:------:|
| `query` | ✅ | ✅ | ✅ |
| `k` (default `5`) | ✅ | ✅ | ✅ |
| `method` (default `"brute"`) | ✅ | ✅ | ✅ |
| `filters` | ✅ | ✅ | ✅ |

### File ingest params (image/audio)

| Param | TypeScript | Go | Python |
|-------|:----------:|:--:|:------:|
| File source | `Blob` / `File` | `io.Reader` | `str` / `Path` / `bytes` |
| `filename` | ✅ | ✅ | ✅ (inferred) |
| `metadata` | ✅ | ✅ | ✅ |
| `vector_id` | ✅ | ✅ | ✅ |

### File search params (image/audio)

| Param | TypeScript | Go | Python |
|-------|:----------:|:--:|:------:|
| File source | `Blob` / `File` | `io.Reader` | `str` / `Path` / `bytes` |
| `k` (default `5`) | ✅ | ✅ | ✅ |
| `method` | ✅ | ✅ | ✅ |
| `filters` | ✅ | ✅ | ✅ |

## LangChain / Ecosystem

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| LangChain VectorStore | ❌ | ❌ | ✅ (`VectorDBVectorStore`) |
| LlamaIndex adapter | ❌ | ❌ | ❌ (in repo, not in SDK) |
| MCP server | ❌ | ❌ | ❌ (in repo, not in SDK) |

## Error Handling

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| `VectorDBError` base class | ✅ | ✅ | ✅ |
| `VectorDBHTTPError` with `statusCode` | ✅ | ✅ | ✅ |
| Non-JSON error body fallback | ✅ | ✅ | ✅ |
| Structured `detail` in error | ✅ | ✅ | ✅ |

## HTTP Layer

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| Underlying library | `fetch` (browser/Node) | `net/http` | `httpx` |
| Query string builder | ✅ (`buildQueryString`) | ✅ (`buildURL`) | ✅ (via `httpx`) |
| JSON POST | ✅ | ✅ | ✅ |
| JSON GET | ✅ | ✅ | ✅ |
| JSON DELETE | ✅ | ✅ | ✅ |
| Multipart POST | ✅ | ✅ | ✅ |
| Network error wrapping | ✅ (`wrapFetch`) | ✅ (`&VectorDBError{...}`) | ❌ (via httpx) |
| Header propagation | ✅ (`mergeOptions`) | ✅ (`mergeOpts`) | ❌ (client level only) |
| Per-request headers | ✅ | ✅ | ❌ |

## Typed Models

| Model | TypeScript | Go | Python |
|-------|:----------:|:--:|:------:|
| `Collection` | ✅ (class) | ✅ (struct) | ✅ (dataclass) |
| `SearchHit` | ✅ (class) | ✅ (struct) | ✅ (dataclass) |
| `SearchResult` | ✅ (class) | ✅ (struct) | ✅ (dataclass) |

## Tests

| Feature | TypeScript | Go | Python |
|---------|:----------:|:--:|:------:|
| Total tests | 91 | 64 | existing |
| Unit tests | ✅ | ✅ | ✅ |
| Integration tests (HTTP server) | ✅ | ✅ | ❌ |
| Edge case tests | ✅ | ✅ | ❌ |
| HTTP error code coverage | 10 codes | 10 codes | minimal |

---

## Summary Gaps

| Gap | SDK(s) affected |
|-----|:---------------:|
| **No `health()` method** | Python |
| **No RAG `query()`** | Python |
| **No per-request option overrides** | Python |
| **No context/AbortSignal support** | Python |
| **No URL encoding on IDs** | Python |
| **No `close()` / cleanup** | TypeScript, Go |
| **No LangChain support** | TypeScript, Go |
| **No integration tests** | Python |
| **No edge case tests** | Python |
