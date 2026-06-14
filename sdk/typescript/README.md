# Vector DB — TypeScript Client

TypeScript/JavaScript client for the Vector Database multimodal API. Works in Node.js (v18+) and modern browsers (fetch API).

## Install

```bash
npm install vector-db-client
```

Or install from the local repo:

```bash
cd sdk/typescript
npm install
npm run build
```

## Quickstart

```typescript
import { VectorDBClient } from "vector-db-client";

const client = new VectorDBClient({ baseUrl: "http://localhost:8000" });

// ── Collections ──────────────────────────────────────

await client.collections.create({
  name: "Docs",
  collectionId: "docs",
  modality: "text",
  dimension: 384,
});

// ── Text ingest & search ─────────────────────────────

await client.multimodal.ingestText("docs", {
  text: "Returns accepted within 30 days.",
  vectorId: "policy-returns",
});

const hits = await client.multimodal.searchText("docs", {
  query: "return policy",
  k: 5,
});
console.log(hits.results[0].vectorId, hits.results[0].distance);

// ── Image ingest & search ────────────────────────────

const imageBytes = new Uint8Array(/* … */);
await client.multimodal.ingestImage("photos", {
  blob: new Blob([imageBytes], { type: "image/jpeg" }),
});

const results = await client.multimodal.searchImage("photos", {
  blob: new Blob([queryBytes], { type: "image/jpeg" }),
  k: 5,
});

// ── Vector CRUD ──────────────────────────────────────

await client.vectors.create({ vector: [0.1, 0.2, 0.3] });
await client.vectors.get("vec_abc123");
await client.vectors.delete("vec_abc123");
await client.vectors.search({ queryVector: [0.1, 0.2, 0.3], k: 10 });

client.health(); // health check
```

## API Surface

| Resource | Methods |
|----------|---------|
| `client.collections` | `create`, `list`, `get`, `delete`, `buildIndex`, `indexStats` |
| `client.vectors` | `create`, `get`, `delete`, `search` |
| `client.multimodal` | `ingestText`, `searchText`, `ingestImage`, `searchImage`, `ingestAudio`, `searchAudio`, `query` (RAG) |

Errors throw `VectorDBHTTPError` with `statusCode` and `detail`.

## Development

```bash
cd sdk/typescript
npm install
npm run build     # compile TypeScript → dist/
npm test          # run vitest tests
npm run typecheck # type-check without emitting
```
