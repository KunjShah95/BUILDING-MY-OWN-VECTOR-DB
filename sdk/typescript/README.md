# Vector DB Client (TypeScript)

The official TypeScript/JavaScript client for the **Massively Scalable ANN Search Engine & Vector Database**.

This client is designed for production use, supporting high-throughput connection configurations, advanced ANN (Approximate Nearest Neighbor) parameters, and multi-modal semantic search.

## Installation

Install via NPM or Yarn:

```bash
npm install vector-db-client
```
or
```bash
yarn add vector-db-client
```

## Quick Start

```typescript
import { VectorDBClient } from 'vector-db-client';

// Initialize the client
const client = new VectorDBClient({
  baseUrl: 'http://localhost:8000',
  timeout: 60000, // Configure timeout for massive queries
});

async function main() {
  // 1. Create a Collection with ANN parameters
  await client.collections.create({
    name: "products",
    dimension: 384, // e.g., for all-MiniLM-L6-v2
    distance_metric: "cosine",
    // Extreme Scale ANN Parameters:
    // M: Number of bi-directional links created for every new element. Higher = better recall, slower indexing.
    // ef_construction: Size of dynamic list for nearest neighbors during index building.
    // Use the backend default (M=16, ef_construction=200) or override if the endpoint supports it.
  });

  // 2. Insert Vectors
  await client.vectors.create({
    collection_name: "products",
    vectors: [
      { id: "1", vector: [0.1, 0.2, ...], metadata: { category: "electronics" } },
      { id: "2", vector: [0.5, 0.1, ...], metadata: { category: "clothing" } }
    ]
  });

  // 3. Search using the ANN Engine
  const results = await client.vectors.search({
    collection_name: "products",
    query_vector: [0.1, 0.2, ...],
    top_k: 10,
    // Provide a metadata filter to narrow search before ANN graph traversal
    filter: { category: "electronics" } 
  });

  console.log("Top matches:", results.hits);
}

main();
```

## Multimodal Search

This database provides an integrated embedding service. Instead of managing embeddings yourself, you can send raw text or images.

```typescript
// Search with a text query
const textResults = await client.multimodal.searchText({
  collection_name: "products",
  text: "High end noise-canceling headphones",
  top_k: 5
});

// Search with an image file
const fs = require('fs');
const imageResults = await client.multimodal.searchFile({
  collection_name: "products",
  file: fs.createReadStream('./query_image.jpg'),
  top_k: 5
});
```

## Production Configuration

When scaling to millions of users, network configurations are critical. By default, `VectorDBClient` uses the native `fetch` API.

In Node.js `v18+`, `fetch` supports HTTP Keep-Alive implicitly, but under extreme load, you may want to pass custom headers or integrate a connection pooling agent via a global dispatcher like `undici` if your framework allows overriding the global fetch dispatcher.

```typescript
const client = new VectorDBClient({
  baseUrl: 'https://api.my-vector-db.internal',
  headers: {
    'Connection': 'keep-alive', // Hint to keep connections open
    'X-API-Key': process.env.VECTOR_DB_API_KEY || ''
  },
  timeout: 10000 // Short fail-fast timeout in high-scale environments
});
```
