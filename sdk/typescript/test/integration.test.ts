import { describe, it, expect, beforeAll, afterAll, beforeEach } from "vitest";
import http from "node:http";
import { AddressInfo } from "node:net";
import { VectorDBClient } from "../src/client";
import { VectorDBHTTPError } from "../src/errors";

/* ------------------------------------------------------------------ */
/*  Local test server                                                  */
/* ------------------------------------------------------------------ */

let server: http.Server;
let port: number;
let baseUrl: string;

/** Accumulated request log for assertions. */
const requests: Array<{
  method: string;
  path: string;
  headers: Record<string, string>;
  body: string;
}> = [];

beforeEach(() => {
  requests.length = 0;
});

/** Helper to parse the body of incoming requests. */
function collectBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (chunk: Buffer) => (data += chunk.toString()));
    req.on("end", () => resolve(data));
  });
}

beforeAll(async () => {
  server = http.createServer(async (req, res) => {
    const body = await collectBody(req);

    // Determine the pathname (strip query params, decode %-encoded chars)
    const rawPath = (req.url ?? "/").split("?")[0];
    const pathname = decodeURIComponent(rawPath);

    // Log every request
    requests.push({
      method: req.method ?? "",
      path: req.url ?? "",
      headers: req.headers as Record<string, string>,
      body,
    });

    // Helper to respond with JSON
    const json = (status: number, data: unknown) => {
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(data));
    };

    // ── Routes ──────────────────────────────────────────────────

    if (req.method === "GET" && pathname === "/health") {
      return json(200, { status: "ok" });
    }

    if (req.method === "POST" && pathname === "/collections") {
      return json(201, {
        collection: {
          collection_id: "demo",
          name: "Demo",
          modality: "text",
          dimension: 384,
          embedding_model: "default",
          distance_metric: "cosine",
        },
      });
    }

    if (req.method === "GET" && pathname === "/collections") {
      return json(200, {
        collections: [
          {
            collection_id: "a",
            name: "A",
            modality: "text",
            dimension: 128,
            embedding_model: "m",
          },
        ],
      });
    }

    if (
      req.method === "GET" &&
      pathname === "/collections/integration-test-col"
    ) {
      return json(200, {
        collection: {
          collection_id: "integration-test-col",
          name: "Integration Test",
          modality: "image",
          dimension: 512,
          embedding_model: "clip",
          description: "A collection for integration testing",
        },
      });
    }

    if (
      req.method === "POST" &&
      pathname === "/collections/integration-test-col/index"
    ) {
      return json(200, { status: "building", method: "hnsw" });
    }

    if (
      req.method === "GET" &&
      pathname === "/collections/integration-test-col/index/stats"
    ) {
      return json(200, {
        vector_count: 42,
        index_type: "hnsw",
        status: "ready",
      });
    }

    if (req.method === "DELETE" && pathname === "/collections/to-delete") {
      return json(200, { deleted: true });
    }

    if (req.method === "POST" && pathname === "/vectors") {
      return json(201, { vector_id: "vec_abc" });
    }

    if (req.method === "GET" && pathname === "/vectors/vec_abc") {
      return json(200, { vector_id: "vec_abc", vector: [0.1, 0.2, 0.3] });
    }

    if (req.method === "DELETE" && pathname === "/vectors/vec_abc") {
      return json(200, { deleted: true });
    }

    if (req.method === "POST" && pathname === "/search") {
      return json(200, {
        success: true,
        results: [
          { vector_id: "v1", distance: 0.1, metadata: { label: "hit" } },
          { vector_id: "v2", distance: 0.2 },
        ],
        total_results: 2,
        search_time: 1.5,
        method: "hnsw",
      });
    }

    // Multimodal text
    if (
      req.method === "POST" &&
      pathname === "/collections/mcol/ingest/text"
    ) {
      return json(200, { vector_id: "v_text" });
    }

    if (
      req.method === "POST" &&
      pathname === "/collections/mcol/search/text"
    ) {
      return json(200, {
        success: true,
        results: [{ vector_id: "v1", distance: 0.05 }],
      });
    }

    // Multimodal image — multipart
    if (
      req.method === "POST" &&
      pathname === "/collections/mcol/ingest/image"
    ) {
      const ct = req.headers["content-type"] ?? "";
      if (!ct.includes("multipart/form-data")) {
        return json(400, { detail: "Expected multipart/form-data" });
      }
      return json(200, { vector_id: "v_img" });
    }

    if (
      req.method === "POST" &&
      pathname === "/collections/mcol/search/image"
    ) {
      const ct = req.headers["content-type"] ?? "";
      if (!ct.includes("multipart/form-data")) {
        return json(400, { detail: "Expected multipart/form-data" });
      }
      return json(200, {
        success: true,
        results: [{ vector_id: "v1", distance: 0.15 }],
      });
    }

    // Audio
    if (
      req.method === "POST" &&
      pathname === "/collections/mcol/ingest/audio"
    ) {
      return json(200, { vector_id: "v_audio" });
    }

    if (
      req.method === "POST" &&
      pathname === "/collections/mcol/search/audio"
    ) {
      return json(200, {
        success: true,
        results: [{ vector_id: "v1", distance: 0.2 }],
      });
    }

    // RAG query
    if (req.method === "POST" && pathname === "/collections/mcol/query") {
      return json(200, { answer: "RAG answer here" });
    }

    // ── Error routes ────────────────────────────────────────────
    // Error routes are accessed directly through the client by hitting
    // /error/{code} which returns the corresponding error status.
    if (pathname === "/error/401") {
      return json(401, { detail: "Invalid API key" });
    }
    if (pathname === "/error/403") {
      return json(403, { detail: "Forbidden" });
    }
    if (pathname === "/error/404") {
      return json(404, { detail: "Not found" });
    }
    if (pathname === "/error/500") {
      return json(500, { error: "Internal server error" });
    }

    // ── URL encoding test ───────────────────────────────────────
    if (req.method === "GET" && pathname === "/collections/col lection id") {
      return json(200, {
        collection: {
          collection_id: "col lection id",
          name: "Spaced",
          modality: "text",
          dimension: 64,
          embedding_model: "m",
        },
      });
    }

    // Fallback 404
    return json(404, { detail: `No route: ${req.method} ${req.url}` });
  });

  await new Promise<void>((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      port = (server.address() as AddressInfo).port;
      baseUrl = `http://127.0.0.1:${port}`;
      resolve();
    });
  });
});

afterAll(() => {
  server?.close();
});

/* ------------------------------------------------------------------ */
/*  Integration Tests                                                  */
/* ------------------------------------------------------------------ */

describe("Integration — health", () => {
  it("returns status ok", async () => {
    const client = new VectorDBClient({ baseUrl });
    const result = await client.health();
    expect(result).toEqual({ status: "ok" });
    expect(requests[0].method).toBe("GET");
    expect(requests[0].path).toBe("/health");
  });
});

describe("Integration — collections", () => {
  it("creates a collection", async () => {
    const client = new VectorDBClient({ baseUrl });
    const col = await client.collections.create({
      name: "Demo",
      collectionId: "demo",
      modality: "text",
      dimension: 384,
    });
    expect(col.collectionId).toBe("demo");
    expect(col.name).toBe("Demo");
    expect(col.modality).toBe("text");
    expect(col.dimension).toBe(384);
    expect(col.embeddingModel).toBe("default");
    expect(col.distanceMetric).toBe("cosine");
  });

  it("lists collections", async () => {
    const client = new VectorDBClient({ baseUrl });
    const list = await client.collections.list();
    expect(list).toHaveLength(1);
    expect(list[0].collectionId).toBe("a");
  });

  it("gets a collection by id", async () => {
    const client = new VectorDBClient({ baseUrl });
    const col = await client.collections.get("integration-test-col");
    expect(col.collectionId).toBe("integration-test-col");
    expect(col.description).toBe("A collection for integration testing");
  });

  it("deletes a collection", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.collections.delete("to-delete");
    expect(res).toEqual({ deleted: true });
  });

  it("builds an index", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.collections.buildIndex("integration-test-col", {
      method: "hnsw",
      m: 32,
    });
    expect(res.status).toBe("building");
  });

  it("gets index stats", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.collections.indexStats("integration-test-col");
    expect(res.vector_count).toBe(42);
    expect(res.status).toBe("ready");
  });
});

describe("Integration — vectors", () => {
  it("creates a vector", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.vectors.create({
      vector: [0.1, 0.2, 0.3],
      vectorId: "vec_abc",
      metadata: { tag: "test" },
    });
    expect(res).toEqual({ vector_id: "vec_abc" });
  });

  it("gets a vector", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.vectors.get("vec_abc");
    expect(res.vector_id).toBe("vec_abc");
  });

  it("deletes a vector", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.vectors.delete("vec_abc");
    expect(res).toEqual({ deleted: true });
  });

  it("searches vectors", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.vectors.search({
      queryVector: [0.1, 0.2],
      k: 3,
      method: "hnsw",
    });
    expect(res.success).toBe(true);
    expect((res.results as any[])).toHaveLength(2);
  });
});

describe("Integration — multimodal", () => {
  it("ingests text", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.multimodal.ingestText("mcol", {
      text: "hello world",
      vectorId: "v_text",
    });
    expect(res).toEqual({ vector_id: "v_text" });
  });

  it("searches text and returns typed SearchResult", async () => {
    const client = new VectorDBClient({ baseUrl });
    const result = await client.multimodal.searchText("mcol", {
      query: "hello",
      k: 5,
    });
    expect(result.success).toBe(true);
    expect(result.results).toHaveLength(1);
    expect(result.results[0].vectorId).toBe("v1");
    expect(result.results[0].distance).toBeCloseTo(0.05);
  });

  it("ingests image via multipart", async () => {
    const client = new VectorDBClient({ baseUrl });
    const blob = new Blob(["fake-image-bytes"], { type: "image/jpeg" });
    const res = await client.multimodal.ingestImage("mcol", { blob });
    expect(res).toEqual({ vector_id: "v_img" });
  });

  it("searches image via multipart and returns typed SearchResult", async () => {
    const client = new VectorDBClient({ baseUrl });
    const blob = new Blob(["query-image"], { type: "image/jpeg" });
    const result = await client.multimodal.searchImage("mcol", {
      blob,
      k: 3,
    });
    expect(result.success).toBe(true);
    expect(result.results[0].vectorId).toBe("v1");
    expect(result.results[0].distance).toBeCloseTo(0.15);
  });

  it("ingests audio via multipart", async () => {
    const client = new VectorDBClient({ baseUrl });
    const blob = new Blob(["fake-audio"], { type: "audio/wav" });
    const res = await client.multimodal.ingestAudio("mcol", { blob });
    expect(res).toEqual({ vector_id: "v_audio" });
  });

  it("searches audio via multipart and returns typed SearchResult", async () => {
    const client = new VectorDBClient({ baseUrl });
    const blob = new Blob(["query-audio"], { type: "audio/wav" });
    const result = await client.multimodal.searchAudio("mcol", {
      blob,
      k: 5,
      method: "brute",
    });
    expect(result.success).toBe(true);
    expect(result.results[0].distance).toBeCloseTo(0.2);
  });

  it("sends a RAG query", async () => {
    const client = new VectorDBClient({ baseUrl });
    const res = await client.multimodal.query("mcol", "What is?");
    expect(res).toEqual({ answer: "RAG answer here" });
  });
});

describe("Integration — typed models", () => {
  it("SearchResult has correct types from searchText", async () => {
    const client = new VectorDBClient({ baseUrl });
    const result = await client.multimodal.searchText("mcol", {
      query: "test",
      k: 10,
    });
    expect(result.success).toBe(true);
    expect(result.totalResults).toBe(1);
    expect(typeof result.searchTime).toBe("number");
    expect(typeof result.method).toBe("string");
    expect(result.raw).toBeDefined();
  });

  it("SearchHit carries metadata when present", async () => {
    const client = new VectorDBClient({ baseUrl });
    const result = await client.vectors.search({
      queryVector: [0.1, 0.2],
    });
    const hits = result.results as any[];
    expect(hits[0].metadata).toEqual({ label: "hit" });
  });
});

describe("Integration — HTTP errors throw VectorDBHTTPError", () => {
  it("401", async () => {
    // Use a client pointed at the error sub-path to trigger a 401
    const client = new VectorDBClient({ baseUrl: `${baseUrl}` });
    // The health method calls GET /health, but we can test /error/401
    // by directly using the collection get which uses path building
    const { jsonGet } = await import("../src/http");
    await expect(jsonGet(baseUrl, "/error/401")).rejects.toThrow(VectorDBHTTPError);
  });

  it("403 returns VectorDBHTTPError with correct status", async () => {
    const { jsonGet } = await import("../src/http");
    try {
      await jsonGet(baseUrl, "/error/403");
    } catch (e: unknown) {
      expect(e).toBeInstanceOf(VectorDBHTTPError);
      expect((e as VectorDBHTTPError).statusCode).toBe(403);
    }
  });

  it("404", async () => {
    const { jsonGet } = await import("../src/http");
    await expect(jsonGet(baseUrl, "/error/404")).rejects.toThrow("HTTP 404");
  });

  it("500", async () => {
    const { jsonGet } = await import("../src/http");
    await expect(jsonGet(baseUrl, "/error/500")).rejects.toThrow("HTTP 500");
  });

  it("carries statusCode and detail on 401", async () => {
    const { jsonGet } = await import("../src/http");
    try {
      await jsonGet(baseUrl, "/error/401");
    } catch (e: unknown) {
      expect(e).toBeInstanceOf(VectorDBHTTPError);
      const err = e as VectorDBHTTPError;
      expect(err.statusCode).toBe(401);
      expect(err.detail).toBe("Invalid API key");
    }
  });

  it("carries statusCode and detail on 500 with fallback payload", async () => {
    const { jsonGet } = await import("../src/http");
    try {
      await jsonGet(baseUrl, "/error/500");
    } catch (e: unknown) {
      expect(e).toBeInstanceOf(VectorDBHTTPError);
      const err = e as VectorDBHTTPError;
      expect(err.statusCode).toBe(500);
      // When there's no 'detail' key, the entire payload becomes the detail
      expect(err.detail).toEqual({ error: "Internal server error" });
    }
  });
});

describe("Integration — URL encoding", () => {
  it("encodes collection IDs with special characters", async () => {
    const client = new VectorDBClient({ baseUrl });
    const col = await client.collections.get("col lection id");
    expect(col.collectionId).toBe("col lection id");
  });
});

describe("Integration — custom headers", () => {
  it("passes client-level and per-request headers", async () => {
    const client = new VectorDBClient({
      baseUrl,
      headers: { "X-Custom": "client-level" },
    });
    await client.health({ headers: { "X-Request": "per-request" } });
    expect(requests[0].headers["x-custom"]).toBe("client-level");
    expect(requests[0].headers["x-request"]).toBe("per-request");
  });
});
