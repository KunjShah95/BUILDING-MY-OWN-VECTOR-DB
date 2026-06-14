import { describe, it, expect, vi, beforeEach } from "vitest";
import { VectorDBClient } from "../src/client";
import { VectorDBHTTPError } from "../src/errors";

/* ------------------------------------------------------------------ */
/*  Mock fetch globally                                                */
/* ------------------------------------------------------------------ */

const mockFetch = vi.fn() as ReturnType<typeof vi.fn>;
vi.stubGlobal("fetch", mockFetch);

function mockJsonResponse(status: number, body: Record<string, unknown>) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  mockFetch.mockReset();
});

/* ------------------------------------------------------------------ */
/*  Client                                                             */
/* ------------------------------------------------------------------ */

describe("VectorDBClient (unit)", () => {
  it("strips trailing slash from baseUrl", () => {
    const c = new VectorDBClient({ baseUrl: "http://localhost:8000//" });
    expect(c.baseUrl).toBe("http://localhost:8000");
  });

  it("health check calls GET /health", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { status: "ok" }));
    const c = new VectorDBClient();
    const result = await c.health();
    expect(result).toEqual({ status: "ok" });
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/health",
      expect.objectContaining({ method: "GET" }),
    );
  });
});

/* ------------------------------------------------------------------ */
/*  Collections                                                       */
/* ------------------------------------------------------------------ */

describe("CollectionsAPI", () => {
  it("create sends POST /collections", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(201, {
        collection: {
          collection_id: "demo",
          name: "Demo",
          modality: "text",
          dimension: 384,
          embedding_model: "default",
        },
      }),
    );

    const c = new VectorDBClient();
    const col = await c.collections.create({ name: "Demo", collectionId: "demo" });

    expect(col.collectionId).toBe("demo");
    expect(col.name).toBe("Demo");

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/collections",
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining('"name":"Demo"'),
      }),
    );
  });

  it("list calls GET /collections", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { collections: [] }));
    const c = new VectorDBClient();
    const list = await c.collections.list();
    expect(list).toEqual([]);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/collections?limit=100&offset=0",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("get calls GET /collections/{id}", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(200, {
        collection: {
          collection_id: "x",
          name: "X",
          modality: "text",
          dimension: 128,
          embedding_model: "m",
        },
      }),
    );
    const c = new VectorDBClient();
    const col = await c.collections.get("x");
    expect(col.collectionId).toBe("x");
  });

  it("delete calls DELETE /collections/{id}", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { deleted: true }));
    const c = new VectorDBClient();
    const res = await c.collections.delete("x");
    expect(res).toEqual({ deleted: true });
  });
});

/* ------------------------------------------------------------------ */
/*  Vectors                                                           */
/* ------------------------------------------------------------------ */

describe("VectorsAPI", () => {
  it("create sends POST /vectors", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(201, { vector_id: "v1" }),
    );
    const c = new VectorDBClient();
    const res = await c.vectors.create({ vector: [0.1, 0.2], vectorId: "v1" });
    expect(res).toEqual({ vector_id: "v1" });
  });

  it("search sends POST /search", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(200, { results: [], success: true }),
    );
    const c = new VectorDBClient();
    const res = await c.vectors.search({ queryVector: [0.1, 0.2], k: 3 });
    expect(res.success).toBe(true);
  });
});

/* ------------------------------------------------------------------ */
/*  Multimodal                                                        */
/* ------------------------------------------------------------------ */

describe("MultimodalAPI", () => {
  it("ingestText sends POST /collections/{id}/ingest/text", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { vector_id: "v1" }));
    const c = new VectorDBClient();
    const res = await c.multimodal.ingestText("docs", {
      text: "hello",
      vectorId: "v1",
    });
    expect(res).toEqual({ vector_id: "v1" });
    const callUrl = mockFetch.mock.calls[0][0] as string;
    expect(callUrl).toContain("/collections/docs/ingest/text");
  });

  it("searchText sends POST /collections/{id}/search/text", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(200, {
        success: true,
        results: [{ vector_id: "v1", distance: 0.1 }],
      }),
    );
    const c = new VectorDBClient();
    const result = await c.multimodal.searchText("docs", {
      query: "hello",
      k: 5,
    });
    expect(result.success).toBe(true);
    expect(result.results[0].vectorId).toBe("v1");
  });

  it("ingestImage rejects without blob", async () => {
    const c = new VectorDBClient();
    await expect(
      c.multimodal.ingestImage("photos", {} as any),
    ).rejects.toThrow("Provide blob");
  });

  it("query sends POST /collections/{id}/query", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(200, { answer: "some answer" }),
    );
    const c = new VectorDBClient();
    const res = await c.multimodal.query("docs", "what is?");
    expect(res).toEqual({ answer: "some answer" });
  });
});

/* ------------------------------------------------------------------ */
/*  Errors                                                            */
/* ------------------------------------------------------------------ */

describe("VectorDBHTTPError", () => {
  it("is thrown on non-2xx with status in message", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(401, { detail: "bad key" }),
    );
    const c = new VectorDBClient();
    await expect(c.health()).rejects.toThrow(VectorDBHTTPError);
  });

  it("carries statusCode and detail string", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(403, { detail: "forbidden" }),
    );
    const c = new VectorDBClient();
    try {
      await c.health();
    } catch (e: unknown) {
      expect(e).toBeInstanceOf(VectorDBHTTPError);
      const err = e as VectorDBHTTPError;
      expect(err.statusCode).toBe(403);
      // raiseForStatus extracts payload.detail, which is the string "forbidden"
      expect(err.detail).toBe("forbidden");
    }
  });
});
