import { describe, it, expect, vi, beforeEach } from "vitest";
import { VectorDBClient } from "../src/client";

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

describe("AnnAPI", () => {
  it("createIndex sends POST /api/v1/ann/index", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(200, { success: true, message: "Created hnsw index" }),
    );

    const c = new VectorDBClient();
    const res = await c.ann.createIndex({ indexType: "hnsw", m: 16, efConstruction: 200 });

    expect(res.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/ann/index",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ index_type: "hnsw", m: 16, ef_construction: 200 }),
      }),
    );
  });

  it("getIndexInfo sends GET /api/v1/ann/index", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { success: true }));

    const c = new VectorDBClient();
    const res = await c.ann.getIndexInfo("hnsw");

    expect(res.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/ann/index?index_type=hnsw",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("saveIndex sends POST /api/v1/ann/index/save", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { success: true }));

    const c = new VectorDBClient();
    const res = await c.ann.saveIndex("hnsw");

    expect(res.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/ann/index/save?index_type=hnsw",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("loadIndex sends POST /api/v1/ann/index/load", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { success: true }));

    const c = new VectorDBClient();
    const res = await c.ann.loadIndex("hnsw");

    expect(res.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/ann/index/load?index_type=hnsw",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("compareSearch sends GET /api/v1/ann/search/compare", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { success: true }));

    const c = new VectorDBClient();
    const res = await c.ann.compareSearch({ queryVector: [0.1, 0.2], k: 5 });

    expect(res.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/ann/search/compare?query_vector=0.1%2C0.2&k=5",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("getStatistics sends GET /api/v1/ann/stats", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { success: true }));

    const c = new VectorDBClient();
    const res = await c.ann.getStatistics();

    expect(res.success).toBe(true);
    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/ann/stats",
      expect.objectContaining({ method: "GET" }),
    );
  });
});
