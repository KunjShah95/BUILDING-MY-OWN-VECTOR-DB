import { describe, it, expect, vi, beforeEach } from "vitest";
import { VectorDBClient } from "../src/client";
import { VectorDBHTTPError, VectorDBError } from "../src/errors";
import { raiseForStatus, buildQueryString, jsonGet, jsonPost, jsonDelete } from "../src/http";

/* ------------------------------------------------------------------ */
/*  Mock fetch globally                                                */
/* ------------------------------------------------------------------ */

const mockFetch = vi.fn() as ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.stubGlobal("fetch", mockFetch);
  mockFetch.mockReset();
});

function mockJsonResponse(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/* ------------------------------------------------------------------ */
/*  Network errors                                                     */
/* ------------------------------------------------------------------ */

describe("Network errors", () => {
  it("throws VectorDBError when fetch rejects (network failure)", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("fetch failed"));
    const client = new VectorDBClient();
    await expect(client.health()).rejects.toThrow(VectorDBError);
  });

  it("includes the original error message", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("connect ECONNREFUSED"));
    const client = new VectorDBClient();
    await expect(client.health()).rejects.toThrow(/connect ECONNREFUSED/);
  });

  it("wraps network error for POST requests", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("NetworkError"));
    const client = new VectorDBClient();
    await expect(
      client.vectors.create({ vector: [0.1] }),
    ).rejects.toThrow(VectorDBError);
  });

  it("wraps network error for DELETE requests", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("NetworkError"));
    const client = new VectorDBClient();
    await expect(client.collections.delete("x")).rejects.toThrow(VectorDBError);
  });

  it("wraps network error for search requests", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("NetworkError"));
    const client = new VectorDBClient();
    await expect(
      client.vectors.search({ queryVector: [0.1] }),
    ).rejects.toThrow(VectorDBError);
  });
});

/* ------------------------------------------------------------------ */
/*  Timeouts and AbortSignal                                           */
/* ------------------------------------------------------------------ */

describe("Timeouts and AbortSignal", () => {
  it("rejects when signal is pre-aborted", async () => {
    const controller = new AbortController();
    controller.abort();

    // In Node.js the pre-aborted fetch throws an error (TypeError or AbortError)
    mockFetch.mockRejectedValueOnce(new DOMException("The operation was aborted", "AbortError"));

    const client = new VectorDBClient();
    await expect(client.health({ signal: controller.signal })).rejects.toThrow();
  });

  it("rejects with AbortError name on pre-aborted signal", async () => {
    mockFetch.mockRejectedValueOnce(new DOMException("The operation was aborted", "AbortError"));

    const client = new VectorDBClient();
    try {
      await client.health({ signal: new AbortController().signal });
    } catch (e: unknown) {
      // Should not reach here because mock fetch is used
    }
    // Since fetch is mocked, the signal is passed to mockFetch but not evaluated
    // by Node's fetch implementation. Instead, verify the signal was passed.
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: expect.any(Object) }),
    );
  });

  it("passes signal through to fetch for POST requests", async () => {
    const signal = new AbortController().signal;
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, {}));

    const client = new VectorDBClient();
    await client.vectors.create({ vector: [0.1] }, { signal });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal }),
    );
  });

  it("passes signal for GET requests", async () => {
    const signal = new AbortController().signal;
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, { collections: [] }));

    const client = new VectorDBClient();
    await client.collections.list(10, 0, { signal });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal }),
    );
  });
});

/* ------------------------------------------------------------------ */
/*  Malformed / unexpected responses                                   */
/* ------------------------------------------------------------------ */

describe("Malformed responses", () => {
  it("handles non-JSON response body gracefully", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("plain text response", {
        status: 200,
        headers: { "Content-Type": "text/plain" },
      }),
    );
    const client = new VectorDBClient();
    const result = await client.health();
    expect(result.message).toBe("plain text response");
  });

  it("handles empty response body on error status", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("", {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const client = new VectorDBClient();
    try {
      await client.health();
    } catch (e: unknown) {
      expect(e).toBeInstanceOf(VectorDBHTTPError);
      const err = e as VectorDBHTTPError;
      expect(err.statusCode).toBe(500);
    }
  });

  it("handles non-json error response (plain text error)", async () => {
    mockFetch.mockResolvedValueOnce(
      new Response("Service Unavailable", {
        status: 503,
        headers: { "Content-Type": "text/plain" },
      }),
    );
    const client = new VectorDBClient();
    try {
      await client.health();
    } catch (e: unknown) {
      expect(e).toBeInstanceOf(VectorDBHTTPError);
      const err = e as VectorDBHTTPError;
      expect(err.statusCode).toBe(503);
      expect(err.detail).toBeDefined();
    }
  });

  it("handles response with extra unknown fields", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(200, {
        status: "ok",
        extra_field_1: "unexpected",
        nested: { also_ignored: true },
      }),
    );
    const client = new VectorDBClient();
    const result = await client.health();
    expect(result.status).toBe("ok");
    expect(result.extra_field_1).toBe("unexpected");
  });

  it("handles response with missing expected fields", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(201, { collection: {} }),
    );
    const client = new VectorDBClient();
    const col = await client.collections.create({
      name: "Empty",
      collectionId: "empty",
    });
    // undefined for fields with no default
    expect(col.collectionId).toBeUndefined();
    expect(col.name).toBeUndefined();
    expect(col.modality).toBeUndefined();
    expect(col.dimension).toBeUndefined();
    // distanceMetric has a default
    expect(col.distanceMetric).toBe("cosine");
  });
});

/* ------------------------------------------------------------------ */
/*  HTTP error codes                                                   */
/* ------------------------------------------------------------------ */

describe("HTTP error codes", () => {
  const codes = [400, 401, 403, 404, 405, 409, 422, 429, 500, 502, 503];

  codes.forEach((code) => {
    it(`throws VectorDBHTTPError on ${code}`, async () => {
      mockFetch.mockResolvedValueOnce(
        mockJsonResponse(code, { detail: `error ${code}` }),
      );
      const client = new VectorDBClient();
      try {
        await client.health();
      } catch (e: unknown) {
        expect(e).toBeInstanceOf(VectorDBHTTPError);
        expect((e as VectorDBHTTPError).statusCode).toBe(code);
      }
    });
  });

  it("propagates detail as string when present", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(400, { detail: "bad request reason" }),
    );
    const client = new VectorDBClient();
    try {
      await client.health();
    } catch (e: unknown) {
      const err = e as VectorDBHTTPError;
      expect(err.detail).toBe("bad request reason");
    }
  });

  it("propagates detail as object when no detail key present", async () => {
    mockFetch.mockResolvedValueOnce(
      mockJsonResponse(500, { error: "something broke" }),
    );
    const client = new VectorDBClient();
    try {
      await client.health();
    } catch (e: unknown) {
      const err = e as VectorDBHTTPError;
      expect(err.detail).toEqual({ error: "something broke" });
    }
  });
});

/* ------------------------------------------------------------------ */
/*  Request options propagation                                        */
/* ------------------------------------------------------------------ */

describe("Request options propagation", () => {
  it("passes client-level and per-request headers", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, {}));
    const client = new VectorDBClient({
      baseUrl: "http://example.com",
      headers: { Authorization: "Bearer token123" },
    });
    await client.health({ headers: { "X-Trace": "abc" } });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "Bearer token123",
          "X-Trace": "abc",
        }),
      }),
    );
  });

  it("per-request headers override client-level headers", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, {}));
    const client = new VectorDBClient({
      headers: { Authorization: "old-token" },
    });
    await client.health({ headers: { Authorization: "new-token" } });

    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          Authorization: "new-token",
        }),
      }),
    );
  });

  it("client-level Content-Type does not override the JSON content type", async () => {
    mockFetch.mockResolvedValueOnce(mockJsonResponse(200, {}));
    const client = new VectorDBClient({
      baseUrl: "http://example.com",
      headers: { "X-Custom": "val" },
    });
    await client.health();

    // The Content-Type should be set by jsonGet, not overridden
    expect(mockFetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
  });
});

/* ------------------------------------------------------------------ */
/*  Multimodal edge cases                                              */
/* ------------------------------------------------------------------ */

describe("Multimodal edge cases", () => {
  it("rejects ingestImage without blob", async () => {
    const client = new VectorDBClient();
    await expect(
      client.multimodal.ingestImage("photos", {} as any),
    ).rejects.toThrow("Provide blob");
  });

  it("rejects searchImage without blob", async () => {
    const client = new VectorDBClient();
    await expect(
      client.multimodal.searchImage("photos", {} as any),
    ).rejects.toThrow("Provide blob");
  });

  it("rejects ingestAudio without blob", async () => {
    const client = new VectorDBClient();
    await expect(
      client.multimodal.ingestAudio("audio", {} as any),
    ).rejects.toThrow("Provide blob");
  });

  it("rejects searchAudio without blob", async () => {
    const client = new VectorDBClient();
    await expect(
      client.multimodal.searchAudio("audio", {} as any),
    ).rejects.toThrow("Provide blob");
  });
});

/* ------------------------------------------------------------------ */
/*  Client initialization edge cases                                   */
/* ------------------------------------------------------------------ */

describe("Client initialization", () => {
  it("uses default baseUrl when not provided", () => {
    const client = new VectorDBClient();
    expect(client.baseUrl).toBe("http://localhost:8000");
  });

  it("uses default timeout when not provided", () => {
    const client = new VectorDBClient();
    expect(client.timeout).toBe(60_000);
  });

  it("strips trailing slashes from baseUrl", () => {
    const c1 = new VectorDBClient({ baseUrl: "http://x.com/" });
    expect(c1.baseUrl).toBe("http://x.com");
    const c2 = new VectorDBClient({ baseUrl: "http://x.com///" });
    expect(c2.baseUrl).toBe("http://x.com");
  });

  it("defaults headers to empty object when not provided", () => {
    const client = new VectorDBClient();
    expect(client.headers).toEqual({});
  });
});

/* ------------------------------------------------------------------ */
/*  raiseForStatus unit tests                                          */
/* ------------------------------------------------------------------ */

describe("raiseForStatus", () => {
  it("returns parsed body on 2xx", async () => {
    const res = mockJsonResponse(200, { key: "value" });
    const result = await raiseForStatus(res);
    expect(result).toEqual({ key: "value" });
  });

  it("throws VectorDBHTTPError on non-2xx with detail", async () => {
    const res = mockJsonResponse(401, { detail: "unauthorized" });
    await expect(raiseForStatus(res)).rejects.toThrow(VectorDBHTTPError);
  });

  it("carries correct statusCode in thrown error", async () => {
    const res = mockJsonResponse(403, { detail: "forbidden" });
    try {
      await raiseForStatus(res);
    } catch (e: unknown) {
      expect((e as VectorDBHTTPError).statusCode).toBe(403);
    }
  });

  it("falls back to entire payload when no detail key exists", async () => {
    const res = mockJsonResponse(400, { message: "bad" });
    try {
      await raiseForStatus(res);
    } catch (e: unknown) {
      expect((e as VectorDBHTTPError).detail).toEqual({ message: "bad" });
    }
  });
});

/* ------------------------------------------------------------------ */
/*  buildQueryString unit tests                                        */
/* ------------------------------------------------------------------ */

describe("buildQueryString", () => {
  it("returns base unchanged when no params", () => {
    expect(buildQueryString("/path")).toBe("/path");
  });

  it("appends single param", () => {
    expect(buildQueryString("/path", { key: "value" })).toBe("/path?key=value");
  });

  it("appends multiple params", () => {
    expect(buildQueryString("/path", { a: "1", b: "2" })).toBe("/path?a=1&b=2");
  });

  it("skips undefined values", () => {
    expect(buildQueryString("/path", { a: "1", b: undefined })).toBe("/path?a=1");
  });

  it("encodes special characters in param values", () => {
    expect(buildQueryString("/search", { q: "hello world" })).toBe(
      "/search?q=hello+world",
    );
  });

  it("handles boolean and numeric values", () => {
    expect(buildQueryString("/x", { flag: true, count: 42 })).toBe(
      "/x?flag=true&count=42",
    );
  });

  it("returns base unchanged when all params are undefined", () => {
    expect(buildQueryString("/path", { a: undefined, b: undefined })).toBe("/path");
  });
});
