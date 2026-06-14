/**
 * Low-level HTTP helpers used by the API resource classes.
 */

import { VectorDBHTTPError } from "./errors";

export interface RequestOptions {
  headers?: Record<string, string>;
  params?: Record<string, string | number | boolean | undefined>;
  signal?: AbortSignal;
}

/**
 * Parse the response body and raise VectorDBHTTPError on non-2xx status.
 */
export async function raiseForStatus(res: Response): Promise<Record<string, unknown>> {
  // Read the body as text first so we can try JSON parsing without
  // consuming the stream (which would make res.text() return "").
  const text = await res.text();
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(text) as Record<string, unknown>;
  } catch {
    payload = { message: text };
  }

  if (!res.ok) {
    throw new VectorDBHTTPError(res.status, payload.detail ?? payload);
  }
  return payload;
}

/**
 * Build a URL query string from an options object, skipping undefined values.
 */
export function buildQueryString(
  base: string,
  params?: Record<string, string | number | boolean | undefined>,
): string {
  if (!params) return base;
  const entries = Object.entries(params).filter(
    ([, v]) => v !== undefined,
  ) as [string, string][];
  if (entries.length === 0) return base;
  const qs = new URLSearchParams(entries).toString();
  return `${base}?${qs}`;
}

import { VectorDBError } from "./errors";

/**
 * Wrap a fetch call so that network-level rejections are re-thrown as
 * VectorDBError (which is a subclass of Error).
 */
async function wrapFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (err: unknown) {
    if (err instanceof VectorDBError) throw err;
    const message = err instanceof Error ? err.message : String(err);
    throw new VectorDBError(`Network error: ${message}`);
  }
}

/**
 * Perform a JSON POST request and parse the response.
 */
export async function jsonPost(
  baseUrl: string,
  path: string,
  body?: Record<string, unknown>,
  options?: RequestOptions,
): Promise<Record<string, unknown>> {
  const url = `${baseUrl}${path}`;
  const res = await wrapFetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal: options?.signal,
  });
  return raiseForStatus(res);
}

/**
 * Perform a multipart/form-data POST request.
 */
export async function multipartPost(
  baseUrl: string,
  path: string,
  formData: FormData,
  options?: RequestOptions,
): Promise<Record<string, unknown>> {
  const url = `${baseUrl}${path}`;
  const res = await wrapFetch(url, {
    method: "POST",
    // Let fetch set Content-Type with boundary
    headers: options?.headers,
    body: formData,
    signal: options?.signal,
  });
  return raiseForStatus(res);
}

/**
 * Perform a GET request.
 */
export async function jsonGet(
  baseUrl: string,
  path: string,
  options?: RequestOptions,
): Promise<Record<string, unknown>> {
  const url = buildQueryString(`${baseUrl}${path}`, options?.params);
  const res = await wrapFetch(url, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    signal: options?.signal,
  });
  return raiseForStatus(res);
}

/**
 * Perform a DELETE request.
 */
export async function jsonDelete(
  baseUrl: string,
  path: string,
  options?: RequestOptions,
): Promise<Record<string, unknown>> {
  const url = `${baseUrl}${path}`;
  const res = await wrapFetch(url, {
    method: "DELETE",
    headers: {
      ...options?.headers,
    },
    signal: options?.signal,
  });
  return raiseForStatus(res);
}
