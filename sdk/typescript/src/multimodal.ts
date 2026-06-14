import type { VectorDBClient } from "./client";
import type { RequestOptions } from "./http";
import { jsonPost, multipartPost } from "./http";
import { SearchResult, type SearchResultData } from "./models";

export interface IngestTextParams {
  text: string;
  metadata?: Record<string, unknown>;
  vectorId?: string;
}

export interface IngestFileParams {
  /** Path (Node.js only — Blob/File in browser) */
  blob?: Blob | File;
  /** Optional metadata encoded as JSON string in the form */
  metadata?: Record<string, unknown>;
  vectorId?: string;
  filename?: string;
}

export interface SearchTextParams {
  query: string;
  k?: number;
  method?: string;
  filters?: Record<string, unknown>;
}

export interface SearchFileParams {
  blob?: Blob | File;
  k?: number;
  method?: string;
  filters?: Record<string, unknown>;
}

export class MultimodalAPI {
  constructor(private readonly client: VectorDBClient) {}

  // ── Text ────────────────────────────────────────────────────────────

  async ingestText(
    collectionId: string,
    params: IngestTextParams,
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { text: params.text };
    if (params.metadata) body.metadata = params.metadata;
    if (params.vectorId) body.vector_id = params.vectorId;
    return jsonPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/ingest/text`,
      body,
      this.client.mergeOptions(options),
    );
  }

  async searchText(
    collectionId: string,
    params: SearchTextParams,
    options?: RequestOptions,
  ): Promise<SearchResult> {
    const body: Record<string, unknown> = {
      query: params.query,
      k: params.k ?? 5,
      method: params.method ?? "brute",
    };
    if (params.filters) body.filters = params.filters;
    const data = await jsonPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/search/text`,
      body,
      this.client.mergeOptions(options),
    );
    return SearchResult.fromApi(data as unknown as SearchResultData);
  }

  // ── Image ───────────────────────────────────────────────────────────

  private makeFileForm(
    blob: Blob | File,
    extra?: { metadata?: Record<string, unknown>; vectorId?: string; k?: number; method?: string; filters?: Record<string, unknown> },
  ): FormData {
    const fd = new FormData();
    fd.append("file", blob);
    if (extra?.metadata) fd.append("metadata", JSON.stringify(extra.metadata));
    if (extra?.vectorId) fd.append("vector_id", extra.vectorId);
    if (extra?.k !== undefined) fd.append("k", String(extra.k));
    if (extra?.method) fd.append("method", extra.method);
    if (extra?.filters) fd.append("filters", JSON.stringify(extra.filters));
    return fd;
  }

  async ingestImage(
    collectionId: string,
    params: IngestFileParams,
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    if (!params.blob) throw new Error("Provide blob for image ingest");
    const fd = this.makeFileForm(params.blob, {
      metadata: params.metadata,
      vectorId: params.vectorId,
    });
    return multipartPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/ingest/image`,
      fd,
      this.client.mergeOptions(options),
    );
  }

  async searchImage(
    collectionId: string,
    params: SearchFileParams,
    options?: RequestOptions,
  ): Promise<SearchResult> {
    if (!params.blob) throw new Error("Provide blob for image search");
    const fd = this.makeFileForm(params.blob, {
      k: params.k ?? 5,
      method: params.method ?? "brute",
      filters: params.filters,
    });
    const data = await multipartPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/search/image`,
      fd,
      this.client.mergeOptions(options),
    );
    return SearchResult.fromApi(data as unknown as SearchResultData);
  }

  // ── Audio ───────────────────────────────────────────────────────────

  async ingestAudio(
    collectionId: string,
    params: IngestFileParams,
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    if (!params.blob) throw new Error("Provide blob for audio ingest");
    const fd = this.makeFileForm(params.blob, {
      metadata: params.metadata,
      vectorId: params.vectorId,
    });
    return multipartPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/ingest/audio`,
      fd,
      this.client.mergeOptions(options),
    );
  }

  async searchAudio(
    collectionId: string,
    params: SearchFileParams,
    options?: RequestOptions,
  ): Promise<SearchResult> {
    if (!params.blob) throw new Error("Provide blob for audio search");
    const fd = this.makeFileForm(params.blob, {
      k: params.k ?? 5,
      method: params.method ?? "brute",
      filters: params.filters,
    });
    const data = await multipartPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/search/audio`,
      fd,
      this.client.mergeOptions(options),
    );
    return SearchResult.fromApi(data as unknown as SearchResultData);
  }

  // ── RAG ─────────────────────────────────────────────────────────────

  async query(
    collectionId: string,
    query: string,
    params?: { k?: number },
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { query };
    if (params?.k) body.k = params.k;
    return jsonPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/query`,
      body,
      this.client.mergeOptions(options),
    );
  }
}
