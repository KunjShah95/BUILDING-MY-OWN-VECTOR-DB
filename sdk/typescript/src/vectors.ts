import type { VectorDBClient } from "./client";
import type { RequestOptions } from "./http";
import { jsonPost, jsonGet, jsonDelete } from "./http";

export interface CreateVectorParams {
  vector: number[];
  metadata?: Record<string, unknown>;
  vectorId?: string;
}

export interface SearchVectorsParams {
  queryVector: number[];
  k?: number;
  method?: string;
  filters?: Record<string, unknown>;
}

export class VectorsAPI {
  constructor(private readonly client: VectorDBClient) {}

  async create(
    params: CreateVectorParams,
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { vector: params.vector };
    if (params.metadata) body.metadata = params.metadata;
    if (params.vectorId) body.vector_id = params.vectorId;
    return jsonPost(this.client.baseUrl, "/vectors", body, this.client.mergeOptions(options));
  }

  async get(vectorId: string, options?: RequestOptions): Promise<Record<string, unknown>> {
    return jsonGet(
      this.client.baseUrl,
      `/vectors/${encodeURIComponent(vectorId)}`,
      this.client.mergeOptions(options),
    );
  }

  async delete(vectorId: string, options?: RequestOptions): Promise<Record<string, unknown>> {
    return jsonDelete(
      this.client.baseUrl,
      `/vectors/${encodeURIComponent(vectorId)}`,
      this.client.mergeOptions(options),
    );
  }

  async search(
    params: SearchVectorsParams,
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      query_vector: params.queryVector,
      k: params.k ?? 5,
      method: params.method ?? "hnsw",
    };
    if (params.filters) body.filters = params.filters;
    return jsonPost(this.client.baseUrl, "/search", body, this.client.mergeOptions(options));
  }
}
