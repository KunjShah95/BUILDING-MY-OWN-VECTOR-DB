import type { VectorDBClient } from "./client";
import type { RequestOptions } from "./http";
import { jsonPost, jsonGet, buildQueryString } from "./http";

export interface CreateIndexParams {
  indexType: "hnsw" | "ivf" | "brute";
  metric?: string;
  m?: number;
  m0?: number;
  efConstruction?: number;
  nClusters?: number;
  nProbes?: number;
}

export class AnnAPI {
  constructor(private readonly client: VectorDBClient) {}

  /**
   * Build and populate an index from all vectors currently stored in the DB.
   */
  async createIndex(
    params: CreateIndexParams,
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      index_type: params.indexType,
    };
    if (params.metric !== undefined) body.metric = params.metric;
    if (params.m !== undefined) body.m = params.m;
    if (params.m0 !== undefined) body.m0 = params.m0;
    if (params.efConstruction !== undefined) body.ef_construction = params.efConstruction;
    if (params.nClusters !== undefined) body.n_clusters = params.nClusters;
    if (params.nProbes !== undefined) body.n_probes = params.nProbes;

    return jsonPost(
      this.client.baseUrl,
      "/api/v1/ann/index",
      body,
      this.client.mergeOptions(options),
    );
  }

  /**
   * Return stats for all loaded indexes, or just one if indexType is given.
   */
  async getIndexInfo(
    indexType?: "hnsw" | "ivf" | "brute",
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const merged = this.client.mergeOptions(options);
    const params: Record<string, string | number | boolean | undefined> = { ...merged?.params };
    if (indexType) {
      params.index_type = indexType;
    }
    return jsonGet(
      this.client.baseUrl,
      "/api/v1/ann/index",
      {
        ...merged,
        params,
      },
    );
  }

  /**
   * Persist an in-memory index to disk (indexes/ann/<type>_index.json).
   */
  async saveIndex(
    indexType: "hnsw" | "ivf" | "brute",
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const merged = this.client.mergeOptions(options);
    const params = { ...merged?.params, index_type: indexType };
    const path = buildQueryString("/api/v1/ann/index/save", params);
    return jsonPost(
      this.client.baseUrl,
      path,
      undefined,
      {
        ...merged,
        params: undefined, // Cleared because they are now in the path
      },
    );
  }

  /**
   * Restore a previously saved index from disk into memory.
   */
  async loadIndex(
    indexType: "hnsw" | "ivf" | "brute",
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const merged = this.client.mergeOptions(options);
    const params = { ...merged?.params, index_type: indexType };
    const path = buildQueryString("/api/v1/ann/index/load", params);
    return jsonPost(
      this.client.baseUrl,
      path,
      undefined,
      {
        ...merged,
        params: undefined, // Cleared because they are now in the path
      },
    );
  }

  /**
   * Run the same query against every loaded index (HNSW, IVF, BruteForce)
   * and return results + wall-clock search time for each.
   */
  async compareSearch(
    params: { queryVector: number[]; k?: number },
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const merged = this.client.mergeOptions(options);
    const queryParams: Record<string, string | number | boolean | undefined> = {
      ...merged?.params,
      query_vector: params.queryVector.join(","),
    };
    if (params.k !== undefined) {
      queryParams.k = params.k;
    }
    return jsonGet(
      this.client.baseUrl,
      "/api/v1/ann/search/compare",
      {
        ...merged,
        params: queryParams,
      },
    );
  }

  /**
   * Return DB vector count and stats for all loaded indexes.
   */
  async getStatistics(options?: RequestOptions): Promise<Record<string, unknown>> {
    return jsonGet(
      this.client.baseUrl,
      "/api/v1/ann/stats",
      this.client.mergeOptions(options),
    );
  }
}
