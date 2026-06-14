import type { VectorDBClient } from "./client";
import type { RequestOptions } from "./http";
import { jsonPost, jsonGet, jsonDelete } from "./http";
import { Collection, type CollectionData } from "./models";

export interface CreateCollectionParams {
  name: string;
  collectionId?: string;
  modality?: string;
  dimension?: number;
  embeddingModel?: string;
  description?: string;
  distanceMetric?: string;
}

export class CollectionsAPI {
  constructor(private readonly client: VectorDBClient) {}

  async create(
    params: CreateCollectionParams,
    options?: RequestOptions,
  ): Promise<Collection> {
    const body: Record<string, unknown> = {
      name: params.name,
      modality: params.modality ?? "text",
      distance_metric: params.distanceMetric ?? "cosine",
    };
    if (params.collectionId) body.collection_id = params.collectionId;
    if (params.dimension !== undefined) body.dimension = params.dimension;
    if (params.embeddingModel) body.embedding_model = params.embeddingModel;
    if (params.description) body.description = params.description;

    const data = await jsonPost(this.client.baseUrl, "/collections", body, this.client.mergeOptions(options));
    const collData = data.collection as CollectionData;
    return Collection.fromApi(collData);
  }

  async list(
    limit = 100,
    offset = 0,
    options?: RequestOptions,
  ): Promise<Collection[]> {
    const merged = this.client.mergeOptions(options);
    const data = await jsonGet(this.client.baseUrl, "/collections", {
      ...merged,
      params: { limit, offset, ...merged?.params },
    });
    const collections = (data.collections ?? []) as CollectionData[];
    return collections.map(Collection.fromApi);
  }

  async get(collectionId: string, options?: RequestOptions): Promise<Collection> {
    const data = await jsonGet(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}`,
      this.client.mergeOptions(options),
    );
    const collData = data.collection as CollectionData;
    return Collection.fromApi(collData);
  }

  async delete(collectionId: string, options?: RequestOptions): Promise<Record<string, unknown>> {
    return jsonDelete(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}`,
      this.client.mergeOptions(options),
    );
  }

  async buildIndex(
    collectionId: string,
    params?: {
      method?: string;
      m?: number;
      m0?: number;
      efConstruction?: number;
      nClusters?: number;
      nProbes?: number;
    },
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = {
      method: params?.method ?? "hnsw",
      m: params?.m ?? 16,
      ef_construction: params?.efConstruction ?? 200,
      n_clusters: params?.nClusters ?? 100,
      n_probes: params?.nProbes ?? 10,
    };
    if (params?.m0 !== undefined) body.m0 = params.m0;
    return jsonPost(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/index`,
      body,
      this.client.mergeOptions(options),
    );
  }

  async indexStats(
    collectionId: string,
    options?: RequestOptions,
  ): Promise<Record<string, unknown>> {
    return jsonGet(
      this.client.baseUrl,
      `/collections/${encodeURIComponent(collectionId)}/index/stats`,
      options,
    );
  }
}
