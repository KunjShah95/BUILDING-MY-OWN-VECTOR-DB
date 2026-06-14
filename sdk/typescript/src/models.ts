/**
 * Data models for the Vector Database API responses.
 */

export interface CollectionData {
  collection_id: string;
  name: string;
  modality: string;
  dimension: number;
  embedding_model: string;
  distance_metric?: string;
  description?: string;
  [key: string]: unknown;
}

export class Collection {
  constructor(
    public readonly collectionId: string,
    public readonly name: string,
    public readonly modality: string,
    public readonly dimension: number,
    public readonly embeddingModel: string,
    public readonly distanceMetric: string = "cosine",
    public readonly description?: string,
    public readonly raw: Record<string, unknown> = {},
  ) {}

  static fromApi(data: CollectionData): Collection {
    return new Collection(
      data.collection_id,
      data.name,
      data.modality,
      data.dimension,
      data.embedding_model,
      data.distance_metric ?? "cosine",
      data.description,
      data as Record<string, unknown>,
    );
  }
}

export interface SearchHitData {
  vector_id: string;
  distance: number;
  metadata?: Record<string, unknown> | null;
  meta_data?: Record<string, unknown> | null;
}

export class SearchHit {
  constructor(
    public readonly vectorId: string,
    public readonly distance: number,
    public readonly metadata?: Record<string, unknown> | null,
  ) {}

  static fromApi(row: SearchHitData): SearchHit {
    return new SearchHit(
      row.vector_id,
      row.distance,
      row.metadata ?? row.meta_data ?? null,
    );
  }
}

export interface SearchResultData {
  success: boolean;
  results: SearchHitData[];
  total_results?: number;
  search_time?: number;
  method?: string;
  [key: string]: unknown;
}

export class SearchResult {
  constructor(
    public readonly success: boolean,
    public readonly results: SearchHit[],
    public readonly totalResults: number,
    public readonly searchTime: number,
    public readonly method: string,
    public readonly raw: Record<string, unknown> = {},
  ) {}

  static fromApi(data: SearchResultData): SearchResult {
    const hits = (data.results ?? []).map(SearchHit.fromApi);
    return new SearchResult(
      data.success,
      hits,
      data.total_results ?? hits.length,
      data.search_time ?? 0,
      data.method ?? "",
      data as Record<string, unknown>,
    );
  }
}
