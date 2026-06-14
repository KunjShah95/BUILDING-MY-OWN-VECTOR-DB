import { CollectionsAPI } from "./collections";
import { VectorsAPI } from "./vectors";
import { MultimodalAPI } from "./multimodal";
import type { RequestOptions } from "./http";
import { jsonGet } from "./http";

export interface ClientOptions {
  baseUrl?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

export class VectorDBClient {
  public readonly baseUrl: string;
  public readonly timeout: number;
  public readonly headers: Record<string, string>;
  public readonly collections: CollectionsAPI;
  public readonly vectors: VectorsAPI;
  public readonly multimodal: MultimodalAPI;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://localhost:8000").replace(/\/+$/, "");
    this.timeout = options.timeout ?? 60_000;
    this.headers = { ...options.headers };

    this.collections = new CollectionsAPI(this);
    this.vectors = new VectorsAPI(this);
    this.multimodal = new MultimodalAPI(this);
  }

  /**
   * Merge per-request options with the client's default headers.
   */
  mergeOptions(options?: RequestOptions): RequestOptions | undefined {
    const clientHeaders = this.headers;
    const hasClientHeaders = Object.keys(clientHeaders).length > 0;
    if (!options && !hasClientHeaders) return undefined;
    if (!options) return { headers: { ...clientHeaders } };
    return {
      ...options,
      headers: {
        ...clientHeaders,
        ...options.headers,
      },
    };
  }

  /**
   * Health check — returns a promise that resolves when the API is reachable.
   */
  async health(options?: RequestOptions): Promise<Record<string, unknown>> {
    return jsonGet(this.baseUrl, "/health", this.mergeOptions(options));
  }
}
