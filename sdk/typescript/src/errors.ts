/**
 * Custom error types for the Vector DB client.
 */

export class VectorDBError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "VectorDBError";
  }
}

export class VectorDBHTTPError extends VectorDBError {
  public readonly statusCode: number;
  public readonly detail: unknown;

  constructor(statusCode: number, detail: unknown) {
    const message = `HTTP ${statusCode}: ${JSON.stringify(detail)}`;
    super(message);
    this.name = "VectorDBHTTPError";
    this.statusCode = statusCode;
    this.detail = detail;
  }
}
