class VectorDBError(Exception):
    """Base error for the Vector DB Python client."""


class VectorDBHTTPError(VectorDBError):
    """Raised when the API returns a non-success HTTP status."""

    def __init__(self, status_code: int, detail):
        self.status_code = status_code
        self.detail = detail
        message = f"HTTP {status_code}: {detail}"
        super().__init__(message)
