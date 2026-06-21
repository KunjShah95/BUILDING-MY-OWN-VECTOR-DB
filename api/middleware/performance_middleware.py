"""Middleware for tracking request performance and recording slow queries."""
import logging
import time
from fastapi import Request, Response
from services.slow_query_analyzer import slow_query_analyzer
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics
HTTP_REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency in seconds", ["method", "endpoint"])

async def performance_middleware(request: Request, call_next):
    """Track request duration and record slow query events."""
    start = time.perf_counter()
    method = request.method
    endpoint = request.url.path

    try:
        response: Response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        # Still record failure in prometheus
        elapsed_sec = time.perf_counter() - start
        HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=status_code).inc()
        HTTP_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed_sec)
        raise e

    elapsed_sec = time.perf_counter() - start
    elapsed_ms = elapsed_sec * 1000

    # Record Prometheus metrics
    HTTP_REQUESTS.labels(method=method, endpoint=endpoint, status=status_code).inc()
    HTTP_LATENCY.labels(method=method, endpoint=endpoint).observe(elapsed_sec)

    if "/search" in endpoint and elapsed_ms > slow_query_analyzer.threshold_ms:
        tenant_id = getattr(request.state, "tenant_id", "unknown")
        slow_query_analyzer.record(
            collection_id=request.path_params.get("collection_id", "unknown"),
            method=method,
            latency_ms=elapsed_ms,
            query_preview=str(request.query_params)[:200],
            tenant_id=tenant_id,
        )

    response.headers.append("Server-Timing", f"total;dur={elapsed_ms:.1f}")
    return response
