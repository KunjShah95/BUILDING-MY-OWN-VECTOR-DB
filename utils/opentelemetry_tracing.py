"""
OpenTelemetry Distributed Tracing (Phase 8: Observability).

Extends the existing ``utils/telemetry.py`` with full OpenTelemetry spans
across API → service → index → DB boundaries, enabling tracing of the
complete search/ingest request path in distributed deployments.

Configuration:
  - Reads OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME from env
  - Defaults to console export if no OTLP endpoint configured
  - Supports Jaeger, Zipkin, and any OTLP-compatible backend
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy OpenTelemetry initialization (avoids hard dependency)
# ---------------------------------------------------------------------------

_OTLP_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    _OTLP_AVAILABLE = True
except ImportError:
    pass

_tracer: Any = None
_tracer_provider: Any = None


def init_tracing(
    service_name: str = "vector-db",
    otlp_endpoint: Optional[str] = None,
    environment: str = "production",
):
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name for this service instance.
        otlp_endpoint: OTLP gRPC endpoint (e.g. "localhost:4317").
            Falls back to console exporter if not set.
        environment: Deployment environment tag.
    """
    global _tracer, _tracer_provider

    if not _OTLP_AVAILABLE:
        logger.warning(
            "OpenTelemetry not installed. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )
        return

    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": environment,
    })

    provider = TracerProvider(resource=resource)

    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            logger.info("OTLP tracing configured for endpoint %s", otlp_endpoint)
        except ImportError:
            exporter = ConsoleSpanExporter()
            logger.warning("OTLP exporter not installed, falling back to console")
    else:
        exporter = ConsoleSpanExporter()
        logger.info("OTLP endpoint not set, using console exporter")

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    _tracer_provider = provider
    _tracer = trace.get_tracer(service_name)


def get_tracer() -> Any:
    """Get the global OpenTelemetry tracer."""
    global _tracer
    if _tracer is None:
        # Auto-init with defaults
        init_tracing()
    return _tracer


# ---------------------------------------------------------------------------
# Decorator for auto-tracing
# ---------------------------------------------------------------------------


def trace_span(name: Optional[str] = None, attributes: Optional[Dict] = None):
    """Decorator that wraps a function in an OpenTelemetry span.

    Usage::

        @trace_span("search_vectors")
        def search(query, k=10):
            ...
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = name or func.__name__
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, dict):
                        span.set_attribute("success", result.get("success", True))
                    return result
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_attribute("error", True)
                    raise
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Context manager for manual span creation
# ---------------------------------------------------------------------------


class SpanContext:
    """Context manager for manual tracing spans.

    Usage::

        with SpanContext("process_batch") as span:
            # do work
            span.set_attribute("batch_size", 100)
    """

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}

    def __enter__(self):
        tracer = get_tracer()
        self._span = tracer.start_span(self.name)
        for k, v in self.attributes.items():
            self._span.set_attribute(k, v)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._span.record_exception(exc_val)
            self._span.set_attribute("error", True)
        self._span.end()


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def shutdown_tracing():
    """Flush and shut down the tracer provider."""
    global _tracer_provider
    if _tracer_provider:
        _tracer_provider.shutdown()
        logger.info("Tracing shut down")


def is_tracing_enabled() -> bool:
    return _OTLP_AVAILABLE and _tracer is not None
