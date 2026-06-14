"""Tests for OpenTelemetry Distributed Tracing (Phase 8)."""

from unittest.mock import MagicMock, patch

import pytest

from utils.opentelemetry_tracing import (
    init_tracing,
    get_tracer,
    trace_span,
    SpanContext,
    shutdown_tracing,
    is_tracing_enabled,
)


class TestTracingInit:
    def test_init_without_otel(self):
        with patch.dict("sys.modules", {"opentelemetry": None}):
            init_tracing(service_name="test-svc")
            # Should not crash, just log a warning

    def test_init_with_defaults(self):
        init_tracing(service_name="test-svc", environment="test")
        # Should not raise
        shutdown_tracing()

    def test_is_tracing_enabled(self):
        enabled = is_tracing_enabled()
        assert isinstance(enabled, bool)


class TestGetTracer:
    def test_get_tracer_returns_tracer(self):
        tracer = get_tracer()
        assert tracer is not None


class TestTraceSpanDecorator:
    def test_trace_span_decorates_sync_fn(self):
        @trace_span("test_operation")
        def my_func(x, y):
            return x + y

        result = my_func(3, 4)
        assert result == 7

    def test_trace_span_with_attributes(self):
        @trace_span("with_attrs", attributes={"key": "val"})
        def my_func():
            return "done"

        assert my_func() == "done"

    def test_trace_span_handles_exception(self):
        @trace_span("failing")
        def failing_fn():
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            failing_fn()


class TestSpanContext:
    def test_span_context_success(self):
        with SpanContext("test_context"):
            pass  # Should not raise

    def test_span_context_with_attributes(self):
        with SpanContext("test_context", attributes={"count": 42}):
            pass

    def test_span_context_exception(self):
        with pytest.raises(RuntimeError):
            with SpanContext("failing_context"):
                raise RuntimeError("span failed")


class TestTracingLifecycle:
    def test_shutdown_tracing(self):
        # Should not raise even if not initialized
        shutdown_tracing()

    def test_init_twice(self):
        init_tracing(service_name="svc1")
        init_tracing(service_name="svc2")
        shutdown_tracing()
