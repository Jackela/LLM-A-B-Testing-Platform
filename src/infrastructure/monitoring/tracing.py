"""Distributed tracing with OpenTelemetry integration."""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import baggage, trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.trace.status import Status, StatusCode

from .structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class TraceConfig:
    """Configuration for distributed tracing."""

    service_name: str = "llm-ab-testing"
    service_version: str = "1.0.0"
    environment: str = "production"

    # Exporters
    jaeger_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None
    console_export: bool = False

    # Sampling
    sampling_ratio: float = 1.0  # Sample all traces by default

    # Instrumentation
    instrument_fastapi: bool = True
    instrument_sqlalchemy: bool = True
    instrument_redis: bool = True
    instrument_httpx: bool = True


@dataclass
class SpanContext:
    """Enhanced span context information."""

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_span(cls, span: Span) -> "SpanContext":
        """Create SpanContext from OpenTelemetry span."""
        span_context = span.get_span_context()
        return cls(
            trace_id=format(span_context.trace_id, "032x"),
            span_id=format(span_context.span_id, "016x"),
            parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
            baggage=dict(baggage.get_all()),
        )


class TracingManager:
    """Centralized tracing management system."""

    def __init__(self, config: TraceConfig):
        self.config = config
        self.tracer: Optional[trace.Tracer] = None
        self._is_initialized = False
        self._span_processors: List[BatchSpanProcessor] = []

        # Custom span attributes
        self._default_attributes = {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
        }

    def initialize(self):
        """Initialize OpenTelemetry tracing."""
        if self._is_initialized:
            logger.warning("Tracing already initialized")
            return

        # Create resource
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
                ResourceAttributes.SERVICE_VERSION: self.config.service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
            }
        )

        # Set tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Setup exporters
        self._setup_exporters(tracer_provider)

        # Get tracer
        self.tracer = trace.get_tracer(self.config.service_name, self.config.service_version)

        # Setup automatic instrumentation
        if self.config.instrument_fastapi:
            FastAPIInstrumentor().instrument()
            logger.info("FastAPI instrumentation enabled")

        if self.config.instrument_sqlalchemy:
            SQLAlchemyInstrumentor().instrument()
            logger.info("SQLAlchemy instrumentation enabled")

        if self.config.instrument_redis:
            RedisInstrumentor().instrument()
            logger.info("Redis instrumentation enabled")

        if self.config.instrument_httpx:
            HTTPXClientInstrumentor().instrument()
            logger.info("HTTPX instrumentation enabled")

        self._is_initialized = True
        logger.info(f"Distributed tracing initialized for {self.config.service_name}")

    def _setup_exporters(self, tracer_provider: TracerProvider):
        """Setup span exporters."""
        # Jaeger exporter
        if self.config.jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.config.jaeger_endpoint.split(":")[0],
                    agent_port=int(self.config.jaeger_endpoint.split(":")[1]),
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                tracer_provider.add_span_processor(span_processor)
                self._span_processors.append(span_processor)
                logger.info(f"Jaeger exporter configured: {self.config.jaeger_endpoint}")
            except Exception as e:
                logger.error(f"Failed to setup Jaeger exporter: {e}")

        # OTLP exporter
        if self.config.otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                tracer_provider.add_span_processor(span_processor)
                self._span_processors.append(span_processor)
                logger.info(f"OTLP exporter configured: {self.config.otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to setup OTLP exporter: {e}")

        # Console exporter (for development)
        if self.config.console_export:
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(span_processor)
            self._span_processors.append(span_processor)
            logger.info("Console exporter enabled")

    def create_span(
        self,
        name: str,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL,
        attributes: Dict[str, Any] = None,
        parent_context=None,
    ) -> Span:
        """Create a new span."""
        if not self._is_initialized or not self.tracer:
            # Return no-op span if not initialized
            return trace.INVALID_SPAN

        # Merge default attributes with provided attributes
        span_attributes = {**self._default_attributes}
        if attributes:
            span_attributes.update(attributes)

        # Create span
        span = self.tracer.start_span(
            name=name, kind=kind, attributes=span_attributes, context=parent_context
        )

        return span

    @contextmanager
    def trace_operation(
        self, name: str, operation_type: str = "internal", attributes: Dict[str, Any] = None
    ):
        """Context manager for tracing operations."""
        # Determine span kind
        kind_mapping = {
            "http": trace.SpanKind.CLIENT,
            "database": trace.SpanKind.CLIENT,
            "cache": trace.SpanKind.CLIENT,
            "external": trace.SpanKind.CLIENT,
            "internal": trace.SpanKind.INTERNAL,
            "server": trace.SpanKind.SERVER,
        }

        span_kind = kind_mapping.get(operation_type, trace.SpanKind.INTERNAL)

        # Add operation type to attributes
        span_attributes = {"operation.type": operation_type}
        if attributes:
            span_attributes.update(attributes)

        span = self.create_span(name, kind=span_kind, attributes=span_attributes)

        try:
            with trace.use_span(span):
                start_time = time.time()
                yield span

                # Record successful completion
                duration = time.time() - start_time
                span.set_attribute("operation.duration_ms", duration * 1000)
                span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Record error
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            span.end()

    @asynccontextmanager
    async def trace_async_operation(
        self, name: str, operation_type: str = "internal", attributes: Dict[str, Any] = None
    ):
        """Async context manager for tracing operations."""
        # Same implementation as sync version, but async
        kind_mapping = {
            "http": trace.SpanKind.CLIENT,
            "database": trace.SpanKind.CLIENT,
            "cache": trace.SpanKind.CLIENT,
            "external": trace.SpanKind.CLIENT,
            "internal": trace.SpanKind.INTERNAL,
            "server": trace.SpanKind.SERVER,
        }

        span_kind = kind_mapping.get(operation_type, trace.SpanKind.INTERNAL)

        span_attributes = {"operation.type": operation_type}
        if attributes:
            span_attributes.update(attributes)

        span = self.create_span(name, kind=span_kind, attributes=span_attributes)

        try:
            with trace.use_span(span):
                start_time = time.time()
                yield span

                duration = time.time() - start_time
                span.set_attribute("operation.duration_ms", duration * 1000)
                span.set_status(Status(StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            span.end()

    def add_span_event(self, name: str, attributes: Dict[str, Any] = None):
        """Add event to current span."""
        current_span = trace.get_current_span()
        if current_span and current_span != trace.INVALID_SPAN:
            current_span.add_event(name, attributes)

    def set_span_attribute(self, key: str, value: Any):
        """Set attribute on current span."""
        current_span = trace.get_current_span()
        if current_span and current_span != trace.INVALID_SPAN:
            current_span.set_attribute(key, value)

    def get_current_span_context(self) -> Optional[SpanContext]:
        """Get current span context."""
        current_span = trace.get_current_span()
        if current_span and current_span != trace.INVALID_SPAN:
            return SpanContext.from_span(current_span)
        return None

    def set_baggage(self, key: str, value: str):
        """Set baggage value."""
        baggage.set_baggage(key, value)

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage value."""
        return baggage.get_baggage(key)

    def inject_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers."""
        inject(headers)
        return headers

    def extract_headers(self, headers: Dict[str, str]):
        """Extract trace context from headers."""
        return extract(headers)

    def shutdown(self):
        """Shutdown tracing and flush spans."""
        logger.info("Shutting down distributed tracing...")

        # Shutdown all span processors
        for processor in self._span_processors:
            try:
                processor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down span processor: {e}")

        self._is_initialized = False
        logger.info("Distributed tracing shutdown complete")


class TracingDecorators:
    """Decorators for automatic tracing."""

    def __init__(self, tracing_manager: TracingManager):
        self.tracing = tracing_manager

    def trace_function(
        self,
        name: str = None,
        operation_type: str = "internal",
        include_args: bool = False,
        include_result: bool = False,
    ):
        """Decorator to trace function calls."""

        def decorator(func):
            span_name = name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):

                async def async_wrapper(*args, **kwargs):
                    attributes = {"function.name": func.__name__}

                    if include_args:
                        attributes["function.args_count"] = len(args)
                        attributes["function.kwargs_keys"] = list(kwargs.keys())

                    async with self.tracing.trace_async_operation(
                        span_name, operation_type, attributes
                    ) as span:
                        try:
                            result = await func(*args, **kwargs)

                            if include_result and result is not None:
                                # Only include simple result types
                                if isinstance(result, (str, int, float, bool)):
                                    span.set_attribute("function.result", str(result))
                                else:
                                    span.set_attribute(
                                        "function.result_type", type(result).__name__
                                    )

                            return result
                        except Exception as e:
                            span.set_attribute("function.exception", str(e))
                            raise

                return async_wrapper
            else:

                def sync_wrapper(*args, **kwargs):
                    attributes = {"function.name": func.__name__}

                    if include_args:
                        attributes["function.args_count"] = len(args)
                        attributes["function.kwargs_keys"] = list(kwargs.keys())

                    with self.tracing.trace_operation(
                        span_name, operation_type, attributes
                    ) as span:
                        try:
                            result = func(*args, **kwargs)

                            if include_result and result is not None:
                                if isinstance(result, (str, int, float, bool)):
                                    span.set_attribute("function.result", str(result))
                                else:
                                    span.set_attribute(
                                        "function.result_type", type(result).__name__
                                    )

                            return result
                        except Exception as e:
                            span.set_attribute("function.exception", str(e))
                            raise

                return sync_wrapper

        return decorator

    def trace_class(self, operation_type: str = "internal", exclude_methods: List[str] = None):
        """Class decorator to trace all methods."""
        exclude_methods = exclude_methods or ["__init__", "__del__"]

        def decorator(cls):
            for name in dir(cls):
                if (name.startswith("_") and name not in ["__call__"]) or name in exclude_methods:
                    continue

                attr = getattr(cls, name)
                if callable(attr):
                    traced_method = self.trace_function(f"{cls.__name__}.{name}", operation_type)(
                        attr
                    )
                    setattr(cls, name, traced_method)

            return cls

        return decorator


class DistributedTracingMiddleware:
    """FastAPI middleware for distributed tracing."""

    def __init__(self, tracing_manager: TracingManager):
        self.tracing = tracing_manager

    async def __call__(self, request, call_next):
        """Process request with distributed tracing."""
        # Extract trace context from headers
        context = self.tracing.extract_headers(dict(request.headers))

        # Create span for request
        span_name = f"{request.method} {request.url.path}"
        attributes = {
            SpanAttributes.HTTP_METHOD: request.method,
            SpanAttributes.HTTP_URL: str(request.url),
            SpanAttributes.HTTP_SCHEME: request.url.scheme,
            SpanAttributes.HTTP_HOST: request.url.hostname,
            SpanAttributes.HTTP_TARGET: request.url.path,
            "http.user_agent": request.headers.get("user-agent", ""),
            "http.remote_addr": request.client.host if request.client else "",
        }

        # Add request ID if available
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        attributes["http.request_id"] = request_id

        span = self.tracing.create_span(
            span_name, kind=trace.SpanKind.SERVER, attributes=attributes, parent_context=context
        )

        try:
            with trace.use_span(span):
                # Set request ID in baggage
                self.tracing.set_baggage("request.id", request_id)

                # Process request
                start_time = time.time()
                response = await call_next(request)
                duration = time.time() - start_time

                # Add response attributes
                span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, response.status_code)
                span.set_attribute("http.response_size", len(getattr(response, "body", b"")))
                span.set_attribute("http.duration_ms", duration * 1000)

                # Set status
                if 400 <= response.status_code < 600:
                    span.set_status(Status(StatusCode.ERROR, f"HTTP {response.status_code}"))
                else:
                    span.set_status(Status(StatusCode.OK))

                # Inject trace context into response headers
                response_headers = dict(response.headers)
                self.tracing.inject_headers(response_headers)

                # Update response headers
                for key, value in response_headers.items():
                    if key.lower().startswith("trace") or key.lower() == "x-trace-id":
                        response.headers[key] = value

                return response

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()


# Global tracing manager instance
_tracing_manager: Optional[TracingManager] = None
_tracing_decorators: Optional[TracingDecorators] = None


def setup_distributed_tracing(config: TraceConfig) -> TracingManager:
    """Setup distributed tracing system."""
    global _tracing_manager, _tracing_decorators

    _tracing_manager = TracingManager(config)
    _tracing_manager.initialize()

    _tracing_decorators = TracingDecorators(_tracing_manager)

    return _tracing_manager


def get_tracing_manager() -> Optional[TracingManager]:
    """Get global tracing manager."""
    return _tracing_manager


def get_tracing_decorators() -> Optional[TracingDecorators]:
    """Get tracing decorators."""
    return _tracing_decorators


# Convenience functions
def trace_operation(name: str, operation_type: str = "internal", attributes: Dict[str, Any] = None):
    """Context manager for tracing operations."""
    if _tracing_manager:
        return _tracing_manager.trace_operation(name, operation_type, attributes)
    else:
        # Return no-op context manager
        @contextmanager
        def noop():
            yield None

        return noop()


def trace_async_operation(
    name: str, operation_type: str = "internal", attributes: Dict[str, Any] = None
):
    """Async context manager for tracing operations."""
    if _tracing_manager:
        return _tracing_manager.trace_async_operation(name, operation_type, attributes)
    else:
        # Return no-op async context manager
        @asynccontextmanager
        async def noop():
            yield None

        return noop()


def trace_function(
    name: str = None,
    operation_type: str = "internal",
    include_args: bool = False,
    include_result: bool = False,
):
    """Decorator to trace function calls."""
    if _tracing_decorators:
        return _tracing_decorators.trace_function(
            name, operation_type, include_args, include_result
        )
    else:
        # Return no-op decorator
        def noop_decorator(func):
            return func

        return noop_decorator


def set_span_attribute(key: str, value: Any):
    """Set attribute on current span."""
    if _tracing_manager:
        _tracing_manager.set_span_attribute(key, value)


def add_span_event(name: str, attributes: Dict[str, Any] = None):
    """Add event to current span."""
    if _tracing_manager:
        _tracing_manager.add_span_event(name, attributes)


def get_current_trace_id() -> Optional[str]:
    """Get current trace ID."""
    if _tracing_manager:
        span_context = _tracing_manager.get_current_span_context()
        return span_context.trace_id if span_context else None
    return None
