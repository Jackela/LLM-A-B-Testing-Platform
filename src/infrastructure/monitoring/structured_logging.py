"""Structured logging with OpenTelemetry integration."""

import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Dict, List, Optional, Union

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    """Event types for structured logging."""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SYSTEM = "system"
    AUDIT = "audit"


@dataclass
class LogContext:
    """Log context information."""

    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class StructuredLogRecord:
    """Structured log record."""

    timestamp: datetime
    level: LogLevel
    message: str
    event_type: EventType
    context: LogContext
    metadata: Dict[str, Any]
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        record = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "event_type": self.event_type.value,
            "context": self.context.to_dict(),
            "metadata": self.metadata,
        }

        if self.error_details:
            record["error"] = self.error_details

        if self.performance_metrics:
            record["performance"] = self.performance_metrics

        return record

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class ContextManager:
    """Thread-local context management."""

    def __init__(self):
        self._local = threading.local()

    def set_context(self, context: LogContext):
        """Set context for current thread."""
        self._local.context = context

    def get_context(self) -> Optional[LogContext]:
        """Get context for current thread."""
        return getattr(self._local, "context", None)

    def clear_context(self):
        """Clear context for current thread."""
        if hasattr(self._local, "context"):
            delattr(self._local, "context")

    @contextmanager
    def context_scope(self, context: LogContext):
        """Context manager for temporary context."""
        old_context = self.get_context()
        try:
            self.set_context(context)
            yield context
        finally:
            if old_context:
                self.set_context(old_context)
            else:
                self.clear_context()


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, context_manager: ContextManager):
        super().__init__()
        self.context_manager = context_manager

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get or create context
        context = self.context_manager.get_context()
        if not context:
            context = LogContext(correlation_id=str(uuid.uuid4()))

        # Add trace information if available
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            span_context = span.get_span_context()
            context.trace_id = format(span_context.trace_id, "032x")
            context.span_id = format(span_context.span_id, "016x")

        # Determine event type
        event_type = getattr(record, "event_type", EventType.SYSTEM)
        if record.levelno >= logging.ERROR:
            event_type = EventType.ERROR

        # Extract metadata
        metadata = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "funcName",
                "lineno",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                metadata[key] = value

        # Add source information
        metadata.update(
            {
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "thread": record.threadName,
                "process": record.processName,
            }
        )

        # Handle errors
        error_details = None
        if record.exc_info:
            error_details = {
                "exception_type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "exception_message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }

        # Create structured record
        structured_record = StructuredLogRecord(
            timestamp=datetime.fromtimestamp(record.created),
            level=LogLevel(record.levelname),
            message=record.getMessage(),
            event_type=event_type,
            context=context,
            metadata=metadata,
            error_details=error_details,
        )

        return structured_record.to_json()


class StructuredLogger:
    """Enhanced structured logger with context management."""

    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        self.context_manager = ContextManager()

        # Configure JSON formatter
        self.json_formatter = JSONFormatter(self.context_manager)

        # Setup handlers
        self._setup_handlers()

        # Initialize OpenTelemetry instrumentation
        LoggingInstrumentor().instrument(set_logging_format=False)

    def _setup_handlers(self):
        """Setup log handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(console_handler)

        # File handlers
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # General log file with rotation
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, f"{self.name}.log"),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(file_handler)

        # Error log file
        error_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, f"{self.name}_errors.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(error_handler)

        # Security log file
        security_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, "security.log"),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding="utf-8",
        )
        security_handler.setFormatter(self.json_formatter)

        # Add filter for security events
        class SecurityFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, "event_type") and record.event_type == EventType.SECURITY

        security_handler.addFilter(SecurityFilter())
        self.logger.addHandler(security_handler)

    def set_context(self, **kwargs):
        """Set logging context."""
        context = self.context_manager.get_context()
        if context:
            # Update existing context
            for key, value in kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)
        else:
            # Create new context
            context = LogContext(
                correlation_id=kwargs.get("correlation_id", str(uuid.uuid4())),
                user_id=kwargs.get("user_id"),
                session_id=kwargs.get("session_id"),
                request_id=kwargs.get("request_id"),
                trace_id=kwargs.get("trace_id"),
                span_id=kwargs.get("span_id"),
            )

        self.context_manager.set_context(context)

    @contextmanager
    def context_scope(self, **kwargs):
        """Create temporary context scope."""
        context = LogContext(
            correlation_id=kwargs.get("correlation_id", str(uuid.uuid4())),
            user_id=kwargs.get("user_id"),
            session_id=kwargs.get("session_id"),
            request_id=kwargs.get("request_id"),
        )

        with self.context_manager.context_scope(context):
            yield context

    def debug(self, message: str, event_type: EventType = EventType.SYSTEM, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra={"event_type": event_type, **kwargs})

    def info(self, message: str, event_type: EventType = EventType.SYSTEM, **kwargs):
        """Log info message."""
        self.logger.info(message, extra={"event_type": event_type, **kwargs})

    def warning(self, message: str, event_type: EventType = EventType.SYSTEM, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra={"event_type": event_type, **kwargs})

    def error(
        self, message: str, event_type: EventType = EventType.ERROR, exc_info: bool = True, **kwargs
    ):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info, extra={"event_type": event_type, **kwargs})

    def critical(
        self, message: str, event_type: EventType = EventType.ERROR, exc_info: bool = True, **kwargs
    ):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info, extra={"event_type": event_type, **kwargs})

    def log_request(self, method: str, path: str, status_code: int, duration: float, **kwargs):
        """Log HTTP request."""
        self.info(
            f"{method} {path} - {status_code} in {duration:.3f}s",
            event_type=EventType.REQUEST,
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration * 1000,
            **kwargs,
        )

    def log_security_event(
        self, event: str, severity: str = "medium", ip_address: str = None, **kwargs
    ):
        """Log security event."""
        level = (
            LogLevel.WARNING
            if severity == "low"
            else LogLevel.ERROR if severity == "high" else LogLevel.WARNING
        )

        log_method = getattr(self, level.value.lower())
        log_method(
            f"Security event: {event}",
            event_type=EventType.SECURITY,
            security_event=event,
            severity=severity,
            ip_address=ip_address,
            **kwargs,
        )

    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.info(
            f"Performance: {operation} took {duration:.3f}s",
            event_type=EventType.PERFORMANCE,
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs,
        )

    def log_business_event(self, event: str, **kwargs):
        """Log business event."""
        self.info(
            f"Business event: {event}",
            event_type=EventType.BUSINESS,
            business_event=event,
            **kwargs,
        )

    def log_audit(self, action: str, resource: str, user_id: str = None, **kwargs):
        """Log audit event."""
        self.info(
            f"Audit: {action} on {resource}",
            event_type=EventType.AUDIT,
            audit_action=action,
            audit_resource=resource,
            audit_user_id=user_id,
            **kwargs,
        )


class OpenTelemetrySetup:
    """OpenTelemetry configuration and setup."""

    def __init__(self, service_name: str, jaeger_endpoint: str = None):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint or os.getenv(
            "JAEGER_ENDPOINT", "http://localhost:14268/api/traces"
        )

    def setup(self):
        """Setup OpenTelemetry tracing."""
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())

        # Set up Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )

        # Set up span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        # Get tracer
        return trace.get_tracer(self.service_name)


# Global logger instances
_loggers: Dict[str, StructuredLogger] = {}
_context_manager = ContextManager()


def get_logger(name: str, level: LogLevel = LogLevel.INFO) -> StructuredLogger:
    """Get or create structured logger."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level)
    return _loggers[name]


def setup_application_logging(
    service_name: str = "llm-ab-testing",
    log_level: LogLevel = LogLevel.INFO,
    enable_tracing: bool = True,
) -> StructuredLogger:
    """Setup application logging with OpenTelemetry."""
    # Setup OpenTelemetry if enabled
    if enable_tracing:
        otel_setup = OpenTelemetrySetup(service_name)
        otel_setup.setup()

    # Get main application logger
    logger = get_logger(service_name, log_level)

    # Configure root logger to prevent duplicate logs
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)  # Only show warnings and above from other libraries

    return logger


# Request logging decorator
def log_function_call(logger: StructuredLogger = None):
    """Decorator to log function calls."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                _logger.log_performance(
                    operation=f"{func.__module__}.{func.__name__}",
                    duration=duration,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys()),
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                _logger.error(
                    f"Function {func.__module__}.{func.__name__} failed after {duration:.3f}s: {e}",
                    operation=f"{func.__module__}.{func.__name__}",
                    duration=duration,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise

        return wrapper

    return decorator
