"""Logging middleware for FastAPI."""

import json
import logging
import time
from typing import Any, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""

    def __init__(self, app, log_body: bool = False):
        """Initialize logging middleware.

        Args:
            app: FastAPI application
            log_body: Whether to log request/response bodies (for debugging)
        """
        super().__init__(app)
        self.log_body = log_body

    async def dispatch(self, request: Request, call_next):
        """Process request through logging middleware."""
        start_time = time.time()
        request.state.start_time = start_time

        # Log request
        await self._log_request(request)

        try:
            response = await call_next(request)

            # Log response
            process_time = time.time() - start_time
            await self._log_response(request, response, process_time)

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Request failed - "
                f"Path: {request.url.path}, "
                f"Method: {request.method}, "
                f"Error: {str(e)}, "
                f"Time: {process_time:.3f}s"
            )
            raise

    async def _log_request(self, request: Request):
        """Log incoming request."""
        log_data = {
            "event": "request_received",
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
        }

        # Add auth info if available
        if hasattr(request.state, "auth_token"):
            log_data["auth_token_preview"] = request.state.auth_token

        # Log request body for debugging (if enabled and safe)
        if self.log_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await self._get_request_body(request)
                if body and len(body) < 1000:  # Limit body size
                    log_data["body_preview"] = body[:500]
            except Exception:
                log_data["body_preview"] = "Unable to read body"

        logger.info(f"Request: {json.dumps(log_data, default=str)}")

    async def _log_response(self, request: Request, response: Response, process_time: float):
        """Log outgoing response."""
        log_data = {
            "event": "response_sent",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000, 2),
            "response_size": response.headers.get("content-length"),
        }

        # Determine log level based on status code
        if response.status_code >= 500:
            log_level = "error"
        elif response.status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"

        getattr(logger, log_level)(f"Response: {json.dumps(log_data, default=str)}")

    async def _get_request_body(self, request: Request) -> str:
        """Safely get request body for logging."""
        try:
            body = await request.body()
            if body:
                # Try to decode as JSON for prettier logging
                try:
                    json_body = json.loads(body)
                    # Remove sensitive fields
                    if isinstance(json_body, dict):
                        json_body = self._sanitize_body(json_body)
                    return json.dumps(json_body)
                except json.JSONDecodeError:
                    # Return as string if not JSON
                    return body.decode("utf-8", errors="ignore")[:500]
            return ""
        except Exception:
            return "Unable to read body"

    def _sanitize_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from request body."""
        sensitive_fields = {
            "password",
            "token",
            "secret",
            "key",
            "authorization",
            "api_key",
            "refresh_token",
            "access_token",
        }

        sanitized = {}
        for key, value in body.items():
            if key.lower() in sensitive_fields:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_body(value)
            else:
                sanitized[key] = value

        return sanitized
