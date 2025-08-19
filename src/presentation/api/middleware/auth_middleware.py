"""Authentication middleware for FastAPI."""

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware."""

    # Paths that don't require authentication
    EXEMPT_PATHS = {
        "/",
        "/health",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/docs",
        "/api/v1/redoc",
        "/api/v1/openapi.json",
    }

    async def dispatch(self, request: Request, call_next):
        """Process request through authentication middleware."""
        start_time = time.time()

        # Skip authentication for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            response = await call_next(request)
            return response

        # Skip authentication for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            response = await call_next(request)
            return response

        # Extract authorization header
        authorization: str = request.headers.get("Authorization")

        if not authorization:
            # Allow request to continue - individual endpoints will handle auth
            response = await call_next(request)
            return response

        # Add auth info to request state for logging
        if authorization.startswith("Bearer "):
            token = authorization.split(" ")[1][:20] + "..."  # Truncate for logging
            request.state.auth_token = token

        try:
            response = await call_next(request)

            # Log authentication info
            process_time = time.time() - start_time
            logger.info(
                f"Auth middleware - Path: {request.url.path}, "
                f"Method: {request.method}, "
                f"Status: {response.status_code}, "
                f"Time: {process_time:.3f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Auth middleware error: {e}")
            raise
