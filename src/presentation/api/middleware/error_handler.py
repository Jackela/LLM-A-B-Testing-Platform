"""Error handling middleware for FastAPI."""

import logging
import traceback
from typing import Any, Dict

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""

    async def dispatch(self, request: Request, call_next):
        """Process request through error handling middleware."""
        try:
            response = await call_next(request)
            return response

        except HTTPException as e:
            # FastAPI HTTPExceptions are handled properly
            return await self._handle_http_exception(request, e)

        except ValueError as e:
            # Validation errors
            return await self._handle_validation_error(request, e)

        except PermissionError as e:
            # Permission errors
            return await self._handle_permission_error(request, e)

        except ConnectionError as e:
            # Connection errors (database, external APIs)
            return await self._handle_connection_error(request, e)

        except Exception as e:
            # Unexpected errors
            return await self._handle_unexpected_error(request, e)

    async def _handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        error_response = {
            "error": {
                "type": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "method": request.method,
            }
        }

        logger.warning(
            f"HTTP exception - {exc.status_code}: {exc.detail} "
            f"Path: {request.url.path}, Method: {request.method}"
        )

        return JSONResponse(status_code=exc.status_code, content=error_response)

    async def _handle_validation_error(self, request: Request, exc: ValueError) -> JSONResponse:
        """Handle validation errors."""
        error_response = {
            "error": {
                "type": "validation_error",
                "message": str(exc),
                "status_code": 400,
                "path": str(request.url.path),
                "method": request.method,
            }
        }

        logger.warning(
            f"Validation error: {exc} " f"Path: {request.url.path}, Method: {request.method}"
        )

        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=error_response)

    async def _handle_permission_error(
        self, request: Request, exc: PermissionError
    ) -> JSONResponse:
        """Handle permission errors."""
        error_response = {
            "error": {
                "type": "permission_error",
                "message": "Insufficient permissions to perform this operation",
                "status_code": 403,
                "path": str(request.url.path),
                "method": request.method,
            }
        }

        logger.warning(
            f"Permission error: {exc} " f"Path: {request.url.path}, Method: {request.method}"
        )

        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=error_response)

    async def _handle_connection_error(
        self, request: Request, exc: ConnectionError
    ) -> JSONResponse:
        """Handle connection errors."""
        error_response = {
            "error": {
                "type": "connection_error",
                "message": "Service temporarily unavailable. Please try again later.",
                "status_code": 503,
                "path": str(request.url.path),
                "method": request.method,
            }
        }

        logger.error(
            f"Connection error: {exc} " f"Path: {request.url.path}, Method: {request.method}"
        )

        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=error_response)

    async def _handle_unexpected_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        error_id = f"ERR-{int(request.state.__dict__.get('start_time', 0))}"

        error_response = {
            "error": {
                "type": "internal_error",
                "message": "An unexpected error occurred. Please contact support.",
                "error_id": error_id,
                "status_code": 500,
                "path": str(request.url.path),
                "method": request.method,
            }
        }

        # Log full error details
        logger.error(
            f"Unexpected error [{error_id}]: {exc} "
            f"Path: {request.url.path}, Method: {request.method}\n"
            f"Traceback: {traceback.format_exc()}"
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response
        )
