"""FastAPI application configuration and setup."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .dependencies.container import Container
from .documentation.openapi_config import get_custom_openapi_schema
from .middleware.auth_middleware import AuthMiddleware
from .middleware.error_handler import ErrorHandlerMiddleware
from .middleware.logging_middleware import LoggingMiddleware
from .performance_setup import (
    add_performance_middleware,
    add_performance_routes,
    performance_lifespan,
)
from .routes.analytics import router as analytics_router
from .routes.auth import router as auth_router
from .routes.evaluation import router as evaluation_router
from .routes.providers import router as providers_router
from .routes.tests import router as tests_router
from .security_setup import add_security_middleware, add_security_routes, security_lifespan

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Dependency injection container
container = Container()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager with performance and security."""
    # Startup
    logger.info("Starting LLM A/B Testing Platform API")
    await container.wire_dependencies()

    # Initialize performance and security systems
    async with performance_lifespan(app):
        async with security_lifespan(app):
            yield

    # Shutdown
    logger.info("Shutting down LLM A/B Testing Platform API")
    await container.cleanup()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="LLM A/B Testing Platform API",
        description="REST API for managing LLM model comparisons and A/B testing",
        version="1.0.0",
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url="/api/v1/redoc",
        lifespan=lifespan,
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Security middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8501"],  # Frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)

    # Security middleware (must be added before performance middleware)
    add_security_middleware(app)

    # Performance middleware
    add_performance_middleware(app)

    # Routes
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(tests_router, prefix="/api/v1/tests", tags=["Test Management"])
    app.include_router(providers_router, prefix="/api/v1/providers", tags=["Model Providers"])
    app.include_router(evaluation_router, prefix="/api/v1/evaluation", tags=["Evaluation"])
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["Analytics"])

    # Security monitoring routes
    add_security_routes(app)

    # Performance monitoring routes
    add_performance_routes(app)

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": int(time.time()), "version": "1.0.0"}

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> Dict[str, str]:
        """Root endpoint with API information."""
        return {
            "message": "LLM A/B Testing Platform API",
            "docs": "/api/v1/docs",
            "version": "1.0.0",
        }

    # Set custom OpenAPI schema
    app.openapi = lambda: get_custom_openapi_schema(app)

    return app


# Create app instance
app = create_app()
