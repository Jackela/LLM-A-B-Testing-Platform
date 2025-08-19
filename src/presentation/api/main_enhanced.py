import time
import uuid
from datetime import datetime

"""Enhanced FastAPI application with comprehensive security and monitoring."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

from ...infrastructure.monitoring.alerting import get_alert_manager
from ...infrastructure.monitoring.health import (
    get_health_checker,
    get_health_monitor,
    health_status_changed,
)
from ...infrastructure.monitoring.metrics import get_metrics_collector, get_performance_tracker

# Monitoring imports
from ...infrastructure.monitoring.structured_logging import (
    EventType,
    get_logger,
    setup_application_logging,
)
from ...infrastructure.monitoring.tracing import (
    DistributedTracingMiddleware,
    TraceConfig,
    setup_distributed_tracing,
)

# Database and other imports
from ...infrastructure.persistence.database import get_database_session

# Security imports
from ...infrastructure.security.auth import Permission, User, get_auth_system
from ...infrastructure.security.middleware import RateLimitPresets, SecurityMiddleware
from ...infrastructure.security.secrets_manager import get_env_secrets

# Initialize components
logger = setup_application_logging("llm-ab-testing", enable_tracing=True)
secrets = get_env_secrets()
auth_system = get_auth_system()
metrics = get_metrics_collector()
performance_tracker = get_performance_tracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with monitoring and alerting."""
    logger.info("Starting LLM A/B Testing Platform with enhanced security and monitoring")

    try:
        # Setup distributed tracing
        tracing_config = TraceConfig(
            service_name="llm-ab-testing",
            service_version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "production"),
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            sampling_ratio=float(os.getenv("TRACE_SAMPLING_RATIO", "0.1")),
        )
        setup_distributed_tracing(tracing_config)

        # Setup health monitoring
        health_checker = get_health_checker(
            database_session_factory=get_database_session,
            redis_url=secrets.get_redis_url(),
            external_services={
                "openai": "https://api.openai.com",
                "anthropic": "https://api.anthropic.com",
            },
        )

        # Setup health monitor with alerting
        health_monitor = get_health_monitor(
            health_checker=health_checker, check_interval=60, alert_callback=health_status_changed
        )

        # Setup alert manager
        alert_manager = get_alert_manager()

        # Start background services
        await health_monitor.start()
        await alert_manager.start()

        logger.info("All monitoring and security services started successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to start application services: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.info("Shutting down application services")
        try:
            await health_monitor.stop()
            await alert_manager.stop()
            logger.info("Application services shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="LLM A/B Testing Platform",
    description="Enterprise-grade platform for conducting A/B tests on Large Language Models",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)

# Configure CORS
allowed_origins = secrets.get_secret(
    "allowed_origins", "ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
)

# Add security middleware
environment = os.getenv("ENVIRONMENT", "production")
security_middleware = SecurityMiddleware(
    app,
    rate_limit_config=(
        RateLimitPresets.PRODUCTION if environment == "production" else RateLimitPresets.DEVELOPMENT
    ),
    enable_ip_filtering=environment == "production",
    enable_input_validation=True,
)
app.add_middleware(type(security_middleware), **security_middleware.__dict__)

# Add distributed tracing middleware
tracing_middleware = DistributedTracingMiddleware
app.add_middleware(tracing_middleware)

# Authentication dependency
security = HTTPBearer(auto_error=False)


async def get_current_user(request: Request, token=Depends(security)) -> User:
    """Get current authenticated user with enhanced security."""
    # Try API key first
    api_key = request.headers.get("X-API-Key")
    if api_key:
        user_data = auth_system.verify_api_key(api_key)
        if user_data:
            user = auth_system.users.get(user_data["username"])
            if user and user.is_active:
                # Log API access
                logger.info(
                    f"API key authentication successful for {user.username}",
                    event_type=EventType.SECURITY,
                    user_id=user.id,
                    username=user.username,
                    authentication_method="api_key",
                    ip_address=request.client.host if request.client else None,
                )
                # Record metrics
                metrics.record_auth_attempt("success", "api_key", 0.001)
                return user

    # Try JWT token
    if token:
        from ...presentation.api.auth.jwt_handler import verify_token

        token_data = verify_token(token.credentials)

        if token_data:
            user = auth_system.users.get(token_data.username)
            if user and user.is_active:
                # Log JWT access
                logger.info(
                    f"JWT authentication successful for {user.username}",
                    event_type=EventType.SECURITY,
                    user_id=user.id,
                    username=user.username,
                    authentication_method="jwt",
                    ip_address=request.client.host if request.client else None,
                )
                # Record metrics
                metrics.record_auth_attempt("success", "jwt", 0.005)
                return user

    # Authentication failed
    ip_address = request.client.host if request.client else "unknown"
    logger.warning(
        "Authentication failed - no valid credentials",
        event_type=EventType.SECURITY,
        ip_address=ip_address,
        user_agent=request.headers.get("user-agent", "unknown"),
        authentication_method="unknown",
    )

    # Record failed attempt
    metrics.record_auth_attempt("failed", "unknown", 0.001)
    metrics.record_security_event("auth_failure", "medium")

    raise HTTPException(
        status_code=401,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permissions(*permissions: Permission):
    """Dependency to require specific permissions."""

    def permission_checker(user: User = Depends(get_current_user)):
        for permission in permissions:
            if not user.has_permission(permission):
                logger.warning(
                    f"Permission denied: {user.username} lacks {permission.value}",
                    event_type=EventType.SECURITY,
                    user_id=user.id,
                    username=user.username,
                    required_permission=permission.value,
                )
                metrics.record_security_event("permission_denied", "medium")
                raise HTTPException(
                    status_code=403, detail=f"Permission '{permission.value}' required"
                )
        return user

    return permission_checker


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "llm-ab-testing"}


@app.get("/health/detailed", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with all components."""
    health_checker = get_health_checker()
    health_status = await health_checker.check_health()
    return health_status.to_dict()


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    metrics_data = metrics.get_metrics()
    return JSONResponse(content=metrics_data, media_type=metrics.get_content_type())


# Authentication endpoints
@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login(request: Request, credentials: Dict[str, str]):
    """Enhanced login with security logging and metrics."""
    username = credentials.get("username")
    password = credentials.get("password")
    mfa_code = credentials.get("mfa_code")

    if not username or not password:
        logger.warning("Login attempt with missing credentials", event_type=EventType.SECURITY)
        raise HTTPException(status_code=400, detail="Username and password required")

    ip_address = request.client.host if request.client else "unknown"

    with performance_tracker.start_operation(
        f"auth_login_{username}", "authentication", username=username, ip_address=ip_address
    ):
        success, message, user = auth_system.authenticate_user(
            username, password, ip_address, mfa_code
        )

        if success and user:
            # Create tokens
            tokens = auth_system.create_tokens(user)

            logger.info(
                f"Login successful for {username}",
                event_type=EventType.SECURITY,
                user_id=user.id,
                username=username,
                ip_address=ip_address,
                mfa_used=bool(mfa_code),
            )

            metrics.record_auth_attempt("success", "password", 0.5)
            metrics.record_user_action("login", user.role.value)

            return {
                "access_token": tokens["access_token"],
                "refresh_token": tokens["refresh_token"],
                "token_type": tokens["token_type"],
                "expires_in": tokens["expires_in"],
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "permissions": [p.value for p in user.permissions],
                },
            }
        else:
            logger.warning(
                f"Login failed for {username}: {message}",
                event_type=EventType.SECURITY,
                username=username,
                ip_address=ip_address,
                failure_reason=message,
            )

            metrics.record_auth_attempt("failed", "password", 0.1)
            metrics.record_security_event("auth_failure", "medium")

            raise HTTPException(status_code=401, detail=message)


@app.post("/api/v1/auth/refresh", tags=["Authentication"])
async def refresh_token(request: Request, refresh_data: Dict[str, str]):
    """Refresh access token."""
    refresh_token = refresh_data.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=400, detail="Refresh token required")

    tokens = auth_system.refresh_access_token(refresh_token)
    if tokens:
        logger.info("Token refresh successful", event_type=EventType.SECURITY)
        metrics.record_auth_attempt("success", "refresh", 0.01)
        return tokens
    else:
        logger.warning("Token refresh failed", event_type=EventType.SECURITY)
        metrics.record_auth_attempt("failed", "refresh", 0.01)
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@app.post("/api/v1/auth/mfa/enable", tags=["Authentication"])
async def enable_mfa(user: User = Depends(get_current_user)):
    """Enable MFA for current user."""
    success, qr_uri, backup_codes = auth_system.enable_mfa(user.username)

    if success:
        logger.info(
            f"MFA enabled for user {user.username}",
            event_type=EventType.SECURITY,
            user_id=user.id,
            username=user.username,
        )

        metrics.record_user_action("enable_mfa", user.role.value)

        return {
            "qr_code_uri": qr_uri,
            "backup_codes": backup_codes,
            "message": "MFA enabled successfully",
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to enable MFA")


@app.post("/api/v1/auth/api-keys", tags=["Authentication"])
async def create_api_key(
    key_data: Dict[str, str], user: User = Depends(require_permissions(Permission.API_ACCESS))
):
    """Create API key for user."""
    name = key_data.get("name", "API Key")

    success, api_key = auth_system.create_api_key(user.id, name)

    if success:
        logger.info(
            f"API key created for user {user.username}",
            event_type=EventType.SECURITY,
            user_id=user.id,
            username=user.username,
            key_name=name,
        )

        metrics.record_user_action("create_api_key", user.role.value)

        return {"api_key": api_key, "name": name, "message": "API key created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create API key")


# Example protected endpoint
@app.get("/api/v1/tests", tags=["Tests"])
async def list_tests(user: User = Depends(require_permissions(Permission.READ_TEST))):
    """List tests with security and monitoring."""
    with performance_tracker.start_operation(
        "list_tests", "database", user_id=user.id, username=user.username
    ):
        # Simulate test listing
        tests = [
            {"id": "test_1", "name": "Test 1", "status": "active"},
            {"id": "test_2", "name": "Test 2", "status": "completed"},
        ]

        logger.info(
            f"Tests listed by user {user.username}",
            event_type=EventType.BUSINESS,
            user_id=user.id,
            username=user.username,
            tests_count=len(tests),
        )

        metrics.record_user_action("list_tests", user.role.value)

        return {"tests": tests, "total": len(tests)}


@app.post("/api/v1/tests", tags=["Tests"])
async def create_test(
    test_data: Dict[str, Any], user: User = Depends(require_permissions(Permission.CREATE_TEST))
):
    """Create new test with monitoring."""
    with performance_tracker.start_operation(
        "create_test", "business", user_id=user.id, username=user.username
    ):
        # Simulate test creation
        test_id = f"test_{int(time.time())}"

        logger.info(
            f"Test created by user {user.username}",
            event_type=EventType.BUSINESS,
            user_id=user.id,
            username=user.username,
            test_id=test_id,
            test_name=test_data.get("name"),
        )

        metrics.record_test_created(user.role.value)
        metrics.record_user_action("create_test", user.role.value)

        return {
            "id": test_id,
            "name": test_data.get("name"),
            "status": "created",
            "created_by": user.username,
        }


# Global exception handler with logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with security logging."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        event_type=EventType.ERROR,
        exception_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        ip_address=request.client.host if request.client else None,
        exc_info=True,
    )

    metrics.record_error("global", type(exc).__name__)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "code": "INTERNAL_ERROR",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# Request/response logging middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all HTTP requests with performance metrics."""
    start_time = time.time()

    # Generate request ID if not present
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Set logging context
    logger.set_context(request_id=request_id, correlation_id=request_id)

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Log request
        logger.log_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
            request_id=request_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent", "unknown"),
        )

        # Record metrics
        metrics.record_http_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration,
        )

        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"

        return response

    except Exception as e:
        duration = time.time() - start_time

        logger.error(
            f"Request failed: {str(e)}",
            event_type=EventType.ERROR,
            method=request.method,
            path=request.url.path,
            duration=duration,
            request_id=request_id,
            ip_address=request.client.host if request.client else None,
            exc_info=True,
        )

        metrics.record_http_request(
            method=request.method, endpoint=request.url.path, status_code=500, duration=duration
        )

        raise


if __name__ == "__main__":
    # Run with enhanced monitoring
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main_enhanced:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") != "production",
        log_config=None,  # Use our custom logging
        access_log=False,  # We handle this in middleware
    )
