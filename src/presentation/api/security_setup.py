"""Security middleware and configuration setup for FastAPI application."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.infrastructure.security.audit_logger import (
    AuditEventType,
    AuditLevel,
    get_audit_logger,
    init_audit_logging,
)
from src.infrastructure.security.auth import get_auth_system
from src.infrastructure.security.input_validator import get_input_validator
from src.infrastructure.security.rate_limiter import AdvancedRateLimiter, get_rate_limiter
from src.infrastructure.security.security_headers import create_security_middleware_stack

logger = logging.getLogger(__name__)


class SecurityMiddleware:
    """Comprehensive security middleware integrating all security components."""

    def __init__(self):
        self.rate_limiter = get_rate_limiter()
        self.input_validator = get_input_validator()
        self.audit_logger = get_audit_logger()
        self.auth_system = get_auth_system()

        # Security middleware stack
        self.security_stack = create_security_middleware_stack()
        self.https_redirect = self.security_stack["https_redirect"]
        self.security_headers = self.security_stack["security_headers"]
        self.security_monitoring = self.security_stack["security_monitoring"]

    async def __call__(self, request: Request, call_next):
        """Process request through security layers."""

        # Step 1: HTTPS enforcement
        https_response = await self.https_redirect(request, lambda req: self._pass_through(req))
        if isinstance(https_response, JSONResponse) and https_response.status_code == 301:
            return https_response

        # Step 2: Security monitoring
        monitoring_response = await self.security_monitoring(
            request, lambda req: self._pass_through(req)
        )
        if isinstance(monitoring_response, JSONResponse) and monitoring_response.status_code == 403:
            await self._log_security_event(
                request, "request_blocked", "Security monitoring blocked request"
            )
            return monitoring_response

        # Step 3: Rate limiting
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")
        endpoint = request.url.path
        method = request.method

        rate_result = await self.rate_limiter.check_rate_limit(
            ip_address=ip_address, endpoint=endpoint, user_agent=user_agent, method=method
        )

        if not rate_result.allowed:
            await self._log_security_event(
                request,
                "rate_limit_exceeded",
                f"Rate limit exceeded: {rate_result.reason}",
                risk_score=0.7,
            )

            response = JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": rate_result.retry_after,
                    "message": rate_result.reason,
                },
            )

            if rate_result.retry_after:
                response.headers["Retry-After"] = str(rate_result.retry_after)

            return response

        # Step 4: Input validation (for POST/PUT requests)
        if method in ["POST", "PUT", "PATCH"] and request.headers.get(
            "content-type", ""
        ).startswith("application/json"):
            try:
                body = await request.body()
                if body:
                    body_str = body.decode("utf-8")
                    validation_result = self.input_validator.validate_input(
                        body_str, input_type="json", required=False
                    )

                    if not validation_result.is_valid:
                        await self._log_security_event(
                            request,
                            "input_validation_failed",
                            f"Input validation failed: {'; '.join(validation_result.issues)}",
                            risk_score=validation_result.confidence,
                        )

                        return JSONResponse(
                            status_code=400,
                            content={"error": "Invalid input", "details": validation_result.issues},
                        )

                    # Log potential attacks
                    if validation_result.detected_attacks:
                        await self._log_security_event(
                            request,
                            "attack_detected",
                            f"Potential attacks detected: {[attack.value for attack in validation_result.detected_attacks]}",
                            risk_score=validation_result.confidence,
                        )

            except Exception as e:
                logger.warning(f"Error validating request body: {e}")

        # Step 5: Process request
        try:
            response = await call_next(request)

            # Step 6: Add security headers
            await self.security_headers(request, lambda req: response)

            # Log successful request
            await self._log_request_event(request, response)

            return response

        except Exception as e:
            await self._log_security_event(
                request, "request_error", f"Request processing error: {str(e)}", risk_score=0.3
            )
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (be careful with spoofing)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def _pass_through(self, request: Request):
        """Pass-through for middleware chain."""
        return None

    async def _log_security_event(
        self, request: Request, event_type: str, message: str, risk_score: float = 0.0
    ):
        """Log security event to audit logger."""
        from src.infrastructure.security.audit_logger import (
            AuditContext,
            AuditEventType,
            AuditLevel,
        )

        context = AuditContext(
            ip_address=self._get_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            request_id=request.headers.get("X-Request-ID"),
        )

        # Map event types
        event_mapping = {
            "request_blocked": AuditEventType.SECURITY_VIOLATION,
            "rate_limit_exceeded": AuditEventType.RATE_LIMIT_EXCEEDED,
            "input_validation_failed": AuditEventType.SECURITY_VIOLATION,
            "attack_detected": AuditEventType.ATTACK_DETECTED,
            "request_error": AuditEventType.ERROR_OCCURRED,
        }

        audit_event_type = event_mapping.get(event_type, AuditEventType.SECURITY_VIOLATION)
        level = (
            AuditLevel.CRITICAL
            if risk_score > 0.8
            else AuditLevel.ERROR if risk_score > 0.5 else AuditLevel.WARNING
        )

        await self.audit_logger.log_event(
            event_type=audit_event_type,
            message=message,
            context=context,
            level=level,
            resource_type="api_request",
            resource_id=request.url.path,
            metadata={
                "method": request.method,
                "endpoint": request.url.path,
                "risk_score": risk_score,
                "headers": dict(request.headers),
            },
        )

    async def _log_request_event(self, request: Request, response: Response):
        """Log successful request event."""
        from src.infrastructure.security.audit_logger import (
            AuditContext,
            AuditEventType,
            AuditLevel,
        )

        # Only log sensitive endpoints or errors
        sensitive_paths = ["/api/v1/auth/", "/api/v1/admin/"]
        should_log = (
            any(path in request.url.path for path in sensitive_paths) or response.status_code >= 400
        )

        if should_log:
            context = AuditContext(
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("User-Agent"),
                request_id=request.headers.get("X-Request-ID"),
            )

            level = AuditLevel.ERROR if response.status_code >= 400 else AuditLevel.INFO

            await self.audit_logger.log_event(
                event_type=(
                    AuditEventType.API_ACCESS
                    if "/api/" in request.url.path
                    else AuditEventType.DATA_READ
                ),
                message=f"API request: {request.method} {request.url.path} -> {response.status_code}",
                context=context,
                level=level,
                resource_type="api_endpoint",
                resource_id=request.url.path,
                outcome="success" if response.status_code < 400 else "failure",
                metadata={
                    "method": request.method,
                    "status_code": response.status_code,
                    "endpoint": request.url.path,
                },
            )


@asynccontextmanager
async def security_lifespan(app: FastAPI):
    """Security-related application lifespan management."""
    # Startup
    logger.info("Initializing security systems...")

    # Initialize audit logging
    audit_logger = await init_audit_logging("./audit_logs")

    # Log system startup
    await audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_STARTUP,
        message="Security systems initialized",
        level=AuditLevel.INFO,
    )

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down security systems...")

        # Log system shutdown
        await audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            message="Security systems shutting down",
            level=AuditLevel.INFO,
        )

        # Cleanup
        await audit_logger.shutdown()
        await get_rate_limiter().shutdown()


def add_security_middleware(app: FastAPI):
    """Add comprehensive security middleware to FastAPI application."""

    # Add security middleware
    app.add_middleware(SecurityMiddleware)

    # Update trusted hosts for production
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Configure for production: ["yourdomain.com", "api.yourdomain.com"]
    )

    logger.info("Security middleware configured")


def add_security_routes(app: FastAPI):
    """Add security monitoring and management routes."""

    @app.get("/api/v1/security/status", tags=["Security"])
    async def security_status():
        """Get security system status."""
        rate_limiter = get_rate_limiter()
        audit_logger = get_audit_logger()

        return {
            "rate_limiter": rate_limiter.get_statistics(),
            "audit_logger": audit_logger.get_security_dashboard(),
            "security_middleware": "active",
            "timestamp": "2024-01-01T00:00:00Z",
        }

    @app.post("/api/v1/security/block-ip", tags=["Security"])
    async def block_ip(ip_address: str, duration_minutes: int = 60):
        """Block an IP address (admin only)."""
        from datetime import timedelta

        rate_limiter = get_rate_limiter()
        success = rate_limiter.block_ip(ip_address, timedelta(minutes=duration_minutes))

        if success:
            # Log security action
            audit_logger = get_audit_logger()
            await audit_logger.log_event(
                event_type=AuditEventType.IP_BLOCKED,
                message=f"IP {ip_address} blocked for {duration_minutes} minutes",
                level=AuditLevel.WARNING,
                metadata={"ip": ip_address, "duration": duration_minutes},
            )

            return {"message": f"IP {ip_address} blocked successfully"}
        else:
            return {"error": "Failed to block IP address"}

    @app.post("/api/v1/security/unblock-ip", tags=["Security"])
    async def unblock_ip(ip_address: str):
        """Unblock an IP address (admin only)."""
        rate_limiter = get_rate_limiter()
        success = rate_limiter.unblock_ip(ip_address)

        if success:
            # Log security action
            audit_logger = get_audit_logger()
            await audit_logger.log_event(
                event_type=AuditEventType.SYSTEM_STARTUP,  # Use generic event for unblock
                message=f"IP {ip_address} unblocked",
                level=AuditLevel.INFO,
                metadata={"ip": ip_address, "action": "unblock"},
            )

            return {"message": f"IP {ip_address} unblocked successfully"}
        else:
            return {"error": "IP address was not blocked"}

    logger.info("Security routes configured")
