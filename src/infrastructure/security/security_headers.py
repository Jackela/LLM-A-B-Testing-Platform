"""Security headers and HTTPS enforcement middleware."""

import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class SecurityHeadersConfig:
    """Configuration for security headers."""

    def __init__(self):
        # Content Security Policy
        self.csp_policy = {
            "default-src": ["'self'"],
            "script-src": ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
            "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            "font-src": ["'self'", "https://fonts.gstatic.com"],
            "img-src": ["'self'", "data:", "https:"],
            "connect-src": ["'self'"],
            "frame-ancestors": ["'none'"],
            "form-action": ["'self'"],
            "base-uri": ["'self'"],
            "object-src": ["'none'"],
            "media-src": ["'self'"],
            "worker-src": ["'self'"],
            "manifest-src": ["'self'"],
        }

        # HTTP Strict Transport Security
        self.hsts_max_age = 31536000  # 1 year
        self.hsts_include_subdomains = True
        self.hsts_preload = True

        # X-Frame-Options
        self.x_frame_options = "DENY"

        # X-Content-Type-Options
        self.x_content_type_options = "nosniff"

        # X-XSS-Protection
        self.x_xss_protection = "1; mode=block"

        # Referrer Policy
        self.referrer_policy = "strict-origin-when-cross-origin"

        # Permissions Policy (formerly Feature Policy)
        self.permissions_policy = {
            "camera": [],
            "microphone": [],
            "geolocation": [],
            "payment": [],
            "usb": [],
            "magnetometer": [],
            "accelerometer": [],
            "gyroscope": [],
            "speaker": [],
            "vibrate": [],
            "fullscreen": ["'self'"],
            "picture-in-picture": [],
        }

        # Cross-Origin policies
        self.cross_origin_embedder_policy = "require-corp"
        self.cross_origin_opener_policy = "same-origin"
        self.cross_origin_resource_policy = "same-site"

        # Cache control for sensitive pages
        self.cache_control_sensitive = "no-store, no-cache, must-revalidate, private"
        self.cache_control_static = "public, max-age=31536000, immutable"

        # Nonce generation for CSP
        self.enable_nonce = True
        self.nonce_length = 16


class SecurityHeadersMiddleware:
    """Middleware for adding security headers."""

    def __init__(self, config: SecurityHeadersConfig = None):
        self.config = config or SecurityHeadersConfig()
        self.nonces: Dict[str, datetime] = {}  # session_id -> nonce creation time
        self.session_nonces: Dict[str, str] = {}  # session_id -> nonce

    def _generate_nonce(self, session_id: str = None) -> str:
        """Generate a new nonce for CSP."""
        nonce = secrets.token_urlsafe(self.config.nonce_length)

        if session_id:
            self.session_nonces[session_id] = nonce
            self.nonces[session_id] = datetime.utcnow()

            # Cleanup old nonces (older than 1 hour)
            cutoff = datetime.utcnow() - timedelta(hours=1)
            expired_sessions = [sid for sid, created in self.nonces.items() if created < cutoff]

            for sid in expired_sessions:
                self.nonces.pop(sid, None)
                self.session_nonces.pop(sid, None)

        return nonce

    def _build_csp_header(self, nonce: str = None) -> str:
        """Build Content Security Policy header."""
        directives = []

        for directive, sources in self.config.csp_policy.items():
            sources_str = " ".join(sources)

            # Add nonce to script-src and style-src if enabled
            if nonce and directive in ["script-src", "style-src"]:
                sources_str += f" 'nonce-{nonce}'"

            directives.append(f"{directive} {sources_str}")

        return "; ".join(directives)

    def _build_permissions_policy_header(self) -> str:
        """Build Permissions Policy header."""
        policies = []

        for feature, allowlist in self.config.permissions_policy.items():
            if allowlist:
                allowlist_str = "(" + " ".join(allowlist) + ")"
            else:
                allowlist_str = "()"

            policies.append(f"{feature}={allowlist_str}")

        return ", ".join(policies)

    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if endpoint contains sensitive data."""
        sensitive_patterns = [
            "/api/v1/auth/",
            "/api/v1/admin/",
            "/api/v1/users/",
            "/api/v1/analytics/",
            "/api/v1/tests/",
        ]

        return any(pattern in path for pattern in sensitive_patterns)

    def _is_static_resource(self, path: str) -> bool:
        """Check if path is a static resource."""
        static_extensions = {
            ".css",
            ".js",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            ".woff",
            ".woff2",
            ".ttf",
        }
        return any(path.endswith(ext) for ext in static_extensions)

    async def __call__(self, request: Request, call_next):
        """Process request and add security headers."""

        # Get session ID from request (if available)
        session_id = request.headers.get("X-Session-ID") or request.cookies.get("session_id")

        # Generate nonce for this request
        nonce = self._generate_nonce(session_id) if self.config.enable_nonce else None

        # Check for HTTPS in production
        if request.url.scheme != "https" and not self._is_development_environment(request):
            # Redirect to HTTPS
            https_url = request.url.replace(scheme="https")
            return JSONResponse(
                status_code=301,
                content={"message": "Redirecting to HTTPS"},
                headers={"Location": str(https_url)},
            )

        # Process the request
        response = await call_next(request)

        # Add security headers
        self._add_security_headers(response, request, nonce)

        return response

    def _is_development_environment(self, request: Request) -> bool:
        """Check if running in development environment."""
        # Allow HTTP for localhost and development
        host = request.client.host if request.client else ""
        return (
            host in ["127.0.0.1", "localhost", "::1"]
            or request.headers.get("X-Development") == "true"
        )

    def _add_security_headers(self, response: Response, request: Request, nonce: str = None):
        """Add all security headers to response."""

        # Content Security Policy
        if nonce or not self.config.enable_nonce:
            csp_header = self._build_csp_header(nonce)
            response.headers["Content-Security-Policy"] = csp_header

        # HTTP Strict Transport Security (only over HTTPS)
        if request.url.scheme == "https":
            hsts_parts = [f"max-age={self.config.hsts_max_age}"]
            if self.config.hsts_include_subdomains:
                hsts_parts.append("includeSubDomains")
            if self.config.hsts_preload:
                hsts_parts.append("preload")

            response.headers["Strict-Transport-Security"] = "; ".join(hsts_parts)

        # X-Frame-Options
        response.headers["X-Frame-Options"] = self.config.x_frame_options

        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = self.config.x_content_type_options

        # X-XSS-Protection
        response.headers["X-XSS-Protection"] = self.config.x_xss_protection

        # Referrer Policy
        response.headers["Referrer-Policy"] = self.config.referrer_policy

        # Permissions Policy
        permissions_header = self._build_permissions_policy_header()
        response.headers["Permissions-Policy"] = permissions_header

        # Cross-Origin policies
        response.headers["Cross-Origin-Embedder-Policy"] = self.config.cross_origin_embedder_policy
        response.headers["Cross-Origin-Opener-Policy"] = self.config.cross_origin_opener_policy
        response.headers["Cross-Origin-Resource-Policy"] = self.config.cross_origin_resource_policy

        # Cache Control based on content type
        path = request.url.path
        if self._is_sensitive_endpoint(path):
            response.headers["Cache-Control"] = self.config.cache_control_sensitive
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        elif self._is_static_resource(path):
            response.headers["Cache-Control"] = self.config.cache_control_static

        # Server header (hide server information)
        response.headers["Server"] = "API-Server"

        # X-Powered-By (remove if exists)
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]

        # Add custom security headers
        response.headers["X-Security-Headers"] = "enabled"
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", secrets.token_hex(8))

        # Add nonce to response if used
        if nonce:
            response.headers["X-CSP-Nonce"] = nonce


class HTTPSRedirectMiddleware:
    """Middleware to enforce HTTPS in production."""

    def __init__(self, force_https: bool = True, exclude_paths: List[str] = None):
        self.force_https = force_https
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]

    async def __call__(self, request: Request, call_next):
        """Redirect HTTP to HTTPS if needed."""

        # Skip HTTPS redirect for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Skip in development
        if self._is_development_environment(request):
            return await call_next(request)

        # Redirect to HTTPS if not already
        if self.force_https and request.url.scheme != "https":
            https_url = request.url.replace(scheme="https")
            return JSONResponse(
                status_code=301,
                content={"message": "HTTPS required"},
                headers={"Location": str(https_url)},
            )

        return await call_next(request)

    def _is_development_environment(self, request: Request) -> bool:
        """Check if running in development environment."""
        host = request.client.host if request.client else ""
        return (
            host in ["127.0.0.1", "localhost", "::1"]
            or request.headers.get("X-Development") == "true"
            or request.headers.get("Host", "").startswith("localhost")
        )


class SecurityMonitoringMiddleware:
    """Middleware for security monitoring and logging."""

    def __init__(self):
        self.suspicious_patterns = [
            # Common attack patterns in URLs
            r"\.\./",
            r"<script",
            r"javascript:",
            r"vbscript:",
            r"onload=",
            r"onerror=",
            r"union.*select",
            r"drop.*table",
            r"/etc/passwd",
            r"/proc/self/environ",
            r"cmd.exe",
            r"powershell",
        ]

        self.blocked_user_agents = [
            "sqlmap",
            "nikto",
            "nmap",
            "masscan",
            "nessus",
            "openvas",
            "zap",
            "burp",
        ]

    async def __call__(self, request: Request, call_next):
        """Monitor request for security issues."""
        start_time = time.time()

        # Log request details
        self._log_request_details(request)

        # Check for suspicious patterns
        security_issues = self._check_security_issues(request)

        if security_issues:
            logger.warning(
                f"Security issues detected in request from {request.client.host}: {security_issues}"
            )

            # For high-risk requests, return 403
            if any("high-risk" in issue for issue in security_issues):
                return JSONResponse(
                    status_code=403, content={"error": "Request blocked for security reasons"}
                )

        # Process request
        response = await call_next(request)

        # Log response details
        processing_time = time.time() - start_time
        self._log_response_details(request, response, processing_time, security_issues)

        return response

    def _log_request_details(self, request: Request):
        """Log detailed request information."""
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'} "
            f"User-Agent: {request.headers.get('User-Agent', 'unknown')}"
        )

    def _check_security_issues(self, request: Request) -> List[str]:
        """Check request for security issues."""
        issues = []

        # Check URL for suspicious patterns
        url_str = str(request.url)
        for pattern in self.suspicious_patterns:
            import re

            if re.search(pattern, url_str, re.IGNORECASE):
                issues.append(f"Suspicious URL pattern: {pattern}")

        # Check User-Agent
        user_agent = request.headers.get("User-Agent", "").lower()
        for blocked_agent in self.blocked_user_agents:
            if blocked_agent in user_agent:
                issues.append(f"high-risk: Blocked user agent: {blocked_agent}")

        # Check for common attack headers
        dangerous_headers = ["X-Forwarded-For", "X-Real-IP", "X-Originating-IP"]
        for header in dangerous_headers:
            if header in request.headers:
                value = request.headers[header]
                if any(char in value for char in ["<", ">", '"', "'"]):
                    issues.append(f"Suspicious characters in {header} header")

        # Check for excessive header count (potential DoS)
        if len(request.headers) > 50:
            issues.append("Excessive header count")

        return issues

    def _log_response_details(
        self,
        request: Request,
        response: Response,
        processing_time: float,
        security_issues: List[str],
    ):
        """Log response details and security events."""

        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "processing_time": f"{processing_time:.3f}s",
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", "unknown")[:100],
            "security_issues": len(security_issues),
        }

        if security_issues:
            log_data["security_details"] = security_issues

        # Log based on status code and security issues
        if response.status_code >= 500:
            logger.error(f"Server error: {log_data}")
        elif response.status_code >= 400:
            logger.warning(f"Client error: {log_data}")
        elif security_issues:
            logger.warning(f"Security event: {log_data}")
        else:
            logger.info(f"Request processed: {log_data}")


def create_security_middleware_stack():
    """Create a complete security middleware stack."""

    # Configuration
    headers_config = SecurityHeadersConfig()

    # Middleware instances
    https_redirect = HTTPSRedirectMiddleware(force_https=False)  # Disabled for development
    security_headers = SecurityHeadersMiddleware(headers_config)
    security_monitoring = SecurityMonitoringMiddleware()

    return {
        "https_redirect": https_redirect,
        "security_headers": security_headers,
        "security_monitoring": security_monitoring,
    }
