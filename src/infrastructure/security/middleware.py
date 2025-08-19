"""Security middleware for API protection."""

import hashlib
import ipaddress
import json
import logging
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """Rate limiting configuration."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10,
        window_size: int = 60,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.window_size = window_size


class RateLimiter:
    """Token bucket rate limiter with sliding window."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.clients: Dict[str, Dict] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()

    def _cleanup_old_entries(self):
        """Clean up old rate limit entries."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        cutoff = now - 3600  # Remove entries older than 1 hour
        clients_to_remove = []

        for client_id, data in self.clients.items():
            if data.get("last_request", 0) < cutoff:
                clients_to_remove.append(client_id)

        for client_id in clients_to_remove:
            del self.clients[client_id]

        self.last_cleanup = now
        logger.debug(f"Cleaned up {len(clients_to_remove)} old rate limit entries")

    def is_allowed(self, client_id: str, endpoint: str = None) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limits."""
        self._cleanup_old_entries()

        now = time.time()

        # Initialize client data if not exists
        if client_id not in self.clients:
            self.clients[client_id] = {
                "minute_requests": deque(),
                "hour_requests": deque(),
                "tokens": self.config.burst_size,
                "last_refill": now,
                "last_request": now,
            }

        client_data = self.clients[client_id]
        client_data["last_request"] = now

        # Clean up old requests
        minute_cutoff = now - 60
        hour_cutoff = now - 3600

        while client_data["minute_requests"] and client_data["minute_requests"][0] < minute_cutoff:
            client_data["minute_requests"].popleft()

        while client_data["hour_requests"] and client_data["hour_requests"][0] < hour_cutoff:
            client_data["hour_requests"].popleft()

        # Check minute and hour limits
        if len(client_data["minute_requests"]) >= self.config.requests_per_minute:
            return False, {
                "error": "Rate limit exceeded",
                "limit_type": "per_minute",
                "retry_after": 60 - (now - client_data["minute_requests"][0]),
            }

        if len(client_data["hour_requests"]) >= self.config.requests_per_hour:
            return False, {
                "error": "Rate limit exceeded",
                "limit_type": "per_hour",
                "retry_after": 3600 - (now - client_data["hour_requests"][0]),
            }

        # Token bucket for burst control
        time_passed = now - client_data["last_refill"]
        tokens_to_add = time_passed / 60 * self.config.requests_per_minute
        client_data["tokens"] = min(self.config.burst_size, client_data["tokens"] + tokens_to_add)
        client_data["last_refill"] = now

        if client_data["tokens"] < 1:
            return False, {
                "error": "Rate limit exceeded",
                "limit_type": "burst",
                "retry_after": 60 / self.config.requests_per_minute,
            }

        # Allow request
        client_data["tokens"] -= 1
        client_data["minute_requests"].append(now)
        client_data["hour_requests"].append(now)

        return True, {
            "remaining_minute": self.config.requests_per_minute
            - len(client_data["minute_requests"]),
            "remaining_hour": self.config.requests_per_hour - len(client_data["hour_requests"]),
            "tokens": int(client_data["tokens"]),
        }


class SecurityHeaders:
    """Security headers management."""

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @staticmethod
    def add_security_headers(response: Response) -> Response:
        """Add security headers to response."""
        for header, value in SecurityHeaders.SECURITY_HEADERS.items():
            response.headers[header] = value
        return response


class InputValidator:
    """Input validation and sanitization."""

    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(['\";])",
        r"(--|\*\/|\*)",
        r"(\bOR\b.*=.*|1=1|1=1--|' OR '1'='1)",
    ]

    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>.*?</embed>",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\b(rm|del|format|shutdown|reboot|kill|pkill)\b",
        r"(\.\./|\.\.\\)",
        r"(/etc/passwd|/etc/shadow|C:\\Windows\\System32)",
    ]

    def __init__(self):
        self.sql_regex = re.compile("|".join(self.SQL_INJECTION_PATTERNS), re.IGNORECASE)
        self.xss_regex = re.compile("|".join(self.XSS_PATTERNS), re.IGNORECASE)
        self.cmd_regex = re.compile("|".join(self.COMMAND_INJECTION_PATTERNS), re.IGNORECASE)

    def validate_input(self, data: Any, field_name: str = "") -> tuple[bool, str]:
        """Validate input data for security threats."""
        if isinstance(data, dict):
            for key, value in data.items():
                valid, message = self.validate_input(value, f"{field_name}.{key}")
                if not valid:
                    return False, message

        elif isinstance(data, list):
            for i, item in enumerate(data):
                valid, message = self.validate_input(item, f"{field_name}[{i}]")
                if not valid:
                    return False, message

        elif isinstance(data, str):
            # Check for SQL injection
            if self.sql_regex.search(data):
                return False, f"Potential SQL injection detected in {field_name}"

            # Check for XSS
            if self.xss_regex.search(data):
                return False, f"Potential XSS attack detected in {field_name}"

            # Check for command injection
            if self.cmd_regex.search(data):
                return False, f"Potential command injection detected in {field_name}"

            # Check string length
            if len(data) > 10000:  # 10KB limit
                return False, f"Input too long in {field_name}"

        return True, ""

    def sanitize_string(self, data: str) -> str:
        """Sanitize string input."""
        if not data:
            return data

        # Remove null bytes
        data = data.replace("\x00", "")

        # Escape HTML entities
        html_escape_table = {
            "&": "&amp;",
            '"': "&quot;",
            "'": "&#39;",
            "<": "&lt;",
            ">": "&gt;",
        }

        for char, escape in html_escape_table.items():
            data = data.replace(char, escape)

        return data


class IPWhitelist:
    """IP address whitelist/blacklist management."""

    def __init__(self):
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.whitelist_networks: List = []
        self.blacklist_networks: List = []

    def add_to_whitelist(self, ip_or_network: str):
        """Add IP or network to whitelist."""
        try:
            network = ipaddress.ip_network(ip_or_network, strict=False)
            self.whitelist_networks.append(network)
        except ipaddress.AddressValueError:
            self.whitelist.add(ip_or_network)

    def add_to_blacklist(self, ip_or_network: str):
        """Add IP or network to blacklist."""
        try:
            network = ipaddress.ip_network(ip_or_network, strict=False)
            self.blacklist_networks.append(network)
        except ipaddress.AddressValueError:
            self.blacklist.add(ip_or_network)

    def is_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        # Check blacklist first
        if ip_address in self.blacklist:
            return False

        try:
            ip = ipaddress.ip_address(ip_address)

            # Check blacklist networks
            for network in self.blacklist_networks:
                if ip in network:
                    return False

            # If whitelist is empty, allow all non-blacklisted IPs
            if not self.whitelist and not self.whitelist_networks:
                return True

            # Check whitelist
            if ip_address in self.whitelist:
                return True

            # Check whitelist networks
            for network in self.whitelist_networks:
                if ip in network:
                    return True

            return False

        except ipaddress.AddressValueError:
            return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware."""

    def __init__(
        self,
        app: ASGIApp,
        rate_limit_config: RateLimitConfig = None,
        enable_ip_filtering: bool = True,
        enable_input_validation: bool = True,
    ):
        super().__init__(app)
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.input_validator = InputValidator()
        self.ip_whitelist = IPWhitelist()
        self.enable_ip_filtering = enable_ip_filtering
        self.enable_input_validation = enable_input_validation

        # Exempt paths from rate limiting
        self.exempt_paths = {
            "/health",
            "/api/v1/health",
            "/metrics",
            "/api/v1/auth/login",  # Allow login attempts
        }

        # Add common private networks to whitelist by default
        self.ip_whitelist.add_to_whitelist("127.0.0.0/8")  # Localhost
        self.ip_whitelist.add_to_whitelist("10.0.0.0/8")  # Private
        self.ip_whitelist.add_to_whitelist("172.16.0.0/12")  # Private
        self.ip_whitelist.add_to_whitelist("192.168.0.0/16")  # Private

    async def dispatch(self, request: Request, call_next: Callable):
        """Process request through security middleware."""
        start_time = time.time()

        # Get client IP
        client_ip = self._get_client_ip(request)

        # IP filtering
        if self.enable_ip_filtering and not self.ip_whitelist.is_allowed(client_ip):
            logger.warning(f"Blocked request from blacklisted IP: {client_ip}")
            return JSONResponse(
                status_code=403, content={"error": "Access denied", "code": "IP_BLOCKED"}
            )

        # Skip security checks for exempt paths
        if request.url.path in self.exempt_paths:
            response = await call_next(request)
            return SecurityHeaders.add_security_headers(response)

        # Rate limiting
        client_id = self._get_client_identifier(request, client_ip)
        allowed, rate_info = self.rate_limiter.is_allowed(client_id, request.url.path)

        if not allowed:
            logger.warning(f"Rate limit exceeded for client {client_id}: {rate_info}")
            response = JSONResponse(
                status_code=429,
                content={
                    "error": rate_info.get("error", "Too Many Requests"),
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": int(rate_info.get("retry_after", 60)),
                },
            )
            response.headers["Retry-After"] = str(int(rate_info.get("retry_after", 60)))
            return SecurityHeaders.add_security_headers(response)

        # Input validation for requests with body
        if self.enable_input_validation and request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Get request body
                body = await request.body()
                if body:
                    try:
                        json_data = json.loads(body)
                        valid, message = self.input_validator.validate_input(
                            json_data, "request_body"
                        )
                        if not valid:
                            logger.warning(f"Input validation failed for {client_ip}: {message}")
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "error": "Invalid input detected",
                                    "code": "INVALID_INPUT",
                                    "details": message,
                                },
                            )
                    except json.JSONDecodeError:
                        pass  # Not JSON, skip validation

                # Recreate request with original body
                request._body = body

            except Exception as e:
                logger.error(f"Error validating input: {e}")

        # Process request
        try:
            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Remaining-Minute"] = str(
                rate_info.get("remaining_minute", 0)
            )
            response.headers["X-RateLimit-Remaining-Hour"] = str(rate_info.get("remaining_hour", 0))

            # Add security headers
            response = SecurityHeaders.add_security_headers(response)

            # Log successful request
            process_time = time.time() - start_time
            logger.info(
                f"Security middleware - IP: {client_ip}, "
                f"Path: {request.url.path}, "
                f"Method: {request.method}, "
                f"Status: {response.status_code}, "
                f"Time: {process_time:.3f}s"
            )

            return response

        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            raise

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        # Check X-Forwarded-For header (common in load balancers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to client host
        return request.client.host if request.client else "unknown"

    def _get_client_identifier(self, request: Request, ip: str) -> str:
        """Get unique client identifier for rate limiting."""
        # Use API key if available
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Use bearer token if available
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            return f"token:{hashlib.sha256(token.encode()).hexdigest()[:16]}"

        # Use IP address as fallback
        return f"ip:{ip}"


# Configuration presets
class RateLimitPresets:
    """Predefined rate limiting configurations."""

    DEVELOPMENT = RateLimitConfig(requests_per_minute=300, requests_per_hour=5000, burst_size=50)

    PRODUCTION = RateLimitConfig(requests_per_minute=60, requests_per_hour=1000, burst_size=10)

    API_HEAVY = RateLimitConfig(requests_per_minute=120, requests_per_hour=2000, burst_size=20)

    STRICT = RateLimitConfig(requests_per_minute=30, requests_per_hour=500, burst_size=5)


def create_security_middleware(environment: str = "production") -> SecurityMiddleware:
    """Create security middleware with environment-specific configuration."""
    if environment == "development":
        config = RateLimitPresets.DEVELOPMENT
        enable_ip_filtering = False
    elif environment == "production":
        config = RateLimitPresets.PRODUCTION
        enable_ip_filtering = True
    else:
        config = RateLimitPresets.PRODUCTION
        enable_ip_filtering = True

    return SecurityMiddleware(
        app=None,  # Will be set by FastAPI
        rate_limit_config=config,
        enable_ip_filtering=enable_ip_filtering,
        enable_input_validation=True,
    )
