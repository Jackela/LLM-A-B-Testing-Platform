"""Advanced rate limiting and DDoS protection system."""

import asyncio
import hashlib
import ipaddress
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class LimitType(str, Enum):
    """Rate limit types."""

    REQUESTS_PER_SECOND = "rps"
    REQUESTS_PER_MINUTE = "rpm"
    REQUESTS_PER_HOUR = "rph"
    REQUESTS_PER_DAY = "rpd"
    BANDWIDTH_PER_MINUTE = "bpm"
    CONCURRENT_CONNECTIONS = "cc"


class ActionType(str, Enum):
    """Actions to take when limits are exceeded."""

    BLOCK = "block"
    DELAY = "delay"
    THROTTLE = "throttle"
    ALERT = "alert"
    CAPTCHA = "captcha"


@dataclass
class RateLimit:
    """Rate limit configuration."""

    limit_type: LimitType
    limit: int
    window_seconds: int
    action: ActionType = ActionType.BLOCK
    delay_seconds: float = 1.0
    burst_limit: Optional[int] = None
    reset_after_seconds: Optional[int] = None


@dataclass
class ClientInfo:
    """Client tracking information."""

    ip_address: str
    user_id: Optional[str] = None
    user_agent: Optional[str] = None
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    blocked_until: Optional[datetime] = None
    violation_count: int = 0
    request_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    bandwidth_used: int = 0
    concurrent_connections: int = 0
    trust_score: float = 1.0  # 0.0 = completely untrusted, 1.0 = fully trusted


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    action: ActionType
    retry_after: Optional[int] = None
    remaining: Optional[int] = None
    reset_time: Optional[datetime] = None
    reason: Optional[str] = None


class ThreatLevel(str, Enum):
    """Threat assessment levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ThreatAssessment:
    """Threat assessment for a client."""

    level: ThreatLevel
    score: float  # 0.0 - 1.0
    indicators: List[str]
    recommendations: List[str]


class AdvancedRateLimiter:
    """Advanced rate limiting with DDoS protection and threat assessment."""

    def __init__(self):
        self.clients: Dict[str, ClientInfo] = {}
        self.blocked_ips: Set[str] = set()
        self.whitelisted_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[RateLimit]] = {
            "default": [
                RateLimit(LimitType.REQUESTS_PER_SECOND, 10, 1),
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 300, 60),
                RateLimit(LimitType.REQUESTS_PER_HOUR, 5000, 3600),
            ],
            "authenticated": [
                RateLimit(LimitType.REQUESTS_PER_SECOND, 20, 1),
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 1000, 60),
                RateLimit(LimitType.REQUESTS_PER_HOUR, 10000, 3600),
            ],
            "api_user": [
                RateLimit(LimitType.REQUESTS_PER_SECOND, 50, 1),
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 2000, 60),
                RateLimit(LimitType.REQUESTS_PER_HOUR, 50000, 3600),
            ],
            "admin": [
                RateLimit(LimitType.REQUESTS_PER_SECOND, 100, 1),
                RateLimit(LimitType.REQUESTS_PER_MINUTE, 5000, 60),
                RateLimit(LimitType.REQUESTS_PER_HOUR, 100000, 3600),
            ],
        }

        # DDoS protection settings
        self.ddos_threshold_rps = 100
        self.ddos_threshold_concurrent = 50
        self.ddos_ban_duration = timedelta(hours=1)
        self.suspicious_patterns = {
            "rapid_requests": {"threshold": 50, "window": 10},
            "user_agent_rotation": {"threshold": 5, "window": 60},
            "multiple_endpoints": {"threshold": 20, "window": 60},
            "error_rate": {"threshold": 0.5, "window": 60},
        }

        # Initialize whitelisted IPs (private networks, etc.)
        self._initialize_whitelist()

        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()

    def _initialize_whitelist(self):
        """Initialize whitelist with common safe IP ranges."""
        safe_ranges = [
            "127.0.0.0/8",  # Localhost
            "10.0.0.0/8",  # Private network
            "172.16.0.0/12",  # Private network
            "192.168.0.0/16",  # Private network
        ]

        for range_str in safe_ranges:
            try:
                network = ipaddress.ip_network(range_str)
                self.whitelisted_ips.update(str(ip) for ip in network)
            except ValueError:
                logger.warning(f"Invalid IP range in whitelist: {range_str}")

    def _start_cleanup_task(self):
        """Start background cleanup task."""

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Cleanup every 5 minutes
                    await self._cleanup_old_entries()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def _cleanup_old_entries(self):
        """Clean up old client entries and expired blocks."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=24)

        # Remove old client entries
        old_clients = [ip for ip, client in self.clients.items() if client.last_seen < cutoff_time]

        for ip in old_clients:
            del self.clients[ip]

        # Remove expired blocks
        self.blocked_ips = {
            ip
            for ip in self.blocked_ips
            if self.clients.get(ip, {}).get("blocked_until", current_time) > current_time
        }

        logger.info(f"Cleaned up {len(old_clients)} old client entries")

    def _get_client_info(
        self, ip_address: str, user_id: str = None, user_agent: str = None
    ) -> ClientInfo:
        """Get or create client information."""
        if ip_address not in self.clients:
            self.clients[ip_address] = ClientInfo(
                ip_address=ip_address, user_id=user_id, user_agent=user_agent
            )

        client = self.clients[ip_address]
        client.last_seen = datetime.utcnow()
        client.request_count += 1

        # Update user info if provided
        if user_id and not client.user_id:
            client.user_id = user_id
        if user_agent and not client.user_agent:
            client.user_agent = user_agent

        return client

    def _calculate_trust_score(self, client: ClientInfo) -> float:
        """Calculate trust score for a client."""
        score = 1.0
        current_time = time.time()

        # Age factor (older clients are more trusted)
        age_hours = (datetime.utcnow() - client.first_seen).total_seconds() / 3600
        age_factor = min(age_hours / 168, 1.0)  # Max trust after 1 week
        score *= 0.5 + 0.5 * age_factor

        # Violation factor
        if client.violation_count > 0:
            violation_factor = max(0.1, 1.0 - (client.violation_count * 0.2))
            score *= violation_factor

        # Request pattern factor
        if len(client.request_history) > 10:
            recent_requests = [
                req
                for req in client.request_history
                if current_time - req["timestamp"] < 300  # Last 5 minutes
            ]

            if len(recent_requests) > 50:  # Very high request rate
                score *= 0.5
            elif len(recent_requests) > 20:  # High request rate
                score *= 0.7

        # Authentication factor
        if client.user_id:
            score *= 1.2  # Authenticated users get bonus

        client.trust_score = max(0.0, min(1.0, score))
        return client.trust_score

    def _check_rate_limits(self, client: ClientInfo, limits: List[RateLimit]) -> RateLimitResult:
        """Check if client exceeds any rate limits."""
        current_time = time.time()

        for limit in limits:
            window_start = current_time - limit.window_seconds

            # Count requests in window
            requests_in_window = sum(
                1 for req in client.request_history if req["timestamp"] >= window_start
            )

            # Check limit
            effective_limit = limit.limit
            if limit.burst_limit and client.trust_score > 0.8:
                effective_limit = limit.burst_limit

            if requests_in_window >= effective_limit:
                # Calculate reset time
                oldest_request = min(
                    req["timestamp"]
                    for req in client.request_history
                    if req["timestamp"] >= window_start
                )
                reset_time = datetime.fromtimestamp(oldest_request + limit.window_seconds)

                return RateLimitResult(
                    allowed=False,
                    action=limit.action,
                    retry_after=int(reset_time.timestamp() - current_time),
                    remaining=0,
                    reset_time=reset_time,
                    reason=f"Rate limit exceeded: {requests_in_window}/{effective_limit} {limit.limit_type.value}",
                )

        # All limits passed
        return RateLimitResult(
            allowed=True,
            action=ActionType.ALERT,
            remaining=min(
                limit.limit
                - sum(
                    1
                    for req in client.request_history
                    if req["timestamp"] >= current_time - limit.window_seconds
                )
                for limit in limits
            ),
        )

    def _assess_threat_level(self, client: ClientInfo) -> ThreatAssessment:
        """Assess threat level for a client."""
        indicators = []
        score = 0.0

        current_time = time.time()

        # Check rapid requests
        recent_requests = [
            req for req in client.request_history if current_time - req["timestamp"] < 10
        ]

        if len(recent_requests) > self.suspicious_patterns["rapid_requests"]["threshold"]:
            indicators.append(f"Rapid requests: {len(recent_requests)} in 10 seconds")
            score += 0.3

        # Check user agent rotation
        if len(client.request_history) > 10:
            user_agents = set(
                req.get("user_agent")
                for req in client.request_history[-50:]
                if req.get("user_agent")
            )
            if len(user_agents) > self.suspicious_patterns["user_agent_rotation"]["threshold"]:
                indicators.append(f"User agent rotation: {len(user_agents)} different agents")
                score += 0.2

        # Check endpoint diversity
        endpoints = set(
            req.get("endpoint") for req in client.request_history[-100:] if req.get("endpoint")
        )
        if len(endpoints) > self.suspicious_patterns["multiple_endpoints"]["threshold"]:
            indicators.append(f"High endpoint diversity: {len(endpoints)} endpoints")
            score += 0.1

        # Check error rate
        recent_errors = sum(
            1 for req in client.request_history[-50:] if req.get("status_code", 200) >= 400
        )
        error_rate = recent_errors / min(len(client.request_history), 50)
        if error_rate > self.suspicious_patterns["error_rate"]["threshold"]:
            indicators.append(f"High error rate: {error_rate:.1%}")
            score += 0.2

        # Check violation history
        if client.violation_count > 3:
            indicators.append(f"Multiple violations: {client.violation_count}")
            score += 0.3

        # Determine threat level
        if score >= 0.8:
            level = ThreatLevel.CRITICAL
        elif score >= 0.6:
            level = ThreatLevel.HIGH
        elif score >= 0.3:
            level = ThreatLevel.MEDIUM
        else:
            level = ThreatLevel.LOW

        # Generate recommendations
        recommendations = []
        if score > 0.5:
            recommendations.append("Increase monitoring frequency")
        if score > 0.7:
            recommendations.append("Consider temporary rate limit reduction")
        if score > 0.8:
            recommendations.append("Implement CAPTCHA challenge")
        if score > 0.9:
            recommendations.append("Consider temporary IP blocking")

        return ThreatAssessment(
            level=level, score=score, indicators=indicators, recommendations=recommendations
        )

    async def check_rate_limit(
        self,
        ip_address: str,
        endpoint: str = "/",
        user_id: str = None,
        user_agent: str = None,
        request_size: int = 0,
        method: str = "GET",
    ) -> RateLimitResult:
        """Check rate limits for a request."""

        # Check if IP is whitelisted
        if ip_address in self.whitelisted_ips:
            return RateLimitResult(allowed=True, action=ActionType.ALERT)

        # Check if IP is blocked
        if ip_address in self.blocked_ips:
            return RateLimitResult(
                allowed=False, action=ActionType.BLOCK, reason="IP address is blocked"
            )

        # Get client information
        client = self._get_client_info(ip_address, user_id, user_agent)

        # Check if client is temporarily blocked
        if client.blocked_until and datetime.utcnow() < client.blocked_until:
            return RateLimitResult(
                allowed=False,
                action=ActionType.BLOCK,
                retry_after=int((client.blocked_until - datetime.utcnow()).total_seconds()),
                reason="Client temporarily blocked",
            )

        # Add request to history
        client.request_history.append(
            {
                "timestamp": time.time(),
                "endpoint": endpoint,
                "user_agent": user_agent,
                "method": method,
                "size": request_size,
            }
        )

        # Calculate trust score
        trust_score = self._calculate_trust_score(client)

        # Determine rate limit category
        if user_id:
            if "admin" in (user_agent or "").lower():
                limit_category = "admin"
            elif "api" in (user_agent or "").lower():
                limit_category = "api_user"
            else:
                limit_category = "authenticated"
        else:
            limit_category = "default"

        # Check rate limits
        limits = self.rate_limits[limit_category]
        result = self._check_rate_limits(client, limits)

        # Assess threat level
        threat = self._assess_threat_level(client)

        # Handle high threat levels
        if threat.level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            logger.warning(f"High threat level detected for {ip_address}: {threat.indicators}")

            if threat.level == ThreatLevel.CRITICAL:
                # Temporarily block client
                client.blocked_until = datetime.utcnow() + timedelta(minutes=15)
                client.violation_count += 1

                return RateLimitResult(
                    allowed=False,
                    action=ActionType.BLOCK,
                    retry_after=900,  # 15 minutes
                    reason=f"Critical threat level: {', '.join(threat.indicators)}",
                )

        # Check for DDoS patterns
        if not result.allowed:
            client.violation_count += 1

            # Auto-block after multiple violations
            if client.violation_count >= 5:
                client.blocked_until = datetime.utcnow() + self.ddos_ban_duration
                self.blocked_ips.add(ip_address)
                logger.warning(
                    f"IP {ip_address} auto-blocked after {client.violation_count} violations"
                )

        return result

    def whitelist_ip(self, ip_address: str) -> bool:
        """Add IP to whitelist."""
        try:
            ipaddress.ip_address(ip_address)
            self.whitelisted_ips.add(ip_address)
            if ip_address in self.blocked_ips:
                self.blocked_ips.remove(ip_address)
            logger.info(f"IP {ip_address} added to whitelist")
            return True
        except ValueError:
            logger.error(f"Invalid IP address: {ip_address}")
            return False

    def block_ip(self, ip_address: str, duration: timedelta = None) -> bool:
        """Block an IP address."""
        try:
            ipaddress.ip_address(ip_address)
            self.blocked_ips.add(ip_address)

            if duration:
                client = self._get_client_info(ip_address)
                client.blocked_until = datetime.utcnow() + duration

            logger.warning(
                f"IP {ip_address} blocked" + (f" for {duration}" if duration else " permanently")
            )
            return True
        except ValueError:
            logger.error(f"Invalid IP address: {ip_address}")
            return False

    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address."""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)

            if ip_address in self.clients:
                self.clients[ip_address].blocked_until = None
                self.clients[ip_address].violation_count = 0

            logger.info(f"IP {ip_address} unblocked")
            return True
        return False

    def get_statistics(self) -> Dict:
        """Get rate limiter statistics."""
        current_time = datetime.utcnow()

        # Active clients (seen in last hour)
        active_clients = sum(
            1
            for client in self.clients.values()
            if (current_time - client.last_seen).total_seconds() < 3600
        )

        # Blocked clients
        blocked_clients = len(self.blocked_ips)
        temp_blocked = sum(
            1
            for client in self.clients.values()
            if client.blocked_until and client.blocked_until > current_time
        )

        # Threat levels
        threat_counts = {level.value: 0 for level in ThreatLevel}
        for client in self.clients.values():
            if (current_time - client.last_seen).total_seconds() < 3600:
                threat = self._assess_threat_level(client)
                threat_counts[threat.level.value] += 1

        return {
            "total_clients": len(self.clients),
            "active_clients": active_clients,
            "blocked_ips": blocked_clients,
            "temporarily_blocked": temp_blocked,
            "whitelisted_ips": len(self.whitelisted_ips),
            "threat_levels": threat_counts,
            "total_requests": sum(client.request_count for client in self.clients.values()),
        }

    async def shutdown(self):
        """Shutdown rate limiter."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


# Global rate limiter instance
_rate_limiter: Optional[AdvancedRateLimiter] = None


def get_rate_limiter() -> AdvancedRateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = AdvancedRateLimiter()
    return _rate_limiter
