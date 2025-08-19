"""Comprehensive audit logging and security monitoring system."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"

    # Data events
    DATA_CREATED = "data_created"
    DATA_READ = "data_read"
    DATA_UPDATED = "data_updated"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"

    # Security events
    SECURITY_VIOLATION = "security_violation"
    ATTACK_DETECTED = "attack_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    IP_BLOCKED = "ip_blocked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGED = "configuration_changed"
    ERROR_OCCURRED = "error_occurred"

    # API events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_RATE_LIMIT = "api_rate_limit"

    # Test management events
    TEST_CREATED = "test_created"
    TEST_STARTED = "test_started"
    TEST_STOPPED = "test_stopped"
    TEST_DELETED = "test_deleted"
    TEST_RESULTS_ACCESSED = "test_results_accessed"


class AuditLevel(str, Enum):
    """Audit logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceStandard(str, Enum):
    """Compliance standards for audit requirements."""

    SOX = "sox"  # Sarbanes-Oxley
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    ISO27001 = "iso27001"  # ISO/IEC 27001
    SOC2 = "soc2"  # Service Organization Control 2


@dataclass
class AuditContext:
    """Context information for audit events."""

    user_id: Optional[str] = None
    username: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    api_key_id: Optional[str] = None
    correlation_id: Optional[str] = None
    source_system: str = "llm_ab_platform"


@dataclass
class AuditEvent:
    """Comprehensive audit event record."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: AuditEventType = AuditEventType.SYSTEM_STARTUP
    level: AuditLevel = AuditLevel.INFO
    message: str = ""

    # Context information
    context: AuditContext = field(default_factory=AuditContext)

    # Event details
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"  # success, failure, pending

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Security information
    risk_score: float = 0.0  # 0.0 - 1.0
    compliance_tags: Set[ComplianceStandard] = field(default_factory=set)

    # Data protection
    contains_pii: bool = False
    contains_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        data["timestamp"] = self.timestamp.isoformat()
        # Convert sets to lists for JSON serialization
        data["compliance_tags"] = list(self.compliance_tags)
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=None)


class SecurityIncident:
    """Security incident tracking."""

    def __init__(self, incident_id: str, severity: str, description: str):
        self.incident_id = incident_id
        self.severity = severity  # low, medium, high, critical
        self.description = description
        self.created_at = datetime.utcnow()
        self.events: List[AuditEvent] = []
        self.status = "open"  # open, investigating, resolved, closed
        self.assigned_to: Optional[str] = None
        self.resolution: Optional[str] = None
        self.resolved_at: Optional[datetime] = None

    def add_event(self, event: AuditEvent):
        """Add related audit event to incident."""
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "incident_id": self.incident_id,
            "severity": self.severity,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "assigned_to": self.assigned_to,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "event_count": len(self.events),
            "latest_event": self.events[-1].timestamp.isoformat() if self.events else None,
        }


class AuditLogger:
    """Comprehensive audit logging system."""

    def __init__(
        self,
        log_directory: str = "./audit_logs",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        retention_days: int = 90,
    ):
        self.log_directory = Path(log_directory)
        self.max_file_size = max_file_size
        self.retention_days = retention_days

        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)

        # Current log file
        self.current_log_file = None
        self._rotate_log_file()

        # In-memory incident tracking
        self.incidents: Dict[str, SecurityIncident] = {}
        self.event_buffer: List[AuditEvent] = []
        self.buffer_size = 1000

        # Security monitoring
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_ips: Set[str] = set()
        self.threat_indicators: Dict[str, int] = {}

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        asyncio.create_task(self._periodic_flush())
        asyncio.create_task(self._periodic_cleanup())
        asyncio.create_task(self._security_monitoring())

    async def _periodic_flush(self):
        """Periodically flush events to disk."""
        while True:
            try:
                await asyncio.sleep(30)  # Flush every 30 seconds
                await self._flush_events()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic flush: {e}")

    async def _periodic_cleanup(self):
        """Periodically clean up old logs and data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                await self._cleanup_old_logs()
                self._cleanup_old_incidents()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def _security_monitoring(self):
        """Monitor for security patterns and incidents."""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                await self._analyze_security_patterns()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")

    def _rotate_log_file(self):
        """Rotate log file if needed."""
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        log_filename = f"audit_{current_date}.jsonl"
        log_path = self.log_directory / log_filename

        # Check if current file needs rotation
        if self.current_log_file != log_path or (
            log_path.exists() and log_path.stat().st_size > self.max_file_size
        ):

            # Create new file with timestamp if size exceeded
            if log_path.exists() and log_path.stat().st_size > self.max_file_size:
                timestamp = datetime.utcnow().strftime("%H%M%S")
                log_filename = f"audit_{current_date}_{timestamp}.jsonl"
                log_path = self.log_directory / log_filename

            self.current_log_file = log_path

    async def _flush_events(self):
        """Flush buffered events to disk."""
        if not self.event_buffer:
            return

        self._rotate_log_file()

        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                for event in self.event_buffer:
                    f.write(event.to_json() + "\n")

            logger.debug(
                f"Flushed {len(self.event_buffer)} audit events to {self.current_log_file}"
            )
            self.event_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")

    async def _cleanup_old_logs(self):
        """Clean up old log files."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        try:
            for log_file in self.log_directory.glob("audit_*.jsonl"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Deleted old audit log: {log_file}")
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")

    def _cleanup_old_incidents(self):
        """Clean up old resolved incidents."""
        cutoff_date = datetime.utcnow() - timedelta(days=30)

        old_incidents = [
            incident_id
            for incident_id, incident in self.incidents.items()
            if (
                incident.status in ["resolved", "closed"]
                and incident.resolved_at
                and incident.resolved_at < cutoff_date
            )
        ]

        for incident_id in old_incidents:
            del self.incidents[incident_id]

        if old_incidents:
            logger.info(f"Cleaned up {len(old_incidents)} old incidents")

    async def _analyze_security_patterns(self):
        """Analyze patterns for security incidents."""
        current_time = datetime.utcnow()

        # Analyze failed login attempts
        for ip, attempts in list(self.failed_login_attempts.items()):
            # Remove old attempts (older than 1 hour)
            recent_attempts = [
                attempt for attempt in attempts if (current_time - attempt).total_seconds() < 3600
            ]

            if recent_attempts:
                self.failed_login_attempts[ip] = recent_attempts

                # Check for brute force attack
                if len(recent_attempts) >= 10:
                    await self._create_security_incident(
                        severity="high",
                        description=f"Potential brute force attack from IP {ip}",
                        metadata={"ip": ip, "attempts": len(recent_attempts)},
                    )
                    self.suspicious_ips.add(ip)
            else:
                del self.failed_login_attempts[ip]

    async def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        context: AuditContext = None,
        level: AuditLevel = AuditLevel.INFO,
        resource_type: str = None,
        resource_id: str = None,
        action: str = None,
        outcome: str = "success",
        metadata: Dict[str, Any] = None,
        compliance_tags: Set[ComplianceStandard] = None,
        contains_pii: bool = False,
        contains_sensitive: bool = False,
    ):
        """Log an audit event."""

        event = AuditEvent(
            event_type=event_type,
            level=level,
            message=message,
            context=context or AuditContext(),
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            metadata=metadata or {},
            compliance_tags=compliance_tags or set(),
            contains_pii=contains_pii,
            contains_sensitive=contains_sensitive,
        )

        # Calculate risk score
        event.risk_score = self._calculate_risk_score(event)

        # Add to buffer
        self.event_buffer.append(event)

        # Flush if buffer is full
        if len(self.event_buffer) >= self.buffer_size:
            await self._flush_events()

        # Handle security events
        if event.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
            await self._handle_security_event(event)

        # Track specific events for monitoring
        await self._track_security_metrics(event)

        logger.debug(f"Audit event logged: {event_type.value} - {message}")

    def _calculate_risk_score(self, event: AuditEvent) -> float:
        """Calculate risk score for an event."""
        score = 0.0

        # Base score by event type
        high_risk_events = {
            AuditEventType.LOGIN_FAILURE: 0.3,
            AuditEventType.ACCESS_DENIED: 0.4,
            AuditEventType.SECURITY_VIOLATION: 0.8,
            AuditEventType.ATTACK_DETECTED: 0.9,
            AuditEventType.SUSPICIOUS_ACTIVITY: 0.7,
        }

        score += high_risk_events.get(event.event_type, 0.1)

        # Increase score for failure outcomes
        if event.outcome == "failure":
            score += 0.3

        # Increase score for sensitive data
        if event.contains_pii:
            score += 0.2
        if event.contains_sensitive:
            score += 0.2

        # Increase score for suspicious IPs
        if event.context.ip_address in self.suspicious_ips:
            score += 0.3

        # Increase score for repeated events from same IP
        if event.context.ip_address:
            ip_events = self.threat_indicators.get(event.context.ip_address, 0)
            score += min(ip_events * 0.1, 0.5)

        return min(score, 1.0)

    async def _handle_security_event(self, event: AuditEvent):
        """Handle high-risk security events."""
        if event.risk_score >= 0.8:
            await self._create_security_incident(
                severity="critical" if event.risk_score >= 0.9 else "high",
                description=f"High-risk security event: {event.message}",
                metadata=event.metadata,
            )

    async def _track_security_metrics(self, event: AuditEvent):
        """Track security metrics for monitoring."""

        # Track failed login attempts
        if event.event_type == AuditEventType.LOGIN_FAILURE and event.context.ip_address:
            ip = event.context.ip_address
            if ip not in self.failed_login_attempts:
                self.failed_login_attempts[ip] = []
            self.failed_login_attempts[ip].append(event.timestamp)

        # Track threat indicators by IP
        if event.context.ip_address and event.risk_score > 0.5:
            ip = event.context.ip_address
            self.threat_indicators[ip] = self.threat_indicators.get(ip, 0) + 1

    async def _create_security_incident(
        self, severity: str, description: str, metadata: Dict[str, Any] = None
    ):
        """Create a new security incident."""
        incident_id = str(uuid.uuid4())
        incident = SecurityIncident(incident_id, severity, description)

        if metadata:
            incident.metadata = metadata

        self.incidents[incident_id] = incident

        # Log the incident creation (without triggering security event handling to prevent recursion)
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            message=f"Security incident created: {description}",
            level=AuditLevel.CRITICAL,
            metadata={"incident_id": incident_id, "severity": severity},
        )

        # Add directly to buffer without triggering security handlers
        self.event_buffer.append(event)

        logger.warning(f"Security incident created: {incident_id} - {description}")

    async def log_authentication_event(
        self,
        event_type: AuditEventType,
        username: str,
        ip_address: str,
        outcome: str = "success",
        additional_info: Dict[str, Any] = None,
    ):
        """Log authentication-related events."""
        context = AuditContext(
            username=username, ip_address=ip_address, source_system="auth_system"
        )

        await self.log_event(
            event_type=event_type,
            message=f"Authentication event: {username} from {ip_address}",
            context=context,
            level=AuditLevel.WARNING if outcome == "failure" else AuditLevel.INFO,
            action="authenticate",
            outcome=outcome,
            metadata=additional_info or {},
            compliance_tags={ComplianceStandard.SOX, ComplianceStandard.SOC2},
        )

    async def log_data_access(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        outcome: str = "success",
        contains_pii: bool = False,
    ):
        """Log data access events."""
        context = AuditContext(user_id=user_id)

        await self.log_event(
            event_type=(
                AuditEventType.DATA_READ if action == "read" else AuditEventType.DATA_UPDATED
            ),
            message=f"Data {action}: {resource_type}#{resource_id}",
            context=context,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            contains_pii=contains_pii,
            compliance_tags={ComplianceStandard.GDPR} if contains_pii else set(),
        )

    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security monitoring dashboard data."""
        current_time = datetime.utcnow()

        # Recent incidents (last 24 hours)
        recent_incidents = [
            incident.to_dict()
            for incident in self.incidents.values()
            if (current_time - incident.created_at).total_seconds() < 86400
        ]

        # Failed login attempts (last hour)
        recent_failed_logins = sum(
            len(
                [attempt for attempt in attempts if (current_time - attempt).total_seconds() < 3600]
            )
            for attempts in self.failed_login_attempts.values()
        )

        # Threat indicators
        high_risk_ips = [ip for ip, count in self.threat_indicators.items() if count >= 5]

        return {
            "dashboard_generated": current_time.isoformat(),
            "incidents": {
                "total_open": len([i for i in self.incidents.values() if i.status == "open"]),
                "recent_24h": len(recent_incidents),
                "critical_open": len(
                    [
                        i
                        for i in self.incidents.values()
                        if i.status == "open" and i.severity == "critical"
                    ]
                ),
            },
            "authentication": {
                "failed_logins_last_hour": recent_failed_logins,
                "suspicious_ips": len(self.suspicious_ips),
                "blocked_ips": len(high_risk_ips),
            },
            "monitoring": {
                "events_buffered": len(self.event_buffer),
                "threat_indicators": len(self.threat_indicators),
                "high_risk_ips": len(high_risk_ips),
            },
            "compliance": {
                "audit_retention_days": self.retention_days,
                "log_directory": str(self.log_directory),
                "current_log_file": str(self.current_log_file) if self.current_log_file else None,
            },
        }

    async def shutdown(self):
        """Shutdown audit logger and flush remaining events."""
        logger.info("Shutting down audit logger...")
        await self._flush_events()

        # Log shutdown event
        await self.log_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            message="Audit logging system shutdown",
            level=AuditLevel.INFO,
        )

        # Final flush
        await self._flush_events()


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


async def init_audit_logging(log_directory: str = "./audit_logs") -> AuditLogger:
    """Initialize audit logging system."""
    global _audit_logger
    _audit_logger = AuditLogger(log_directory)

    # Log system startup
    await _audit_logger.log_event(
        event_type=AuditEventType.SYSTEM_STARTUP,
        message="Audit logging system initialized",
        level=AuditLevel.INFO,
        compliance_tags={ComplianceStandard.SOX, ComplianceStandard.SOC2},
    )

    return _audit_logger
