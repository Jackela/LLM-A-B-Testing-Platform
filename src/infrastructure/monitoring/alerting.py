"""Multi-channel alerting system with escalation and correlation."""

import asyncio
import json
import logging
import smtplib
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import aiofiles
import httpx

from ..security.secrets_manager import get_env_secrets
from .health import HealthStatus, SystemHealth
from .structured_logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertChannel(str, Enum):
    """Alert channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    description: str
    condition: str  # Condition expression
    severity: AlertSeverity
    channels: List[AlertChannel]
    threshold: float = 0.0
    duration_minutes: int = 5  # How long condition must be true
    cooldown_minutes: int = 30  # Minimum time between alerts
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Alert:
    """Alert instance."""

    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    description: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""  # For deduplication

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        if self.resolved_at:
            result["resolved_at"] = self.resolved_at.isoformat()
        return result


@dataclass
class NotificationConfig:
    """Notification channel configuration."""

    channel: AlertChannel
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit_per_hour: int = 100


class AlertCorrelator:
    """Alert correlation and noise reduction."""

    def __init__(self, correlation_window_minutes: int = 10):
        self.correlation_window_minutes = correlation_window_minutes
        self.recent_alerts = deque()  # Recent alerts for correlation
        self.alert_groups = defaultdict(list)  # Grouped alerts by pattern

    def should_suppress_alert(self, alert: Alert) -> bool:
        """Determine if alert should be suppressed due to correlation."""
        now = datetime.utcnow()

        # Clean old alerts
        cutoff = now - timedelta(minutes=self.correlation_window_minutes)
        while self.recent_alerts and self.recent_alerts[0].created_at < cutoff:
            old_alert = self.recent_alerts.popleft()
            # Remove from groups
            for group_key in list(self.alert_groups.keys()):
                if old_alert in self.alert_groups[group_key]:
                    self.alert_groups[group_key].remove(old_alert)
                if not self.alert_groups[group_key]:
                    del self.alert_groups[group_key]

        # Check for similar alerts
        similar_count = 0
        for existing_alert in self.recent_alerts:
            if self._are_alerts_similar(alert, existing_alert):
                similar_count += 1

        # Suppress if too many similar alerts
        if similar_count >= 5:  # More than 5 similar alerts in window
            return True

        # Add to recent alerts
        self.recent_alerts.append(alert)

        # Group by pattern
        group_key = self._get_alert_group_key(alert)
        self.alert_groups[group_key].append(alert)

        return False

    def _are_alerts_similar(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are similar."""
        # Same rule and severity
        if alert1.rule_name == alert2.rule_name and alert1.severity == alert2.severity:
            return True

        # Similar labels
        common_labels = set(alert1.labels.keys()) & set(alert2.labels.keys())
        if len(common_labels) >= 2:  # At least 2 common labels
            matching_values = sum(
                1 for key in common_labels if alert1.labels[key] == alert2.labels[key]
            )
            if matching_values / len(common_labels) >= 0.8:  # 80% matching values
                return True

        return False

    def _get_alert_group_key(self, alert: Alert) -> str:
        """Get grouping key for alert."""
        # Group by rule name and key labels
        key_labels = ["service", "component", "instance"]
        group_parts = [alert.rule_name]

        for label in key_labels:
            if label in alert.labels:
                group_parts.append(f"{label}={alert.labels[label]}")

        return ":".join(group_parts)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alert groups."""
        summary = {}
        for group_key, alerts in self.alert_groups.items():
            summary[group_key] = {
                "count": len(alerts),
                "severities": [alert.severity.value for alert in alerts],
                "latest": max(alerts, key=lambda a: a.created_at).created_at.isoformat(),
            }
        return summary


class NotificationChannel:
    """Base class for notification channels."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.rate_limiter = deque()  # Recent notifications for rate limiting

    def _check_rate_limit(self) -> bool:
        """Check if we can send notification based on rate limits."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=1)

        # Remove old notifications
        while self.rate_limiter and self.rate_limiter[0] < cutoff:
            self.rate_limiter.popleft()

        # Check limit
        if len(self.rate_limiter) >= self.config.rate_limit_per_hour:
            return False

        # Add current notification
        self.rate_limiter.append(now)
        return True

    async def send_notification(self, alert: Alert) -> bool:
        """Send notification for alert."""
        if not self.config.enabled:
            return False

        if not self._check_rate_limit():
            logger.warning(f"Rate limit exceeded for {self.config.channel}")
            return False

        try:
            await self._send_notification(alert)
            return True
        except Exception as e:
            logger.error(f"Failed to send {self.config.channel} notification: {e}")
            return False

    async def _send_notification(self, alert: Alert):
        """Override this method in subclasses."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    async def _send_notification(self, alert: Alert):
        """Send email notification."""
        config = self.config.config

        # Create message
        msg = MIMEMultipart()
        msg["From"] = config["from_email"]
        msg["To"] = ", ".join(config["to_emails"])
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.message}"

        # Create email body
        body = f"""
Alert: {alert.message}

Severity: {alert.severity.upper()}
Rule: {alert.rule_name}
Created: {alert.created_at.isoformat()}
Status: {alert.status.upper()}

Description:
{alert.description}

Labels:
{json.dumps(alert.labels, indent=2)}

Alert ID: {alert.id}
"""

        msg.attach(MIMEText(body, "plain"))

        # Send email
        with smtplib.SMTP(config["smtp_host"], config["smtp_port"]) as server:
            if config.get("use_tls", True):
                server.starttls()
            if config.get("username") and config.get("password"):
                server.login(config["username"], config["password"])
            server.send_message(msg)

        logger.info(f"Email alert sent for {alert.id}")


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""

    async def _send_notification(self, alert: Alert):
        """Send Slack notification."""
        config = self.config.config

        # Color based on severity
        color_map = {
            AlertSeverity.LOW: "#36a64f",  # Green
            AlertSeverity.MEDIUM: "#ff9500",  # Orange
            AlertSeverity.HIGH: "#ff0000",  # Red
            AlertSeverity.CRITICAL: "#800080",  # Purple
        }

        # Create Slack message
        message = {
            "channel": config["channel"],
            "username": config.get("username", "AlertBot"),
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "title": f"[{alert.severity.upper()}] {alert.message}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Rule", "value": alert.rule_name, "short": True},
                        {"title": "Status", "value": alert.status.upper(), "short": True},
                        {"title": "Created", "value": alert.created_at.isoformat(), "short": True},
                        {"title": "Alert ID", "value": alert.id, "short": True},
                    ],
                    "footer": "LLM A/B Testing Platform",
                    "ts": int(alert.created_at.timestamp()),
                }
            ],
        }

        # Send to Slack
        async with httpx.AsyncClient() as client:
            response = await client.post(config["webhook_url"], json=message, timeout=10.0)
            response.raise_for_status()

        logger.info(f"Slack alert sent for {alert.id}")


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""

    async def _send_notification(self, alert: Alert):
        """Send webhook notification."""
        config = self.config.config

        # Prepare payload
        payload = alert.to_dict()

        # Add custom fields if configured
        if "custom_fields" in config:
            payload.update(config["custom_fields"])

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LLM-AB-Testing-AlertManager/1.0",
        }

        # Add custom headers if configured
        if "headers" in config:
            headers.update(config["headers"])

        # Send webhook
        async with httpx.AsyncClient() as client:
            response = await client.post(config["url"], json=payload, headers=headers, timeout=10.0)
            response.raise_for_status()

        logger.info(f"Webhook alert sent for {alert.id}")


class AlertManager:
    """Centralized alert management system."""

    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: Dict[AlertChannel, NotificationChannel] = {}
        self.correlator = AlertCorrelator()
        self.secrets = get_env_secrets()

        # Alert processing
        self._alert_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Setup default rules
        self._setup_default_rules()

        # Setup notification channels
        self._setup_notification_channels()

    def _setup_default_rules(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                description="CPU usage is above 90%",
                condition="cpu_percent > 90",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                threshold=90.0,
                duration_minutes=5,
                labels={"component": "system", "resource": "cpu"},
            ),
            AlertRule(
                name="high_memory_usage",
                description="Memory usage is above 90%",
                condition="memory_percent > 90",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                threshold=90.0,
                duration_minutes=5,
                labels={"component": "system", "resource": "memory"},
            ),
            AlertRule(
                name="database_connection_failed",
                description="Database connection failure",
                condition="database_health == 'unhealthy'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
                duration_minutes=1,
                labels={"component": "database", "service": "postgresql"},
            ),
            AlertRule(
                name="redis_connection_failed",
                description="Redis connection failure",
                condition="redis_health == 'unhealthy'",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                duration_minutes=2,
                labels={"component": "cache", "service": "redis"},
            ),
            AlertRule(
                name="high_error_rate",
                description="High error rate in API requests",
                condition="error_rate > 0.05",  # 5% error rate
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK],
                threshold=0.05,
                duration_minutes=10,
                labels={"component": "api"},
            ),
            AlertRule(
                name="security_incident",
                description="Security incident detected",
                condition="security_event == 'high'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK],
                duration_minutes=0,  # Immediate
                labels={"component": "security"},
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    def _setup_notification_channels(self):
        """Setup notification channels from configuration."""
        # Email channel
        email_config = {
            "smtp_host": self.secrets.get_secret("smtp_host", "SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(self.secrets.get_secret("smtp_port", "SMTP_PORT", "587")),
            "use_tls": True,
            "username": self.secrets.get_secret("smtp_username", "SMTP_USERNAME", required=False),
            "password": self.secrets.get_secret("smtp_password", "SMTP_PASSWORD", required=False),
            "from_email": self.secrets.get_secret(
                "alert_from_email", "ALERT_FROM_EMAIL", "alerts@example.com"
            ),
            "to_emails": self.secrets.get_secret(
                "alert_to_emails", "ALERT_TO_EMAILS", "admin@example.com"
            ).split(","),
        }

        self.notification_channels[AlertChannel.EMAIL] = EmailNotificationChannel(
            NotificationConfig(
                channel=AlertChannel.EMAIL, config=email_config, rate_limit_per_hour=50
            )
        )

        # Slack channel
        slack_webhook = self.secrets.get_secret(
            "slack_webhook_url", "SLACK_WEBHOOK_URL", required=False
        )
        if slack_webhook:
            slack_config = {
                "webhook_url": slack_webhook,
                "channel": self.secrets.get_secret("slack_channel", "SLACK_CHANNEL", "#alerts"),
                "username": "AlertBot",
            }

            self.notification_channels[AlertChannel.SLACK] = SlackNotificationChannel(
                NotificationConfig(
                    channel=AlertChannel.SLACK, config=slack_config, rate_limit_per_hour=100
                )
            )

        # Webhook channel
        webhook_url = self.secrets.get_secret(
            "alert_webhook_url", "ALERT_WEBHOOK_URL", required=False
        )
        if webhook_url:
            webhook_config = {
                "url": webhook_url,
                "headers": {
                    "Authorization": f"Bearer {self.secrets.get_secret('webhook_token', 'WEBHOOK_TOKEN', '')}",
                },
            }

            self.notification_channels[AlertChannel.WEBHOOK] = WebhookNotificationChannel(
                NotificationConfig(
                    channel=AlertChannel.WEBHOOK, config=webhook_config, rate_limit_per_hour=200
                )
            )

    async def start(self):
        """Start alert processing."""
        if self._is_running:
            return

        self._is_running = True
        self._processing_task = asyncio.create_task(self._process_alerts())
        logger.info("Alert manager started")

    async def stop(self):
        """Stop alert processing."""
        self._is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")

    async def trigger_alert(
        self,
        rule_name: str,
        message: str,
        labels: Dict[str, str] = None,
        annotations: Dict[str, Any] = None,
    ):
        """Trigger an alert."""
        if rule_name not in self.alert_rules:
            logger.warning(f"Unknown alert rule: {rule_name}")
            return

        rule = self.alert_rules[rule_name]
        if not rule.enabled:
            return

        # Create alert
        alert_id = f"{rule_name}_{int(time.time())}"
        alert = Alert(
            id=alert_id,
            rule_name=rule_name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            description=rule.description,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            labels={**rule.labels, **(labels or {})},
            annotations=annotations or {},
            fingerprint=self._generate_fingerprint(rule_name, labels or {}),
        )

        # Add to queue
        await self._alert_queue.put(alert)

    async def _process_alerts(self):
        """Process alerts from queue."""
        while self._is_running:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)

                # Check correlation
                if self.correlator.should_suppress_alert(alert):
                    logger.info(f"Alert {alert.id} suppressed due to correlation")
                    continue

                # Store alert
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)

                # Send notifications
                await self._send_notifications(alert)

                logger.info(
                    f"Alert processed: {alert.id}",
                    event_type="system",
                    alert_id=alert.id,
                    rule_name=alert.rule_name,
                    severity=alert.severity.value,
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    async def _send_notifications(self, alert: Alert):
        """Send notifications for alert."""
        rule = self.alert_rules[alert.rule_name]

        # Send to configured channels
        for channel in rule.channels:
            if channel in self.notification_channels:
                try:
                    success = await self.notification_channels[channel].send_notification(alert)
                    if success:
                        logger.info(f"Notification sent via {channel} for alert {alert.id}")
                    else:
                        logger.warning(
                            f"Failed to send notification via {channel} for alert {alert.id}"
                        )
                except Exception as e:
                    logger.error(f"Error sending {channel} notification for alert {alert.id}: {e}")

    def _generate_fingerprint(self, rule_name: str, labels: Dict[str, str]) -> str:
        """Generate alert fingerprint for deduplication."""
        import hashlib

        fingerprint_data = f"{rule_name}:{json.dumps(labels, sort_keys=True)}"
        return hashlib.md5(fingerprint_data.encode()).hexdigest()

    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()

            del self.active_alerts[alert_id]

            logger.info(f"Alert resolved: {alert_id}")

    async def acknowledge_alert(self, alert_id: str):
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.updated_at = datetime.utcnow()

            logger.info(f"Alert acknowledged: {alert_id}")

    def add_alert_rule(self, rule: AlertRule):
        """Add or update alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Alert rule added/updated: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Alert rule removed: {rule_name}")

    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)

        alerts_24h = [a for a in self.alert_history if a.created_at >= last_24h]
        alerts_7d = [a for a in self.alert_history if a.created_at >= last_7d]

        return {
            "active_alerts": len(self.active_alerts),
            "alerts_24h": len(alerts_24h),
            "alerts_7d": len(alerts_7d),
            "alerts_by_severity": {
                severity.value: len(
                    [a for a in self.active_alerts.values() if a.severity == severity]
                )
                for severity in AlertSeverity
            },
            "correlation_summary": self.correlator.get_alert_summary(),
        }


# Integration with health monitoring
async def health_status_changed(old_status: HealthStatus, health: SystemHealth):
    """Handle health status changes and trigger alerts."""
    alert_manager = get_alert_manager()

    # Trigger alerts based on health status
    if health.status == HealthStatus.UNHEALTHY:
        failed_checks = [c.name for c in health.checks if c.status == HealthStatus.UNHEALTHY]

        for check_name in failed_checks:
            rule_name = f"{check_name}_failure"
            if rule_name in alert_manager.alert_rules:
                await alert_manager.trigger_alert(
                    rule_name,
                    f"{check_name.title()} health check failed",
                    labels={"component": check_name, "health_check": "failed"},
                    annotations={"health_status": health.status.value, "message": health.message},
                )

    elif health.status == HealthStatus.DEGRADED:
        degraded_checks = [c.name for c in health.checks if c.status == HealthStatus.DEGRADED]

        for check_name in degraded_checks:
            rule_name = f"{check_name}_degraded"
            if rule_name in alert_manager.alert_rules:
                await alert_manager.trigger_alert(
                    rule_name,
                    f"{check_name.title()} health check degraded",
                    labels={"component": check_name, "health_check": "degraded"},
                    annotations={"health_status": health.status.value, "message": health.message},
                )


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
