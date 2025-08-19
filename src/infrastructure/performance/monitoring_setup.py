"""Monitoring and alerting setup for performance optimization."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .metrics_collector import MetricsCollector
from .performance_manager import get_performance_manager


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertRule:
    """Performance alert rule configuration."""

    name: str
    description: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    severity: AlertSeverity
    evaluation_window_seconds: int = 300  # 5 minutes
    evaluation_interval_seconds: int = 60  # 1 minute
    min_occurrences: int = 3  # Alert after N consecutive breaches
    cooldown_seconds: int = 900  # 15 minutes between alerts
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Performance alert instance."""

    rule: AlertRule
    value: float
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    message: str = ""
    resolution_timestamp: Optional[datetime] = None
    acknowledgment_timestamp: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class PerformanceMonitor:
    """Comprehensive performance monitoring and alerting system."""

    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Monitoring state
        self._monitoring_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._rule_states: Dict[str, Dict[str, Any]] = {}

        self.logger = logging.getLogger(__name__)

        # Setup default alert rules
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self) -> None:
        """Setup default performance alert rules."""

        # Response time alerts
        self.add_alert_rule(
            AlertRule(
                name="high_response_time_p95",
                description="95th percentile response time is above threshold",
                metric_name="http_request_duration_p95",
                condition="gt",
                threshold=2.0,  # 2 seconds
                severity=AlertSeverity.WARNING,
                evaluation_window_seconds=300,
                min_occurrences=3,
                labels={"category": "performance", "type": "response_time"},
            )
        )

        self.add_alert_rule(
            AlertRule(
                name="critical_response_time_p95",
                description="95th percentile response time is critically high",
                metric_name="http_request_duration_p95",
                condition="gt",
                threshold=5.0,  # 5 seconds
                severity=AlertSeverity.CRITICAL,
                evaluation_window_seconds=180,
                min_occurrences=2,
                labels={"category": "performance", "type": "response_time"},
            )
        )

        # Error rate alerts
        self.add_alert_rule(
            AlertRule(
                name="high_error_rate",
                description="HTTP error rate is above acceptable threshold",
                metric_name="http_error_rate",
                condition="gt",
                threshold=0.05,  # 5%
                severity=AlertSeverity.WARNING,
                evaluation_window_seconds=300,
                min_occurrences=3,
                labels={"category": "reliability", "type": "error_rate"},
            )
        )

        self.add_alert_rule(
            AlertRule(
                name="critical_error_rate",
                description="HTTP error rate is critically high",
                metric_name="http_error_rate",
                condition="gt",
                threshold=0.20,  # 20%
                severity=AlertSeverity.CRITICAL,
                evaluation_window_seconds=180,
                min_occurrences=2,
                labels={"category": "reliability", "type": "error_rate"},
            )
        )

        # Cache performance alerts
        self.add_alert_rule(
            AlertRule(
                name="low_cache_hit_rate",
                description="Cache hit rate is below optimal threshold",
                metric_name="cache_hit_rate",
                condition="lt",
                threshold=0.7,  # 70%
                severity=AlertSeverity.WARNING,
                evaluation_window_seconds=600,
                min_occurrences=5,
                labels={"category": "performance", "type": "cache"},
            )
        )

        # Memory usage alerts
        self.add_alert_rule(
            AlertRule(
                name="high_memory_usage",
                description="Memory usage is approaching limits",
                metric_name="memory_usage_percent",
                condition="gt",
                threshold=80.0,  # 80%
                severity=AlertSeverity.WARNING,
                evaluation_window_seconds=300,
                min_occurrences=3,
                labels={"category": "resources", "type": "memory"},
            )
        )

        self.add_alert_rule(
            AlertRule(
                name="critical_memory_usage",
                description="Memory usage is critically high",
                metric_name="memory_usage_percent",
                condition="gt",
                threshold=95.0,  # 95%
                severity=AlertSeverity.CRITICAL,
                evaluation_window_seconds=120,
                min_occurrences=2,
                labels={"category": "resources", "type": "memory"},
            )
        )

        # Database performance alerts
        self.add_alert_rule(
            AlertRule(
                name="slow_database_queries",
                description="Database query response time is high",
                metric_name="database_query_duration_p95",
                condition="gt",
                threshold=0.5,  # 500ms
                severity=AlertSeverity.WARNING,
                evaluation_window_seconds=300,
                min_occurrences=3,
                labels={"category": "performance", "type": "database"},
            )
        )

        # External service alerts
        self.add_alert_rule(
            AlertRule(
                name="external_service_failures",
                description="External service failure rate is high",
                metric_name="external_service_error_rate",
                condition="gt",
                threshold=0.10,  # 10%
                severity=AlertSeverity.WARNING,
                evaluation_window_seconds=300,
                min_occurrences=3,
                labels={"category": "reliability", "type": "external_service"},
            )
        )

        # Circuit breaker alerts
        self.add_alert_rule(
            AlertRule(
                name="circuit_breaker_open",
                description="Circuit breaker has opened",
                metric_name="circuit_breaker_state",
                condition="eq",
                threshold=1.0,  # 1 = open
                severity=AlertSeverity.ERROR,
                evaluation_window_seconds=60,
                min_occurrences=1,
                labels={"category": "reliability", "type": "circuit_breaker"},
            )
        )

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.alert_rules.append(rule)
        self._rule_states[rule.name] = {
            "consecutive_breaches": 0,
            "last_alert_time": None,
            "last_evaluation": None,
        }
        self.logger.info(f"Added alert rule: {rule.name} ({rule.severity.value})")

    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule."""
        for i, rule in enumerate(self.alert_rules):
            if rule.name == rule_name:
                del self.alert_rules[i]
                self._rule_states.pop(rule_name, None)
                self.logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)

    async def start_monitoring(self) -> None:
        """Start performance monitoring and alerting."""
        if self._monitoring_enabled:
            return

        self._monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring_enabled = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        self.logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_enabled:
            try:
                await self._evaluate_alert_rules()
                await self._cleanup_resolved_alerts()
                await asyncio.sleep(30)  # Evaluate every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _evaluate_alert_rules(self) -> None:
        """Evaluate all alert rules against current metrics."""
        if not self.metrics_collector:
            return

        current_time = datetime.utcnow()

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            try:
                # Get current metric value
                metric_value = await self._get_metric_value(rule.metric_name)
                if metric_value is None:
                    continue

                # Evaluate condition
                condition_met = self._evaluate_condition(
                    metric_value, rule.condition, rule.threshold
                )

                # Get rule state
                rule_state = self._rule_states[rule.name]

                if condition_met:
                    rule_state["consecutive_breaches"] += 1

                    # Check if we should trigger an alert
                    if rule_state[
                        "consecutive_breaches"
                    ] >= rule.min_occurrences and self._should_trigger_alert(
                        rule, rule_state, current_time
                    ):

                        await self._trigger_alert(rule, metric_value, current_time)
                        rule_state["last_alert_time"] = current_time
                else:
                    # Reset consecutive breaches
                    if rule_state["consecutive_breaches"] > 0:
                        rule_state["consecutive_breaches"] = 0

                        # Check if we should resolve existing alerts
                        await self._resolve_alerts_for_rule(rule, current_time)

                rule_state["last_evaluation"] = current_time

            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule.name}: {e}", exc_info=True)

    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric."""
        if not self.metrics_collector:
            return None

        try:
            # Get current metrics summary
            metrics_summary = self.metrics_collector.get_metrics_summary()

            # Handle different metric paths
            if metric_name == "http_request_duration_p95":
                return metrics_summary.get("avg_response_time", 0.0)

            elif metric_name == "http_error_rate":
                total_requests = metrics_summary.get("total_requests", 0)
                failed_requests = metrics_summary.get("failed_requests", 0)
                return failed_requests / max(total_requests, 1)

            elif metric_name == "cache_hit_rate":
                # Get from performance manager
                perf_manager = get_performance_manager()
                if perf_manager and perf_manager.cache_manager:
                    cache_stats = await perf_manager.cache_manager.get_stats()
                    cache_metrics = cache_stats.get("metrics", {})
                    hits = cache_metrics.get("hits", 0)
                    misses = cache_metrics.get("misses", 0)
                    return hits / max(hits + misses, 1)
                return 0.0

            elif metric_name == "memory_usage_percent":
                return metrics_summary.get("memory_usage_mb", 0) / 1024  # Convert to %

            elif metric_name == "database_query_duration_p95":
                return metrics_summary.get("avg_response_time", 0.0)  # Placeholder

            elif metric_name == "external_service_error_rate":
                # This would come from external service metrics
                return 0.0  # Placeholder

            elif metric_name == "circuit_breaker_state":
                # This would come from circuit breaker manager
                return 0.0  # 0 = closed, 1 = open, 2 = half-open

            else:
                return metrics_summary.get(metric_name, 0.0)

        except Exception as e:
            self.logger.error(f"Error getting metric value for {metric_name}: {e}")
            return None

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate if a condition is met."""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001  # Float equality
        else:
            return False

    def _should_trigger_alert(
        self, rule: AlertRule, rule_state: Dict[str, Any], current_time: datetime
    ) -> bool:
        """Check if an alert should be triggered based on cooldown period."""
        last_alert_time = rule_state.get("last_alert_time")
        if last_alert_time is None:
            return True

        cooldown_elapsed = (current_time - last_alert_time).total_seconds()
        return cooldown_elapsed >= rule.cooldown_seconds

    async def _trigger_alert(self, rule: AlertRule, value: float, timestamp: datetime) -> None:
        """Trigger an alert for a rule."""
        # Check if there's already an active alert for this rule
        existing_alert = None
        for alert in self.active_alerts:
            if alert.rule.name == rule.name and alert.status == AlertStatus.ACTIVE:
                existing_alert = alert
                break

        if existing_alert:
            # Update existing alert
            existing_alert.value = value
            existing_alert.timestamp = timestamp
        else:
            # Create new alert
            message = f"{rule.description}: {rule.metric_name} = {value:.3f} (threshold: {rule.threshold})"

            alert = Alert(rule=rule, value=value, timestamp=timestamp, message=message)

            self.active_alerts.append(alert)
            self.alert_history.append(alert)

            self.logger.warning(f"ALERT TRIGGERED [{rule.severity.value.upper()}]: {message}")

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert callback: {e}", exc_info=True)

    async def _resolve_alerts_for_rule(self, rule: AlertRule, timestamp: datetime) -> None:
        """Resolve active alerts for a rule."""
        for alert in self.active_alerts:
            if alert.rule.name == rule.name and alert.status == AlertStatus.ACTIVE:

                alert.status = AlertStatus.RESOLVED
                alert.resolution_timestamp = timestamp

                self.logger.info(f"ALERT RESOLVED: {alert.rule.name}")

    async def _cleanup_resolved_alerts(self) -> None:
        """Remove old resolved alerts from active list."""
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=1)  # Keep resolved alerts for 1 hour

        self.active_alerts = [
            alert
            for alert in self.active_alerts
            if (
                alert.status == AlertStatus.ACTIVE
                or (alert.resolution_timestamp and alert.resolution_timestamp > cutoff_time)
            )
        ]

    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if 0 <= alert_id < len(self.active_alerts):
            alert = self.active_alerts[alert_id]
            if alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledgment_timestamp = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by

                self.logger.info(f"Alert acknowledged: {alert.rule.name} by {acknowledged_by}")
                return True

        return False

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = [alert for alert in self.active_alerts if alert.status == AlertStatus.ACTIVE]

        if severity:
            alerts = [alert for alert in alerts if alert.rule.severity == severity]

        return sorted(alerts, key=lambda x: (x.rule.severity.value, x.timestamp), reverse=True)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status."""
        active_alerts = self.get_active_alerts()

        severity_counts = {severity: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.rule.severity] += 1

        return {
            "monitoring_enabled": self._monitoring_enabled,
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules if r.enabled]),
            "active_alerts": len(active_alerts),
            "severity_breakdown": {
                severity.value: count for severity, count in severity_counts.items()
            },
            "recent_alerts": [
                {
                    "rule_name": alert.rule.name,
                    "severity": alert.rule.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "status": alert.status.value,
                }
                for alert in active_alerts[:10]  # Last 10 alerts
            ],
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring system."""
        return {
            "status": "healthy" if self._monitoring_enabled else "disabled",
            "monitoring_enabled": self._monitoring_enabled,
            "alert_rules": len(self.alert_rules),
            "active_alerts": len(self.get_active_alerts()),
            "critical_alerts": len(self.get_active_alerts(AlertSeverity.CRITICAL)),
            "metrics_collector_available": self.metrics_collector is not None,
        }


# Default alert callback implementations


def console_alert_callback(alert: Alert) -> None:
    """Print alerts to console."""
    severity_icons = {
        AlertSeverity.INFO: "ℹ️",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.ERROR: "❌",
        AlertSeverity.CRITICAL: "🚨",
    }

    icon = severity_icons.get(alert.rule.severity, "📊")
    print(f"{icon} [{alert.rule.severity.value.upper()}] {alert.message}")


def logging_alert_callback(alert: Alert) -> None:
    """Log alerts using Python logging."""
    logger = logging.getLogger("performance_alerts")

    severity_levels = {
        AlertSeverity.INFO: logging.INFO,
        AlertSeverity.WARNING: logging.WARNING,
        AlertSeverity.ERROR: logging.ERROR,
        AlertSeverity.CRITICAL: logging.CRITICAL,
    }

    level = severity_levels.get(alert.rule.severity, logging.INFO)
    logger.log(level, f"ALERT: {alert.message}")


# Global monitoring instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """Get global performance monitor instance."""
    return _performance_monitor


async def init_performance_monitor(
    metrics_collector: Optional[MetricsCollector] = None,
    enable_console_alerts: bool = True,
    enable_logging_alerts: bool = True,
) -> PerformanceMonitor:
    """Initialize global performance monitor."""
    global _performance_monitor

    _performance_monitor = PerformanceMonitor(metrics_collector)

    # Add default callbacks
    if enable_console_alerts:
        _performance_monitor.add_alert_callback(console_alert_callback)

    if enable_logging_alerts:
        _performance_monitor.add_alert_callback(logging_alert_callback)

    await _performance_monitor.start_monitoring()

    return _performance_monitor


async def shutdown_performance_monitor() -> None:
    """Shutdown global performance monitor."""
    global _performance_monitor
    if _performance_monitor:
        await _performance_monitor.stop_monitoring()
        _performance_monitor = None
