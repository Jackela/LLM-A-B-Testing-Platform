"""Health status enumeration."""

from enum import Enum


class HealthStatus(Enum):
    """Enumeration of provider health statuses."""

    UNKNOWN = ("unknown", 0)
    HEALTHY = ("healthy", 1)
    DEGRADED = ("degraded", 2)
    UNHEALTHY = ("unhealthy", 3)

    def __init__(self, value: str, severity: int):
        self.status_value = value
        self.severity = severity

    def __str__(self) -> str:
        """String representation."""
        return f"HealthStatus.{self.name}"

    @property
    def is_operational(self) -> bool:
        """Check if provider is operational (healthy or degraded)."""
        return self in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    @property
    def is_healthy(self) -> bool:
        """Check if provider is fully healthy."""
        return self == HealthStatus.HEALTHY

    @classmethod
    def from_response_time(cls, response_time_ms: int) -> "HealthStatus":
        """Determine health status from response time."""
        if response_time_ms < 1000:  # < 1 second
            return cls.HEALTHY
        elif response_time_ms < 5000:  # < 5 seconds
            return cls.DEGRADED
        else:
            return cls.UNHEALTHY

    @classmethod
    def from_error_rate(cls, error_rate: float) -> "HealthStatus":
        """Determine health status from error rate (0.0 to 1.0)."""
        if error_rate < 0.01:  # < 1% error rate
            return cls.HEALTHY
        elif error_rate < 0.05:  # < 5% error rate
            return cls.DEGRADED
        else:
            return cls.UNHEALTHY
