"""Rate limiting value objects."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..exceptions import ValidationError
from .provider_type import ProviderType


@dataclass
class RateLimits:
    """Rate limiting configuration and state."""

    requests_per_minute: int
    requests_per_day: int
    current_minute_count: int = 0
    current_day_count: int = 0
    last_reset_minute: Optional[datetime] = None
    last_reset_day: Optional[datetime] = None

    def __post_init__(self):
        """Initialize rate limits after creation."""
        if self.requests_per_minute < 0:
            raise ValidationError("requests_per_minute cannot be negative")

        if self.requests_per_day < 0:
            raise ValidationError("requests_per_day cannot be negative")

        if self.current_minute_count < 0:
            raise ValidationError("current_minute_count cannot be negative")

        if self.current_day_count < 0:
            raise ValidationError("current_day_count cannot be negative")

        # Initialize reset times if not provided
        now = datetime.utcnow()
        if self.last_reset_minute is None:
            self.last_reset_minute = now
        if self.last_reset_day is None:
            self.last_reset_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits."""
        # Auto-reset counters if time has passed
        self._auto_reset()

        return (
            self.current_minute_count < self.requests_per_minute
            and self.current_day_count < self.requests_per_day
        )

    def record_request(self) -> None:
        """Record a request and update counters."""
        if not self.can_make_request():
            raise ValidationError("Cannot record request when rate limit exceeded")

        self.current_minute_count += 1
        self.current_day_count += 1

    def reset_minute_counter(self) -> None:
        """Reset the minute counter."""
        self.current_minute_count = 0
        self.last_reset_minute = datetime.utcnow()

    def reset_day_counter(self) -> None:
        """Reset the day counter."""
        self.current_day_count = 0
        self.last_reset_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    def time_until_reset(self) -> int:
        """Get seconds until next minute reset."""
        now = datetime.utcnow()
        next_minute = self.last_reset_minute + timedelta(minutes=1)

        if now >= next_minute:
            return 0

        return int((next_minute - now).total_seconds())

    def time_until_day_reset(self) -> int:
        """Get seconds until next day reset."""
        now = datetime.utcnow()
        next_day = self.last_reset_day + timedelta(days=1)

        if now >= next_day:
            return 0

        return int((next_day - now).total_seconds())

    def get_remaining_requests_minute(self) -> int:
        """Get remaining requests for current minute."""
        self._auto_reset()
        return max(0, self.requests_per_minute - self.current_minute_count)

    def get_remaining_requests_day(self) -> int:
        """Get remaining requests for current day."""
        self._auto_reset()
        return max(0, self.requests_per_day - self.current_day_count)

    def _auto_reset(self) -> None:
        """Automatically reset counters if time has passed."""
        now = datetime.utcnow()

        # Check if minute should reset
        if now >= (self.last_reset_minute + timedelta(minutes=1)):
            self.reset_minute_counter()

        # Check if day should reset
        next_day = self.last_reset_day + timedelta(days=1)
        if now >= next_day:
            self.reset_day_counter()

    @classmethod
    def default_for_provider(cls, provider_type: ProviderType) -> "RateLimits":
        """Get default rate limits for a provider type."""
        default_limits = {
            ProviderType.OPENAI: cls(requests_per_minute=3500, requests_per_day=10000),
            ProviderType.ANTHROPIC: cls(requests_per_minute=1000, requests_per_day=5000),
            ProviderType.GOOGLE: cls(requests_per_minute=300, requests_per_day=1500),
            ProviderType.BAIDU: cls(requests_per_minute=100, requests_per_day=1000),
            ProviderType.ALIBABA: cls(requests_per_minute=200, requests_per_day=2000),
        }

        return default_limits.get(provider_type, cls(requests_per_minute=60, requests_per_day=1000))

    @classmethod
    def unlimited(cls) -> "RateLimits":
        """Create unlimited rate limits (for testing)."""
        return cls(requests_per_minute=999999, requests_per_day=999999)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"RateLimits(per_minute={self.requests_per_minute}, "
            f"per_day={self.requests_per_day}, "
            f"used_minute={self.current_minute_count}, "
            f"used_day={self.current_day_count})"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_day": self.requests_per_day,
            "current_minute_count": self.current_minute_count,
            "current_day_count": self.current_day_count,
            "remaining_minute": self.get_remaining_requests_minute(),
            "remaining_day": self.get_remaining_requests_day(),
            "reset_minute_in": self.time_until_reset(),
            "reset_day_in": self.time_until_day_reset(),
        }
