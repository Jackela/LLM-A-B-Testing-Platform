"""Tests for Model Provider value objects."""

from decimal import Decimal

import pytest

from src.domain.model_provider.exceptions import RateLimitExceeded, ValidationError
from src.domain.model_provider.value_objects.health_status import HealthStatus
from src.domain.model_provider.value_objects.model_category import ModelCategory
from src.domain.model_provider.value_objects.money import Money
from src.domain.model_provider.value_objects.provider_type import ProviderType
from src.domain.model_provider.value_objects.rate_limits import RateLimits


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_provider_type_values(self):
        """Test all provider type values are available."""
        assert ProviderType.OPENAI
        assert ProviderType.ANTHROPIC
        assert ProviderType.GOOGLE
        assert ProviderType.BAIDU
        assert ProviderType.ALIBABA

    def test_provider_type_string_representation(self):
        """Test string representation of provider types."""
        assert str(ProviderType.OPENAI) == "ProviderType.OPENAI"
        assert ProviderType.OPENAI.value == "openai"


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self):
        """Test all health status values are available."""
        assert HealthStatus.UNKNOWN
        assert HealthStatus.HEALTHY
        assert HealthStatus.UNHEALTHY
        assert HealthStatus.DEGRADED

    def test_health_status_ordering(self):
        """Test health status ordering for severity."""
        assert HealthStatus.UNHEALTHY.severity > HealthStatus.DEGRADED.severity
        assert HealthStatus.DEGRADED.severity > HealthStatus.HEALTHY.severity
        assert HealthStatus.HEALTHY.severity > HealthStatus.UNKNOWN.severity


class TestModelCategory:
    """Tests for ModelCategory enum."""

    def test_model_category_values(self):
        """Test all model category values are available."""
        assert ModelCategory.TEXT_GENERATION
        assert ModelCategory.CHAT_COMPLETION
        assert ModelCategory.CODE_GENERATION
        assert ModelCategory.EMBEDDING


class TestRateLimits:
    """Tests for RateLimits value object."""

    def test_create_rate_limits(self):
        """Test creating rate limits."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=0,
            current_day_count=0,
        )

        assert rate_limits.requests_per_minute == 60
        assert rate_limits.requests_per_day == 1000
        assert rate_limits.current_minute_count == 0
        assert rate_limits.current_day_count == 0

    def test_can_make_request_within_limits(self):
        """Test can_make_request returns True when within limits."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=30,
            current_day_count=500,
        )

        assert rate_limits.can_make_request() is True

    def test_can_make_request_minute_limit_exceeded(self):
        """Test can_make_request returns False when minute limit exceeded."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=60,
            current_day_count=500,
        )

        assert rate_limits.can_make_request() is False

    def test_can_make_request_day_limit_exceeded(self):
        """Test can_make_request returns False when day limit exceeded."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=30,
            current_day_count=1000,
        )

        assert rate_limits.can_make_request() is False

    def test_record_request(self):
        """Test recording a request updates counters."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=30,
            current_day_count=500,
        )

        rate_limits.record_request()

        assert rate_limits.current_minute_count == 31
        assert rate_limits.current_day_count == 501

    def test_reset_minute_counter(self):
        """Test resetting minute counter."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=30,
            current_day_count=500,
        )

        rate_limits.reset_minute_counter()
        assert rate_limits.current_minute_count == 0

    def test_reset_day_counter(self):
        """Test resetting day counter."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=30,
            current_day_count=500,
        )

        rate_limits.reset_day_counter()
        assert rate_limits.current_day_count == 0

    def test_default_for_provider_openai(self):
        """Test default rate limits for OpenAI provider."""
        rate_limits = RateLimits.default_for_provider(ProviderType.OPENAI)

        assert rate_limits.requests_per_minute == 3500
        assert rate_limits.requests_per_day == 10000
        assert rate_limits.current_minute_count == 0
        assert rate_limits.current_day_count == 0

    def test_default_for_provider_anthropic(self):
        """Test default rate limits for Anthropic provider."""
        rate_limits = RateLimits.default_for_provider(ProviderType.ANTHROPIC)

        assert rate_limits.requests_per_minute == 1000
        assert rate_limits.requests_per_day == 5000
        assert rate_limits.current_minute_count == 0
        assert rate_limits.current_day_count == 0

    def test_time_until_reset_minute(self):
        """Test calculating time until minute reset."""
        rate_limits = RateLimits(
            requests_per_minute=60,
            requests_per_day=1000,
            current_minute_count=60,
            current_day_count=500,
        )

        time_until_reset = rate_limits.time_until_reset()
        assert time_until_reset <= 60  # Should be within current minute


class TestMoney:
    """Tests for Money value object."""

    def test_create_money(self):
        """Test creating money value object."""
        money = Money(Decimal("10.50"), "USD")

        assert money.amount == Decimal("10.50")
        assert money.currency == "USD"

    def test_money_equality(self):
        """Test money equality comparison."""
        money1 = Money(Decimal("10.50"), "USD")
        money2 = Money(Decimal("10.50"), "USD")
        money3 = Money(Decimal("10.51"), "USD")
        money4 = Money(Decimal("10.50"), "EUR")

        assert money1 == money2
        assert money1 != money3
        assert money1 != money4

    def test_money_addition(self):
        """Test money addition."""
        money1 = Money(Decimal("10.50"), "USD")
        money2 = Money(Decimal("5.25"), "USD")

        result = money1 + money2
        assert result.amount == Decimal("15.75")
        assert result.currency == "USD"

    def test_money_addition_different_currencies_raises_error(self):
        """Test that adding money with different currencies raises error."""
        money1 = Money(Decimal("10.50"), "USD")
        money2 = Money(Decimal("5.25"), "EUR")

        with pytest.raises(ValidationError, match="Cannot add money with different currencies"):
            money1 + money2

    def test_money_subtraction(self):
        """Test money subtraction."""
        money1 = Money(Decimal("10.50"), "USD")
        money2 = Money(Decimal("5.25"), "USD")

        result = money1 - money2
        assert result.amount == Decimal("5.25")
        assert result.currency == "USD"

    def test_money_multiplication(self):
        """Test money multiplication."""
        money = Money(Decimal("10.50"), "USD")

        result = money * 2
        assert result.amount == Decimal("21.00")
        assert result.currency == "USD"

    def test_money_string_representation(self):
        """Test money string representation."""
        money = Money(Decimal("10.50"), "USD")

        assert str(money) == "10.50 USD"

    def test_money_zero(self):
        """Test creating zero money."""
        money = Money.zero("USD")

        assert money.amount == Decimal("0.00")
        assert money.currency == "USD"

    def test_money_is_zero(self):
        """Test checking if money is zero."""
        zero_money = Money.zero("USD")
        non_zero_money = Money(Decimal("10.50"), "USD")

        assert zero_money.is_zero() is True
        assert non_zero_money.is_zero() is False

    def test_money_is_positive(self):
        """Test checking if money is positive."""
        positive_money = Money(Decimal("10.50"), "USD")
        zero_money = Money.zero("USD")
        negative_money = Money(Decimal("-5.00"), "USD")

        assert positive_money.is_positive() is True
        assert zero_money.is_positive() is False
        assert negative_money.is_positive() is False
