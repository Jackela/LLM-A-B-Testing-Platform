"""Cost calculation service for model provider usage."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

from ....domain.model_provider.value_objects.provider_type import ProviderType
from ...dto.model_response_dto import ModelResponseDTO

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for model usage."""

    input_tokens: int
    output_tokens: int
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    provider_id: str
    model_id: str
    calculated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost_per_token": str(self.input_cost_per_token),
            "output_cost_per_token": str(self.output_cost_per_token),
            "input_cost": str(self.input_cost),
            "output_cost": str(self.output_cost),
            "total_cost": str(self.total_cost),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class UsageMetrics:
    """Usage metrics for cost tracking."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: Decimal = Decimal("0.0000")
    average_cost_per_request: Decimal = Decimal("0.0000")
    most_expensive_request: Optional[Decimal] = None
    least_expensive_request: Optional[Decimal] = None

    def update(self, cost: Decimal):
        """Update metrics with new cost."""
        self.total_requests += 1
        self.total_cost += cost
        self.average_cost_per_request = self.total_cost / self.total_requests

        if self.most_expensive_request is None or cost > self.most_expensive_request:
            self.most_expensive_request = cost

        if self.least_expensive_request is None or cost < self.least_expensive_request:
            self.least_expensive_request = cost


class CostCalculator:
    """Service for calculating model usage costs with precise decimal arithmetic."""

    # Default pricing per 1000 tokens (updated as of 2024)
    # These would ideally be loaded from configuration or external pricing API
    DEFAULT_PRICING = {
        ProviderType.OPENAI: {
            "gpt-4-turbo": {
                "input": Decimal("0.0100"),  # $0.01 per 1K input tokens
                "output": Decimal("0.0300"),  # $0.03 per 1K output tokens
            },
            "gpt-4": {"input": Decimal("0.0300"), "output": Decimal("0.0600")},
            "gpt-3.5-turbo": {"input": Decimal("0.0005"), "output": Decimal("0.0015")},
            "gpt-3.5-turbo-16k": {"input": Decimal("0.0030"), "output": Decimal("0.0040")},
        },
        ProviderType.ANTHROPIC: {
            "claude-3-opus": {"input": Decimal("0.0150"), "output": Decimal("0.0750")},
            "claude-3-sonnet": {"input": Decimal("0.0030"), "output": Decimal("0.0150")},
            "claude-3-haiku": {"input": Decimal("0.0002"), "output": Decimal("0.0012")},
        },
        ProviderType.GOOGLE: {
            "gemini-pro": {"input": Decimal("0.0002"), "output": Decimal("0.0006")},
            "gemini-pro-vision": {"input": Decimal("0.0002"), "output": Decimal("0.0006")},
        },
    }

    # Fallback pricing for unknown models
    FALLBACK_PRICING = {
        "input": Decimal("0.0010"),  # Conservative estimate
        "output": Decimal("0.0020"),
    }

    def __init__(self, custom_pricing: Optional[Dict] = None):
        self.pricing = self.DEFAULT_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)

        # Usage tracking
        self.usage_metrics: Dict[str, UsageMetrics] = {}

        self.logger = logger

    def calculate_cost(
        self,
        provider_type: ProviderType,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        custom_rates: Optional[Dict[str, Decimal]] = None,
    ) -> CostBreakdown:
        """
        Calculate precise cost for model usage.

        Args:
            provider_type: Type of provider
            model_id: ID of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            custom_rates: Optional custom pricing rates

        Returns:
            Detailed cost breakdown
        """
        # Get pricing rates
        if custom_rates:
            input_rate = custom_rates.get("input", self.FALLBACK_PRICING["input"])
            output_rate = custom_rates.get("output", self.FALLBACK_PRICING["output"])
        else:
            input_rate, output_rate = self._get_model_pricing(provider_type, model_id)

        # Calculate costs (pricing is per 1000 tokens)
        input_cost = (Decimal(input_tokens) * input_rate) / 1000
        output_cost = (Decimal(output_tokens) * output_rate) / 1000
        total_cost = input_cost + output_cost

        # Round to 4 decimal places for precision
        input_cost = input_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        output_cost = output_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        total_cost = total_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        breakdown = CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_token=input_rate,
            output_cost_per_token=output_rate,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            provider_id=str(provider_type.value),
            model_id=model_id,
            calculated_at=datetime.utcnow(),
        )

        # Update usage metrics
        self._update_usage_metrics(provider_type, model_id, breakdown)

        self.logger.debug(
            f"Cost calculated for {provider_type.value}/{model_id}: "
            f"${total_cost} ({input_tokens} input + {output_tokens} output tokens)",
            extra={"cost_breakdown": breakdown.to_dict()},
        )

        return breakdown

    def calculate_cost_from_response(
        self,
        response: ModelResponseDTO,
        provider_type: ProviderType,
        custom_rates: Optional[Dict[str, Decimal]] = None,
    ) -> Optional[CostBreakdown]:
        """
        Calculate cost from a model response DTO.

        Args:
            response: Model response containing token usage
            provider_type: Type of provider
            custom_rates: Optional custom pricing rates

        Returns:
            Cost breakdown if token information is available, None otherwise
        """
        if response.input_tokens is None or response.output_tokens is None:
            self.logger.warning(
                f"Cannot calculate cost for {response.provider_id}/{response.model_id}: "
                "token information not available",
                extra={"response_id": response.response_id},
            )
            return None

        return self.calculate_cost(
            provider_type,
            response.model_id,
            response.input_tokens,
            response.output_tokens,
            custom_rates,
        )

    def estimate_cost(
        self,
        provider_type: ProviderType,
        model_id: str,
        prompt: str,
        max_tokens: int = 1000,
        custom_rates: Optional[Dict[str, Decimal]] = None,
    ) -> CostBreakdown:
        """
        Estimate cost before making API call.

        Args:
            provider_type: Type of provider
            model_id: ID of the model
            prompt: Input prompt text
            max_tokens: Maximum output tokens requested
            custom_rates: Optional custom pricing rates

        Returns:
            Estimated cost breakdown
        """
        # Estimate input tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_input_tokens = max(1, len(prompt) // 4)

        # Use max_tokens as output estimate (conservative)
        estimated_output_tokens = max_tokens

        return self.calculate_cost(
            provider_type, model_id, estimated_input_tokens, estimated_output_tokens, custom_rates
        )

    def calculate_batch_cost(
        self,
        responses: List[ModelResponseDTO],
        provider_type: ProviderType,
        custom_rates: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate total cost for a batch of responses.

        Args:
            responses: List of model responses
            provider_type: Type of provider
            custom_rates: Optional custom pricing rates

        Returns:
            Batch cost summary
        """
        total_cost = Decimal("0.0000")
        total_input_tokens = 0
        total_output_tokens = 0
        cost_breakdowns = []
        failed_calculations = 0

        for response in responses:
            try:
                breakdown = self.calculate_cost_from_response(response, provider_type, custom_rates)
                if breakdown:
                    cost_breakdowns.append(breakdown)
                    total_cost += breakdown.total_cost
                    total_input_tokens += breakdown.input_tokens
                    total_output_tokens += breakdown.output_tokens
                else:
                    failed_calculations += 1
            except Exception as e:
                self.logger.error(
                    f"Error calculating cost for response {response.response_id}: {e}",
                    exc_info=True,
                )
                failed_calculations += 1

        average_cost = (total_cost / len(cost_breakdowns)) if cost_breakdowns else Decimal("0.0000")

        return {
            "total_cost": str(total_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "average_cost_per_request": str(
                average_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            ),
            "successful_calculations": len(cost_breakdowns),
            "failed_calculations": failed_calculations,
            "cost_breakdowns": [breakdown.to_dict() for breakdown in cost_breakdowns],
            "calculated_at": datetime.utcnow().isoformat(),
        }

    def get_usage_metrics(
        self, provider_type: Optional[ProviderType] = None, model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get usage metrics for tracking and analysis.

        Args:
            provider_type: Optional provider filter
            model_id: Optional model filter

        Returns:
            Usage metrics summary
        """
        if provider_type and model_id:
            key = f"{provider_type.value}_{model_id}"
            metrics = self.usage_metrics.get(key)
            if metrics:
                return {
                    "provider": provider_type.value,
                    "model": model_id,
                    "metrics": {
                        "total_requests": metrics.total_requests,
                        "total_input_tokens": metrics.total_input_tokens,
                        "total_output_tokens": metrics.total_output_tokens,
                        "total_cost": str(metrics.total_cost),
                        "average_cost_per_request": str(metrics.average_cost_per_request),
                        "most_expensive_request": (
                            str(metrics.most_expensive_request)
                            if metrics.most_expensive_request
                            else None
                        ),
                        "least_expensive_request": (
                            str(metrics.least_expensive_request)
                            if metrics.least_expensive_request
                            else None
                        ),
                    },
                }
            else:
                return {"error": f"No metrics found for {provider_type.value}/{model_id}"}

        # Return all metrics
        all_metrics = {}
        for key, metrics in self.usage_metrics.items():
            parts = key.split("_", 1)
            if len(parts) == 2:
                provider, model = parts
                all_metrics[key] = {
                    "provider": provider,
                    "model": model,
                    "total_requests": metrics.total_requests,
                    "total_input_tokens": metrics.total_input_tokens,
                    "total_output_tokens": metrics.total_output_tokens,
                    "total_cost": str(metrics.total_cost),
                    "average_cost_per_request": str(metrics.average_cost_per_request),
                    "most_expensive_request": (
                        str(metrics.most_expensive_request)
                        if metrics.most_expensive_request
                        else None
                    ),
                    "least_expensive_request": (
                        str(metrics.least_expensive_request)
                        if metrics.least_expensive_request
                        else None
                    ),
                }

        return {"metrics": all_metrics, "total_tracked_models": len(all_metrics)}

    def reset_usage_metrics(self):
        """Reset all usage metrics."""
        self.usage_metrics.clear()
        self.logger.info("Usage metrics have been reset")

    def _get_model_pricing(
        self, provider_type: ProviderType, model_id: str
    ) -> tuple[Decimal, Decimal]:
        """Get input and output token pricing for a model."""
        provider_pricing = self.pricing.get(provider_type, {})
        model_pricing = provider_pricing.get(model_id)

        if model_pricing:
            return model_pricing["input"], model_pricing["output"]

        # Try to find similar model (e.g., gpt-4-0613 -> gpt-4)
        for known_model, pricing in provider_pricing.items():
            if model_id.startswith(known_model):
                self.logger.info(
                    f"Using pricing for {known_model} for similar model {model_id}",
                    extra={"provider": provider_type.value, "model": model_id},
                )
                return pricing["input"], pricing["output"]

        # Use fallback pricing
        self.logger.warning(
            f"No pricing found for {provider_type.value}/{model_id}, using fallback rates",
            extra={"provider": provider_type.value, "model": model_id},
        )
        return self.FALLBACK_PRICING["input"], self.FALLBACK_PRICING["output"]

    def _update_usage_metrics(
        self, provider_type: ProviderType, model_id: str, breakdown: CostBreakdown
    ):
        """Update usage metrics with new cost breakdown."""
        key = f"{provider_type.value}_{model_id}"

        if key not in self.usage_metrics:
            self.usage_metrics[key] = UsageMetrics()

        metrics = self.usage_metrics[key]
        metrics.total_input_tokens += breakdown.input_tokens
        metrics.total_output_tokens += breakdown.output_tokens
        metrics.update(breakdown.total_cost)

    def update_pricing(
        self, provider_type: ProviderType, model_id: str, input_rate: Decimal, output_rate: Decimal
    ):
        """Update pricing for a specific model."""
        if provider_type not in self.pricing:
            self.pricing[provider_type] = {}

        self.pricing[provider_type][model_id] = {"input": input_rate, "output": output_rate}

        self.logger.info(
            f"Updated pricing for {provider_type.value}/{model_id}: "
            f"input=${input_rate}/1K, output=${output_rate}/1K"
        )

    def get_model_pricing(
        self, provider_type: ProviderType, model_id: str
    ) -> Optional[Dict[str, Decimal]]:
        """Get current pricing for a model."""
        provider_pricing = self.pricing.get(provider_type, {})
        return provider_pricing.get(model_id)
