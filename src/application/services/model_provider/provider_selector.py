"""Intelligent provider selection service for optimal model routing."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ....domain.model_provider.entities.model_provider import ModelProvider
from ....domain.model_provider.value_objects.health_status import HealthStatus
from ....domain.model_provider.value_objects.provider_type import ProviderType

logger = logging.getLogger(__name__)


class SelectionCriteria(Enum):
    """Criteria for provider selection."""

    COST_OPTIMIZATION = "cost_optimization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    RELIABILITY_OPTIMIZATION = "reliability_optimization"
    BALANCED = "balanced"


@dataclass
class ProviderScore:
    """Score for a provider based on selection criteria."""

    provider: ModelProvider
    total_score: float
    cost_score: float
    performance_score: float
    reliability_score: float
    availability_score: float
    reasons: List[str]


@dataclass
class SelectionContext:
    """Context for provider selection decisions."""

    model_id: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    priority_level: str = "normal"  # low, normal, high, critical
    max_latency_ms: Optional[int] = None
    max_cost_per_request: Optional[float] = None
    required_features: List[str] = None
    selection_criteria: SelectionCriteria = SelectionCriteria.BALANCED


class ProviderSelector:
    """Service for intelligent provider selection and routing."""

    # Default weights for balanced selection
    DEFAULT_WEIGHTS = {"cost": 0.25, "performance": 0.25, "reliability": 0.25, "availability": 0.25}

    # Criteria-specific weights
    CRITERIA_WEIGHTS = {
        SelectionCriteria.COST_OPTIMIZATION: {
            "cost": 0.50,
            "performance": 0.15,
            "reliability": 0.20,
            "availability": 0.15,
        },
        SelectionCriteria.PERFORMANCE_OPTIMIZATION: {
            "cost": 0.15,
            "performance": 0.50,
            "reliability": 0.20,
            "availability": 0.15,
        },
        SelectionCriteria.RELIABILITY_OPTIMIZATION: {
            "cost": 0.15,
            "performance": 0.20,
            "reliability": 0.50,
            "availability": 0.15,
        },
        SelectionCriteria.BALANCED: DEFAULT_WEIGHTS,
    }

    def __init__(self):
        self.logger = logger
        self.performance_history: Dict[str, List[float]] = {}  # provider_id -> latency history
        self.reliability_history: Dict[str, List[bool]] = {}  # provider_id -> success history
        self.cost_history: Dict[str, List[float]] = {}  # provider_id -> cost history

    def select_provider(
        self, available_providers: List[ModelProvider], context: SelectionContext
    ) -> Optional[ProviderScore]:
        """
        Select the best provider based on the given context and criteria.

        Args:
            available_providers: List of available providers
            context: Selection context with requirements and preferences

        Returns:
            Best provider with score details, or None if no suitable provider
        """
        if not available_providers:
            self.logger.warning("No providers available for selection")
            return None

        # Filter providers that support the requested model
        suitable_providers = self._filter_suitable_providers(available_providers, context)

        if not suitable_providers:
            self.logger.warning(
                f"No providers support model {context.model_id}",
                extra={
                    "model_id": context.model_id,
                    "available_providers": len(available_providers),
                },
            )
            return None

        # Score each suitable provider
        scored_providers = []
        for provider in suitable_providers:
            score = self._score_provider(provider, context)
            if score:
                scored_providers.append(score)

        if not scored_providers:
            self.logger.warning("No providers received positive scores")
            return None

        # Sort by total score (descending)
        scored_providers.sort(key=lambda x: x.total_score, reverse=True)

        best_provider = scored_providers[0]

        self.logger.info(
            f"Selected provider {best_provider.provider.name} for model {context.model_id} "
            f"(score: {best_provider.total_score:.3f})",
            extra={
                "provider": best_provider.provider.name,
                "model": context.model_id,
                "score": best_provider.total_score,
                "reasons": best_provider.reasons,
            },
        )

        return best_provider

    def select_providers_for_load_balancing(
        self,
        available_providers: List[ModelProvider],
        context: SelectionContext,
        max_providers: int = 3,
    ) -> List[ProviderScore]:
        """
        Select multiple providers for load balancing.

        Args:
            available_providers: List of available providers
            context: Selection context
            max_providers: Maximum number of providers to select

        Returns:
            List of providers sorted by score
        """
        # Get all suitable providers with scores
        suitable_providers = self._filter_suitable_providers(available_providers, context)

        scored_providers = []
        for provider in suitable_providers:
            score = self._score_provider(provider, context)
            if score and score.total_score > 0.3:  # Minimum threshold for load balancing
                scored_providers.append(score)

        # Sort by score and return top N
        scored_providers.sort(key=lambda x: x.total_score, reverse=True)
        selected = scored_providers[:max_providers]

        self.logger.info(
            f"Selected {len(selected)} providers for load balancing: "
            f"{[p.provider.name for p in selected]}",
            extra={
                "model": context.model_id,
                "providers": [{"name": p.provider.name, "score": p.total_score} for p in selected],
            },
        )

        return selected

    def _filter_suitable_providers(
        self, providers: List[ModelProvider], context: SelectionContext
    ) -> List[ModelProvider]:
        """Filter providers that meet the basic requirements."""
        suitable = []

        for provider in providers:
            # Check if provider supports the model
            model_config = provider.find_model_config(context.model_id)
            if not model_config:
                continue

            # Check availability and health
            if not provider.health_status.is_operational:
                continue

            # Check rate limits
            if not provider.rate_limits.can_make_request():
                continue

            # Check required features
            if context.required_features:
                if not all(
                    model_config.supports_parameter(feature)
                    for feature in context.required_features
                ):
                    continue

            # Check token limits
            total_estimated_tokens = (
                context.estimated_input_tokens + context.estimated_output_tokens
            )
            if total_estimated_tokens > model_config.max_tokens:
                continue

            suitable.append(provider)

        return suitable

    def _score_provider(
        self, provider: ModelProvider, context: SelectionContext
    ) -> Optional[ProviderScore]:
        """Score a provider based on the selection criteria."""
        try:
            # Get model configuration
            model_config = provider.find_model_config(context.model_id)
            if not model_config:
                return None

            # Calculate individual scores
            cost_score = self._calculate_cost_score(provider, model_config, context)
            performance_score = self._calculate_performance_score(provider, context)
            reliability_score = self._calculate_reliability_score(provider, context)
            availability_score = self._calculate_availability_score(provider, context)

            # Get weights for the selection criteria
            weights = self.CRITERIA_WEIGHTS.get(context.selection_criteria, self.DEFAULT_WEIGHTS)

            # Calculate weighted total score
            total_score = (
                cost_score * weights["cost"]
                + performance_score * weights["performance"]
                + reliability_score * weights["reliability"]
                + availability_score * weights["availability"]
            )

            # Collect reasons for the score
            reasons = []
            if cost_score > 0.8:
                reasons.append("low cost")
            if performance_score > 0.8:
                reasons.append("high performance")
            if reliability_score > 0.8:
                reasons.append("high reliability")
            if availability_score > 0.8:
                reasons.append("high availability")

            return ProviderScore(
                provider=provider,
                total_score=total_score,
                cost_score=cost_score,
                performance_score=performance_score,
                reliability_score=reliability_score,
                availability_score=availability_score,
                reasons=reasons,
            )

        except Exception as e:
            self.logger.error(
                f"Error scoring provider {provider.name}: {e}",
                extra={"provider": provider.name, "model": context.model_id},
                exc_info=True,
            )
            return None

    def _calculate_cost_score(
        self, provider: ModelProvider, model_config, context: SelectionContext
    ) -> float:
        """Calculate cost score (1.0 = lowest cost, 0.0 = highest cost)."""
        try:
            # Calculate estimated cost for this request
            input_cost = (context.estimated_input_tokens * model_config.cost_per_input_token) / 1000
            output_cost = (
                context.estimated_output_tokens * model_config.cost_per_output_token
            ) / 1000
            total_cost = input_cost + output_cost

            # Check against maximum cost constraint
            if context.max_cost_per_request and total_cost > context.max_cost_per_request:
                return 0.0  # Exceeds cost limit

            # Score based on historical cost data or model configuration
            # For now, use a simple inverse relationship with cost
            # In production, you might compare against other providers

            # Normalize cost (lower cost = higher score)
            # This is a simplified scoring - you'd want to compare against market rates
            if total_cost <= 0.001:  # Very cheap
                return 1.0
            elif total_cost <= 0.01:  # Reasonable
                return 0.8
            elif total_cost <= 0.1:  # Moderate
                return 0.6
            elif total_cost <= 1.0:  # Expensive
                return 0.4
            else:  # Very expensive
                return 0.2

        except Exception as e:
            self.logger.warning(f"Error calculating cost score for {provider.name}: {e}")
            return 0.5  # Neutral score

    def _calculate_performance_score(
        self, provider: ModelProvider, context: SelectionContext
    ) -> float:
        """Calculate performance score (1.0 = best performance, 0.0 = worst)."""
        provider_id = str(provider.id)

        # Check historical performance data
        if provider_id in self.performance_history:
            recent_latencies = self.performance_history[provider_id][-10:]  # Last 10 requests
            if recent_latencies:
                avg_latency = sum(recent_latencies) / len(recent_latencies)

                # Check against latency constraint
                if context.max_latency_ms and avg_latency > context.max_latency_ms:
                    return 0.0

                # Score based on latency (lower = better)
                if avg_latency <= 1000:  # Under 1s
                    return 1.0
                elif avg_latency <= 3000:  # Under 3s
                    return 0.8
                elif avg_latency <= 10000:  # Under 10s
                    return 0.6
                elif avg_latency <= 30000:  # Under 30s
                    return 0.4
                else:  # Over 30s
                    return 0.2

        # Default score based on provider health status
        if provider.health_status == HealthStatus.HEALTHY:
            return 0.8
        elif provider.health_status == HealthStatus.DEGRADED:
            return 0.6
        else:
            return 0.4

    def _calculate_reliability_score(
        self, provider: ModelProvider, context: SelectionContext
    ) -> float:
        """Calculate reliability score (1.0 = most reliable, 0.0 = least reliable)."""
        provider_id = str(provider.id)

        # Check historical reliability data
        if provider_id in self.reliability_history:
            recent_attempts = self.reliability_history[provider_id][-20:]  # Last 20 requests
            if recent_attempts:
                success_rate = sum(recent_attempts) / len(recent_attempts)

                if success_rate >= 0.98:  # 98%+ success rate
                    return 1.0
                elif success_rate >= 0.95:  # 95%+ success rate
                    return 0.9
                elif success_rate >= 0.90:  # 90%+ success rate
                    return 0.8
                elif success_rate >= 0.80:  # 80%+ success rate
                    return 0.6
                else:  # Below 80%
                    return 0.3

        # Default score based on provider health status
        if provider.health_status == HealthStatus.HEALTHY:
            return 0.9
        elif provider.health_status == HealthStatus.DEGRADED:
            return 0.7
        else:
            return 0.3

    def _calculate_availability_score(
        self, provider: ModelProvider, context: SelectionContext
    ) -> float:
        """Calculate availability score (1.0 = fully available, 0.0 = unavailable)."""
        # Check current rate limit status
        rate_limit_factor = 1.0
        if provider.rate_limits:
            remaining_requests = (
                provider.rate_limits.requests_per_minute - provider.rate_limits.current_minute_count
            )
            if remaining_requests <= 0:
                return 0.0  # No capacity
            elif remaining_requests < 5:
                rate_limit_factor = 0.5  # Low capacity
            elif remaining_requests < 20:
                rate_limit_factor = 0.8  # Moderate capacity

        # Check health status
        health_factor = 1.0
        if provider.health_status == HealthStatus.HEALTHY:
            health_factor = 1.0
        elif provider.health_status == HealthStatus.DEGRADED:
            health_factor = 0.7
        elif provider.health_status == HealthStatus.UNHEALTHY:
            health_factor = 0.3
        else:  # UNKNOWN
            health_factor = 0.5

        return rate_limit_factor * health_factor

    def record_performance_metrics(
        self,
        provider: ModelProvider,
        latency_ms: float,
        success: bool,
        cost: Optional[float] = None,
    ):
        """Record performance metrics for future selection decisions."""
        provider_id = str(provider.id)

        # Record latency
        if provider_id not in self.performance_history:
            self.performance_history[provider_id] = []
        self.performance_history[provider_id].append(latency_ms)

        # Keep only recent history (last 50 requests)
        if len(self.performance_history[provider_id]) > 50:
            self.performance_history[provider_id] = self.performance_history[provider_id][-50:]

        # Record reliability
        if provider_id not in self.reliability_history:
            self.reliability_history[provider_id] = []
        self.reliability_history[provider_id].append(success)

        # Keep only recent history (last 100 requests)
        if len(self.reliability_history[provider_id]) > 100:
            self.reliability_history[provider_id] = self.reliability_history[provider_id][-100:]

        # Record cost if provided
        if cost is not None:
            if provider_id not in self.cost_history:
                self.cost_history[provider_id] = []
            self.cost_history[provider_id].append(cost)

            # Keep only recent history (last 100 requests)
            if len(self.cost_history[provider_id]) > 100:
                self.cost_history[provider_id] = self.cost_history[provider_id][-100:]

        self.logger.debug(
            f"Recorded metrics for {provider.name}: latency={latency_ms}ms, success={success}, cost={cost}",
            extra={"provider": provider.name, "provider_id": provider_id},
        )

    def get_provider_statistics(self, provider: ModelProvider) -> Dict[str, Any]:
        """Get statistics for a provider."""
        provider_id = str(provider.id)

        stats = {
            "provider_name": provider.name,
            "provider_id": provider_id,
            "health_status": provider.health_status.name,
            "rate_limit_status": {
                "requests_per_minute": provider.rate_limits.requests_per_minute,
                "current_count": provider.rate_limits.current_minute_count,
                "remaining": provider.rate_limits.requests_per_minute
                - provider.rate_limits.current_minute_count,
            },
        }

        # Add performance stats
        if provider_id in self.performance_history:
            latencies = self.performance_history[provider_id]
            stats["performance"] = {
                "request_count": len(latencies),
                "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
            }

        # Add reliability stats
        if provider_id in self.reliability_history:
            successes = self.reliability_history[provider_id]
            stats["reliability"] = {
                "request_count": len(successes),
                "success_rate": sum(successes) / len(successes) if successes else 0,
                "recent_failures": sum(1 for s in successes[-10:] if not s),  # Last 10 failures
            }

        # Add cost stats
        if provider_id in self.cost_history:
            costs = self.cost_history[provider_id]
            stats["cost"] = {
                "request_count": len(costs),
                "average_cost": sum(costs) / len(costs) if costs else 0,
                "min_cost": min(costs) if costs else 0,
                "max_cost": max(costs) if costs else 0,
            }

        return stats
