"""Response processing service for standardizing provider responses."""

import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ....domain.model_provider.entities.model_config import ModelConfig
from ....domain.model_provider.value_objects.provider_type import ProviderType
from ...dto.model_request_dto import ModelRequestDTO
from ...dto.model_response_dto import ModelResponseDTO, ResponseStatus
from .cost_calculator import CostCalculator

logger = logging.getLogger(__name__)


class ResponseProcessor:
    """Service for processing and standardizing model provider responses."""

    def __init__(self, cost_calculator: CostCalculator):
        self.cost_calculator = cost_calculator
        self.logger = logger

    async def standardize_response(
        self,
        raw_response: Any,
        request: ModelRequestDTO,
        provider_type: ProviderType,
        model_config: ModelConfig,
        request_start_time: datetime,
    ) -> ModelResponseDTO:
        """
        Standardize a provider response into a ModelResponseDTO.

        Args:
            raw_response: Raw response from the provider
            request: Original request
            provider_type: Type of provider
            model_config: Configuration of the model used
            request_start_time: When the request was initiated

        Returns:
            Standardized response DTO
        """
        try:
            # Convert raw response to dictionary if needed
            if hasattr(raw_response, "__dict__"):
                response_dict = raw_response.__dict__
            elif hasattr(raw_response, "dict"):
                response_dict = raw_response.dict()
            elif isinstance(raw_response, dict):
                response_dict = raw_response
            else:
                response_dict = {"raw": str(raw_response)}

            # Create standardized response
            response_dto = ModelResponseDTO.from_provider_response(
                provider_id=request.provider_id,
                model_id=request.model_id,
                provider_response=response_dict,
                provider_type=provider_type,
                request_start_time=request_start_time,
                request_id=request.request_id,
            )

            # Enhance response with additional processing
            await self._enhance_response(response_dto, request, model_config, provider_type)

            return response_dto

        except Exception as e:
            self.logger.error(
                f"Error standardizing response from {provider_type.value}: {e}",
                extra={
                    "provider": provider_type.value,
                    "model": request.model_id,
                    "request_id": request.request_id,
                },
                exc_info=True,
            )

            return ModelResponseDTO.create_error_response(
                provider_id=request.provider_id,
                model_id=request.model_id,
                error_message=f"Failed to process response: {str(e)}",
                request_id=request.request_id,
            )

    async def _enhance_response(
        self,
        response_dto: ModelResponseDTO,
        request: ModelRequestDTO,
        model_config: ModelConfig,
        provider_type: ProviderType,
    ):
        """Enhance response with additional processing and validation."""

        # Calculate costs if token information is available
        if response_dto.input_tokens and response_dto.output_tokens:
            try:
                cost_breakdown = self.cost_calculator.calculate_cost(
                    provider_type=provider_type,
                    model_id=response_dto.model_id,
                    input_tokens=response_dto.input_tokens,
                    output_tokens=response_dto.output_tokens,
                )

                response_dto.input_cost = cost_breakdown.input_cost
                response_dto.output_cost = cost_breakdown.output_cost
                response_dto.total_cost = cost_breakdown.total_cost

            except Exception as e:
                self.logger.warning(
                    f"Failed to calculate costs for response: {e}",
                    extra={"response_id": response_dto.response_id},
                )

        # Validate response against model configuration
        await self._validate_response(response_dto, model_config)

        # Add retry information if this was a retry
        if request.retry_count > 0:
            response_dto.retry_count = request.retry_count
            response_dto.is_retry = True

        # Enhance with model version if not already present
        if not response_dto.model_version:
            response_dto.model_version = model_config.model_id

    async def _validate_response(self, response_dto: ModelResponseDTO, model_config: ModelConfig):
        """Validate response against model configuration and constraints."""

        # Check token limits
        if response_dto.output_tokens:
            max_tokens = model_config.max_tokens
            if response_dto.output_tokens > max_tokens:
                self.logger.warning(
                    f"Response exceeded model token limit: {response_dto.output_tokens} > {max_tokens}",
                    extra={"model": response_dto.model_id, "response_id": response_dto.response_id},
                )

        # Check for content policy violations (basic check)
        if response_dto.text:
            await self._check_content_policy(response_dto)

    async def _check_content_policy(self, response_dto: ModelResponseDTO):
        """Basic content policy check."""
        # This is a simplified check - in production you might use more sophisticated filters
        potentially_harmful_indicators = [
            "content policy",
            "cannot provide",
            "cannot assist",
            "inappropriate",
            "harmful",
            "unsafe",
        ]

        if response_dto.text and any(
            indicator.lower() in response_dto.text.lower()
            for indicator in potentially_harmful_indicators
        ):
            # Add warning to metadata
            if "content_warnings" not in response_dto.provider_metadata:
                response_dto.provider_metadata["content_warnings"] = []
            response_dto.provider_metadata["content_warnings"].append(
                "Response may contain content policy-related refusal"
            )

    def extract_token_usage(
        self, raw_response: Dict[str, Any], provider_type: ProviderType
    ) -> tuple[Optional[int], Optional[int]]:
        """
        Extract token usage from raw provider response.

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        try:
            if provider_type == ProviderType.OPENAI:
                usage = raw_response.get("usage", {})
                return usage.get("prompt_tokens"), usage.get("completion_tokens")

            elif provider_type == ProviderType.ANTHROPIC:
                usage = raw_response.get("usage", {})
                return usage.get("input_tokens"), usage.get("output_tokens")

            elif provider_type == ProviderType.GOOGLE:
                # Google doesn't always provide token counts directly
                # We might need to estimate or use a different method
                metadata = raw_response.get("usageMetadata", {})
                return metadata.get("promptTokenCount"), metadata.get("candidatesTokenCount")

            else:
                # Generic extraction attempt
                if "usage" in raw_response:
                    usage = raw_response["usage"]
                    return (
                        usage.get("input_tokens") or usage.get("prompt_tokens"),
                        usage.get("output_tokens") or usage.get("completion_tokens"),
                    )

        except Exception as e:
            self.logger.warning(
                f"Failed to extract token usage from {provider_type.value} response: {e}",
                extra={
                    "raw_response_keys": (
                        list(raw_response.keys()) if isinstance(raw_response, dict) else "not_dict"
                    )
                },
            )

        return None, None

    def estimate_token_usage(
        self, prompt: str, response_text: Optional[str], provider_type: ProviderType
    ) -> tuple[int, int]:
        """
        Estimate token usage when not provided by the provider.

        Args:
            prompt: Input prompt
            response_text: Generated response text
            provider_type: Provider type for tokenization specifics

        Returns:
            Tuple of (estimated_input_tokens, estimated_output_tokens)
        """
        # Simple estimation: ~4 characters per token for English text
        # This is approximate and varies by tokenizer and language

        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(response_text) // 4) if response_text else 0

        # Add some buffer for special tokens, system messages, etc.
        input_tokens += 5
        output_tokens += 2

        self.logger.debug(
            f"Estimated token usage for {provider_type.value}: "
            f"{input_tokens} input, {output_tokens} output tokens"
        )

        return input_tokens, output_tokens

    def detect_response_quality_issues(
        self, response_dto: ModelResponseDTO, request: ModelRequestDTO
    ) -> List[str]:
        """
        Detect potential quality issues in the response.

        Returns:
            List of detected issues
        """
        issues = []

        # Check for empty or very short responses
        if not response_dto.text or len(response_dto.text.strip()) < 10:
            issues.append("Response is empty or very short")

        # Check for high latency
        if response_dto.latency_ms and response_dto.latency_ms > 30000:  # 30 seconds
            issues.append(f"High latency: {response_dto.latency_ms}ms")

        # Check for potential errors in response text
        if response_dto.text:
            error_indicators = [
                "error",
                "failed",
                "unable to",
                "cannot process",
                "something went wrong",
                "internal error",
            ]
            if any(
                indicator.lower() in response_dto.text.lower() for indicator in error_indicators
            ):
                issues.append("Response may contain error messages")

        # Check token efficiency
        if (
            response_dto.input_tokens
            and response_dto.output_tokens
            and response_dto.input_tokens > 0
        ):
            ratio = response_dto.output_tokens / response_dto.input_tokens
            if ratio < 0.1:  # Very low output relative to input
                issues.append("Low output token ratio - possible incomplete response")
            elif ratio > 10:  # Very high output relative to input
                issues.append("High output token ratio - possible verbose response")

        # Check for truncation indicators
        if response_dto.finish_reason in ["length", "max_tokens", "truncated"]:
            issues.append(f"Response may be truncated: {response_dto.finish_reason}")

        return issues

    def calculate_response_quality_score(
        self, response_dto: ModelResponseDTO, request: ModelRequestDTO
    ) -> float:
        """
        Calculate a quality score for the response (0.0 to 1.0).

        Returns:
            Quality score where 1.0 is highest quality
        """
        if not response_dto.is_successful():
            return 0.0

        score = 1.0

        # Deduct for quality issues
        issues = self.detect_response_quality_issues(response_dto, request)
        score -= len(issues) * 0.1

        # Bonus for good performance metrics
        if response_dto.latency_ms:
            if response_dto.latency_ms < 2000:  # Under 2 seconds
                score += 0.05
            elif response_dto.latency_ms > 15000:  # Over 15 seconds
                score -= 0.1

        # Bonus for good token efficiency
        if response_dto.tokens_per_second:
            if response_dto.tokens_per_second > 20:  # Good throughput
                score += 0.05
            elif response_dto.tokens_per_second < 5:  # Poor throughput
                score -= 0.05

        # Ensure score is within bounds
        return max(0.0, min(1.0, score))

    async def process_batch_responses(
        self,
        responses: List[tuple[Any, ModelRequestDTO, datetime]],
        provider_type: ProviderType,
        model_config: ModelConfig,
    ) -> List[ModelResponseDTO]:
        """
        Process a batch of responses in parallel.

        Args:
            responses: List of (raw_response, request, start_time) tuples
            provider_type: Provider type
            model_config: Model configuration

        Returns:
            List of standardized response DTOs
        """
        processed_responses = []

        for raw_response, request, start_time in responses:
            try:
                processed_response = await self.standardize_response(
                    raw_response, request, provider_type, model_config, start_time
                )
                processed_responses.append(processed_response)
            except Exception as e:
                self.logger.error(
                    f"Failed to process response in batch: {e}",
                    extra={"request_id": request.request_id},
                    exc_info=True,
                )
                # Create error response
                error_response = ModelResponseDTO.create_error_response(
                    provider_id=request.provider_id,
                    model_id=request.model_id,
                    error_message=f"Batch processing failed: {str(e)}",
                    request_id=request.request_id,
                )
                processed_responses.append(error_response)

        return processed_responses
