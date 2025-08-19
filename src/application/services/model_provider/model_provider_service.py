"""Core Model Provider Service for orchestrating external LLM provider calls."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ....domain.model_provider.entities.model_provider import ModelProvider
from ....domain.model_provider.value_objects.provider_type import ProviderType
from ....domain.test_management.entities.test_sample import TestSample
from ...dto.model_request_dto import BatchModelRequestDTO, ModelRequestDTO
from ...dto.model_response_dto import BatchModelResponseDTO, ModelResponseDTO, ResponseStatus
from ...interfaces.unit_of_work import UnitOfWork
from .circuit_breaker import CircuitBreakerConfig, CircuitBreakerFactory
from .cost_calculator import CostCalculator
from .error_handler import ErrorHandler, ProviderError
from .provider_selector import ProviderSelector, SelectionContext, SelectionCriteria
from .response_processor import ResponseProcessor
from .retry_service import RetryContext, RetryService

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for provider requests."""

    def __init__(self):
        self.request_counts: Dict[str, List[datetime]] = {}
        self.request_limits: Dict[str, int] = {}  # requests per minute

    async def can_proceed(self, provider_id: str, model_id: str) -> bool:
        """Check if request can proceed based on rate limits."""
        key = f"{provider_id}_{model_id}"
        current_time = datetime.utcnow()

        # Initialize if not exists
        if key not in self.request_counts:
            self.request_counts[key] = []
            return True

        # Clean old requests (older than 1 minute)
        minute_ago = current_time.timestamp() - 60
        self.request_counts[key] = [
            req_time for req_time in self.request_counts[key] if req_time.timestamp() > minute_ago
        ]

        # Check limit (default 60 requests per minute if not set)
        limit = self.request_limits.get(key, 60)
        return len(self.request_counts[key]) < limit

    def record_request(self, provider_id: str, model_id: str):
        """Record a request for rate limiting."""
        key = f"{provider_id}_{model_id}"
        current_time = datetime.utcnow()

        if key not in self.request_counts:
            self.request_counts[key] = []

        self.request_counts[key].append(current_time)


class ModelProviderService:
    """Main service for orchestrating model provider calls with full reliability patterns."""

    def __init__(
        self,
        uow: UnitOfWork,
        error_handler: Optional[ErrorHandler] = None,
        circuit_breaker_factory: Optional[CircuitBreakerFactory] = None,
        retry_service: Optional[RetryService] = None,
        response_processor: Optional[ResponseProcessor] = None,
        cost_calculator: Optional[CostCalculator] = None,
        provider_selector: Optional[ProviderSelector] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.uow = uow
        self.error_handler = error_handler or ErrorHandler()
        self.circuit_breaker_factory = circuit_breaker_factory or CircuitBreakerFactory()
        self.retry_service = retry_service or RetryService()
        self.cost_calculator = cost_calculator or CostCalculator()
        self.response_processor = response_processor or ResponseProcessor(self.cost_calculator)
        self.provider_selector = provider_selector or ProviderSelector()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.logger = logger

    async def call_model(
        self,
        provider_id: str,
        model_id: str,
        prompt: str,
        parameters: Dict[str, Any],
        test_context: Optional[Dict[str, Any]] = None,
    ) -> ModelResponseDTO:
        """
        Execute a single model call with full reliability patterns.

        Args:
            provider_id: ID of the provider to use
            model_id: ID of the model to call
            prompt: Input prompt for the model
            parameters: Model parameters (temperature, max_tokens, etc.)
            test_context: Optional context information

        Returns:
            Standardized model response
        """
        request_id = str(uuid4())

        try:
            # Create request DTO
            request = ModelRequestDTO(
                provider_id=provider_id,
                model_id=model_id,
                prompt=prompt,
                parameters=parameters,
                test_context=test_context or {},
                request_id=request_id,
            )

            self.logger.info(
                f"Starting model call: {provider_id}/{model_id}",
                extra={"request_id": request_id, "prompt_length": len(prompt)},
            )

            # Execute with reliability patterns
            response = await self._execute_model_call_with_reliability(request)

            self.logger.info(
                f"Model call completed: {provider_id}/{model_id} (status: {response.status.value})",
                extra={
                    "request_id": request_id,
                    "latency_ms": response.latency_ms,
                    "total_cost": str(response.total_cost) if response.total_cost else None,
                },
            )

            return response

        except Exception as e:
            self.logger.error(
                f"Model call failed: {provider_id}/{model_id}: {e}",
                extra={"request_id": request_id},
                exc_info=True,
            )

            # Create error response
            return ModelResponseDTO.create_error_response(
                provider_id=provider_id,
                model_id=model_id,
                error_message=f"Model call failed: {str(e)}",
                request_id=request_id,
            )

    async def call_model_batch(self, batch_request: BatchModelRequestDTO) -> BatchModelResponseDTO:
        """
        Execute a batch of model calls with parallel processing and reliability patterns.

        Args:
            batch_request: Batch of model requests

        Returns:
            Batch response with all individual results
        """
        batch_id = batch_request.batch_id or str(uuid4())

        self.logger.info(
            f"Starting batch model call: {len(batch_request.requests)} requests",
            extra={"batch_id": batch_id},
        )

        try:
            # Process requests in parallel with concurrency control
            semaphore = asyncio.Semaphore(batch_request.max_parallel_requests)

            async def process_single_request(request: ModelRequestDTO) -> ModelResponseDTO:
                async with semaphore:
                    return await self._execute_model_call_with_reliability(request)

            # Execute all requests
            tasks = [process_single_request(request) for request in batch_request.requests]

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions in the results
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    request = batch_request.requests[i]
                    error_response = ModelResponseDTO.create_error_response(
                        provider_id=request.provider_id,
                        model_id=request.model_id,
                        error_message=f"Batch processing error: {str(response)}",
                        request_id=request.request_id,
                    )
                    processed_responses.append(error_response)
                else:
                    processed_responses.append(response)

            batch_response = BatchModelResponseDTO(batch_id=batch_id, responses=processed_responses)

            self.logger.info(
                f"Batch model call completed: {batch_response.successful_count}/{len(batch_request.requests)} successful",
                extra={
                    "batch_id": batch_id,
                    "success_rate": batch_response.get_success_rate(),
                    "total_cost": (
                        str(batch_response.total_cost) if batch_response.total_cost else None
                    ),
                },
            )

            return batch_response

        except Exception as e:
            self.logger.error(
                f"Batch model call failed: {e}", extra={"batch_id": batch_id}, exc_info=True
            )

            # Create error responses for all requests
            error_responses = [
                ModelResponseDTO.create_error_response(
                    provider_id=request.provider_id,
                    model_id=request.model_id,
                    error_message=f"Batch processing failed: {str(e)}",
                    request_id=request.request_id,
                )
                for request in batch_request.requests
            ]

            return BatchModelResponseDTO(batch_id=batch_id, responses=error_responses)

    async def call_model_with_intelligent_routing(
        self,
        model_id: str,
        prompt: str,
        parameters: Dict[str, Any],
        selection_criteria: SelectionCriteria = SelectionCriteria.BALANCED,
        test_context: Optional[Dict[str, Any]] = None,
    ) -> ModelResponseDTO:
        """
        Call model with intelligent provider selection.

        Args:
            model_id: ID of the model to call
            prompt: Input prompt
            parameters: Model parameters
            selection_criteria: Criteria for provider selection
            test_context: Optional test context

        Returns:
            Model response from the selected provider
        """
        try:
            # Get all available providers
            async with self.uow:
                all_providers = await self.uow.providers.find_all()

            # Filter providers that support the model
            supporting_providers = [
                p for p in all_providers if p.find_model_config(model_id) is not None
            ]

            if not supporting_providers:
                return ModelResponseDTO.create_error_response(
                    provider_id="unknown",
                    model_id=model_id,
                    error_message=f"No providers support model {model_id}",
                )

            # Create selection context
            selection_context = SelectionContext(
                model_id=model_id,
                estimated_input_tokens=len(prompt) // 4,  # Rough estimation
                estimated_output_tokens=parameters.get("max_tokens", 1000),
                selection_criteria=selection_criteria,
            )

            # Select best provider
            selected_provider_score = self.provider_selector.select_provider(
                supporting_providers, selection_context
            )

            if not selected_provider_score:
                return ModelResponseDTO.create_error_response(
                    provider_id="unknown",
                    model_id=model_id,
                    error_message="No suitable provider found",
                )

            # Execute call with selected provider
            response = await self.call_model(
                provider_id=str(selected_provider_score.provider.id),
                model_id=model_id,
                prompt=prompt,
                parameters=parameters,
                test_context=test_context,
            )

            # Record metrics for future selection
            self.provider_selector.record_performance_metrics(
                provider=selected_provider_score.provider,
                latency_ms=response.latency_ms or 0,
                success=response.is_successful(),
                cost=float(response.total_cost) if response.total_cost else None,
            )

            return response

        except Exception as e:
            self.logger.error(f"Intelligent routing failed: {e}", exc_info=True)
            return ModelResponseDTO.create_error_response(
                provider_id="unknown",
                model_id=model_id,
                error_message=f"Intelligent routing failed: {str(e)}",
            )

    async def _execute_model_call_with_reliability(
        self, request: ModelRequestDTO
    ) -> ModelResponseDTO:
        """Execute model call with full reliability patterns."""

        # Get provider and validate
        async with self.uow:
            provider = await self.uow.providers.find_by_id(request.provider_id)
            if not provider:
                raise ProviderError(
                    f"Provider {request.provider_id} not found", provider_name=request.provider_id
                )

            if not provider.health_status.is_operational:
                raise ProviderError(
                    f"Provider {provider.name} is not operational (status: {provider.health_status.name})",
                    provider_name=provider.name,
                )

        # Check rate limits
        if not await self.rate_limiter.can_proceed(request.provider_id, request.model_id):
            raise ProviderError(
                f"Rate limit exceeded for {request.provider_id}/{request.model_id}",
                provider_name=provider.name,
            )

        # Get circuit breaker
        circuit_breaker = self.circuit_breaker_factory.get_circuit_breaker_for_provider(
            request.provider_id,
            request.model_id,
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60.0),
        )

        # Create retry context
        retry_context = self.retry_service.create_context(
            operation_name=f"model_call_{request.provider_id}_{request.model_id}",
            provider_name=provider.name,
            model_id=request.model_id,
            request_id=request.request_id,
            total_timeout=300.0,  # 5 minutes total timeout
        )

        # Execute with circuit breaker and retry
        try:
            response = await circuit_breaker.execute(
                lambda: self.retry_service.execute_with_adaptive_retry(
                    self._make_api_call, retry_context, request, provider
                )
            )

            # Record successful request
            self.rate_limiter.record_request(request.provider_id, request.model_id)

            return response

        except Exception as e:
            # Handle and convert errors
            provider_error = self.error_handler.handle_provider_error(
                e, provider.name, request.model_id, {"request_id": request.request_id}
            )

            # Create error response
            return ModelResponseDTO.create_error_response(
                provider_id=request.provider_id,
                model_id=request.model_id,
                error_message=str(provider_error),
                status=self._convert_error_to_response_status(provider_error),
                request_id=request.request_id,
            )

    async def _make_api_call(
        self, request: ModelRequestDTO, provider: ModelProvider
    ) -> ModelResponseDTO:
        """Make the actual API call to the provider."""

        # Get model configuration
        model_config = provider.find_model_config(request.model_id)
        if not model_config:
            raise ProviderError(
                f"Model {request.model_id} not found in provider {provider.name}",
                provider_name=provider.name,
                model_id=request.model_id,
            )

        # Validate parameters
        for param_name, param_value in request.parameters.items():
            if not model_config.supports_parameter(param_name):
                raise ProviderError(
                    f"Parameter '{param_name}' not supported by model {request.model_id}",
                    provider_name=provider.name,
                    model_id=request.model_id,
                )

        # Record start time for latency measurement
        start_time = datetime.utcnow()

        try:
            # This is where the actual API call would be made
            # In this implementation, we're creating a mock response for demonstration
            # In a real implementation, this would use the appropriate client library

            # Simulate API call duration
            await asyncio.sleep(0.1)  # 100ms simulated latency

            # Create mock response based on provider type
            mock_response = self._create_mock_response(provider, request, start_time)

            # Process and standardize the response
            standardized_response = await self.response_processor.standardize_response(
                raw_response=mock_response,
                request=request,
                provider_type=provider.provider_type,
                model_config=model_config,
                request_start_time=start_time,
            )

            return standardized_response

        except Exception as e:
            # Let the error handler deal with provider-specific errors
            raise e

    def _create_mock_response(
        self, provider: ModelProvider, request: ModelRequestDTO, start_time: datetime
    ) -> Dict[str, Any]:
        """Create a mock response for demonstration purposes."""

        # Calculate simulated metrics
        latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        input_tokens = len(request.prompt) // 4  # Rough estimation
        output_tokens = min(request.parameters.get("max_tokens", 150), 200)

        # Create provider-specific response format
        if provider.provider_type == ProviderType.OPENAI:
            return {
                "id": f"chatcmpl-{str(uuid4())[:8]}",
                "object": "chat.completion",
                "model": request.model_id,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Mock response from {provider.name} for: {request.prompt[:50]}...",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }

        elif provider.provider_type == ProviderType.ANTHROPIC:
            return {
                "id": f"msg_{str(uuid4())}",
                "type": "message",
                "model": request.model_id,
                "content": [
                    {
                        "type": "text",
                        "text": f"Mock response from {provider.name} for: {request.prompt[:50]}...",
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }

        else:
            # Generic response
            return {
                "text": f"Mock response from {provider.name} for: {request.prompt[:50]}...",
                "token_count": output_tokens,
                "model": request.model_id,
            }

    def _convert_error_to_response_status(self, error: ProviderError) -> ResponseStatus:
        """Convert provider error to response status."""
        error_type = getattr(error, "error_type", None)

        if not error_type:
            return ResponseStatus.ERROR

        from .error_handler import ProviderErrorType

        mapping = {
            ProviderErrorType.RATE_LIMIT_EXCEEDED: ResponseStatus.RATE_LIMITED,
            ProviderErrorType.TIMEOUT_ERROR: ResponseStatus.TIMEOUT,
            ProviderErrorType.INVALID_REQUEST: ResponseStatus.INVALID_REQUEST,
            ProviderErrorType.AUTHENTICATION_ERROR: ResponseStatus.ERROR,
            ProviderErrorType.MODEL_NOT_FOUND: ResponseStatus.INVALID_REQUEST,
        }

        return mapping.get(error_type, ResponseStatus.ERROR)

    # Integration methods for Agent 2.1 Test Management Use Cases

    async def process_test_sample(
        self,
        sample: TestSample,
        provider_id: str,
        model_id: str,
        parameters: Dict[str, Any],
        test_context: Optional[Dict[str, Any]] = None,
    ) -> ModelResponseDTO:
        """
        Process a single test sample through a model provider.

        This method integrates with Agent 2.1's ProcessSamplesUseCase.
        """
        return await self.call_model(
            provider_id=provider_id,
            model_id=model_id,
            prompt=sample.prompt,
            parameters=parameters,
            test_context={
                "test_sample_id": str(sample.id),
                "sample_metadata": sample.metadata or {},
                **(test_context or {}),
            },
        )

    async def get_provider_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        async with self.uow:
            providers = await self.uow.providers.find_all()

        health_status = {}
        for provider in providers:
            provider_stats = self.provider_selector.get_provider_statistics(provider)
            health_status[provider.name] = {
                "health_status": provider.health_status.name,
                "is_operational": provider.health_status.is_operational,
                "rate_limit_remaining": (
                    provider.rate_limits.requests_per_minute
                    - provider.rate_limits.current_minute_count
                ),
                "supported_models": [m.model_id for m in provider.supported_models],
                "statistics": provider_stats,
            }

        # Add circuit breaker status
        cb_status = self.circuit_breaker_factory.get_status_summary()
        health_status["circuit_breakers"] = cb_status

        return health_status
