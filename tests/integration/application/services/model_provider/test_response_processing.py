"""Integration tests for response processing and cost calculation."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.dto.model_request_dto import ModelRequestDTO
from src.application.dto.model_response_dto import ModelResponseDTO, ResponseStatus
from src.application.services.model_provider.cost_calculator import CostBreakdown, CostCalculator
from src.application.services.model_provider.response_processor import ResponseProcessor
from src.domain.model_provider.entities.model_config import ModelConfig
from src.domain.model_provider.value_objects.provider_type import ProviderType


class TestResponseProcessorIntegration:
    """Test response processor integration."""

    @pytest.fixture
    def cost_calculator(self):
        """Create cost calculator instance."""
        return CostCalculator()

    @pytest.fixture
    def response_processor(self, cost_calculator):
        """Create response processor instance."""
        return ResponseProcessor(cost_calculator)

    @pytest.fixture
    def mock_model_config(self):
        """Create mock model configuration."""
        return ModelConfig(
            model_id="test-model",
            max_tokens=2000,
            cost_per_input_token=Decimal("0.001"),
            cost_per_output_token=Decimal("0.002"),
            supported_parameters=["temperature", "max_tokens"],
        )

    @pytest.fixture
    def sample_request(self):
        """Create sample request DTO."""
        return ModelRequestDTO(
            provider_id="test-provider",
            model_id="test-model",
            prompt="Test prompt for processing",
            parameters={"temperature": 0.7, "max_tokens": 100},
        )

    @pytest.mark.asyncio
    async def test_openai_response_standardization(
        self, response_processor, sample_request, mock_model_config
    ):
        """Test standardization of OpenAI response format."""
        # Mock OpenAI response
        openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test response from OpenAI.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

        start_time = datetime.utcnow()

        # Process response
        standardized = await response_processor.standardize_response(
            raw_response=openai_response,
            request=sample_request,
            provider_type=ProviderType.OPENAI,
            model_config=mock_model_config,
            request_start_time=start_time,
        )

        # Verify standardization
        assert standardized.provider_id == "test-provider"
        assert standardized.model_id == "test-model"
        assert standardized.status == ResponseStatus.SUCCESS
        assert standardized.text == "This is a test response from OpenAI."
        assert standardized.input_tokens == 15
        assert standardized.output_tokens == 10
        assert standardized.total_tokens == 25
        assert standardized.finish_reason == "stop"
        assert standardized.latency_ms is not None
        assert standardized.latency_ms > 0

    @pytest.mark.asyncio
    async def test_anthropic_response_standardization(
        self, response_processor, sample_request, mock_model_config
    ):
        """Test standardization of Anthropic response format."""
        # Mock Anthropic response
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "model": "test-model",
            "content": [{"type": "text", "text": "This is a test response from Anthropic."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 12},
        }

        start_time = datetime.utcnow()

        # Process response
        standardized = await response_processor.standardize_response(
            raw_response=anthropic_response,
            request=sample_request,
            provider_type=ProviderType.ANTHROPIC,
            model_config=mock_model_config,
            request_start_time=start_time,
        )

        # Verify standardization
        assert standardized.provider_id == "test-provider"
        assert standardized.model_id == "test-model"
        assert standardized.status == ResponseStatus.SUCCESS
        assert standardized.text == "This is a test response from Anthropic."
        assert standardized.input_tokens == 20
        assert standardized.output_tokens == 12
        assert standardized.finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_cost_calculation_integration(
        self, response_processor, sample_request, mock_model_config
    ):
        """Test cost calculation integration in response processing."""
        # Mock response with token usage
        response_with_tokens = {
            "text": "Response with cost calculation",
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        }

        start_time = datetime.utcnow()

        # Process response
        standardized = await response_processor.standardize_response(
            raw_response=response_with_tokens,
            request=sample_request,
            provider_type=ProviderType.OPENAI,
            model_config=mock_model_config,
            request_start_time=start_time,
        )

        # Verify cost calculation
        if standardized.input_tokens and standardized.output_tokens:
            assert standardized.input_cost is not None
            assert standardized.output_cost is not None
            assert standardized.total_cost is not None
            assert standardized.total_cost > 0

    @pytest.mark.asyncio
    async def test_response_validation(self, response_processor, sample_request):
        """Test response validation against model constraints."""
        # Create model config with low token limit
        restrictive_config = ModelConfig(
            model_id="restricted-model",
            max_tokens=10,  # Very low limit
            cost_per_input_token=Decimal("0.001"),
            cost_per_output_token=Decimal("0.002"),
            supported_parameters=["temperature"],
        )

        # Mock response that exceeds token limit
        excessive_response = {
            "text": "This is a very long response that exceeds the model token limit",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 50,  # Exceeds max_tokens
                "total_tokens": 55,
            },
        }

        start_time = datetime.utcnow()

        # Process response
        standardized = await response_processor.standardize_response(
            raw_response=excessive_response,
            request=sample_request,
            provider_type=ProviderType.OPENAI,
            model_config=restrictive_config,
            request_start_time=start_time,
        )

        # Response should still be processed but may have warnings in metadata
        assert standardized is not None
        assert standardized.output_tokens == 50

    @pytest.mark.asyncio
    async def test_quality_issue_detection(
        self, response_processor, sample_request, mock_model_config
    ):
        """Test detection of response quality issues."""
        # Test empty response
        empty_response = {"text": "", "usage": {"prompt_tokens": 10, "completion_tokens": 0}}

        request_empty = ModelRequestDTO(
            provider_id="test-provider",
            model_id="test-model",
            prompt="Empty response test",
            parameters={},
        )

        issues = response_processor.detect_response_quality_issues(
            ModelResponseDTO.create_success_response(
                provider_id="test-provider",
                model_id="test-model",
                text="",
                input_tokens=10,
                output_tokens=0,
                latency_ms=1000,
            ),
            request_empty,
        )

        assert len(issues) > 0
        assert any("empty" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_quality_score_calculation(
        self, response_processor, sample_request, mock_model_config
    ):
        """Test response quality score calculation."""
        # High quality response
        good_response = ModelResponseDTO.create_success_response(
            provider_id="test-provider",
            model_id="test-model",
            text="This is a high quality response with good length and performance.",
            input_tokens=20,
            output_tokens=15,
            latency_ms=1500,  # Good latency
        )
        good_response.tokens_per_second = 25.0  # Good throughput

        good_score = response_processor.calculate_response_quality_score(
            good_response, sample_request
        )

        # Poor quality response
        poor_response = ModelResponseDTO.create_success_response(
            provider_id="test-provider",
            model_id="test-model",
            text="Bad",  # Very short
            input_tokens=100,
            output_tokens=1,
            latency_ms=20000,  # Very slow
        )
        poor_response.tokens_per_second = 2.0  # Poor throughput

        poor_score = response_processor.calculate_response_quality_score(
            poor_response, sample_request
        )

        # Good response should have higher score
        assert good_score > poor_score
        assert 0.0 <= good_score <= 1.0
        assert 0.0 <= poor_score <= 1.0

    @pytest.mark.asyncio
    async def test_token_usage_estimation(self, response_processor):
        """Test token usage estimation when not provided by provider."""
        prompt = "This is a test prompt for token estimation"
        response_text = "This is the generated response text for estimation"

        input_tokens, output_tokens = response_processor.estimate_token_usage(
            prompt, response_text, ProviderType.OPENAI
        )

        # Verify reasonable estimates
        assert input_tokens > 0
        assert output_tokens > 0
        assert input_tokens > len(prompt) // 8  # Should be more than simple division
        assert output_tokens > len(response_text) // 8

    @pytest.mark.asyncio
    async def test_batch_response_processing(self, response_processor, mock_model_config):
        """Test processing multiple responses in batch."""
        responses_data = [
            (
                {"text": f"Response {i}", "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
                ModelRequestDTO(
                    provider_id="test-provider",
                    model_id="test-model",
                    prompt=f"Prompt {i}",
                    parameters={},
                ),
                datetime.utcnow(),
            )
            for i in range(3)
        ]

        processed_responses = await response_processor.process_batch_responses(
            responses_data, ProviderType.OPENAI, mock_model_config
        )

        # Verify all responses processed
        assert len(processed_responses) == 3
        for i, response in enumerate(processed_responses):
            assert response.text == f"Response {i}"
            assert response.is_successful()


class TestCostCalculatorIntegration:
    """Test cost calculator integration."""

    @pytest.fixture
    def cost_calculator(self):
        """Create cost calculator instance."""
        return CostCalculator()

    def test_openai_cost_calculation(self, cost_calculator):
        """Test cost calculation for OpenAI models."""
        breakdown = cost_calculator.calculate_cost(
            provider_type=ProviderType.OPENAI,
            model_id="gpt-4-turbo",
            input_tokens=1000,
            output_tokens=500,
        )

        # Verify calculation
        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.input_tokens == 1000
        assert breakdown.output_tokens == 500
        assert breakdown.input_cost > 0
        assert breakdown.output_cost > 0
        assert breakdown.total_cost == breakdown.input_cost + breakdown.output_cost
        assert breakdown.provider_id == ProviderType.OPENAI.value
        assert breakdown.model_id == "gpt-4-turbo"

    def test_anthropic_cost_calculation(self, cost_calculator):
        """Test cost calculation for Anthropic models."""
        breakdown = cost_calculator.calculate_cost(
            provider_type=ProviderType.ANTHROPIC,
            model_id="claude-3-opus",
            input_tokens=2000,
            output_tokens=1000,
        )

        # Verify calculation
        assert breakdown.input_tokens == 2000
        assert breakdown.output_tokens == 1000
        assert breakdown.input_cost > 0
        assert breakdown.output_cost > 0
        assert breakdown.provider_id == ProviderType.ANTHROPIC.value
        assert breakdown.model_id == "claude-3-opus"

    def test_unknown_model_fallback_pricing(self, cost_calculator):
        """Test fallback pricing for unknown models."""
        breakdown = cost_calculator.calculate_cost(
            provider_type=ProviderType.OPENAI,
            model_id="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should use fallback pricing
        assert breakdown.input_cost > 0
        assert breakdown.output_cost > 0
        assert breakdown.model_id == "unknown-model"

    def test_custom_pricing_rates(self, cost_calculator):
        """Test calculation with custom pricing rates."""
        custom_rates = {"input": Decimal("0.005"), "output": Decimal("0.010")}

        breakdown = cost_calculator.calculate_cost(
            provider_type=ProviderType.OPENAI,
            model_id="test-model",
            input_tokens=1000,
            output_tokens=500,
            custom_rates=custom_rates,
        )

        # Verify custom rates are used
        expected_input_cost = Decimal("5.0000")  # (1000 * 0.005) / 1000 * 1000 = 5
        expected_output_cost = Decimal("5.0000")  # (500 * 0.010) / 1000 * 1000 = 5

        assert breakdown.input_cost == expected_input_cost
        assert breakdown.output_cost == expected_output_cost

    def test_cost_from_response_dto(self, cost_calculator):
        """Test cost calculation from response DTO."""
        response = ModelResponseDTO(
            provider_id="test-provider",
            model_id="gpt-3.5-turbo",
            status=ResponseStatus.SUCCESS,
            text="Test response",
            input_tokens=500,
            output_tokens=300,
        )

        breakdown = cost_calculator.calculate_cost_from_response(response, ProviderType.OPENAI)

        # Verify calculation
        assert breakdown is not None
        assert breakdown.input_tokens == 500
        assert breakdown.output_tokens == 300

    def test_batch_cost_calculation(self, cost_calculator):
        """Test batch cost calculation."""
        responses = [
            ModelResponseDTO(
                provider_id="test-provider",
                model_id="gpt-3.5-turbo",
                status=ResponseStatus.SUCCESS,
                text=f"Response {i}",
                input_tokens=100 + i * 10,
                output_tokens=50 + i * 5,
            )
            for i in range(3)
        ]

        batch_cost = cost_calculator.calculate_batch_cost(responses, ProviderType.OPENAI)

        # Verify batch calculation
        assert "total_cost" in batch_cost
        assert "total_input_tokens" in batch_cost
        assert "total_output_tokens" in batch_cost
        assert "successful_calculations" in batch_cost
        assert batch_cost["successful_calculations"] == 3
        assert batch_cost["total_input_tokens"] == 100 + 110 + 120  # Sum of input tokens
        assert batch_cost["total_output_tokens"] == 50 + 55 + 60  # Sum of output tokens

    def test_usage_metrics_tracking(self, cost_calculator):
        """Test usage metrics tracking."""
        # Calculate costs for multiple requests
        for i in range(3):
            cost_calculator.calculate_cost(
                provider_type=ProviderType.OPENAI,
                model_id="gpt-3.5-turbo",
                input_tokens=100 + i * 50,
                output_tokens=50 + i * 25,
            )

        # Get usage metrics
        metrics = cost_calculator.get_usage_metrics(ProviderType.OPENAI, "gpt-3.5-turbo")

        # Verify metrics tracking
        assert "metrics" in metrics
        assert metrics["metrics"]["total_requests"] == 3
        assert metrics["metrics"]["total_input_tokens"] == 100 + 150 + 200
        assert metrics["metrics"]["total_output_tokens"] == 50 + 75 + 100

    def test_cost_estimation(self, cost_calculator):
        """Test cost estimation before API call."""
        prompt = "This is a test prompt for cost estimation"

        breakdown = cost_calculator.estimate_cost(
            provider_type=ProviderType.OPENAI, model_id="gpt-4-turbo", prompt=prompt, max_tokens=500
        )

        # Verify estimation
        assert breakdown.input_tokens > 0
        assert breakdown.output_tokens == 500  # Should use max_tokens
        assert breakdown.input_cost > 0
        assert breakdown.output_cost > 0

    def test_precision_rounding(self, cost_calculator):
        """Test that cost calculations maintain proper precision."""
        breakdown = cost_calculator.calculate_cost(
            provider_type=ProviderType.OPENAI,
            model_id="gpt-4-turbo",
            input_tokens=1,  # Minimal tokens to test precision
            output_tokens=1,
        )

        # Verify precision (should be rounded to 4 decimal places)
        assert len(str(breakdown.input_cost).split(".")[-1]) <= 4
        assert len(str(breakdown.output_cost).split(".")[-1]) <= 4
        assert len(str(breakdown.total_cost).split(".")[-1]) <= 4

    def test_pricing_updates(self, cost_calculator):
        """Test dynamic pricing updates."""
        original_pricing = cost_calculator.get_model_pricing(ProviderType.OPENAI, "gpt-4-turbo")

        # Update pricing
        new_input_rate = Decimal("0.020")
        new_output_rate = Decimal("0.040")

        cost_calculator.update_pricing(
            ProviderType.OPENAI, "gpt-4-turbo", new_input_rate, new_output_rate
        )

        # Verify pricing updated
        updated_pricing = cost_calculator.get_model_pricing(ProviderType.OPENAI, "gpt-4-turbo")

        assert updated_pricing["input"] == new_input_rate
        assert updated_pricing["output"] == new_output_rate
