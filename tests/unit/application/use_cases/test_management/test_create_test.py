"""Unit tests for CreateTestUseCase."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.use_cases.test_management.create_test import CreateTestUseCase
from src.domain.model_provider.value_objects.money import Money
from src.domain.test_management.value_objects.validation_result import ValidationResult
from tests.factories import CreateTestCommandDTOFactory, ModelProviderFactory, TestFactory


class TestCreateTestUseCase:
    """Unit tests for CreateTestUseCase."""

    @pytest.fixture
    def mock_uow(self):
        """Mock unit of work."""
        uow = AsyncMock()
        uow.tests = AsyncMock()
        uow.providers = AsyncMock()
        uow.__aenter__ = AsyncMock(return_value=uow)
        uow.__aexit__ = AsyncMock(return_value=None)
        return uow

    @pytest.fixture
    def mock_event_publisher(self):
        """Mock event publisher."""
        return AsyncMock()

    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service."""
        service = AsyncMock()
        service.validate_test_creation.return_value = ValidationResult(True, [])
        service.estimate_test_cost.return_value = Money(10.0, "USD")
        service.estimate_test_duration.return_value = 300.0
        return service

    @pytest.fixture
    def mock_provider_service(self):
        """Mock provider service."""
        service = AsyncMock()
        service.verify_model_availability.return_value = {"openai/gpt-4": True}
        service.validate_model_parameters.return_value = {}
        return service

    @pytest.fixture
    def use_case(
        self, mock_uow, mock_event_publisher, mock_validation_service, mock_provider_service
    ):
        """Create use case instance."""
        return CreateTestUseCase(
            uow=mock_uow,
            event_publisher=mock_event_publisher,
            validation_service=mock_validation_service,
            provider_service=mock_provider_service,
        )

    @pytest.mark.asyncio
    async def test_successful_creation(self, use_case, mock_uow):
        """Test successful test creation."""
        # Arrange
        command = CreateTestCommandDTOFactory()

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is True
        assert result.status == "CONFIGURED"
        assert result.test_id is not None
        mock_uow.tests.save.assert_called_once()
        mock_uow.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_validation_failure(self, use_case, mock_validation_service):
        """Test validation failure handling."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["Invalid configuration"]
        )

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"
        assert "Invalid configuration" in result.errors

    @pytest.mark.asyncio
    async def test_provider_unavailable(self, use_case, mock_provider_service):
        """Test handling of unavailable providers."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        mock_provider_service.verify_model_availability.return_value = {"openai/gpt-4": False}

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "business_rule_violation"
        assert any("not available" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_database_error(self, use_case, mock_uow):
        """Test database error handling."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        mock_uow.commit.side_effect = Exception("Database error")

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "system_error"
        assert any("system error" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_event_publishing(self, use_case, mock_event_publisher):
        """Test event publishing after successful creation."""
        # Arrange
        command = CreateTestCommandDTOFactory()

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is True
        mock_event_publisher.publish_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_estimation(self, use_case, mock_validation_service):
        """Test cost estimation integration."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        expected_cost = Money(25.50, "USD")
        mock_validation_service.estimate_test_cost.return_value = expected_cost

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is True
        assert result.estimated_cost == expected_cost

    @pytest.mark.asyncio
    async def test_insufficient_samples(self, use_case, mock_validation_service):
        """Test handling of insufficient samples."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        command.samples = command.samples[:5]  # Only 5 samples
        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["At least 10 samples required"]
        )

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"
        assert "samples required" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_invalid_model_parameters(self, use_case, mock_provider_service):
        """Test invalid model parameter handling."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        mock_provider_service.validate_model_parameters.return_value = {
            "openai/gpt-4": ["max_tokens exceeds limit"]
        }

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "business_rule_violation"
        assert any("parameter validation failed" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, use_case, mock_uow):
        """Test transaction rollback on failure."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        mock_uow.tests.save.side_effect = Exception("Save failed")

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        mock_uow.__aenter__.assert_called_once()
        mock_uow.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_sample_assignment(self, use_case, mock_uow):
        """Test sample assignment to test."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        sample_count = len(command.samples)

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is True
        saved_test = mock_uow.tests.save.call_args[0][0]
        assert len(saved_test.samples) == sample_count

    @pytest.mark.asyncio
    async def test_model_weight_validation(self, use_case, mock_validation_service):
        """Test model weight validation."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        # Set invalid weights
        for model in command.configuration.models:
            model.weight = 0.6  # Total > 1.0

        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["Model weights must sum to 1.0"]
        )

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"
        assert "weights" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_max_cost_validation(self, use_case, mock_validation_service):
        """Test maximum cost validation."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        command.configuration.max_cost = Money(5.0, "USD")  # Low max cost

        mock_validation_service.estimate_test_cost.return_value = Money(100.0, "USD")
        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["Estimated cost exceeds maximum allowed cost"]
        )

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"
        assert "cost exceeds" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_duplicate_model_validation(self, use_case, mock_validation_service):
        """Test duplicate model validation."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        # Add duplicate model
        command.configuration.models.append(command.configuration.models[0])

        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["Duplicate models not allowed"]
        )

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"
        assert "duplicate" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_evaluation_configuration_validation(self, use_case, mock_validation_service):
        """Test evaluation configuration validation."""
        # Arrange
        command = CreateTestCommandDTOFactory()
        command.configuration.evaluation.judge_count = 0

        mock_validation_service.validate_test_creation.return_value = ValidationResult(
            False, ["At least one judge required"]
        )

        # Act
        result = await use_case.execute(command)

        # Assert
        assert result.created_test is False
        assert result.status == "validation_failed"
        assert "judge required" in result.errors[0].lower()
