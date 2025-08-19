"""Unit tests for StartTestUseCase."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.application.use_cases.test_management.start_test import StartTestUseCase
from src.domain.test_management.value_objects.test_status import TestStatus
from src.domain.test_management.value_objects.validation_result import ValidationResult
from tests.factories import ModelProviderFactory, TestFactory


class TestStartTestUseCase:
    """Unit tests for StartTestUseCase."""

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
        service.validate_test_start.return_value = ValidationResult(True, [])
        return service

    @pytest.fixture
    def mock_orchestration_service(self):
        """Mock orchestration service."""
        service = AsyncMock()
        service.start_test_execution.return_value = True
        return service

    @pytest.fixture
    def use_case(
        self, mock_uow, mock_event_publisher, mock_validation_service, mock_orchestration_service
    ):
        """Create use case instance."""
        return StartTestUseCase(
            uow=mock_uow,
            event_publisher=mock_event_publisher,
            validation_service=mock_validation_service,
            orchestration_service=mock_orchestration_service,
        )

    @pytest.fixture
    def configured_test(self):
        """Create a configured test."""
        test = TestFactory()
        test.status = TestStatus.CONFIGURED
        return test

    @pytest.mark.asyncio
    async def test_successful_start(self, use_case, mock_uow, configured_test):
        """Test successful test start."""
        # Arrange
        test_id = configured_test.id
        mock_uow.tests.get_by_id.return_value = configured_test

        # Act
        result = await use_case.execute(test_id)

        # Assert
        assert result.started is True
        assert result.status == "RUNNING"
        assert configured_test.status == TestStatus.RUNNING
        assert configured_test.started_at is not None
        mock_uow.tests.save.assert_called_once_with(configured_test)
        mock_uow.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_not_found(self, use_case, mock_uow):
        """Test handling of non-existent test."""
        # Arrange
        test_id = str(uuid4())
        mock_uow.tests.get_by_id.return_value = None

        # Act
        result = await use_case.execute(test_id)

        # Assert
        assert result.started is False
        assert result.status == "not_found"
        assert "Test not found" in result.errors

    @pytest.mark.asyncio
    async def test_invalid_test_status(self, use_case, mock_uow):
        """Test handling of test with invalid status."""
        # Arrange
        test = TestFactory()
        test.status = TestStatus.RUNNING  # Already running
        mock_uow.tests.get_by_id.return_value = test

        # Act
        result = await use_case.execute(test.id)

        # Assert
        assert result.started is False
        assert result.status == "business_rule_violation"
        assert "cannot be started" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_validation_failure(
        self, use_case, mock_uow, mock_validation_service, configured_test
    ):
        """Test validation failure handling."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test
        mock_validation_service.validate_test_start.return_value = ValidationResult(
            False, ["Insufficient resources"]
        )

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is False
        assert result.status == "validation_failed"
        assert "Insufficient resources" in result.errors

    @pytest.mark.asyncio
    async def test_orchestration_failure(
        self, use_case, mock_uow, mock_orchestration_service, configured_test
    ):
        """Test orchestration service failure."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test
        mock_orchestration_service.start_test_execution.side_effect = Exception(
            "Orchestration failed"
        )

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is False
        assert result.status == "system_error"
        assert any("system error" in error.lower() for error in result.errors)

    @pytest.mark.asyncio
    async def test_event_publishing(
        self, use_case, mock_uow, mock_event_publisher, configured_test
    ):
        """Test event publishing after successful start."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is True
        mock_event_publisher.publish_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_database_error(self, use_case, mock_uow, configured_test):
        """Test database error handling."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test
        mock_uow.commit.side_effect = Exception("Database error")

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is False
        assert result.status == "system_error"

    @pytest.mark.asyncio
    async def test_completed_test_cannot_start(self, use_case, mock_uow):
        """Test that completed test cannot be started."""
        # Arrange
        test = TestFactory()
        test.status = TestStatus.COMPLETED
        mock_uow.tests.get_by_id.return_value = test

        # Act
        result = await use_case.execute(test.id)

        # Assert
        assert result.started is False
        assert result.status == "business_rule_violation"
        assert "cannot be started" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_failed_test_cannot_start(self, use_case, mock_uow):
        """Test that failed test cannot be started."""
        # Arrange
        test = TestFactory()
        test.status = TestStatus.FAILED
        mock_uow.tests.get_by_id.return_value = test

        # Act
        result = await use_case.execute(test.id)

        # Assert
        assert result.started is False
        assert result.status == "business_rule_violation"
        assert "cannot be started" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, use_case, mock_uow, configured_test):
        """Test transaction rollback on failure."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test
        mock_uow.tests.save.side_effect = Exception("Save failed")

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is False
        mock_uow.__aenter__.assert_called_once()
        mock_uow.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_provider_availability_check(
        self, use_case, mock_uow, mock_validation_service, configured_test
    ):
        """Test provider availability validation."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test
        mock_validation_service.validate_test_start.return_value = ValidationResult(
            False, ["Model provider is offline"]
        )

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is False
        assert result.status == "validation_failed"
        assert "provider is offline" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_insufficient_funds_validation(
        self, use_case, mock_uow, mock_validation_service, configured_test
    ):
        """Test insufficient funds validation."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test
        mock_validation_service.validate_test_start.return_value = ValidationResult(
            False, ["Insufficient funds for test execution"]
        )

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is False
        assert result.status == "validation_failed"
        assert "insufficient funds" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_rate_limit_validation(
        self, use_case, mock_uow, mock_validation_service, configured_test
    ):
        """Test rate limit validation."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test
        mock_validation_service.validate_test_start.return_value = ValidationResult(
            False, ["Rate limits exceeded"]
        )

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is False
        assert result.status == "validation_failed"
        assert "rate limits" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_empty_samples_validation(self, use_case, mock_uow, mock_validation_service):
        """Test validation of test with no samples."""
        # Arrange
        test = TestFactory()
        test.status = TestStatus.CONFIGURED
        test.samples = []  # No samples
        mock_uow.tests.get_by_id.return_value = test
        mock_validation_service.validate_test_start.return_value = ValidationResult(
            False, ["Test has no samples to execute"]
        )

        # Act
        result = await use_case.execute(test.id)

        # Assert
        assert result.started is False
        assert result.status == "validation_failed"
        assert "no samples" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_concurrent_start_prevention(self, use_case, mock_uow, configured_test):
        """Test prevention of concurrent test starts."""
        # Arrange
        mock_uow.tests.get_by_id.return_value = configured_test

        # Simulate concurrent start attempt by changing status during execution
        async def change_status(*args, **kwargs):
            configured_test.status = TestStatus.RUNNING

        mock_uow.tests.save.side_effect = change_status

        # Act
        result = await use_case.execute(configured_test.id)

        # Assert
        assert result.started is True  # First call should succeed
        assert configured_test.status == TestStatus.RUNNING
