"""Integration tests for repository implementations."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.domain.model_provider.value_objects.provider_type import ProviderType
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus
from src.infrastructure.persistence.repositories.analytics_repository_impl import (
    AnalyticsRepositoryImpl,
)
from src.infrastructure.persistence.repositories.evaluation_repository_impl import (
    EvaluationRepositoryImpl,
)
from src.infrastructure.persistence.repositories.provider_repository_impl import (
    ProviderRepositoryImpl,
)
from src.infrastructure.persistence.repositories.test_repository_impl import TestRepositoryImpl
from tests.factories import (
    EvaluationResultFactory,
    ModelPerformanceFactory,
    ModelProviderFactory,
    TestFactory,
    TestSampleFactory,
)


@pytest.mark.integration
class TestRepositoryIntegration:
    """Integration tests for repository implementations with real database."""

    @pytest.fixture
    def test_repository(self, async_session):
        """Create test repository with real database session."""
        return TestRepositoryImpl(async_session)

    @pytest.fixture
    def provider_repository(self, async_session):
        """Create provider repository with real database session."""
        return ProviderRepositoryImpl(async_session)

    @pytest.fixture
    def evaluation_repository(self, async_session):
        """Create evaluation repository with real database session."""
        return EvaluationRepositoryImpl(async_session)

    @pytest.fixture
    def analytics_repository(self, async_session):
        """Create analytics repository with real database session."""
        return AnalyticsRepositoryImpl(async_session)

    @pytest.mark.asyncio
    async def test_test_repository_crud_operations(self, test_repository):
        """Test CRUD operations for test repository."""
        # Create
        test = TestFactory()
        test.samples = [TestSampleFactory(test_id=test.id) for _ in range(5)]

        await test_repository.save(test)

        # Read
        retrieved_test = await test_repository.get_by_id(test.id)
        assert retrieved_test is not None
        assert retrieved_test.id == test.id
        assert retrieved_test.name == test.name
        assert len(retrieved_test.samples) == 5

        # Update
        retrieved_test.status = TestStatus.RUNNING
        retrieved_test.started_at = datetime.utcnow()
        await test_repository.save(retrieved_test)

        updated_test = await test_repository.get_by_id(test.id)
        assert updated_test.status == TestStatus.RUNNING
        assert updated_test.started_at is not None

        # Delete
        await test_repository.delete(test.id)
        deleted_test = await test_repository.get_by_id(test.id)
        assert deleted_test is None

    @pytest.mark.asyncio
    async def test_test_repository_filtering_and_pagination(self, test_repository):
        """Test filtering and pagination for test repository."""
        # Create multiple tests with different statuses
        tests = []
        for i in range(15):
            test = TestFactory()
            test.status = TestStatus.CONFIGURED if i < 5 else TestStatus.RUNNING
            test.name = f"Test {i:02d}"
            await test_repository.save(test)
            tests.append(test)

        # Test filtering by status
        configured_tests = await test_repository.get_by_status(TestStatus.CONFIGURED)
        assert len(configured_tests) == 5

        running_tests = await test_repository.get_by_status(TestStatus.RUNNING)
        assert len(running_tests) == 10

        # Test pagination
        page1 = await test_repository.get_paginated(page=1, page_size=5)
        assert len(page1) == 5

        page2 = await test_repository.get_paginated(page=2, page_size=5)
        assert len(page2) == 5

        # Ensure different tests on different pages
        page1_ids = {test.id for test in page1}
        page2_ids = {test.id for test in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_test_repository_complex_queries(self, test_repository):
        """Test complex queries for test repository."""
        # Create tests with different creation dates
        base_date = datetime.utcnow()
        old_test = TestFactory()
        old_test.created_at = base_date - timedelta(days=30)
        old_test.status = TestStatus.COMPLETED
        await test_repository.save(old_test)

        recent_test = TestFactory()
        recent_test.created_at = base_date - timedelta(days=1)
        recent_test.status = TestStatus.RUNNING
        await test_repository.save(recent_test)

        # Test date range filtering
        recent_tests = await test_repository.get_by_date_range(
            start_date=base_date - timedelta(days=7), end_date=base_date
        )
        assert len(recent_tests) == 1
        assert recent_tests[0].id == recent_test.id

        # Test status and date combination
        completed_tests = await test_repository.get_by_status_and_date_range(
            status=TestStatus.COMPLETED,
            start_date=base_date - timedelta(days=60),
            end_date=base_date,
        )
        assert len(completed_tests) == 1
        assert completed_tests[0].id == old_test.id

    @pytest.mark.asyncio
    async def test_provider_repository_crud_operations(self, provider_repository):
        """Test CRUD operations for provider repository."""
        # Create
        provider = ModelProviderFactory()
        provider.provider_type = ProviderType.OPENAI

        await provider_repository.save(provider)

        # Read
        retrieved_provider = await provider_repository.get_by_id(provider.id)
        assert retrieved_provider is not None
        assert retrieved_provider.id == provider.id
        assert retrieved_provider.name == provider.name
        assert retrieved_provider.provider_type == ProviderType.OPENAI

        # Update
        retrieved_provider.is_active = False
        await provider_repository.save(retrieved_provider)

        updated_provider = await provider_repository.get_by_id(provider.id)
        assert updated_provider.is_active is False

        # Test get by name
        found_provider = await provider_repository.get_by_name(provider.name)
        assert found_provider is not None
        assert found_provider.id == provider.id

    @pytest.mark.asyncio
    async def test_provider_repository_active_providers_query(self, provider_repository):
        """Test querying active providers."""
        # Create active and inactive providers
        active_providers = []
        for i in range(3):
            provider = ModelProviderFactory()
            provider.name = f"active_provider_{i}"
            provider.is_active = True
            await provider_repository.save(provider)
            active_providers.append(provider)

        inactive_providers = []
        for i in range(2):
            provider = ModelProviderFactory()
            provider.name = f"inactive_provider_{i}"
            provider.is_active = False
            await provider_repository.save(provider)
            inactive_providers.append(provider)

        # Query active providers
        active_results = await provider_repository.get_active_providers()
        active_ids = {provider.id for provider in active_results}
        expected_active_ids = {provider.id for provider in active_providers}

        assert active_ids == expected_active_ids

    @pytest.mark.asyncio
    async def test_provider_repository_health_status_updates(self, provider_repository):
        """Test health status updates for providers."""
        # Create provider
        provider = ModelProviderFactory()
        provider.health_status.is_healthy = True
        provider.health_status.response_time_ms = 100

        await provider_repository.save(provider)

        # Update health status
        provider.health_status.is_healthy = False
        provider.health_status.response_time_ms = 5000
        provider.health_status.error_rate = 0.1
        provider.health_status.last_check = datetime.utcnow()

        await provider_repository.save(provider)

        # Verify update
        updated_provider = await provider_repository.get_by_id(provider.id)
        assert updated_provider.health_status.is_healthy is False
        assert updated_provider.health_status.response_time_ms == 5000
        assert updated_provider.health_status.error_rate == 0.1

    @pytest.mark.asyncio
    async def test_evaluation_repository_crud_operations(self, evaluation_repository):
        """Test CRUD operations for evaluation repository."""
        # Create
        evaluation_result = EvaluationResultFactory()

        await evaluation_repository.save(evaluation_result)

        # Read
        retrieved_result = await evaluation_repository.get_by_id(evaluation_result.id)
        assert retrieved_result is not None
        assert retrieved_result.id == evaluation_result.id
        assert retrieved_result.scores == evaluation_result.scores

        # Update
        retrieved_result.confidence = 0.95
        retrieved_result.reasoning = "Updated reasoning"
        await evaluation_repository.save(retrieved_result)

        updated_result = await evaluation_repository.get_by_id(evaluation_result.id)
        assert updated_result.confidence == 0.95
        assert updated_result.reasoning == "Updated reasoning"

    @pytest.mark.asyncio
    async def test_evaluation_repository_complex_queries(self, evaluation_repository):
        """Test complex queries for evaluation repository."""
        # Create evaluations for different samples and judges
        sample_id_1 = str(uuid4())
        sample_id_2 = str(uuid4())
        judge_id_1 = str(uuid4())
        judge_id_2 = str(uuid4())

        evaluations = []

        # Sample 1 evaluations
        for judge_id in [judge_id_1, judge_id_2]:
            eval_result = EvaluationResultFactory()
            eval_result.sample_id = sample_id_1
            eval_result.judge_id = judge_id
            eval_result.scores = {"accuracy": 8.0, "relevance": 7.5}
            await evaluation_repository.save(eval_result)
            evaluations.append(eval_result)

        # Sample 2 evaluations
        eval_result = EvaluationResultFactory()
        eval_result.sample_id = sample_id_2
        eval_result.judge_id = judge_id_1
        eval_result.scores = {"accuracy": 9.0, "relevance": 8.5}
        await evaluation_repository.save(eval_result)
        evaluations.append(eval_result)

        # Query by sample
        sample1_evaluations = await evaluation_repository.get_by_sample_id(sample_id_1)
        assert len(sample1_evaluations) == 2

        sample2_evaluations = await evaluation_repository.get_by_sample_id(sample_id_2)
        assert len(sample2_evaluations) == 1

        # Query by judge
        judge1_evaluations = await evaluation_repository.get_by_judge_id(judge_id_1)
        assert len(judge1_evaluations) == 2

        judge2_evaluations = await evaluation_repository.get_by_judge_id(judge_id_2)
        assert len(judge2_evaluations) == 1

    @pytest.mark.asyncio
    async def test_analytics_repository_crud_operations(self, analytics_repository):
        """Test CRUD operations for analytics repository."""
        # Create
        model_performance = ModelPerformanceFactory()

        await analytics_repository.save(model_performance)

        # Read
        retrieved_performance = await analytics_repository.get_by_id(model_performance.id)
        assert retrieved_performance is not None
        assert retrieved_performance.id == model_performance.id
        assert retrieved_performance.test_id == model_performance.test_id

        # Update
        retrieved_performance.total_samples = 100
        retrieved_performance.successful_samples = 95
        await analytics_repository.save(retrieved_performance)

        updated_performance = await analytics_repository.get_by_id(model_performance.id)
        assert updated_performance.total_samples == 100
        assert updated_performance.successful_samples == 95

    @pytest.mark.asyncio
    async def test_analytics_repository_aggregation_queries(self, analytics_repository):
        """Test aggregation queries for analytics repository."""
        # Create multiple performance records for same test
        test_id = str(uuid4())
        performances = []

        for i in range(3):
            performance = ModelPerformanceFactory()
            performance.test_id = test_id
            performance.model_config_id = str(uuid4())
            performance.total_samples = 100
            performance.successful_samples = 90 + i
            await analytics_repository.save(performance)
            performances.append(performance)

        # Query performance by test
        test_performances = await analytics_repository.get_by_test_id(test_id)
        assert len(test_performances) == 3

        # Test aggregation
        avg_success_rate = await analytics_repository.get_average_success_rate(test_id)
        expected_avg = (90 + 91 + 92) / (100 * 3)
        assert abs(avg_success_rate - expected_avg) < 0.001

    @pytest.mark.asyncio
    async def test_repository_transaction_handling(self, test_repository, async_session):
        """Test transaction handling in repositories."""
        # Start a transaction
        test = TestFactory()

        try:
            # Save test
            await test_repository.save(test)

            # Verify it exists in the session but not committed
            session_test = await test_repository.get_by_id(test.id)
            assert session_test is not None

            # Simulate an error that would cause rollback
            raise Exception("Simulated error")

        except Exception:
            # Rollback should occur automatically
            await async_session.rollback()

        # After rollback, test should not exist
        rolledback_test = await test_repository.get_by_id(test.id)
        assert rolledback_test is None

    @pytest.mark.asyncio
    async def test_repository_bulk_operations(self, test_repository):
        """Test bulk operations for repositories."""
        # Create multiple tests
        tests = [TestFactory() for _ in range(10)]

        # Bulk save
        await test_repository.bulk_save(tests)

        # Verify all were saved
        for test in tests:
            retrieved_test = await test_repository.get_by_id(test.id)
            assert retrieved_test is not None

        # Bulk update
        for test in tests:
            test.status = TestStatus.RUNNING

        await test_repository.bulk_save(tests)

        # Verify all were updated
        for test in tests:
            retrieved_test = await test_repository.get_by_id(test.id)
            assert retrieved_test.status == TestStatus.RUNNING

    @pytest.mark.asyncio
    async def test_repository_concurrent_access(self, test_repository):
        """Test concurrent access to repositories."""
        import asyncio

        # Create test
        test = TestFactory()
        await test_repository.save(test)

        # Define concurrent update function
        async def update_test(field_value):
            retrieved_test = await test_repository.get_by_id(test.id)
            retrieved_test.name = f"Updated {field_value}"
            await test_repository.save(retrieved_test)

        # Run concurrent updates
        await asyncio.gather(*[update_test(i) for i in range(5)])

        # Verify final state
        final_test = await test_repository.get_by_id(test.id)
        assert final_test is not None
        assert "Updated" in final_test.name

    @pytest.mark.asyncio
    async def test_repository_soft_delete(self, test_repository):
        """Test soft delete functionality."""
        # Create test
        test = TestFactory()
        await test_repository.save(test)

        # Soft delete
        await test_repository.soft_delete(test.id)

        # Should not appear in normal queries
        retrieved_test = await test_repository.get_by_id(test.id)
        assert retrieved_test is None

        # Should appear in deleted queries
        deleted_test = await test_repository.get_deleted_by_id(test.id)
        assert deleted_test is not None
        assert deleted_test.deleted_at is not None

    @pytest.mark.asyncio
    async def test_repository_search_functionality(self, test_repository):
        """Test search functionality in repositories."""
        # Create tests with searchable content
        test1 = TestFactory()
        test1.name = "Machine Learning Evaluation"
        test1.configuration.description = "Testing ML models for accuracy"
        await test_repository.save(test1)

        test2 = TestFactory()
        test2.name = "Natural Language Processing"
        test2.configuration.description = "NLP model comparison"
        await test_repository.save(test2)

        test3 = TestFactory()
        test3.name = "Computer Vision"
        test3.configuration.description = "Image classification test"
        await test_repository.save(test3)

        # Search by name
        ml_tests = await test_repository.search_by_name("Machine")
        assert len(ml_tests) == 1
        assert ml_tests[0].id == test1.id

        # Search by description
        model_tests = await test_repository.search_by_description("model")
        assert len(model_tests) == 2

        # Full text search
        all_matching = await test_repository.full_text_search("test")
        assert len(all_matching) >= 2
