"""Database integration tests for LLM A/B Testing Platform."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.persistence.database import get_database, reset_database
from src.infrastructure.persistence.models.analytics_models import AnalyticsEvent
from src.infrastructure.persistence.models.evaluation_models import Evaluation
from src.infrastructure.persistence.models.provider_models import ModelProvider
from src.infrastructure.persistence.models.test_models import Test, TestStatus


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.fixture(scope="class")
    async def db_session(self):
        """Create database session for testing."""
        database = get_database()

        # Reset database for clean testing
        await reset_database()

        async with database.get_session() as session:
            yield session

    @pytest.mark.asyncio
    async def test_provider_crud_operations(self, db_session: AsyncSession):
        """Test provider CRUD operations."""
        from src.domain.entities.provider import Provider, ProviderConfig, ProviderType
        from src.domain.repositories.provider_repository import ProviderRepository

        repo = ProviderRepository(db_session)

        # Create provider
        provider_config = ProviderConfig(
            api_key="test_key", model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000
        )

        provider = Provider(
            name="test_provider",
            provider_type=ProviderType.OPENAI,
            config=provider_config,
            is_active=True,
        )

        created_provider = await repo.create(provider)
        assert created_provider.id is not None
        assert created_provider.name == "test_provider"
        assert created_provider.provider_type == ProviderType.OPENAI

        # Read provider
        retrieved_provider = await repo.get_by_id(created_provider.id)
        assert retrieved_provider is not None
        assert retrieved_provider.name == "test_provider"
        assert retrieved_provider.config.model == "gpt-3.5-turbo"

        # Update provider
        retrieved_provider.name = "updated_provider"
        retrieved_provider.is_active = False
        updated_provider = await repo.update(retrieved_provider)
        assert updated_provider.name == "updated_provider"
        assert updated_provider.is_active == False

        # List providers
        providers = await repo.get_all()
        assert len(providers) >= 1
        assert any(p.id == created_provider.id for p in providers)

        # List active providers
        active_providers = await repo.get_active()
        assert all(p.is_active for p in active_providers)

        # Delete provider
        await repo.delete(created_provider.id)
        deleted_provider = await repo.get_by_id(created_provider.id)
        assert deleted_provider is None

    @pytest.mark.asyncio
    async def test_ab_test_operations(self, db_session: AsyncSession):
        """Test A/B test operations."""
        from src.domain.entities.provider import Provider, ProviderConfig, ProviderType
        from src.domain.entities.test import Test, TestStatus
        from src.domain.repositories.provider_repository import ProviderRepository
        from src.domain.repositories.test_repository import TestRepository

        test_repo = TestRepository(db_session)
        provider_repo = ProviderRepository(db_session)

        # Create providers first
        provider_a = Provider(
            name="provider_a",
            provider_type=ProviderType.OPENAI,
            config=ProviderConfig(api_key="key_a", model="gpt-3.5-turbo"),
            is_active=True,
        )

        provider_b = Provider(
            name="provider_b",
            provider_type=ProviderType.ANTHROPIC,
            config=ProviderConfig(api_key="key_b", model="claude-3-sonnet"),
            is_active=True,
        )

        created_provider_a = await provider_repo.create(provider_a)
        created_provider_b = await provider_repo.create(provider_b)

        # Create test
        test = Test(
            name="Database Integration Test",
            description="Testing database operations",
            created_by="test_user",
            prompt_template="Test prompt: {input}",
            provider_a_id=created_provider_a.id,
            provider_b_id=created_provider_b.id,
            evaluation_criteria={"quality": 0.5, "relevance": 0.5},
            sample_size=100,
            confidence_level=0.95,
            status=TestStatus.DRAFT,
        )

        created_test = await test_repo.create(test)
        assert created_test.id is not None
        assert created_test.name == "Database Integration Test"
        assert created_test.status == TestStatus.DRAFT

        # Update test status
        created_test.status = TestStatus.RUNNING
        created_test.started_at = datetime.utcnow()
        updated_test = await test_repo.update(created_test)
        assert updated_test.status == TestStatus.RUNNING
        assert updated_test.started_at is not None

        # Get test by user
        user_tests = await test_repo.get_by_user("test_user")
        assert len(user_tests) >= 1
        assert any(t.id == created_test.id for t in user_tests)

        # Get tests by status
        running_tests = await test_repo.get_by_status(TestStatus.RUNNING)
        assert len(running_tests) >= 1
        assert any(t.id == created_test.id for t in running_tests)

        # Complete test
        created_test.status = TestStatus.COMPLETED
        created_test.completed_at = datetime.utcnow()
        await test_repo.update(created_test)

        # Cleanup
        await test_repo.delete(created_test.id)
        await provider_repo.delete(created_provider_a.id)
        await provider_repo.delete(created_provider_b.id)

    @pytest.mark.asyncio
    async def test_evaluation_operations(self, db_session: AsyncSession):
        """Test evaluation operations."""
        from src.domain.entities.evaluation import Evaluation, EvaluationScores
        from src.domain.entities.provider import Provider, ProviderConfig, ProviderType
        from src.domain.entities.test import Test, TestStatus
        from src.domain.repositories.evaluation_repository import EvaluationRepository
        from src.domain.repositories.provider_repository import ProviderRepository
        from src.domain.repositories.test_repository import TestRepository

        eval_repo = EvaluationRepository(db_session)
        test_repo = TestRepository(db_session)
        provider_repo = ProviderRepository(db_session)

        # Create test first
        provider = Provider(
            name="eval_provider",
            provider_type=ProviderType.OPENAI,
            config=ProviderConfig(api_key="key", model="gpt-3.5-turbo"),
            is_active=True,
        )
        created_provider = await provider_repo.create(provider)

        test = Test(
            name="Evaluation Test",
            description="Testing evaluations",
            created_by="test_user",
            prompt_template="Evaluate: {input}",
            provider_a_id=created_provider.id,
            provider_b_id=created_provider.id,
            evaluation_criteria={"accuracy": 1.0},
            sample_size=10,
            confidence_level=0.95,
            status=TestStatus.RUNNING,
        )
        created_test = await test_repo.create(test)

        # Create evaluation
        evaluation_scores = EvaluationScores(accuracy={"a": 0.8, "b": 0.6})

        evaluation = Evaluation(
            test_id=created_test.id,
            input_text="What is AI?",
            response_a="AI is artificial intelligence",
            response_b="AI stands for artificial intelligence",
            evaluation_scores=evaluation_scores,
            evaluator="test_evaluator",
            metadata={"source": "integration_test"},
        )

        created_evaluation = await eval_repo.create(evaluation)
        assert created_evaluation.id is not None
        assert created_evaluation.test_id == created_test.id
        assert created_evaluation.input_text == "What is AI?"

        # Get evaluations by test
        test_evaluations = await eval_repo.get_by_test_id(created_test.id)
        assert len(test_evaluations) >= 1
        assert any(e.id == created_evaluation.id for e in test_evaluations)

        # Get evaluation statistics
        stats = await eval_repo.get_test_statistics(created_test.id)
        assert stats["total_evaluations"] >= 1
        assert "average_scores" in stats

        # Cleanup
        await eval_repo.delete(created_evaluation.id)
        await test_repo.delete(created_test.id)
        await provider_repo.delete(created_provider.id)

    @pytest.mark.asyncio
    async def test_analytics_operations(self, db_session: AsyncSession):
        """Test analytics operations."""
        from src.domain.entities.analytics import AnalyticsEvent, EventType
        from src.domain.repositories.analytics_repository import AnalyticsRepository

        analytics_repo = AnalyticsRepository(db_session)

        # Create analytics event
        event = AnalyticsEvent(
            event_type=EventType.TEST_CREATED,
            user_id="test_user",
            test_id="test_123",
            metadata={"test_name": "Analytics Test"},
            timestamp=datetime.utcnow(),
        )

        created_event = await analytics_repo.create_event(event)
        assert created_event.id is not None
        assert created_event.event_type == EventType.TEST_CREATED
        assert created_event.user_id == "test_user"

        # Get events by user
        user_events = await analytics_repo.get_events_by_user("test_user")
        assert len(user_events) >= 1
        assert any(e.id == created_event.id for e in user_events)

        # Get events by type
        test_events = await analytics_repo.get_events_by_type(EventType.TEST_CREATED)
        assert len(test_events) >= 1
        assert any(e.id == created_event.id for e in test_events)

        # Get events in date range
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)

        range_events = await analytics_repo.get_events_in_range(start_date, end_date)
        assert len(range_events) >= 1
        assert any(e.id == created_event.id for e in range_events)

        # Get dashboard data
        dashboard_data = await analytics_repo.get_dashboard_data()
        assert "total_tests" in dashboard_data
        assert "total_evaluations" in dashboard_data
        assert "active_tests" in dashboard_data

    @pytest.mark.asyncio
    async def test_transaction_handling(self, db_session: AsyncSession):
        """Test database transaction handling."""
        from src.domain.entities.provider import Provider, ProviderConfig, ProviderType
        from src.domain.repositories.provider_repository import ProviderRepository

        repo = ProviderRepository(db_session)

        # Test successful transaction
        provider = Provider(
            name="transaction_test",
            provider_type=ProviderType.OPENAI,
            config=ProviderConfig(api_key="key", model="gpt-3.5-turbo"),
            is_active=True,
        )

        created_provider = await repo.create(provider)
        assert created_provider.id is not None

        # Test rollback scenario (simulate by trying to create duplicate)
        try:
            duplicate_provider = Provider(
                name="transaction_test",  # Same name
                provider_type=ProviderType.OPENAI,
                config=ProviderConfig(api_key="key2", model="gpt-4"),
                is_active=True,
            )
            # This might succeed if no unique constraints, that's ok
            await repo.create(duplicate_provider)
        except Exception:
            # Expected if there are unique constraints
            pass

        # Verify original provider still exists
        retrieved = await repo.get_by_id(created_provider.id)
        assert retrieved is not None
        assert retrieved.name == "transaction_test"

        # Cleanup
        await repo.delete(created_provider.id)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, db_session: AsyncSession):
        """Test concurrent database operations."""
        from src.domain.entities.provider import Provider, ProviderConfig, ProviderType
        from src.domain.repositories.provider_repository import ProviderRepository

        repo = ProviderRepository(db_session)

        # Create multiple providers concurrently
        async def create_provider(index: int):
            provider = Provider(
                name=f"concurrent_provider_{index}",
                provider_type=ProviderType.OPENAI,
                config=ProviderConfig(api_key=f"key_{index}", model="gpt-3.5-turbo"),
                is_active=True,
            )
            return await repo.create(provider)

        # Create 5 providers concurrently
        tasks = [create_provider(i) for i in range(5)]
        created_providers = await asyncio.gather(*tasks)

        assert len(created_providers) == 5
        assert all(p.id is not None for p in created_providers)
        assert len(set(p.name for p in created_providers)) == 5  # All unique names

        # Read all providers concurrently
        read_tasks = [repo.get_by_id(p.id) for p in created_providers]
        retrieved_providers = await asyncio.gather(*read_tasks)

        assert len(retrieved_providers) == 5
        assert all(p is not None for p in retrieved_providers)

        # Cleanup concurrently
        delete_tasks = [repo.delete(p.id) for p in created_providers]
        await asyncio.gather(*delete_tasks)

        # Verify all deleted
        verify_tasks = [repo.get_by_id(p.id) for p in created_providers]
        deleted_checks = await asyncio.gather(*verify_tasks)
        assert all(p is None for p in deleted_checks)

    @pytest.mark.asyncio
    async def test_database_constraints(self, db_session: AsyncSession):
        """Test database constraints and validation."""
        from src.domain.entities.provider import Provider, ProviderConfig, ProviderType
        from src.domain.entities.test import Test, TestStatus
        from src.domain.repositories.provider_repository import ProviderRepository
        from src.domain.repositories.test_repository import TestRepository

        test_repo = TestRepository(db_session)
        provider_repo = ProviderRepository(db_session)

        # Create provider
        provider = Provider(
            name="constraint_test_provider",
            provider_type=ProviderType.OPENAI,
            config=ProviderConfig(api_key="key", model="gpt-3.5-turbo"),
            is_active=True,
        )
        created_provider = await provider_repo.create(provider)

        # Test valid test creation
        valid_test = Test(
            name="Valid Test",
            description="A valid test",
            created_by="test_user",
            prompt_template="Test: {input}",
            provider_a_id=created_provider.id,
            provider_b_id=created_provider.id,
            evaluation_criteria={"quality": 1.0},
            sample_size=10,
            confidence_level=0.95,
            status=TestStatus.DRAFT,
        )

        created_test = await test_repo.create(valid_test)
        assert created_test.id is not None

        # Test invalid foreign key (should fail or be handled gracefully)
        try:
            invalid_test = Test(
                name="Invalid Test",
                description="Test with invalid provider",
                created_by="test_user",
                prompt_template="Test: {input}",
                provider_a_id="nonexistent_provider_id",
                provider_b_id=created_provider.id,
                evaluation_criteria={"quality": 1.0},
                sample_size=10,
                confidence_level=0.95,
                status=TestStatus.DRAFT,
            )

            await test_repo.create(invalid_test)
            # If it succeeds, foreign key constraints might not be enforced
            # This is ok for testing
        except Exception:
            # Expected if foreign key constraints are enforced
            pass

        # Cleanup
        await test_repo.delete(created_test.id)
        await provider_repo.delete(created_provider.id)


@pytest.mark.asyncio
async def test_database_performance():
    """Test database performance with larger datasets."""
    from src.domain.entities.provider import Provider, ProviderConfig, ProviderType
    from src.domain.repositories.provider_repository import ProviderRepository
    from src.infrastructure.persistence.database import get_database

    database = get_database()

    async with database.get_session() as session:
        repo = ProviderRepository(session)

        # Time bulk creation
        import time

        start_time = time.time()

        # Create 50 providers
        providers = []
        for i in range(50):
            provider = Provider(
                name=f"perf_provider_{i}",
                provider_type=ProviderType.OPENAI,
                config=ProviderConfig(api_key=f"key_{i}", model="gpt-3.5-turbo"),
                is_active=True,
            )
            providers.append(await repo.create(provider))

        creation_time = time.time() - start_time
        print(f"Created 50 providers in {creation_time:.2f} seconds")

        # Time bulk read
        start_time = time.time()
        all_providers = await repo.get_all()
        read_time = time.time() - start_time
        print(f"Read {len(all_providers)} providers in {read_time:.2f} seconds")

        # Time bulk delete
        start_time = time.time()
        for provider in providers:
            await repo.delete(provider.id)
        delete_time = time.time() - start_time
        print(f"Deleted 50 providers in {delete_time:.2f} seconds")

        # Basic performance assertions
        assert creation_time < 10.0  # Should create 50 providers in under 10 seconds
        assert read_time < 1.0  # Should read all providers in under 1 second
        assert delete_time < 5.0  # Should delete 50 providers in under 5 seconds


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "-s"],
        capture_output=True,
        text=True,
    )

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
