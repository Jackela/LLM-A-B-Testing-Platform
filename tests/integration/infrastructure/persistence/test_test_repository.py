"""Integration tests for TestRepository implementation."""

from datetime import datetime
from uuid import uuid4

import pytest
import pytest_asyncio

from src.domain.test_management.entities.test import Test
from src.domain.test_management.entities.test_configuration import TestConfiguration
from src.domain.test_management.entities.test_sample import TestSample
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus
from src.infrastructure.persistence.database import DatabaseConfig, DatabaseManager
from src.infrastructure.persistence.repositories.test_repository_impl import TestRepositoryImpl


@pytest_asyncio.fixture
async def database_manager():
    """Create test database manager."""
    config = DatabaseConfig(database_url="sqlite+aiosqlite:///:memory:", echo=False)
    manager = DatabaseManager(config)

    # Create tables
    from src.infrastructure.persistence.database import Base

    async with manager.get_async_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield manager

    await manager.close()


@pytest_asyncio.fixture
async def repository(database_manager):
    """Create test repository."""
    session_factory = database_manager.get_async_session_factory()
    return TestRepositoryImpl(session_factory)


@pytest_asyncio.fixture
async def sample_test():
    """Create a sample test for testing."""
    configuration = TestConfiguration(
        models=["gpt-4", "claude-3"],
        evaluation_templates=[uuid4()],
        randomization_seed="test-seed",
        parallel_executions={"default": 2},
        timeout_seconds={"default": 30},
        retry_config={"max_retries": 3},
    )

    test = Test.create(name="Test Integration Test", configuration=configuration)

    # Add some samples
    samples = [
        TestSample(
            prompt="Test prompt 1",
            difficulty=DifficultyLevel.EASY,
            expected_output="Expected output 1",
        ),
        TestSample(
            prompt="Test prompt 2",
            difficulty=DifficultyLevel.MEDIUM,
            expected_output="Expected output 2",
        ),
        TestSample(
            prompt="Test prompt 3",
            difficulty=DifficultyLevel.HARD,
            expected_output="Expected output 3",
        ),
    ]

    for sample in samples:
        test.add_sample(sample)

    return test


class TestTestRepository:
    """Test cases for TestRepository implementation."""

    async def test_save_and_find_by_id(self, repository, sample_test):
        """Test saving and retrieving a test by ID."""
        # Save the test
        await repository.save(sample_test)

        # Retrieve the test
        retrieved_test = await repository.find_by_id(sample_test.id)

        assert retrieved_test is not None
        assert retrieved_test.id == sample_test.id
        assert retrieved_test.name == sample_test.name
        assert retrieved_test.status == sample_test.status
        assert len(retrieved_test.samples) == len(sample_test.samples)

    async def test_find_by_id_not_found(self, repository):
        """Test finding non-existent test returns None."""
        non_existent_id = uuid4()
        result = await repository.find_by_id(non_existent_id)
        assert result is None

    async def test_find_by_status(self, repository, sample_test):
        """Test finding tests by status."""
        # Save test in draft status
        await repository.save(sample_test)

        # Find by status
        draft_tests = await repository.find_by_status(TestStatus.DRAFT)
        assert len(draft_tests) == 1
        assert draft_tests[0].id == sample_test.id

        # Test status that should have no results
        running_tests = await repository.find_by_status(TestStatus.RUNNING)
        assert len(running_tests) == 0

    async def test_find_active_tests(self, repository, sample_test):
        """Test finding active tests."""
        # Save test in draft status (active)
        await repository.save(sample_test)

        active_tests = await repository.find_active_tests()
        assert len(active_tests) == 1
        assert active_tests[0].id == sample_test.id

    async def test_count_by_status(self, repository, sample_test):
        """Test counting tests by status."""
        # Save test
        await repository.save(sample_test)

        # Count by status
        draft_count = await repository.count_by_status(TestStatus.DRAFT)
        assert draft_count == 1

        running_count = await repository.count_by_status(TestStatus.RUNNING)
        assert running_count == 0

    async def test_find_by_name_pattern(self, repository, sample_test):
        """Test finding tests by name pattern."""
        # Save test
        await repository.save(sample_test)

        # Find by pattern
        matches = await repository.find_by_name_pattern("Integration")
        assert len(matches) == 1
        assert matches[0].id == sample_test.id

        # Test pattern that should not match
        no_matches = await repository.find_by_name_pattern("NonExistent")
        assert len(no_matches) == 0

    async def test_update_test_status(self, repository, sample_test):
        """Test updating test status efficiently."""
        # Save test
        await repository.save(sample_test)

        # Update status
        success = await repository.update_test_status(sample_test.id, TestStatus.CONFIGURED)
        assert success is True

        # Verify update
        updated_test = await repository.find_by_id(sample_test.id)
        assert updated_test.status == TestStatus.CONFIGURED

    async def test_exists(self, repository, sample_test):
        """Test checking if test exists."""
        # Test non-existent
        exists_before = await repository.exists(sample_test.id)
        assert exists_before is False

        # Save test
        await repository.save(sample_test)

        # Test exists
        exists_after = await repository.exists(sample_test.id)
        assert exists_after is True

    async def test_delete(self, repository, sample_test):
        """Test deleting a test."""
        # Save test
        await repository.save(sample_test)

        # Verify exists
        exists_before = await repository.exists(sample_test.id)
        assert exists_before is True

        # Delete test
        await repository.delete(sample_test.id)

        # Verify deleted
        exists_after = await repository.exists(sample_test.id)
        assert exists_after is False

    async def test_get_test_summary(self, repository, sample_test):
        """Test getting test summary."""
        # Save test
        await repository.save(sample_test)

        # Get summary
        summary = await repository.get_test_summary(sample_test.id)

        assert summary is not None
        assert summary["id"] == str(sample_test.id)
        assert summary["name"] == sample_test.name
        assert summary["status"] == sample_test.status.value
        assert summary["sample_count"] == len(sample_test.samples)

    async def test_get_tests_summary(self, repository, sample_test):
        """Test getting tests summary list."""
        # Save test
        await repository.save(sample_test)

        # Get summaries
        summaries = await repository.get_tests_summary()

        assert len(summaries) == 1
        assert summaries[0]["id"] == str(sample_test.id)
        assert summaries[0]["name"] == sample_test.name

    async def test_find_tests_with_samples_count(self, repository, sample_test):
        """Test finding tests by sample count."""
        # Save test with 3 samples
        await repository.save(sample_test)

        # Find tests with at least 2 samples
        tests = await repository.find_tests_with_samples_count(min_samples=2)
        assert len(tests) == 1
        assert tests[0].id == sample_test.id

        # Find tests with at least 5 samples (should be empty)
        tests = await repository.find_tests_with_samples_count(min_samples=5)
        assert len(tests) == 0

    async def test_find_by_ids(self, repository, sample_test):
        """Test finding multiple tests by IDs."""
        # Save test
        await repository.save(sample_test)

        # Find by IDs
        test_ids = [sample_test.id, uuid4()]  # One existing, one non-existing
        tests = await repository.find_by_ids(test_ids)

        assert len(tests) == 1
        assert tests[0].id == sample_test.id

    async def test_domain_events_cleared_after_load(self, repository, sample_test):
        """Test that domain events are cleared when loading from database."""
        # Ensure test has domain events
        assert len(sample_test.get_domain_events()) > 0

        # Save test
        await repository.save(sample_test)

        # Load test
        loaded_test = await repository.find_by_id(sample_test.id)

        # Domain events should be cleared
        assert len(loaded_test.get_domain_events()) == 0

    async def test_sample_evaluation_results_preserved(self, repository, sample_test):
        """Test that sample evaluation results are preserved."""
        # Add evaluation results to samples
        for i, sample in enumerate(sample_test.samples):
            sample.add_evaluation_result(
                f"model_{i}", {"score": 0.8 + i * 0.05, "feedback": f"Good response {i}"}
            )

        # Save test
        await repository.save(sample_test)

        # Load test
        loaded_test = await repository.find_by_id(sample_test.id)

        # Verify evaluation results are preserved
        for i, sample in enumerate(loaded_test.samples):
            result = sample.get_evaluation_result(f"model_{i}")
            assert result is not None
            assert result["score"] == 0.8 + i * 0.05
            assert result["feedback"] == f"Good response {i}"

    async def test_concurrent_access(self, repository, sample_test):
        """Test concurrent access to repository."""
        import asyncio

        # Save initial test
        await repository.save(sample_test)

        async def update_test_name(suffix):
            test = await repository.find_by_id(sample_test.id)
            test.name = f"{test.name} - {suffix}"
            await repository.save(test)
            return test.name

        # Perform concurrent updates
        tasks = [update_test_name(f"Update{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)

        # Verify all updates succeeded
        assert len(results) == 3
        assert all("Update" in name for name in results)

        # Verify final state
        final_test = await repository.find_by_id(sample_test.id)
        assert "Update" in final_test.name
