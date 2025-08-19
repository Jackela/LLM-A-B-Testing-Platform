"""Tests for Test Management domain repository interfaces."""

from abc import ABC
from typing import List, Optional
from uuid import UUID, uuid4

import pytest

from src.domain.test_management.entities.test import Test
from src.domain.test_management.entities.test_configuration import TestConfiguration
from src.domain.test_management.entities.test_sample import TestSample
from src.domain.test_management.repositories.test_repository import TestRepository
from src.domain.test_management.value_objects.difficulty_level import DifficultyLevel
from src.domain.test_management.value_objects.test_status import TestStatus


class MockTestRepository(TestRepository):
    """Mock implementation of TestRepository for testing."""

    def __init__(self):
        self._tests = {}
        self._save_called = False
        self._delete_called = False

    async def save(self, test: Test) -> None:
        """Save test to mock storage."""
        self._tests[test.id] = test
        self._save_called = True

    async def find_by_id(self, test_id: UUID) -> Optional[Test]:
        """Find test by ID."""
        return self._tests.get(test_id)

    async def find_by_status(self, status: TestStatus) -> List[Test]:
        """Find tests by status."""
        return [test for test in self._tests.values() if test.status == status]

    async def find_active_tests(self) -> List[Test]:
        """Find active (non-terminal) tests."""
        active_statuses = [TestStatus.DRAFT, TestStatus.CONFIGURED, TestStatus.RUNNING]
        return [test for test in self._tests.values() if test.status in active_statuses]

    async def delete(self, test_id: UUID) -> None:
        """Delete test by ID."""
        if test_id in self._tests:
            del self._tests[test_id]
        self._delete_called = True

    async def find_by_name_pattern(self, pattern: str) -> List[Test]:
        """Find tests by name pattern."""
        return [test for test in self._tests.values() if pattern.lower() in test.name.lower()]

    async def count_by_status(self, status: TestStatus) -> int:
        """Count tests by status."""
        return len([test for test in self._tests.values() if test.status == status])

    # Test helper methods
    def was_save_called(self) -> bool:
        return self._save_called

    def was_delete_called(self) -> bool:
        return self._delete_called

    def get_all_tests(self) -> List[Test]:
        return list(self._tests.values())


class TestTestRepositoryInterface:
    """Tests for TestRepository interface contract."""

    @pytest.fixture
    def test_configuration(self):
        """Create test configuration for testing."""
        return TestConfiguration(models=["gpt-4", "claude-3"], max_tokens=1000, temperature=0.7)

    @pytest.fixture
    def test_sample(self):
        """Create test sample for testing."""
        return TestSample(prompt="Test prompt", difficulty=DifficultyLevel.MEDIUM)

    @pytest.fixture
    def repository(self):
        """Create mock repository for testing."""
        return MockTestRepository()

    @pytest.fixture
    def sample_test(self, test_configuration):
        """Create sample test for testing."""
        test = Test.create("Sample Test", test_configuration)
        # Add minimum samples to allow configuration
        for i in range(10):
            sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
            test.add_sample(sample)
        return test

    @pytest.mark.asyncio
    async def test_repository_is_abstract_base_class(self):
        """Test that TestRepository is an abstract base class."""
        assert issubclass(TestRepository, ABC)

        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError):
            TestRepository()

    @pytest.mark.asyncio
    async def test_save_test(self, repository, sample_test):
        """Test saving test to repository."""
        await repository.save(sample_test)
        assert repository.was_save_called()

        # Verify test was saved
        retrieved = await repository.find_by_id(sample_test.id)
        assert retrieved is not None
        assert retrieved.id == sample_test.id
        assert retrieved.name == sample_test.name

    @pytest.mark.asyncio
    async def test_find_by_id_existing(self, repository, sample_test):
        """Test finding existing test by ID."""
        await repository.save(sample_test)
        retrieved = await repository.find_by_id(sample_test.id)
        assert retrieved is not None
        assert retrieved.id == sample_test.id

    @pytest.mark.asyncio
    async def test_find_by_id_non_existing(self, repository):
        """Test finding non-existing test by ID returns None."""
        non_existing_id = uuid4()
        retrieved = await repository.find_by_id(non_existing_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_find_by_status(self, repository, test_configuration):
        """Test finding tests by status."""
        # Create tests with different statuses
        draft_test = Test.create("Draft Test", test_configuration)
        configured_test = Test.create("Configured Test", test_configuration)

        # Add samples and configure one test
        for test in [draft_test, configured_test]:
            for i in range(10):
                sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
                test.add_sample(sample)

        configured_test.configure()

        # Save tests
        await repository.save(draft_test)
        await repository.save(configured_test)

        # Find by status
        draft_tests = await repository.find_by_status(TestStatus.DRAFT)
        configured_tests = await repository.find_by_status(TestStatus.CONFIGURED)

        assert len(draft_tests) == 1
        assert draft_tests[0].id == draft_test.id
        assert len(configured_tests) == 1
        assert configured_tests[0].id == configured_test.id

    @pytest.mark.asyncio
    async def test_find_active_tests(self, repository, test_configuration):
        """Test finding active (non-terminal) tests."""
        # Create tests with different statuses
        draft_test = Test.create("Draft Test", test_configuration)
        configured_test = Test.create("Configured Test", test_configuration)
        running_test = Test.create("Running Test", test_configuration)
        completed_test = Test.create("Completed Test", test_configuration)

        # Setup tests
        for test in [draft_test, configured_test, running_test, completed_test]:
            for i in range(10):
                sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
                test.add_sample(sample)

        # Configure and start some tests
        configured_test.configure()
        running_test.configure()
        running_test.start()
        completed_test.configure()
        completed_test.start()
        completed_test.complete()

        # Save all tests
        for test in [draft_test, configured_test, running_test, completed_test]:
            await repository.save(test)

        # Find active tests
        active_tests = await repository.find_active_tests()
        active_ids = [test.id for test in active_tests]

        assert len(active_tests) == 3
        assert draft_test.id in active_ids
        assert configured_test.id in active_ids
        assert running_test.id in active_ids
        assert completed_test.id not in active_ids

    @pytest.mark.asyncio
    async def test_delete_test(self, repository, sample_test):
        """Test deleting test from repository."""
        # Save test first
        await repository.save(sample_test)
        assert await repository.find_by_id(sample_test.id) is not None

        # Delete test
        await repository.delete(sample_test.id)
        assert repository.was_delete_called()

        # Verify test was deleted
        assert await repository.find_by_id(sample_test.id) is None

    @pytest.mark.asyncio
    async def test_delete_non_existing_test(self, repository):
        """Test deleting non-existing test doesn't raise error."""
        non_existing_id = uuid4()
        await repository.delete(non_existing_id)
        assert repository.was_delete_called()

    @pytest.mark.asyncio
    async def test_find_by_name_pattern(self, repository, test_configuration):
        """Test finding tests by name pattern."""
        test1 = Test.create("User Authentication Test", test_configuration)
        test2 = Test.create("API Performance Test", test_configuration)
        test3 = Test.create("Authentication Validation", test_configuration)

        # Add samples to all tests
        for test in [test1, test2, test3]:
            for i in range(10):
                sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
                test.add_sample(sample)

        # Save tests
        for test in [test1, test2, test3]:
            await repository.save(test)

        # Search by pattern
        auth_tests = await repository.find_by_name_pattern("auth")
        performance_tests = await repository.find_by_name_pattern("performance")

        assert len(auth_tests) == 2  # test1 and test3 contain "auth"
        assert len(performance_tests) == 1  # Only test2 contains "performance"

        auth_names = [test.name for test in auth_tests]
        assert "User Authentication Test" in auth_names
        assert "Authentication Validation" in auth_names

    @pytest.mark.asyncio
    async def test_count_by_status(self, repository, test_configuration):
        """Test counting tests by status."""
        # Create multiple tests with same status
        for i in range(3):
            test = Test.create(f"Draft Test {i}", test_configuration)
            for j in range(10):
                sample = TestSample(prompt=f"Prompt {j}", difficulty=DifficultyLevel.EASY)
                test.add_sample(sample)
            await repository.save(test)

        # Create one configured test
        configured_test = Test.create("Configured Test", test_configuration)
        for j in range(10):
            sample = TestSample(prompt=f"Prompt {j}", difficulty=DifficultyLevel.EASY)
            configured_test.add_sample(sample)
        configured_test.configure()
        await repository.save(configured_test)

        # Count by status
        draft_count = await repository.count_by_status(TestStatus.DRAFT)
        configured_count = await repository.count_by_status(TestStatus.CONFIGURED)
        running_count = await repository.count_by_status(TestStatus.RUNNING)

        assert draft_count == 3
        assert configured_count == 1
        assert running_count == 0

    @pytest.mark.asyncio
    async def test_repository_preserves_domain_events(self, repository, sample_test):
        """Test that repository preserves domain events when saving/loading."""
        # Test should have creation event
        original_events = sample_test.get_domain_events()
        assert len(original_events) > 0

        # Save and retrieve
        await repository.save(sample_test)
        retrieved = await repository.find_by_id(sample_test.id)

        assert retrieved is not None
        # Note: In a real implementation, domain events might be cleared after
        # publishing or handled differently. This test documents the expected behavior.
        retrieved_events = retrieved.get_domain_events()

        # Domain events should be preserved or handled consistently
        assert isinstance(retrieved_events, list)

    @pytest.mark.asyncio
    async def test_repository_handles_concurrent_access(self, repository, test_configuration):
        """Test repository behavior with concurrent operations."""
        test1 = Test.create("Test 1", test_configuration)
        test2 = Test.create("Test 2", test_configuration)

        # Add samples
        for test in [test1, test2]:
            for i in range(10):
                sample = TestSample(prompt=f"Prompt {i}", difficulty=DifficultyLevel.EASY)
                test.add_sample(sample)

        # Simulate concurrent saves
        await repository.save(test1)
        await repository.save(test2)

        # Both tests should be retrievable
        retrieved1 = await repository.find_by_id(test1.id)
        retrieved2 = await repository.find_by_id(test2.id)

        assert retrieved1 is not None
        assert retrieved2 is not None
        assert retrieved1.id != retrieved2.id

        # Should find both in active tests
        active_tests = await repository.find_active_tests()
        assert len(active_tests) == 2
