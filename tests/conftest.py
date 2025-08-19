"""Enhanced global test configuration and fixtures."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Environment Configuration
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment configuration."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    os.environ["DISABLE_AUTHENTICATION"] = "true"
    os.environ["MOCK_EXTERNAL_SERVICES"] = "true"

    # Ensure test directories exist
    test_dirs = ["tests/temp", "tests/fixtures", "tests/logs", "tests/reports"]

    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    yield

    # Cleanup test environment
    for dir_path in test_dirs:
        try:
            import shutil

            if Path(dir_path).exists():
                shutil.rmtree(dir_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {dir_path}: {e}")


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainer, None, None]:
    """Provide PostgreSQL test container for integration tests."""
    with PostgresContainer("postgres:15-alpine") as postgres:
        # Wait for container to be ready
        postgres.get_connection_url()
        yield postgres


@pytest.fixture(scope="session")
def redis_container() -> Generator[RedisContainer, None, None]:
    """Provide Redis test container for integration tests."""
    with RedisContainer("redis:7-alpine") as redis:
        # Wait for container to be ready
        redis.get_connection_url()
        yield redis


@pytest.fixture(scope="function")
def test_database(postgres_container: PostgresContainer):
    """Provide isolated test database for each test function."""
    # Create test database URL
    db_url = postgres_container.get_connection_url()

    # Create engine and session for this test
    engine = create_engine(db_url, echo=False)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    from src.infrastructure.persistence.models.base import Base

    Base.metadata.create_all(bind=engine)

    # Provide session
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop all tables for clean state
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture(scope="function")
def test_redis(redis_container: RedisContainer):
    """Provide Redis connection for each test function."""
    import redis

    redis_url = redis_container.get_connection_url()
    redis_client = redis.from_url(redis_url)

    try:
        yield redis_client
    finally:
        redis_client.flushdb()  # Clear all data
        redis_client.close()


# =============================================================================
# Application Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def test_app():
    """Provide test FastAPI application."""
    from src.presentation.api.app import create_app

    # Override dependencies for testing
    app = create_app()

    # Mock external dependencies
    from src.presentation.api.dependencies.singleton_container import reset_container

    reset_container()

    return app


@pytest.fixture(scope="function")
def test_client(test_app):
    """Provide test client for API testing."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture(scope="function")
def authenticated_client(test_client: TestClient):
    """Provide authenticated test client."""
    # Create test user and get token
    test_user_data = {
        "username": "test_user",
        "password": "test_password",
        "email": "test@example.com",
    }

    # Register user
    register_response = test_client.post("/auth/register", json=test_user_data)

    # Login to get token
    login_response = test_client.post(
        "/auth/login",
        data={"username": test_user_data["username"], "password": test_user_data["password"]},
    )

    if login_response.status_code == 200:
        token = login_response.json()["access_token"]
        test_client.headers.update({"Authorization": f"Bearer {token}"})

    yield test_client


# =============================================================================
# Mock Service Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def mock_llm_providers():
    """Mock LLM provider services for testing."""
    from unittest.mock import MagicMock, patch

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="Mock OpenAI response"))
    ]

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value.content = [
        MagicMock(text="Mock Anthropic response")
    ]

    with (
        patch("openai.OpenAI", return_value=mock_openai),
        patch("anthropic.Anthropic", return_value=mock_anthropic),
    ):
        yield {"openai": mock_openai, "anthropic": mock_anthropic}


@pytest.fixture(scope="function")
def mock_external_services():
    """Mock all external services for isolated testing."""
    from unittest.mock import MagicMock, patch

    # Mock email service
    mock_email = MagicMock()
    mock_email.send_email.return_value = True

    # Mock monitoring service
    mock_monitoring = MagicMock()
    mock_monitoring.track_event.return_value = None

    # Mock file storage
    mock_storage = MagicMock()
    mock_storage.upload_file.return_value = "mock_file_url"

    with (
        patch("src.infrastructure.external.email_service", mock_email),
        patch("src.infrastructure.external.monitoring_service", mock_monitoring),
        patch("src.infrastructure.external.file_storage", mock_storage),
    ):
        yield {"email": mock_email, "monitoring": mock_monitoring, "storage": mock_storage}


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def sample_test_configuration():
    """Provide sample test configuration for testing."""
    return {
        "name": "Sample A/B Test",
        "description": "Test for comparing model performance",
        "model_a": {
            "model_id": "gpt-4",
            "provider": "openai",
            "parameters": {"temperature": 0.7, "max_tokens": 1000},
        },
        "model_b": {
            "model_id": "claude-3-opus",
            "provider": "anthropic",
            "parameters": {"temperature": 0.7, "max_tokens": 1000},
        },
        "evaluation_config": {
            "dimensions": ["accuracy", "helpfulness", "relevance"],
            "judges": ["gpt-4", "human"],
            "criteria": "Compare responses for accuracy and helpfulness",
        },
        "samples": [
            {
                "prompt": "What is the capital of France?",
                "expected_response": "Paris",
                "category": "factual",
            },
            {
                "prompt": "Explain quantum computing in simple terms",
                "expected_response": None,
                "category": "explanatory",
            },
        ],
    }


@pytest.fixture(scope="function")
def sample_evaluation_results():
    """Provide sample evaluation results for testing."""
    return [
        {
            "sample_id": "sample_1",
            "model_a_response": "Paris is the capital of France.",
            "model_b_response": "The capital of France is Paris.",
            "evaluation": {
                "accuracy": {"model_a": 1.0, "model_b": 1.0},
                "helpfulness": {"model_a": 0.9, "model_b": 0.95},
                "relevance": {"model_a": 1.0, "model_b": 1.0},
            },
            "winner": "model_b",
            "confidence": 0.6,
        },
        {
            "sample_id": "sample_2",
            "model_a_response": "Quantum computing uses quantum bits...",
            "model_b_response": "Quantum computing is a type of computing...",
            "evaluation": {
                "accuracy": {"model_a": 0.85, "model_b": 0.9},
                "helpfulness": {"model_a": 0.8, "model_b": 0.85},
                "relevance": {"model_a": 0.9, "model_b": 0.95},
            },
            "winner": "model_b",
            "confidence": 0.7,
        },
    ]


@pytest.fixture(scope="function")
def temp_directory():
    """Provide temporary directory for test file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# =============================================================================
# Async Test Support
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Provide event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def async_test_client():
    """Provide async test client for async API testing."""
    from httpx import AsyncClient

    from src.presentation.api.app import create_app

    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# =============================================================================
# Performance Testing Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def performance_monitor():
    """Provide performance monitoring for tests."""
    import time

    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.metrics = {}

        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss

        def stop(self):
            if self.start_time:
                self.metrics["execution_time"] = time.time() - self.start_time
                self.metrics["memory_delta"] = (
                    psutil.Process().memory_info().rss - self.start_memory
                )
            return self.metrics

    return PerformanceMonitor()


# =============================================================================
# Error Injection Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def error_injector():
    """Provide error injection utilities for testing error scenarios."""
    from unittest.mock import patch

    class ErrorInjector:
        def __init__(self):
            self.active_patches = []

        def inject_network_error(self, target_function):
            """Inject network error for testing."""

            def side_effect(*args, **kwargs):
                raise ConnectionError("Simulated network error")

            patcher = patch(target_function, side_effect=side_effect)
            mock = patcher.start()
            self.active_patches.append(patcher)
            return mock

        def inject_timeout_error(self, target_function, timeout=5.0):
            """Inject timeout error for testing."""
            import time

            def side_effect(*args, **kwargs):
                time.sleep(timeout + 0.1)  # Simulate timeout
                raise TimeoutError("Simulated timeout")

            patcher = patch(target_function, side_effect=side_effect)
            mock = patcher.start()
            self.active_patches.append(patcher)
            return mock

        def inject_authentication_error(self, target_function):
            """Inject authentication error for testing."""

            def side_effect(*args, **kwargs):
                raise PermissionError("Simulated authentication error")

            patcher = patch(target_function, side_effect=side_effect)
            mock = patcher.start()
            self.active_patches.append(patcher)
            return mock

        def cleanup(self):
            """Clean up all active patches."""
            for patcher in self.active_patches:
                patcher.stop()
            self.active_patches.clear()

    injector = ErrorInjector()
    yield injector
    injector.cleanup()


# =============================================================================
# Test Markers and Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "functional: Functional tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "external: Tests requiring external services")
    config.addinivalue_line("markers", "manual: Manual tests")


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Skip external tests if in CI without external services
    if "external" in item.keywords and os.getenv("SKIP_EXTERNAL_TESTS"):
        pytest.skip("Skipping external tests in this environment")

    # Skip slow tests in fast mode
    if "slow" in item.keywords and os.getenv("FAST_TESTS_ONLY"):
        pytest.skip("Skipping slow tests in fast mode")


def pytest_runtest_teardown(item):
    """Teardown for each test run."""
    # Clear any remaining mocks or patches
    from unittest.mock import patch

    patch.stopall()

    # Clear test-specific environment variables
    test_env_vars = [var for var in os.environ if var.startswith("TEST_")]
    for var in test_env_vars:
        del os.environ[var]


# =============================================================================
# Test Utilities
# =============================================================================


@pytest.fixture(scope="function")
def test_logger():
    """Provide test-specific logger."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)

    # Add memory handler to capture logs
    from logging.handlers import MemoryHandler

    memory_handler = MemoryHandler(capacity=1000)
    logger.addHandler(memory_handler)

    yield logger

    # Clean up handler
    logger.removeHandler(memory_handler)


@pytest.fixture(scope="function")
def assert_helpers():
    """Provide enhanced assertion helpers for tests."""

    class AssertHelpers:
        @staticmethod
        def assert_valid_uuid(uuid_string):
            """Assert that string is a valid UUID."""
            import uuid

            try:
                uuid.UUID(uuid_string)
            except ValueError:
                pytest.fail(f"'{uuid_string}' is not a valid UUID")

        @staticmethod
        def assert_valid_timestamp(timestamp_string):
            """Assert that string is a valid ISO timestamp."""
            from datetime import datetime

            try:
                datetime.fromisoformat(timestamp_string.replace("Z", "+00:00"))
            except ValueError:
                pytest.fail(f"'{timestamp_string}' is not a valid ISO timestamp")

        @staticmethod
        def assert_response_structure(response_data, expected_structure):
            """Assert that response has expected structure."""

            def check_structure(data, structure):
                if isinstance(structure, dict):
                    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
                    for key, value_type in structure.items():
                        assert key in data, f"Missing key: {key}"
                        check_structure(data[key], value_type)
                elif isinstance(structure, list):
                    assert isinstance(data, list), f"Expected list, got {type(data)}"
                    if structure and len(data) > 0:
                        check_structure(data[0], structure[0])
                elif isinstance(structure, type):
                    assert isinstance(data, structure), f"Expected {structure}, got {type(data)}"

            check_structure(response_data, expected_structure)

        @staticmethod
        def assert_performance_within_limits(execution_time, max_time, operation_name="Operation"):
            """Assert that operation completed within time limits."""
            assert (
                execution_time <= max_time
            ), f"{operation_name} took {execution_time:.3f}s, expected <= {max_time}s"

        @staticmethod
        def assert_no_memory_leaks(initial_memory, final_memory, tolerance_mb=10):
            """Assert that memory usage didn't increase significantly."""
            memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)
            assert (
                memory_increase_mb <= tolerance_mb
            ), f"Memory increased by {memory_increase_mb:.2f}MB, tolerance: {tolerance_mb}MB"

    return AssertHelpers()


# =============================================================================
# Session-level cleanup
# =============================================================================


def pytest_sessionfinish(session, exitstatus):
    """Clean up after all tests complete."""
    # Final cleanup of any persistent resources
    logger.info("Test session completed. Performing final cleanup...")

    # Clear any remaining environment variables
    test_vars = [var for var in os.environ if var.startswith(("TEST_", "TESTING"))]
    for var in test_vars:
        if var != "TESTING":  # Keep TESTING flag
            del os.environ[var]
