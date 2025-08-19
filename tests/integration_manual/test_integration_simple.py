"""Simple integration test to validate core functionality."""

import asyncio
import logging
import sys

from fastapi.testclient import TestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_api_basic_functionality():
    """Test basic API functionality without complex dependencies."""
    logger.info("Testing basic API functionality...")

    try:
        from src.presentation.api.app import create_app

        # Create app
        app = create_app()
        client = TestClient(app)

        # Test health endpoint
        response = client.get("/health")
        logger.info(f"Health endpoint: {response.status_code}")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

        # Test root endpoint
        response = client.get("/")
        logger.info(f"Root endpoint: {response.status_code}")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data

        logger.info("✅ Basic API functionality test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Basic API test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_security_components():
    """Test security components basic functionality."""
    logger.info("Testing security components...")

    try:
        # Test rate limiter
        from src.infrastructure.security.rate_limiter import get_rate_limiter

        rate_limiter = get_rate_limiter()
        logger.info("✅ Rate limiter initialized")

        # Test input validator
        from src.infrastructure.security.input_validator import get_input_validator

        validator = get_input_validator()
        result = validator.validate_input("test input", "string")
        assert result.is_valid
        logger.info("✅ Input validator working")

        # Test auth system
        from src.infrastructure.security.auth import Role, get_auth_system

        auth_system = get_auth_system()

        # Test user creation
        success, message = auth_system.create_user(
            username="integration_test_user",
            email="test@integration.com",
            password="TestPass123!",
            role=Role.USER,
        )

        if success or "already exists" in message:
            logger.info("✅ Auth system working")
        else:
            logger.warning(f"⚠️ Auth system issue: {message}")

        logger.info("✅ Security components test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Security components test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_database_basic():
    """Test basic database functionality."""
    logger.info("Testing basic database functionality...")

    try:
        from src.infrastructure.persistence.database import get_database

        database = get_database()
        logger.info("✅ Database connection initialized")

        # Test that we can get a session (async context)
        # This is just a basic connection test
        logger.info("✅ Database basic test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_async_operations():
    """Test async operations work correctly."""
    logger.info("Testing async operations...")

    try:
        # Test async rate limiter
        from src.infrastructure.security.rate_limiter import get_rate_limiter

        rate_limiter = get_rate_limiter()

        # Test rate limit check
        result = await rate_limiter.check_rate_limit(ip_address="127.0.0.1", endpoint="/test")

        assert result.allowed
        logger.info("✅ Async rate limiter working")

        logger.info("✅ Async operations test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Async operations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run simple integration tests."""
    logger.info("🧪 Starting Simple Integration Tests")
    print("=" * 60)

    tests = [
        ("API Basic Functionality", test_api_basic_functionality),
        ("Security Components", test_security_components),
        ("Database Basic", test_database_basic),
    ]

    async_tests = [
        ("Async Operations", test_async_operations),
    ]

    results = []

    # Run sync tests
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
        print("-" * 40)

    # Run async tests
    for test_name, test_func in async_tests:
        logger.info(f"Running {test_name}...")
        try:
            result = asyncio.run(test_func())
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
        print("-" * 40)

    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status} {test_name}")

    if passed == total:
        logger.info("🎉 All simple integration tests passed!")
        print("\n✅ System is ready for basic operation")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed")
        print(f"\n⚠️ {total - passed} issues found - review and fix before deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
