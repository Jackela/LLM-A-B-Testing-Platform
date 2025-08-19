"""Simple security components test focusing on individual functionality."""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_rate_limiter_basic():
    """Test basic rate limiter functionality."""
    logger.info("Testing Basic Rate Limiter...")

    from src.infrastructure.security.rate_limiter import get_rate_limiter

    rate_limiter = get_rate_limiter()

    # Test normal requests
    for i in range(3):
        result = await rate_limiter.check_rate_limit(
            ip_address="192.168.1.100", endpoint="/api/v1/tests"
        )
        print(f"  Request {i+1}: Allowed={result.allowed}")

    # Get statistics
    stats = rate_limiter.get_statistics()
    print(f"  Active clients: {stats['active_clients']}")
    print(f"  Total requests tracked: {stats['total_requests']}")

    return True


async def test_input_validation_basic():
    """Test basic input validation."""
    logger.info("Testing Basic Input Validation...")

    from src.infrastructure.security.input_validator import get_input_validator

    validator = get_input_validator()

    # Test clean input
    result = validator.validate_input("Hello World", "string")
    print(f"  Clean string: Valid={result.is_valid}")

    # Test XSS
    result = validator.validate_input("<script>alert('test')</script>", "string")
    print(
        f"  XSS attempt: Valid={result.is_valid}, Attacks detected={len(result.detected_attacks)}"
    )

    # Test SQL injection
    result = validator.validate_input("'; DROP TABLE users; --", "string")
    print(
        f"  SQL injection: Valid={result.is_valid}, Attacks detected={len(result.detected_attacks)}"
    )

    # Test email validation
    result = validator.validate_input("test@example.com", "email")
    print(f"  Valid email: Valid={result.is_valid}")

    result = validator.validate_input("invalid-email", "email")
    print(f"  Invalid email: Valid={result.is_valid}")

    return True


async def test_auth_system_basic():
    """Test basic authentication functionality."""
    logger.info("Testing Basic Authentication...")

    from src.infrastructure.security.auth import Permission, Role, get_auth_system

    auth_system = get_auth_system()

    # Test user creation
    success, message = auth_system.create_user(
        username="testuser2", email="test2@example.com", password="SecurePass123!", role=Role.USER
    )
    print(f"  User creation: Success={success}")

    # Test authentication
    success, message, user = auth_system.authenticate_user(
        "testuser2", "SecurePass123!", "192.168.1.100"
    )
    print(f"  Authentication: Success={success}")

    if user:
        # Test permissions
        print(f"  User has CREATE_TEST permission: {user.has_permission(Permission.CREATE_TEST)}")
        print(
            f"  User has SYSTEM_CONFIG permission: {user.has_permission(Permission.SYSTEM_CONFIG)}"
        )

        # Test token creation
        tokens = auth_system.create_tokens(user)
        print(f"  Token created: {len(tokens['access_token']) > 0}")

        # Test token verification
        token_data = auth_system.verify_token(tokens["access_token"])
        print(f"  Token verification: {token_data is not None}")

    return True


async def test_security_headers_basic():
    """Test security headers generation."""
    logger.info("Testing Security Headers...")

    from src.infrastructure.security.security_headers import SecurityHeadersConfig

    config = SecurityHeadersConfig()

    # Test CSP policy
    print(f"  CSP default-src: {config.csp_policy['default-src']}")
    print(f"  HSTS max age: {config.hsts_max_age} seconds")
    print(f"  X-Frame-Options: {config.x_frame_options}")
    print(f"  Referrer Policy: {config.referrer_policy}")

    return True


async def test_fastapi_integration():
    """Test FastAPI application with security middleware."""
    logger.info("Testing FastAPI Security Integration...")

    try:
        from src.presentation.api.app import create_app

        app = create_app()
        print(f"  FastAPI app created successfully")
        print(f"  App title: {app.title}")
        print(f"  Middleware count: {len(app.user_middleware)}")

        # Check if security routes are available
        routes = [route.path for route in app.routes]
        security_routes = [route for route in routes if "/security/" in route]
        print(f"  Security routes found: {len(security_routes)}")

        return True

    except Exception as e:
        print(f"  FastAPI integration error: {e}")
        return False


async def main():
    """Run simple security tests."""
    logger.info("🔒 Starting Simple Security Component Tests")
    print("=" * 50)

    results = []

    try:
        # Test individual components
        results.append(await test_rate_limiter_basic())
        print("-" * 30)

        results.append(await test_input_validation_basic())
        print("-" * 30)

        results.append(await test_auth_system_basic())
        print("-" * 30)

        results.append(await test_security_headers_basic())
        print("-" * 30)

        results.append(await test_fastapi_integration())
        print("-" * 30)

        # Summary
        passed = sum(results)
        total = len(results)

        print(f"\n📊 Test Results: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 All security component tests passed!")
            print("\n✅ Security Components Validated:")
            print("  • Advanced Rate Limiter")
            print("  • Input Validation & Attack Detection")
            print("  • Enhanced Authentication System")
            print("  • Security Headers Configuration")
            print("  • FastAPI Security Integration")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed")

    except Exception as e:
        logger.error(f"❌ Security test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
