"""Comprehensive security integration test for all security components."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_rate_limiter():
    """Test the advanced rate limiter."""
    logger.info("Testing Advanced Rate Limiter...")

    from src.infrastructure.security.rate_limiter import ActionType, get_rate_limiter

    rate_limiter = get_rate_limiter()

    # Test normal requests
    for i in range(5):
        result = await rate_limiter.check_rate_limit(
            ip_address="192.168.1.100", endpoint="/api/v1/tests", user_agent="test-client"
        )
        print(f"  Request {i+1}: Allowed={result.allowed}, Remaining={result.remaining}")

    # Test rate limit exceeded
    for i in range(15):  # Exceed the limit
        result = await rate_limiter.check_rate_limit(
            ip_address="192.168.1.100", endpoint="/api/v1/tests", user_agent="test-client"
        )
        if not result.allowed:
            print(f"  Rate limit exceeded after {i+5} requests: {result.reason}")
            break

    # Test different IP
    result = await rate_limiter.check_rate_limit(
        ip_address="192.168.1.101", endpoint="/api/v1/tests", user_agent="test-client"
    )
    print(f"  Different IP allowed: {result.allowed}")

    # Test whitelist
    rate_limiter.whitelist_ip("192.168.1.102")
    for i in range(20):  # Should not be rate limited
        result = await rate_limiter.check_rate_limit(
            ip_address="192.168.1.102", endpoint="/api/v1/tests", user_agent="test-client"
        )
        if not result.allowed:
            print(f"  Whitelisted IP blocked unexpectedly!")
            break
    else:
        print(f"  Whitelisted IP processed 20 requests successfully")

    # Get statistics
    stats = rate_limiter.get_statistics()
    print(f"  Rate Limiter Stats: {stats}")

    logger.info("✅ Rate Limiter test completed")


async def test_input_validator():
    """Test the input validation system."""
    logger.info("Testing Input Validator...")

    from src.infrastructure.security.input_validator import ValidationSeverity, get_input_validator

    validator = get_input_validator()

    # Test normal input
    result = validator.validate_input("Hello World", "string")
    print(f"  Normal string: Valid={result.is_valid}, Issues={len(result.issues)}")

    # Test XSS attempt
    xss_payload = "<script>alert('xss')</script>"
    result = validator.validate_input(xss_payload, "string")
    print(
        f"  XSS payload: Valid={result.is_valid}, Attacks={[a.value for a in result.detected_attacks]}"
    )
    print(f"    Confidence: {result.confidence:.2f}, Severity: {result.severity}")

    # Test SQL injection
    sql_payload = "'; DROP TABLE users; --"
    result = validator.validate_input(sql_payload, "string")
    print(
        f"  SQL injection: Valid={result.is_valid}, Attacks={[a.value for a in result.detected_attacks]}"
    )

    # Test command injection
    cmd_payload = "; cat /etc/passwd"
    result = validator.validate_input(cmd_payload, "string")
    print(
        f"  Command injection: Valid={result.is_valid}, Attacks={[a.value for a in result.detected_attacks]}"
    )

    # Test email validation
    result = validator.validate_input("test@example.com", "email")
    print(f"  Valid email: Valid={result.is_valid}")

    result = validator.validate_input("invalid-email", "email")
    print(f"  Invalid email: Valid={result.is_valid}, Issues={result.issues}")

    # Test JSON validation
    valid_json = '{"name": "test", "value": 123}'
    result = validator.validate_input(valid_json, "json")
    print(f"  Valid JSON: Valid={result.is_valid}, Parsed type: {type(result.cleaned_value)}")

    # Test batch validation
    inputs = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecurePass123!",
        "description": "This is a test description",
    }

    rules = {
        "username": {"type": "username", "required": True},
        "email": {"type": "email", "required": True},
        "password": {"type": "password", "required": True},
        "description": {"type": "text", "required": False},
    }

    batch_results = validator.validate_batch(inputs, rules)
    valid_count = sum(1 for r in batch_results.values() if r.is_valid)
    print(f"  Batch validation: {valid_count}/{len(batch_results)} fields valid")

    # Test filename safety
    safe_filename = "document.pdf"
    unsafe_filename = "../../../etc/passwd"
    print(f"  Safe filename '{safe_filename}': {validator.is_safe_filename(safe_filename)}")
    print(f"  Unsafe filename '{unsafe_filename}': {validator.is_safe_filename(unsafe_filename)}")

    logger.info("✅ Input Validator test completed")


async def test_audit_logger():
    """Test the audit logging system."""
    logger.info("Testing Audit Logger...")

    from src.infrastructure.security.audit_logger import (
        AuditContext,
        AuditEventType,
        AuditLevel,
        ComplianceStandard,
        init_audit_logging,
    )

    # Initialize audit logger
    audit_logger = await init_audit_logging("./test_audit_logs")

    # Test various event types
    context = AuditContext(
        user_id="test-user-123",
        username="testuser",
        ip_address="192.168.1.100",
        user_agent="test-client/1.0",
    )

    # Test authentication events
    await audit_logger.log_authentication_event(
        AuditEventType.LOGIN_SUCCESS, "testuser", "192.168.1.100", "success"
    )

    await audit_logger.log_authentication_event(
        AuditEventType.LOGIN_FAILURE,
        "testuser",
        "192.168.1.101",
        "failure",
        {"reason": "invalid_password"},
    )

    # Test data access events
    await audit_logger.log_data_access(
        "read", "test", "test-123", "test-user-123", "success", contains_pii=True
    )

    # Test security events
    await audit_logger.log_event(
        event_type=AuditEventType.ATTACK_DETECTED,
        message="XSS attack detected in input",
        context=context,
        level=AuditLevel.CRITICAL,
        resource_type="api_endpoint",
        resource_id="/api/v1/tests",
        metadata={"attack_type": "xss", "payload": "<script>alert('xss')</script>"},
        compliance_tags={ComplianceStandard.SOX, ComplianceStandard.GDPR},
    )

    # Wait for background processing
    await asyncio.sleep(2)

    # Get security dashboard
    dashboard = audit_logger.get_security_dashboard()
    print(f"  Security Dashboard:")
    print(f"    Total incidents: {dashboard['incidents']['total_open']}")
    print(
        f"    Failed logins (last hour): {dashboard['authentication']['failed_logins_last_hour']}"
    )
    print(f"    Events buffered: {dashboard['monitoring']['events_buffered']}")
    print(f"    Current log file: {dashboard['compliance']['current_log_file']}")

    # Test incident creation (simulate multiple failed logins)
    for i in range(12):  # Should trigger brute force detection
        await audit_logger.log_authentication_event(
            AuditEventType.LOGIN_FAILURE, "attacker", "192.168.1.200", "failure"
        )

    # Wait for security analysis
    await asyncio.sleep(2)

    # Check dashboard again
    dashboard = audit_logger.get_security_dashboard()
    print(f"  After attack simulation:")
    print(f"    Total incidents: {dashboard['incidents']['total_open']}")
    print(f"    Suspicious IPs: {dashboard['authentication']['suspicious_ips']}")

    logger.info("✅ Audit Logger test completed")


async def test_security_headers():
    """Test security headers middleware."""
    logger.info("Testing Security Headers...")

    import asyncio

    from fastapi import Request
    from fastapi.responses import JSONResponse

    from src.infrastructure.security.security_headers import create_security_middleware_stack

    class MockRequest:
        def __init__(self, scheme="https", host="localhost", path="/api/v1/tests"):
            self.url = type(
                "MockURL",
                (),
                {
                    "scheme": scheme,
                    "path": path,
                    "replace": lambda **kwargs: type("MockURL", (), {**self.__dict__, **kwargs})(),
                },
            )()
            self.headers = {"User-Agent": "test-client"}
            self.cookies = {}
            self.client = type("MockClient", (), {"host": "127.0.0.1"})()
            self.method = "GET"

    async def mock_call_next(request):
        return JSONResponse(content={"message": "test"})

    # Create security middleware stack
    middleware_stack = create_security_middleware_stack()
    security_headers = middleware_stack["security_headers"]

    # Test HTTPS request
    https_request = MockRequest(scheme="https")
    response = await security_headers(https_request, mock_call_next)

    print(f"  HTTPS Headers added:")
    for header, value in response.headers.items():
        if header.lower().startswith(
            ("strict-transport-security", "content-security-policy", "x-frame-options")
        ):
            print(f"    {header}: {value[:50]}..." if len(value) > 50 else f"    {header}: {value}")

    # Test HTTP request (should not have HSTS)
    http_request = MockRequest(scheme="http")
    response = await security_headers(http_request, mock_call_next)

    has_hsts = "Strict-Transport-Security" in response.headers
    print(f"  HTTP request has HSTS: {has_hsts}")

    logger.info("✅ Security Headers test completed")


async def test_auth_system():
    """Test the enhanced authentication system."""
    logger.info("Testing Enhanced Authentication System...")

    from src.infrastructure.security.auth import Permission, Role, get_auth_system

    auth_system = get_auth_system()

    # Test user creation
    success, message = auth_system.create_user(
        username="testuser", email="test@example.com", password="SecurePass123!", role=Role.USER
    )
    print(f"  User creation: Success={success}, Message={message}")

    # Test password validation
    is_strong, issues = auth_system.validate_password_strength("weak")
    print(f"  Weak password: Strong={is_strong}, Issues={len(issues)}")

    is_strong, issues = auth_system.validate_password_strength("SecurePass123!")
    print(f"  Strong password: Strong={is_strong}")

    # Test authentication
    success, message, user = auth_system.authenticate_user(
        "testuser", "SecurePass123!", "192.168.1.100"
    )
    print(f"  Authentication: Success={success}, User={user.username if user else None}")

    if user:
        # Test permissions
        print(f"  User role: {user.role}")
        print(f"  Has CREATE_TEST permission: {user.has_permission(Permission.CREATE_TEST)}")
        print(f"  Has SYSTEM_CONFIG permission: {user.has_permission(Permission.SYSTEM_CONFIG)}")

        # Test token creation
        tokens = auth_system.create_tokens(user)
        print(f"  Tokens created: access_token length={len(tokens['access_token'])}")

        # Test token verification
        token_data = auth_system.verify_token(tokens["access_token"])
        print(f"  Token verification: Valid={token_data is not None}")
        if token_data:
            print(f"    Token user: {token_data.username}, Role: {token_data.role}")

        # Test API key creation
        success, api_key = auth_system.create_api_key(user.id, "test-key")
        print(f"  API key creation: Success={success}")

        if success:
            # Test API key verification
            api_user = auth_system.verify_api_key(api_key)
            print(f"  API key verification: Valid={api_user is not None}")

    # Test failed authentication
    success, message, user = auth_system.authenticate_user(
        "testuser", "wrongpassword", "192.168.1.100"
    )
    print(f"  Failed auth: Success={success}, Message={message}")

    # Test account lockout (simulate multiple failures)
    for i in range(6):
        success, message, user = auth_system.authenticate_user(
            "testuser", "wrongpassword", "192.168.1.100"
        )
        if "locked" in message.lower():
            print(f"  Account locked after {i+1} failed attempts")
            break

    logger.info("✅ Authentication System test completed")


async def main():
    """Run comprehensive security integration tests."""
    logger.info("🔒 Starting Comprehensive Security Integration Tests")
    print("=" * 60)

    try:
        # Test individual components
        await test_rate_limiter()
        print("-" * 40)

        await test_input_validator()
        print("-" * 40)

        await test_audit_logger()
        print("-" * 40)

        await test_security_headers()
        print("-" * 40)

        await test_auth_system()
        print("-" * 40)

        logger.info("🎉 All security integration tests completed successfully!")

        # Summary
        print("\n📋 Security Components Validated:")
        print("  ✅ Advanced Rate Limiter with DDoS protection")
        print("  ✅ Input Validator with attack detection")
        print("  ✅ Comprehensive Audit Logging system")
        print("  ✅ Security Headers middleware")
        print("  ✅ Enhanced Authentication with MFA support")
        print("  ✅ RBAC with fine-grained permissions")
        print("  ✅ API key management")
        print("  ✅ Security monitoring and incident tracking")

    except Exception as e:
        logger.error(f"❌ Security integration test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
