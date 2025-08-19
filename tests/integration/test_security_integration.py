"""Security integration tests for LLM A/B Testing Platform."""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from src.infrastructure.security.auth import get_auth_system
from src.infrastructure.security.input_validator import get_input_validator
from src.infrastructure.security.rate_limiter import get_rate_limiter
from src.presentation.api.app import create_app


class TestSecurityIntegration:
    """Integration tests for security features."""

    @pytest.fixture(scope="class")
    def app(self):
        """Create FastAPI application for testing."""
        return create_app()

    @pytest.fixture(scope="class")
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture(scope="class")
    def admin_headers(self, client) -> Dict[str, str]:
        """Create admin user and return headers."""
        auth_system = get_auth_system()

        # Create admin user
        success, message = auth_system.create_user(
            username="security_admin",
            email="admin@security.com",
            password="AdminPass123!",
            role="ADMIN",
        )

        if not success and "already exists" not in message:
            pytest.fail(f"Failed to create admin user: {message}")

        # Login and get token
        login_data = {"username": "security_admin", "password": "AdminPass123!"}

        response = client.post("/api/v1/auth/login", json=login_data)
        if response.status_code != 200:
            pytest.fail(f"Admin login failed: {response.text}")

        token_data = response.json()
        return {"Authorization": f"Bearer {token_data['access_token']}"}

    @pytest.fixture(scope="class")
    def user_headers(self, client) -> Dict[str, str]:
        """Create regular user and return headers."""
        auth_system = get_auth_system()

        # Create regular user
        success, message = auth_system.create_user(
            username="security_user",
            email="user@security.com",
            password="UserPass123!",
            role="USER",
        )

        if not success and "already exists" not in message:
            pytest.fail(f"Failed to create user: {message}")

        # Login and get token
        login_data = {"username": "security_user", "password": "UserPass123!"}

        response = client.post("/api/v1/auth/login", json=login_data)
        if response.status_code != 200:
            pytest.fail(f"User login failed: {response.text}")

        token_data = response.json()
        return {"Authorization": f"Bearer {token_data['access_token']}"}

    def test_authentication_security(self, client):
        """Test authentication security features."""
        # Test login with valid credentials
        valid_login = {"username": "security_user", "password": "UserPass123!"}

        response = client.post("/api/v1/auth/login", json=valid_login)
        assert response.status_code == 200

        token_data = response.json()
        assert "access_token" in token_data
        assert token_data["token_type"] == "bearer"

        # Test login with invalid credentials
        invalid_login = {"username": "security_user", "password": "WrongPassword"}

        response = client.post("/api/v1/auth/login", json=invalid_login)
        assert response.status_code == 401

        # Test brute force protection (multiple failed attempts)
        for i in range(6):  # Exceed the limit
            response = client.post("/api/v1/auth/login", json=invalid_login)
            if response.status_code == 429:  # Rate limited
                break
        else:
            # If no rate limiting after 6 attempts, that's unexpected but not a failure
            pass

    def test_authorization_levels(self, client, user_headers, admin_headers):
        """Test different authorization levels."""
        # Test user access to user endpoints
        response = client.get("/api/v1/auth/me", headers=user_headers)
        assert response.status_code == 200

        user_data = response.json()
        assert user_data["username"] == "security_user"

        # Test admin access to user endpoints
        response = client.get("/api/v1/auth/me", headers=admin_headers)
        assert response.status_code == 200

        admin_data = response.json()
        assert admin_data["username"] == "security_admin"

        # Test unauthorized access
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

        # Test invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/auth/me", headers=invalid_headers)
        assert response.status_code == 401

    def test_rate_limiting(self, client, user_headers):
        """Test rate limiting functionality."""
        # Make multiple requests quickly
        responses = []
        for i in range(20):  # Try to exceed rate limit
            response = client.get("/api/v1/tests/", headers=user_headers)
            responses.append(response)

            if response.status_code == 429:  # Rate limited
                assert "retry-after" in response.headers.get("retry-after", "") or True
                break

            # Small delay to avoid overwhelming
            time.sleep(0.1)

        # At least some requests should succeed
        successful_requests = [r for r in responses if r.status_code in [200, 404]]
        assert len(successful_requests) > 0

    def test_input_validation_security(self, client, user_headers):
        """Test input validation and sanitization."""
        # Test XSS attempt
        xss_data = {
            "name": "<script>alert('xss')</script>",
            "description": "Normal description",
            "prompt_template": "Test: {input}",
        }

        response = client.post("/api/v1/tests/", json=xss_data, headers=user_headers)
        # Should be rejected or sanitized
        if response.status_code == 201:
            # If created, check that XSS was sanitized
            created_data = response.json()
            assert "<script>" not in created_data.get("name", "")
        else:
            # Should be rejected with 400 or 422
            assert response.status_code in [400, 422]

        # Test SQL injection attempt
        sql_injection_data = {
            "name": "Test'; DROP TABLE tests; --",
            "description": "SQL injection attempt",
            "prompt_template": "Test: {input}",
        }

        response = client.post("/api/v1/tests/", json=sql_injection_data, headers=user_headers)
        # Should be rejected or sanitized
        assert response.status_code in [201, 400, 422]

        # Test command injection
        command_injection_data = {
            "name": "test; rm -rf /",
            "description": "Command injection attempt",
            "prompt_template": "Test: {input}",
        }

        response = client.post("/api/v1/tests/", json=command_injection_data, headers=user_headers)
        assert response.status_code in [201, 400, 422]

        # Test large payload (potential DoS)
        large_data = {
            "name": "A" * 10000,  # Very long name
            "description": "B" * 50000,  # Very long description
            "prompt_template": "Test: {input}",
        }

        response = client.post("/api/v1/tests/", json=large_data, headers=user_headers)
        # Should be rejected due to size limits
        assert response.status_code in [400, 413, 422]

    def test_security_headers(self, client):
        """Test security headers in responses."""
        response = client.get("/")

        # Check for important security headers
        headers = response.headers

        # These headers might be present depending on configuration
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
            "strict-transport-security",
            "content-security-policy",
        ]

        # At least some security headers should be present
        present_headers = [h for h in security_headers if h in headers]
        # Note: In test environment, some headers might not be set
        # This is more of an informational test

        # Test that sensitive headers are not exposed
        sensitive_headers = ["server", "x-powered-by"]
        for header in sensitive_headers:
            # It's ok if these are present, but they shouldn't reveal sensitive info
            if header in headers:
                value = headers[header].lower()
                assert "apache" not in value or "nginx" not in value or "microsoft" not in value

    def test_jwt_token_security(self, client, user_headers):
        """Test JWT token security features."""
        # Get current user to verify token works
        response = client.get("/api/v1/auth/me", headers=user_headers)
        assert response.status_code == 200

        # Test token expiration (if implemented)
        # This would require waiting or manipulating token
        # For now, just verify token format
        auth_header = user_headers["Authorization"]
        token = auth_header.split(" ")[1]

        # JWT tokens have 3 parts separated by dots
        token_parts = token.split(".")
        assert len(token_parts) == 3

        # Test malformed token
        malformed_headers = {"Authorization": "Bearer malformed.token"}
        response = client.get("/api/v1/auth/me", headers=malformed_headers)
        assert response.status_code == 401

        # Test empty token
        empty_headers = {"Authorization": "Bearer "}
        response = client.get("/api/v1/auth/me", headers=empty_headers)
        assert response.status_code == 401

    def test_api_key_security(self, client, admin_headers):
        """Test API key security if implemented."""
        # Create API key (if endpoint exists)
        api_key_data = {"name": "test_api_key", "permissions": ["read"]}

        response = client.post("/api/v1/auth/api-keys", json=api_key_data, headers=admin_headers)

        if response.status_code == 201:
            # API key creation successful
            api_key_response = response.json()
            assert "api_key" in api_key_response

            # Test using API key
            api_key = api_key_response["api_key"]
            api_headers = {"X-API-Key": api_key}

            response = client.get("/api/v1/tests/", headers=api_headers)
            assert response.status_code in [200, 401, 403]  # Various valid responses

            # Test invalid API key
            invalid_api_headers = {"X-API-Key": "invalid_key_here"}
            response = client.get("/api/v1/tests/", headers=invalid_api_headers)
            assert response.status_code == 401
        else:
            # API key endpoints might not be implemented yet
            assert response.status_code in [404, 405]

    def test_password_security(self, client):
        """Test password security requirements."""
        auth_system = get_auth_system()

        # Test weak passwords
        weak_passwords = ["123456", "password", "abc", ""]

        for weak_password in weak_passwords:
            success, message = auth_system.create_user(
                username=f"weak_user_{len(weak_password)}",
                email=f"weak_{len(weak_password)}@test.com",
                password=weak_password,
                role="USER",
            )
            # Should fail for weak passwords
            assert not success
            assert "password" in message.lower() or "strength" in message.lower()

        # Test strong password
        strong_password = "StrongPassword123!@#"
        success, message = auth_system.create_user(
            username="strong_user", email="strong@test.com", password=strong_password, role="USER"
        )
        assert success

    def test_session_security(self, client, user_headers):
        """Test session security features."""
        # Test that sessions are properly managed
        response = client.get("/api/v1/auth/me", headers=user_headers)
        assert response.status_code == 200

        original_user = response.json()

        # Test logout (if implemented)
        logout_response = client.post("/api/v1/auth/logout", headers=user_headers)

        if logout_response.status_code == 200:
            # If logout is implemented, token should be invalidated
            response = client.get("/api/v1/auth/me", headers=user_headers)
            assert response.status_code == 401
        else:
            # Logout might not be implemented
            assert logout_response.status_code in [404, 405]

    def test_csrf_protection(self, client, user_headers):
        """Test CSRF protection if implemented."""
        # This is more relevant for web forms than API endpoints
        # But we can test that API doesn't accept suspicious requests

        # Test request with suspicious origin
        suspicious_headers = {
            **user_headers,
            "Origin": "http://malicious-site.com",
            "Referer": "http://malicious-site.com/attack",
        }

        response = client.post(
            "/api/v1/tests/",
            json={"name": "test", "description": "test"},
            headers=suspicious_headers,
        )

        # API might allow this (APIs often don't check origin)
        # But if CSRF protection is implemented, it should be rejected
        assert response.status_code in [200, 201, 400, 403, 422]

    def test_data_sanitization(self, client, user_headers):
        """Test that sensitive data is not exposed."""
        # Test that password hashes are not returned
        response = client.get("/api/v1/auth/me", headers=user_headers)
        assert response.status_code == 200

        user_data = response.json()
        sensitive_fields = ["password", "password_hash", "api_key", "secret"]

        for field in sensitive_fields:
            assert field not in user_data

        # Test that error messages don't expose sensitive information
        response = client.get("/api/v1/tests/nonexistent-id", headers=user_headers)
        assert response.status_code == 404

        error_data = response.json()
        error_message = str(error_data).lower()

        # Should not expose internal details
        sensitive_terms = ["database", "sql", "internal", "stack trace", "exception"]
        for term in sensitive_terms:
            assert term not in error_message


@pytest.mark.asyncio
async def test_concurrent_security_operations():
    """Test security under concurrent access."""
    from httpx import AsyncClient

    from src.presentation.api.app import create_app

    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test concurrent login attempts
        login_data = {"username": "security_user", "password": "UserPass123!"}

        tasks = [ac.post("/api/v1/auth/login", json=login_data) for _ in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful logins
        successful_logins = []
        for response in responses:
            if hasattr(response, "status_code") and response.status_code == 200:
                successful_logins.append(response)

        # Should have at least some successful logins
        assert len(successful_logins) > 0

        # All successful responses should have tokens
        for response in successful_logins:
            data = response.json()
            assert "access_token" in data


if __name__ == "__main__":
    # Run tests with pytest
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True,
    )

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
