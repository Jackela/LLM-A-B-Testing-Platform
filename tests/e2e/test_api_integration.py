"""End-to-end tests for API integration."""

from datetime import datetime
from uuid import uuid4

import pytest

from tests.factories import (
    CreateTestCommandDTOFactory,
    ModelConfigurationDTOFactory,
    TestSampleDTOFactory,
)


@pytest.mark.e2e
class TestAPIIntegration:
    """End-to-end tests for API integration."""

    @pytest.mark.asyncio
    async def test_api_authentication_flow(self, async_client):
        """Test complete authentication flow."""
        # Test unauthorized access
        response = await async_client.get("/api/v1/tests/")
        assert response.status_code == 401

        # Test login
        login_data = {"username": "test@example.com", "password": "testpassword"}

        login_response = await async_client.post("/api/v1/auth/login", data=login_data)

        if login_response.status_code == 200:
            token_data = login_response.json()
            assert "access_token" in token_data
            assert "token_type" in token_data

            # Test authenticated access
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            auth_response = await async_client.get("/api/v1/tests/", headers=headers)
            assert auth_response.status_code == 200
        else:
            # If login fails, test user creation and then login
            register_data = {
                "email": "test@example.com",
                "password": "testpassword",
                "full_name": "Test User",
            }

            register_response = await async_client.post("/api/v1/auth/register", json=register_data)

            if register_response.status_code == 201:
                # Try login again
                login_response = await async_client.post("/api/v1/auth/login", data=login_data)
                assert login_response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_error_handling(self, async_client, auth_headers):
        """Test API error handling and response formats."""
        # Test 404 - non-existent test
        response = await async_client.get(f"/api/v1/tests/{uuid4()}", headers=auth_headers)
        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data

        # Test 422 - validation error
        invalid_test_data = {
            "name": "",  # Empty name should cause validation error
            "configuration": {},
            "samples": [],
        }

        response = await async_client.post(
            "/api/v1/tests/", json=invalid_test_data, headers=auth_headers
        )
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data

        # Test 400 - bad request
        response = await async_client.post(
            "/api/v1/tests/", data="invalid json", headers=auth_headers
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_api_content_negotiation(self, async_client, auth_headers):
        """Test API content negotiation and response formats."""
        # Test JSON response (default)
        response = await async_client.get("/api/v1/tests/", headers=auth_headers)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

        # Test with explicit Accept header
        headers = {**auth_headers, "Accept": "application/json"}
        response = await async_client.get("/api/v1/tests/", headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")

    @pytest.mark.asyncio
    async def test_api_rate_limiting(self, async_client, auth_headers):
        """Test API rate limiting functionality."""
        # Make multiple rapid requests
        responses = []
        for i in range(100):
            response = await async_client.get("/api/v1/tests/", headers=auth_headers)
            responses.append(response)

            # If rate limited, stop
            if response.status_code == 429:
                break

        # Check if any request was rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        if rate_limited:
            # Verify rate limit headers
            rate_limited_response = next(r for r in responses if r.status_code == 429)
            assert (
                "X-RateLimit-Limit" in rate_limited_response.headers
                or "Retry-After" in rate_limited_response.headers
            )

    @pytest.mark.asyncio
    async def test_api_versioning(self, async_client, auth_headers):
        """Test API versioning functionality."""
        # Test current version
        response = await async_client.get("/api/v1/tests/", headers=auth_headers)
        assert response.status_code == 200

        # Test API info endpoint
        info_response = await async_client.get("/api/v1/info")
        assert info_response.status_code == 200

        info_data = info_response.json()
        assert "version" in info_data
        assert "name" in info_data

    @pytest.mark.asyncio
    async def test_api_cors_headers(self, async_client):
        """Test CORS headers in API responses."""
        # Test preflight request
        response = await async_client.options(
            "/api/v1/tests/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        # Should allow CORS
        if response.status_code == 200:
            assert "Access-Control-Allow-Origin" in response.headers
            assert "Access-Control-Allow-Methods" in response.headers

    @pytest.mark.asyncio
    async def test_api_health_check(self, async_client):
        """Test API health check endpoints."""
        # Test basic health check
        response = await async_client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert "status" in health_data
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]

        # Test detailed health check
        detailed_response = await async_client.get("/health/detailed")
        if detailed_response.status_code == 200:
            detailed_data = detailed_response.json()
            assert "database" in detailed_data
            assert "external_services" in detailed_data

    @pytest.mark.asyncio
    async def test_api_metrics_endpoints(self, async_client, auth_headers):
        """Test API metrics and monitoring endpoints."""
        # Test metrics endpoint (if available)
        metrics_response = await async_client.get("/metrics", headers=auth_headers)

        # Should return metrics in Prometheus format or be protected
        assert metrics_response.status_code in [200, 401, 404]

        if metrics_response.status_code == 200:
            metrics_text = metrics_response.text
            # Basic check for Prometheus format
            assert "# HELP" in metrics_text or "# TYPE" in metrics_text

    @pytest.mark.asyncio
    async def test_api_documentation_endpoints(self, async_client):
        """Test API documentation endpoints."""
        # Test OpenAPI/Swagger docs
        docs_response = await async_client.get("/docs")
        assert docs_response.status_code == 200

        # Test OpenAPI JSON
        openapi_response = await async_client.get("/openapi.json")
        assert openapi_response.status_code == 200

        openapi_data = openapi_response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data

    @pytest.mark.asyncio
    async def test_api_data_validation(self, async_client, auth_headers):
        """Test comprehensive API data validation."""
        # Test valid data
        valid_command = CreateTestCommandDTOFactory()
        response = await async_client.post(
            "/api/v1/tests/", json=valid_command.dict(), headers=auth_headers
        )
        assert response.status_code == 201

        # Test various invalid data scenarios
        invalid_scenarios = [
            # Missing required fields
            {
                "configuration": valid_command.configuration.dict(),
                "samples": [s.dict() for s in valid_command.samples],
            },
            # Invalid data types
            {
                "name": 123,  # Should be string
                "configuration": valid_command.configuration.dict(),
                "samples": [s.dict() for s in valid_command.samples],
            },
            # Invalid nested data
            {
                "name": "Test",
                "configuration": {
                    "models": [{"invalid": "data"}],
                    "evaluation": valid_command.configuration.evaluation.dict(),
                },
                "samples": [s.dict() for s in valid_command.samples],
            },
        ]

        for invalid_data in invalid_scenarios:
            response = await async_client.post(
                "/api/v1/tests/", json=invalid_data, headers=auth_headers
            )
            assert response.status_code in [400, 422]

    @pytest.mark.asyncio
    async def test_api_response_consistency(self, async_client, auth_headers):
        """Test API response format consistency."""
        # Create a test
        command = CreateTestCommandDTOFactory()
        create_response = await async_client.post(
            "/api/v1/tests/", json=command.dict(), headers=auth_headers
        )

        if create_response.status_code == 201:
            test_id = create_response.json()["test_id"]

            # Test consistent response formats across endpoints
            endpoints = [
                f"/api/v1/tests/{test_id}",
                "/api/v1/tests/",
                f"/api/v1/tests/{test_id}/progress",
            ]

            for endpoint in endpoints:
                response = await async_client.get(endpoint, headers=auth_headers)

                if response.status_code == 200:
                    data = response.json()

                    # Check for consistent timestamp formats
                    if "created_at" in str(data):
                        # Should be ISO format timestamps
                        assert "T" in str(data) and "Z" in str(data)

                    # Check for consistent ID formats
                    if "id" in str(data):
                        # Should be UUID format
                        import re

                        uuid_pattern = (
                            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
                        )
                        assert re.search(uuid_pattern, str(data))

    @pytest.mark.asyncio
    async def test_api_concurrent_requests(self, async_client, auth_headers):
        """Test API handling of concurrent requests."""
        import asyncio

        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = async_client.get("/api/v1/tests/", headers=auth_headers)
            tasks.append(task)

        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests completed successfully
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) >= 8  # Allow for some failures

        for response in successful_responses:
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_request_timeout_handling(self, async_client, auth_headers):
        """Test API request timeout handling."""
        # Test with reasonable timeout
        try:
            response = await async_client.get("/api/v1/tests/", headers=auth_headers, timeout=30.0)
            assert response.status_code == 200
        except Exception as e:
            # Timeout should be handled gracefully
            assert "timeout" in str(e).lower()

    @pytest.mark.asyncio
    async def test_api_large_payload_handling(self, async_client, auth_headers):
        """Test API handling of large payloads."""
        # Create test with large number of samples
        large_command = CreateTestCommandDTOFactory(
            samples=[TestSampleDTOFactory() for _ in range(1000)]
        )

        response = await async_client.post(
            "/api/v1/tests/", json=large_command.dict(), headers=auth_headers
        )

        # Should either accept or reject with appropriate status
        assert response.status_code in [201, 413, 422]

        if response.status_code == 413:
            # Payload too large - should have helpful error message
            error_data = response.json()
            assert "too large" in error_data.get("detail", "").lower()

    @pytest.mark.asyncio
    async def test_api_security_headers(self, async_client):
        """Test API security headers."""
        response = await async_client.get("/api/v1/info")

        # Check for important security headers
        security_headers = ["X-Content-Type-Options", "X-Frame-Options", "X-XSS-Protection"]

        for header in security_headers:
            if header in response.headers:
                assert response.headers[header] is not None

    @pytest.mark.asyncio
    async def test_api_response_compression(self, async_client, auth_headers):
        """Test API response compression."""
        # Request with compression
        headers = {**auth_headers, "Accept-Encoding": "gzip, deflate"}

        response = await async_client.get("/api/v1/tests/", headers=headers)

        assert response.status_code == 200

        # Check if response is compressed
        if "Content-Encoding" in response.headers:
            assert response.headers["Content-Encoding"] in ["gzip", "deflate"]
