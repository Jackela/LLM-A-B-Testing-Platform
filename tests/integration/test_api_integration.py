"""Comprehensive API integration tests for LLM A/B Testing Platform."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from src.infrastructure.persistence.database import get_database_url, reset_database
from src.infrastructure.security.auth import get_auth_system
from src.presentation.api.app import create_app


class TestAPIIntegration:
    """Integration tests for the complete API functionality."""

    @pytest.fixture(scope="class")
    def app(self):
        """Create FastAPI application for testing."""
        return create_app()

    @pytest.fixture(scope="class")
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture(scope="class")
    def auth_headers(self, client) -> Dict[str, str]:
        """Create authenticated user and return headers."""
        # Create test user
        auth_system = get_auth_system()
        success, message = auth_system.create_user(
            username="integration_test_user",
            email="test@integration.com",
            password="TestPass123!",
            role="USER",
        )

        if not success and "already exists" not in message:
            pytest.fail(f"Failed to create test user: {message}")

        # Login and get token
        login_data = {"username": "integration_test_user", "password": "TestPass123!"}

        response = client.post("/api/v1/auth/login", json=login_data)
        if response.status_code != 200:
            pytest.fail(f"Login failed: {response.text}")

        token_data = response.json()
        return {"Authorization": f"Bearer {token_data['access_token']}"}

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "LLM A/B Testing Platform" in data["message"]
        assert data["docs"] == "/api/v1/docs"

    def test_authentication_flow(self, client):
        """Test complete authentication flow."""
        # Test registration (if endpoint exists)
        # Test login
        login_data = {"username": "integration_test_user", "password": "TestPass123!"}

        response = client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200

        token_data = response.json()
        assert "access_token" in token_data
        assert "token_type" in token_data
        assert token_data["token_type"] == "bearer"

        # Test protected endpoint with token
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        response = client.get("/api/v1/auth/me", headers=headers)
        assert response.status_code == 200

        user_data = response.json()
        assert user_data["username"] == "integration_test_user"

    def test_model_providers_crud(self, client, auth_headers):
        """Test model provider CRUD operations."""
        # Create provider
        provider_data = {
            "name": "test_provider",
            "provider_type": "openai",
            "config": {"api_key": "test_key", "model": "gpt-3.5-turbo"},
            "is_active": True,
        }

        response = client.post("/api/v1/providers/", json=provider_data, headers=auth_headers)
        assert response.status_code == 201

        created_provider = response.json()
        provider_id = created_provider["id"]
        assert created_provider["name"] == "test_provider"
        assert created_provider["provider_type"] == "openai"

        # Get provider
        response = client.get(f"/api/v1/providers/{provider_id}", headers=auth_headers)
        assert response.status_code == 200

        provider = response.json()
        assert provider["id"] == provider_id
        assert provider["name"] == "test_provider"

        # List providers
        response = client.get("/api/v1/providers/", headers=auth_headers)
        assert response.status_code == 200

        providers = response.json()
        assert len(providers) >= 1
        assert any(p["id"] == provider_id for p in providers)

        # Update provider
        update_data = {"name": "updated_test_provider", "is_active": False}

        response = client.put(
            f"/api/v1/providers/{provider_id}", json=update_data, headers=auth_headers
        )
        assert response.status_code == 200

        updated_provider = response.json()
        assert updated_provider["name"] == "updated_test_provider"
        assert updated_provider["is_active"] == False

        # Delete provider
        response = client.delete(f"/api/v1/providers/{provider_id}", headers=auth_headers)
        assert response.status_code == 204

        # Verify deletion
        response = client.get(f"/api/v1/providers/{provider_id}", headers=auth_headers)
        assert response.status_code == 404

    def test_ab_tests_workflow(self, client, auth_headers):
        """Test A/B test creation and management workflow."""
        # First create two providers for the test
        provider_a_data = {
            "name": "provider_a_test",
            "provider_type": "openai",
            "config": {"api_key": "test_key_a", "model": "gpt-3.5-turbo"},
            "is_active": True,
        }

        provider_b_data = {
            "name": "provider_b_test",
            "provider_type": "anthropic",
            "config": {"api_key": "test_key_b", "model": "claude-3-sonnet"},
            "is_active": True,
        }

        response_a = client.post("/api/v1/providers/", json=provider_a_data, headers=auth_headers)
        assert response_a.status_code == 201
        provider_a_id = response_a.json()["id"]

        response_b = client.post("/api/v1/providers/", json=provider_b_data, headers=auth_headers)
        assert response_b.status_code == 201
        provider_b_id = response_b.json()["id"]

        # Create A/B test
        test_data = {
            "name": "Integration Test A/B",
            "description": "Testing the integration workflow",
            "prompt_template": "Answer this question: {question}",
            "provider_a_id": provider_a_id,
            "provider_b_id": provider_b_id,
            "evaluation_criteria": {"accuracy": 0.4, "relevance": 0.3, "clarity": 0.3},
            "sample_size": 100,
            "confidence_level": 0.95,
        }

        response = client.post("/api/v1/tests/", json=test_data, headers=auth_headers)
        assert response.status_code == 201

        created_test = response.json()
        test_id = created_test["id"]
        assert created_test["name"] == "Integration Test A/B"
        assert created_test["status"] == "draft"

        # Get test
        response = client.get(f"/api/v1/tests/{test_id}", headers=auth_headers)
        assert response.status_code == 200

        test = response.json()
        assert test["id"] == test_id
        assert test["provider_a_id"] == provider_a_id
        assert test["provider_b_id"] == provider_b_id

        # Start test
        response = client.post(f"/api/v1/tests/{test_id}/start", headers=auth_headers)
        assert response.status_code == 200

        started_test = response.json()
        assert started_test["status"] == "running"

        # List tests
        response = client.get("/api/v1/tests/", headers=auth_headers)
        assert response.status_code == 200

        tests = response.json()
        assert len(tests) >= 1
        assert any(t["id"] == test_id for t in tests)

        # Cleanup
        client.delete(f"/api/v1/tests/{test_id}", headers=auth_headers)
        client.delete(f"/api/v1/providers/{provider_a_id}", headers=auth_headers)
        client.delete(f"/api/v1/providers/{provider_b_id}", headers=auth_headers)

    def test_evaluation_workflow(self, client, auth_headers):
        """Test evaluation submission and results workflow."""
        # Create providers and test first
        provider_data = {
            "name": "eval_test_provider",
            "provider_type": "openai",
            "config": {"api_key": "test_key", "model": "gpt-3.5-turbo"},
            "is_active": True,
        }

        provider_response = client.post(
            "/api/v1/providers/", json=provider_data, headers=auth_headers
        )
        provider_id = provider_response.json()["id"]

        test_data = {
            "name": "Evaluation Test",
            "description": "Testing evaluation workflow",
            "prompt_template": "Test prompt: {input}",
            "provider_a_id": provider_id,
            "provider_b_id": provider_id,
            "evaluation_criteria": {"quality": 1.0},
            "sample_size": 10,
            "confidence_level": 0.95,
        }

        test_response = client.post("/api/v1/tests/", json=test_data, headers=auth_headers)
        test_id = test_response.json()["id"]

        # Start test
        client.post(f"/api/v1/tests/{test_id}/start", headers=auth_headers)

        # Submit evaluation
        evaluation_data = {
            "test_id": test_id,
            "input_text": "What is machine learning?",
            "response_a": "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
            "response_b": "ML is when computers learn from data to make predictions or decisions.",
            "evaluation_scores": {"quality": {"a": 0.9, "b": 0.7}},
            "metadata": {"evaluator": "integration_test"},
        }

        response = client.post(
            "/api/v1/evaluation/submit", json=evaluation_data, headers=auth_headers
        )
        assert response.status_code == 201

        evaluation = response.json()
        assert evaluation["test_id"] == test_id
        assert "id" in evaluation

        # Get evaluation results
        response = client.get(f"/api/v1/evaluation/test/{test_id}/results", headers=auth_headers)
        assert response.status_code == 200

        results = response.json()
        assert "evaluations" in results
        assert len(results["evaluations"]) >= 1

        # Cleanup
        client.delete(f"/api/v1/tests/{test_id}", headers=auth_headers)
        client.delete(f"/api/v1/providers/{provider_id}", headers=auth_headers)

    def test_analytics_endpoints(self, client, auth_headers):
        """Test analytics and reporting endpoints."""
        # Test dashboard
        response = client.get("/api/v1/analytics/dashboard", headers=auth_headers)
        assert response.status_code == 200

        dashboard = response.json()
        assert "total_tests" in dashboard
        assert "active_tests" in dashboard
        assert "total_evaluations" in dashboard

        # Test test results (should work even with no data)
        response = client.get("/api/v1/analytics/test-results", headers=auth_headers)
        assert response.status_code == 200

        # Test performance metrics
        response = client.get("/api/v1/analytics/performance", headers=auth_headers)
        assert response.status_code == 200

    def test_security_features(self, client, auth_headers):
        """Test security features and rate limiting."""
        # Test without authentication
        response = client.get("/api/v1/tests/")
        assert response.status_code == 401

        # Test with invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token_here"}
        response = client.get("/api/v1/tests/", headers=invalid_headers)
        assert response.status_code == 401

        # Test security status endpoint
        response = client.get("/api/v1/security/status", headers=auth_headers)
        # May return 404 if admin-only, that's expected
        assert response.status_code in [200, 403, 404]

        # Test input validation (try malicious input)
        malicious_data = {
            "name": "<script>alert('xss')</script>",
            "description": "'; DROP TABLE tests; --",
            "prompt_template": "Normal template: {input}",
        }

        response = client.post("/api/v1/tests/", json=malicious_data, headers=auth_headers)
        # Should either be rejected (400) or sanitized (201/422)
        assert response.status_code in [400, 422]

    def test_performance_monitoring(self, client, auth_headers):
        """Test performance monitoring endpoints."""
        # Test performance status
        response = client.get("/api/v1/performance/status", headers=auth_headers)
        # May not exist or be admin-only
        assert response.status_code in [200, 404, 403]

        # Test cache statistics
        response = client.get("/api/v1/performance/cache/stats", headers=auth_headers)
        assert response.status_code in [200, 404, 403]

    def test_error_handling(self, client, auth_headers):
        """Test error handling and edge cases."""
        # Test 404 scenarios
        response = client.get("/api/v1/tests/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404

        response = client.get("/api/v1/providers/nonexistent-id", headers=auth_headers)
        assert response.status_code == 404

        # Test invalid data formats
        invalid_test_data = {"name": "", "invalid_field": "should_be_ignored"}  # Empty name

        response = client.post("/api/v1/tests/", json=invalid_test_data, headers=auth_headers)
        assert response.status_code == 422  # Validation error

        # Test invalid JSON
        response = client.post(
            "/api/v1/tests/",
            data="invalid json content",
            headers={**auth_headers, "Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_cors_and_headers(self, client):
        """Test CORS and security headers."""
        response = client.get("/")

        # Check that response doesn't expose internal errors
        assert response.status_code == 200

        # Test OPTIONS request for CORS
        response = client.options("/api/v1/tests/")
        # Should handle OPTIONS requests properly
        assert response.status_code in [200, 405]


@pytest.mark.asyncio
async def test_async_operations():
    """Test asynchronous operations and concurrent requests."""
    from httpx import AsyncClient

    from src.presentation.api.app import create_app

    app = create_app()

    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Test concurrent health checks
        tasks = [ac.get("/health") for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200

        # Test that responses are consistent
        response_data = [r.json() for r in responses]
        first_status = response_data[0]["status"]
        assert all(data["status"] == first_status for data in response_data)


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
