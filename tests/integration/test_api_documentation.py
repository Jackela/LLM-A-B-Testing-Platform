"""API documentation validation tests."""

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from src.presentation.api.app import create_app
from src.presentation.api.documentation.api_examples import DocumentationExamples
from src.presentation.api.documentation.validation_schemas import *


class TestAPIDocumentation:
    """Test API documentation accuracy and completeness."""

    @pytest.fixture(scope="class")
    def app(self):
        """Create FastAPI application for testing."""
        return create_app()

    @pytest.fixture(scope="class")
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_openapi_schema_generation(self, client):
        """Test that OpenAPI schema is generated correctly."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()

        # Verify basic schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema

        # Verify API information
        info = schema["info"]
        assert info["title"] == "LLM A/B Testing Platform API"
        assert info["version"] == "1.0.0"
        assert "description" in info

        # Verify security schemes
        components = schema["components"]
        assert "securitySchemes" in components
        assert "BearerAuth" in components["securitySchemes"]
        assert "ApiKeyAuth" in components["securitySchemes"]

        # Verify common schemas are present
        schemas = components.get("schemas", {})
        assert "Error" in schemas
        assert "ValidationError" in schemas
        assert "HealthCheck" in schemas

    def test_docs_endpoint_accessible(self, client):
        """Test that documentation endpoints are accessible."""
        # Test Swagger UI
        response = client.get("/api/v1/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Test ReDoc
        response = client.get("/api/v1/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_health_endpoint_documentation(self, client):
        """Test health endpoint matches documentation."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()

        # Validate against schema
        health_response = HealthCheckResponse(**data)
        assert health_response.status == "healthy"
        assert health_response.version == "1.0.0"
        assert isinstance(health_response.timestamp, int)

    def test_authentication_schema_validation(self):
        """Test authentication request/response schemas."""
        # Test valid login request
        login_data = {"username": "test@example.com", "password": "ValidPassword123!"}

        login_request = LoginRequest(**login_data)
        assert login_request.username == "test@example.com"
        assert login_request.password == "ValidPassword123!"

        # Test login response structure
        response_data = {
            "access_token": "test_token",
            "token_type": "bearer",
            "expires_in": 86400,
            "user": {
                "id": "user-123",
                "username": "test@example.com",
                "email": "test@example.com",
                "role": "USER",
                "permissions": ["CREATE_TEST"],
                "created_at": "2024-01-01T00:00:00Z",
                "account_status": "active",
            },
        }

        login_response = LoginResponse(**response_data)
        assert login_response.token_type == "bearer"
        assert login_response.user.role == UserRole.USER

    def test_provider_schema_validation(self):
        """Test provider request/response schemas."""
        # Test create provider request
        provider_data = {
            "name": "Test Provider",
            "provider_type": "openai",
            "config": {
                "api_key": "sk-test",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            "is_active": True,
            "description": "Test provider description",
        }

        create_request = CreateProviderRequest(**provider_data)
        assert create_request.provider_type == ProviderType.OPENAI
        assert create_request.config.temperature == 0.7

        # Test provider response
        response_data = {
            "id": "provider-123",
            "name": "Test Provider",
            "provider_type": "openai",
            "is_active": True,
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
            "created_by": "user-123",
            "config": {"model": "gpt-4", "temperature": 0.7},
            "description": "Test provider description",
        }

        provider_response = ProviderResponse(**response_data)
        assert provider_response.provider_type == ProviderType.OPENAI
        assert provider_response.is_active == True

    def test_test_schema_validation(self):
        """Test A/B test request/response schemas."""
        # Test create test request
        test_data = {
            "name": "Test A/B Test",
            "description": "Testing schema validation",
            "prompt_template": "Test prompt: {input}",
            "provider_a_id": "provider-123",
            "provider_b_id": "provider-456",
            "evaluation_criteria": {"accuracy": 0.6, "helpfulness": 0.4},
            "sample_size": 100,
            "confidence_level": 0.95,
            "metadata": {"test": "true"},
        }

        create_request = CreateTestRequest(**test_data)
        assert create_request.sample_size == 100
        assert create_request.confidence_level == 0.95
        assert sum(create_request.evaluation_criteria.values()) == 1.0

        # Test invalid criteria (should raise validation error)
        invalid_test_data = test_data.copy()
        invalid_test_data["evaluation_criteria"] = {
            "accuracy": 0.3,
            "helpfulness": 0.3,
        }  # Sum = 0.6

        with pytest.raises(ValueError, match="must sum to 1.0"):
            CreateTestRequest(**invalid_test_data)

        # Test test response
        response_data = {
            "id": "test-123",
            "name": "Test A/B Test",
            "description": "Testing schema validation",
            "status": "draft",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:00:00Z",
            "created_by": "user-123",
            "provider_a_id": "provider-123",
            "provider_b_id": "provider-456",
            "sample_size": 100,
            "confidence_level": 0.95,
            "evaluations_count": 0,
            "progress": 0.0,
        }

        test_response = TestResponse(**response_data)
        assert test_response.status == TestStatus.DRAFT
        assert test_response.progress == 0.0

    def test_evaluation_schema_validation(self):
        """Test evaluation request/response schemas."""
        # Test submit evaluation request
        eval_data = {
            "test_id": "test-123",
            "input_text": "Test input",
            "response_a": "Response from model A",
            "response_b": "Response from model B",
            "evaluation_scores": {
                "accuracy": {"a": 0.8, "b": 0.6},
                "helpfulness": {"a": 0.7, "b": 0.9},
            },
            "evaluator": "test-evaluator",
            "evaluation_time_seconds": 60,
            "metadata": {"difficulty": "easy"},
        }

        eval_request = SubmitEvaluationRequest(**eval_data)
        assert eval_request.test_id == "test-123"
        assert eval_request.evaluation_scores["accuracy"]["a"] == 0.8

        # Test evaluation response
        response_data = {
            "id": "eval-123",
            "test_id": "test-123",
            "status": "processed",
            "submitted_at": "2024-01-01T14:30:00Z",
            "overall_score_a": 0.75,
            "overall_score_b": 0.75,
            "confidence": 0.9,
            "evaluation_number": 1,
        }

        eval_response = EvaluationResponse(**response_data)
        assert eval_response.status == "processed"
        assert eval_response.confidence == 0.9

    def test_analytics_schema_validation(self):
        """Test analytics response schemas."""
        # Test dashboard response
        dashboard_data = {
            "summary": {
                "total_tests": 10,
                "active_tests": 2,
                "completed_tests": 8,
                "draft_tests": 0,
                "total_evaluations": 500,
                "total_providers": 5,
                "active_providers": 4,
            },
            "recent_activity": [
                {
                    "type": "test_completed",
                    "test_id": "test-123",
                    "test_name": "Test A/B",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "result": "Provider A better",
                }
            ],
            "performance_metrics": {
                "average_test_duration": 5.5,
                "average_evaluations_per_test": 150,
                "average_confidence_level": 0.94,
                "most_active_evaluator": "evaluator-1",
            },
        }

        dashboard_response = DashboardResponse(**dashboard_data)
        assert dashboard_response.summary.total_tests == 10
        assert len(dashboard_response.recent_activity) == 1
        assert dashboard_response.performance_metrics.average_confidence_level == 0.94

    def test_error_schema_validation(self):
        """Test error response schemas."""
        # Test standard error response
        error_data = {
            "error": "validation_error",
            "message": "Invalid input provided",
            "details": {"field": "provider_type", "issue": "invalid value"},
        }

        error_response = ErrorResponse(**error_data)
        assert error_response.error == "validation_error"
        assert error_response.details["field"] == "provider_type"

        # Test validation error response
        validation_data = {
            "detail": [
                {"loc": ["body", "name"], "msg": "field required", "type": "value_error.missing"}
            ]
        }

        validation_response = ValidationErrorResponse(**validation_data)
        assert len(validation_response.detail) == 1
        assert validation_response.detail[0].msg == "field required"

    def test_api_examples_completeness(self):
        """Test that API examples are comprehensive."""
        examples = DocumentationExamples.get_all_examples()

        # Verify all categories are present
        expected_categories = [
            ExampleCategory.AUTHENTICATION,
            ExampleCategory.PROVIDERS,
            ExampleCategory.TESTS,
            ExampleCategory.EVALUATIONS,
            ExampleCategory.ANALYTICS,
        ]

        for category in expected_categories:
            assert category in examples, f"Missing examples for category: {category}"
            assert len(examples[category]) > 0, f"No examples for category: {category}"

        # Verify authentication examples
        auth_examples = examples[ExampleCategory.AUTHENTICATION]
        auth_summaries = [ex.summary for ex in auth_examples]
        assert "User Login" in auth_summaries
        assert "Successful Login Response" in auth_summaries

        # Verify provider examples
        provider_examples = examples[ExampleCategory.PROVIDERS]
        provider_summaries = [ex.summary for ex in provider_examples]
        assert "OpenAI Provider Configuration" in provider_summaries
        assert "Anthropic Claude Provider" in provider_summaries

    def test_schema_field_validation(self):
        """Test field validation in schemas."""
        # Test invalid temperature (too high)
        with pytest.raises(ValueError):
            ProviderConfig(api_key="test", model="gpt-4", temperature=3.0)  # Invalid: > 2.0

        # Test invalid sample size (too low)
        with pytest.raises(ValueError):
            CreateTestRequest(
                name="Test",
                description="Test",
                prompt_template="Test",
                provider_a_id="provider-1",
                provider_b_id="provider-2",
                evaluation_criteria={"accuracy": 1.0},
                sample_size=5,  # Invalid: < 10
                confidence_level=0.95,
            )

        # Test invalid confidence level (too low)
        with pytest.raises(ValueError):
            CreateTestRequest(
                name="Test",
                description="Test",
                prompt_template="Test",
                provider_a_id="provider-1",
                provider_b_id="provider-2",
                evaluation_criteria={"accuracy": 1.0},
                sample_size=100,
                confidence_level=0.5,  # Invalid: < 0.8
            )

    def test_enum_validation(self):
        """Test enum field validation."""
        # Test valid provider type
        provider = CreateProviderRequest(
            name="Test",
            provider_type=ProviderType.OPENAI,
            config=ProviderConfig(api_key="test", model="gpt-4"),
            is_active=True,
        )
        assert provider.provider_type == ProviderType.OPENAI

        # Test valid test status
        test_response = TestResponse(
            id="test-123",
            name="Test",
            description="Test",
            status=TestStatus.RUNNING,
            created_at="2024-01-01T12:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
            created_by="user-123",
            provider_a_id="provider-1",
            provider_b_id="provider-2",
            sample_size=100,
            confidence_level=0.95,
            evaluations_count=0,
            progress=25.0,
        )
        assert test_response.status == TestStatus.RUNNING

        # Test valid user role
        user = UserProfile(
            id="user-123",
            username="test@example.com",
            email="test@example.com",
            role=UserRole.ADMIN,
            created_at="2024-01-01T00:00:00Z",
        )
        assert user.role == UserRole.ADMIN


if __name__ == "__main__":
    # Run documentation validation tests
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
