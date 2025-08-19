"""API documentation examples and schemas for LLM A/B Testing Platform."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ExampleCategory(str, Enum):
    """Categories for API examples."""

    AUTHENTICATION = "authentication"
    PROVIDERS = "providers"
    TESTS = "tests"
    EVALUATIONS = "evaluations"
    ANALYTICS = "analytics"
    SECURITY = "security"
    PERFORMANCE = "performance"


class APIExample(BaseModel):
    """Structure for API documentation examples."""

    summary: str
    description: str = ""
    value: Dict[str, Any]
    category: ExampleCategory


class DocumentationExamples:
    """Comprehensive API examples for documentation."""

    @staticmethod
    def get_all_examples() -> Dict[str, List[APIExample]]:
        """Get all API examples organized by category."""

        return {
            ExampleCategory.AUTHENTICATION: [
                APIExample(
                    summary="User Login",
                    description="Standard user authentication with email and password",
                    value={"username": "user@company.com", "password": "SecurePassword123!"},
                    category=ExampleCategory.AUTHENTICATION,
                ),
                APIExample(
                    summary="Successful Login Response",
                    description="Response after successful authentication",
                    value={
                        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGNvbXBhbnkuY29tIiwiZXhwIjoxNjQwOTk1MjAwLCJyb2xlIjoiVVNFUiJ9.signature",
                        "token_type": "bearer",
                        "expires_in": 86400,
                        "refresh_token": "refresh_token_here",
                        "user": {
                            "id": "user-12345",
                            "username": "user@company.com",
                            "email": "user@company.com",
                            "role": "USER",
                            "created_at": "2024-01-01T00:00:00Z",
                            "last_login": "2024-01-01T12:00:00Z",
                        },
                    },
                    category=ExampleCategory.AUTHENTICATION,
                ),
                APIExample(
                    summary="Admin Login",
                    description="Administrator login with elevated privileges",
                    value={"username": "admin@company.com", "password": "AdminSecurePass456!"},
                    category=ExampleCategory.AUTHENTICATION,
                ),
                APIExample(
                    summary="User Profile",
                    description="User profile information returned by /auth/me endpoint",
                    value={
                        "id": "user-12345",
                        "username": "user@company.com",
                        "email": "user@company.com",
                        "role": "USER",
                        "permissions": [
                            "CREATE_TEST",
                            "READ_TEST",
                            "UPDATE_TEST",
                            "READ_PROVIDER",
                            "SUBMIT_EVALUATION",
                        ],
                        "created_at": "2024-01-01T00:00:00Z",
                        "last_login": "2024-01-01T12:00:00Z",
                        "account_status": "active",
                    },
                    category=ExampleCategory.AUTHENTICATION,
                ),
            ],
            ExampleCategory.PROVIDERS: [
                APIExample(
                    summary="OpenAI Provider Configuration",
                    description="Configure OpenAI GPT models for testing",
                    value={
                        "name": "OpenAI GPT-4 Turbo",
                        "provider_type": "openai",
                        "config": {
                            "api_key": "sk-proj-...",
                            "model": "gpt-4-turbo-preview",
                            "temperature": 0.7,
                            "max_tokens": 2000,
                            "top_p": 1.0,
                            "frequency_penalty": 0.0,
                            "presence_penalty": 0.0,
                        },
                        "is_active": True,
                        "description": "Latest GPT-4 Turbo model for high-quality responses",
                    },
                    category=ExampleCategory.PROVIDERS,
                ),
                APIExample(
                    summary="Anthropic Claude Provider",
                    description="Configure Anthropic Claude models",
                    value={
                        "name": "Claude-3 Sonnet",
                        "provider_type": "anthropic",
                        "config": {
                            "api_key": "ant-api03-...",
                            "model": "claude-3-sonnet-20240229",
                            "max_tokens": 2000,
                            "temperature": 0.7,
                            "top_p": 1.0,
                        },
                        "is_active": True,
                        "description": "Claude-3 Sonnet for balanced performance and capability",
                    },
                    category=ExampleCategory.PROVIDERS,
                ),
                APIExample(
                    summary="Google AI Provider",
                    description="Configure Google AI models (Gemini)",
                    value={
                        "name": "Gemini Pro",
                        "provider_type": "google",
                        "config": {
                            "api_key": "AIza...",
                            "model": "gemini-pro",
                            "temperature": 0.7,
                            "max_tokens": 2000,
                            "top_p": 1.0,
                            "top_k": 40,
                        },
                        "is_active": True,
                        "description": "Google's Gemini Pro model",
                    },
                    category=ExampleCategory.PROVIDERS,
                ),
                APIExample(
                    summary="Provider Response",
                    description="Response after creating a provider",
                    value={
                        "id": "provider-abc123",
                        "name": "OpenAI GPT-4 Turbo",
                        "provider_type": "openai",
                        "is_active": True,
                        "created_at": "2024-01-01T12:00:00Z",
                        "updated_at": "2024-01-01T12:00:00Z",
                        "created_by": "user-12345",
                        "config": {
                            "model": "gpt-4-turbo-preview",
                            "temperature": 0.7,
                            "max_tokens": 2000,
                        },
                    },
                    category=ExampleCategory.PROVIDERS,
                ),
            ],
            ExampleCategory.TESTS: [
                APIExample(
                    summary="Customer Support A/B Test",
                    description="A/B test comparing models for customer support responses",
                    value={
                        "name": "Customer Support Response Quality",
                        "description": "Comparing GPT-4 vs Claude-3 for customer support ticket responses",
                        "prompt_template": "You are a professional customer support agent. Respond helpfully and professionally to this customer inquiry: {question}",
                        "provider_a_id": "provider-abc123",
                        "provider_b_id": "provider-def456",
                        "evaluation_criteria": {
                            "helpfulness": 0.35,
                            "professionalism": 0.25,
                            "accuracy": 0.25,
                            "clarity": 0.15,
                        },
                        "sample_size": 200,
                        "confidence_level": 0.95,
                        "metadata": {
                            "department": "customer_support",
                            "priority": "high",
                            "estimated_duration": "7 days",
                        },
                    },
                    category=ExampleCategory.TESTS,
                ),
                APIExample(
                    summary="Content Generation Test",
                    description="A/B test for marketing content generation",
                    value={
                        "name": "Marketing Copy Generation",
                        "description": "Testing different models for creating engaging marketing copy",
                        "prompt_template": "Create compelling marketing copy for: {product_description}. Target audience: {audience}. Tone: {tone}",
                        "provider_a_id": "provider-ghi789",
                        "provider_b_id": "provider-jkl012",
                        "evaluation_criteria": {
                            "creativity": 0.4,
                            "persuasiveness": 0.3,
                            "brand_alignment": 0.2,
                            "clarity": 0.1,
                        },
                        "sample_size": 150,
                        "confidence_level": 0.95,
                    },
                    category=ExampleCategory.TESTS,
                ),
                APIExample(
                    summary="Test Creation Response",
                    description="Response after successfully creating an A/B test",
                    value={
                        "id": "test-xyz789",
                        "name": "Customer Support Response Quality",
                        "description": "Comparing GPT-4 vs Claude-3 for customer support ticket responses",
                        "status": "draft",
                        "created_at": "2024-01-01T12:00:00Z",
                        "created_by": "user-12345",
                        "provider_a_id": "provider-abc123",
                        "provider_b_id": "provider-def456",
                        "sample_size": 200,
                        "confidence_level": 0.95,
                        "evaluations_count": 0,
                        "progress": 0.0,
                        "estimated_completion": "2024-01-08T12:00:00Z",
                    },
                    category=ExampleCategory.TESTS,
                ),
                APIExample(
                    summary="Test Status Update",
                    description="Starting a test and status response",
                    value={
                        "id": "test-xyz789",
                        "status": "running",
                        "started_at": "2024-01-01T13:00:00Z",
                        "progress": 15.5,
                        "evaluations_count": 31,
                        "estimated_completion": "2024-01-06T10:30:00Z",
                    },
                    category=ExampleCategory.TESTS,
                ),
            ],
            ExampleCategory.EVALUATIONS: [
                APIExample(
                    summary="Customer Support Evaluation",
                    description="Evaluation data for customer support responses",
                    value={
                        "test_id": "test-xyz789",
                        "input_text": "I ordered a product 5 days ago but haven't received any shipping updates. Can you help me track my order?",
                        "response_a": "I understand your concern about not receiving shipping updates. Let me help you track your order. To locate your order details, I'll need your order number or the email address used for the purchase. Once I have that information, I can provide you with the current status and tracking information. Our shipping team typically sends updates within 24 hours of processing, so there may have been a delay in our notification system.",
                        "response_b": "I can help you track your order. Please provide your order number and I'll look it up for you right away.",
                        "evaluation_scores": {
                            "helpfulness": {"a": 0.9, "b": 0.7},
                            "professionalism": {"a": 0.95, "b": 0.8},
                            "accuracy": {"a": 0.85, "b": 0.9},
                            "clarity": {"a": 0.8, "b": 0.95},
                        },
                        "evaluator": "expert-evaluator-001",
                        "evaluation_time_seconds": 67,
                        "metadata": {
                            "difficulty": "medium",
                            "category": "order_tracking",
                            "customer_sentiment": "frustrated",
                            "expected_response_length": "medium",
                        },
                    },
                    category=ExampleCategory.EVALUATIONS,
                ),
                APIExample(
                    summary="Marketing Copy Evaluation",
                    description="Evaluation for marketing content generation",
                    value={
                        "test_id": "test-abc456",
                        "input_text": "Product: Wireless noise-canceling headphones. Audience: Young professionals. Tone: Modern and energetic",
                        "response_a": "🎧 Elevate Your Workday! Our premium wireless noise-canceling headphones are designed for the modern professional who demands excellence. Block out distractions, boost productivity, and enjoy crystal-clear audio whether you're in the office or on the go. Style meets performance in these sleek, comfortable headphones that keep up with your dynamic lifestyle.",
                        "response_b": "Professional Wireless Headphones with Active Noise Cancellation. Perfect for busy professionals who need to focus. High-quality audio and comfortable design for all-day wear.",
                        "evaluation_scores": {
                            "creativity": {"a": 0.85, "b": 0.4},
                            "persuasiveness": {"a": 0.9, "b": 0.6},
                            "brand_alignment": {"a": 0.8, "b": 0.7},
                            "clarity": {"a": 0.75, "b": 0.9},
                        },
                        "evaluator": "marketing-expert-003",
                        "evaluation_time_seconds": 89,
                        "metadata": {
                            "campaign_type": "product_launch",
                            "target_platform": "social_media",
                            "character_count_a": 287,
                            "character_count_b": 156,
                        },
                    },
                    category=ExampleCategory.EVALUATIONS,
                ),
                APIExample(
                    summary="Evaluation Submission Response",
                    description="Response after submitting an evaluation",
                    value={
                        "id": "eval-qwe123",
                        "test_id": "test-xyz789",
                        "status": "processed",
                        "submitted_at": "2024-01-01T14:30:00Z",
                        "overall_score_a": 0.875,
                        "overall_score_b": 0.8125,
                        "confidence": 0.94,
                        "evaluation_number": 32,
                    },
                    category=ExampleCategory.EVALUATIONS,
                ),
            ],
            ExampleCategory.ANALYTICS: [
                APIExample(
                    summary="Dashboard Overview",
                    description="Main dashboard analytics data",
                    value={
                        "summary": {
                            "total_tests": 47,
                            "active_tests": 5,
                            "completed_tests": 39,
                            "draft_tests": 3,
                            "total_evaluations": 8429,
                            "total_providers": 12,
                            "active_providers": 9,
                        },
                        "recent_activity": [
                            {
                                "type": "test_completed",
                                "test_id": "test-abc123",
                                "test_name": "Email Response Quality",
                                "timestamp": "2024-01-01T11:45:00Z",
                                "result": "Provider A significantly better",
                            },
                            {
                                "type": "evaluation_submitted",
                                "test_id": "test-xyz789",
                                "evaluator": "expert-evaluator-001",
                                "timestamp": "2024-01-01T11:30:00Z",
                            },
                        ],
                        "performance_metrics": {
                            "average_test_duration": 5.2,
                            "average_evaluations_per_test": 179,
                            "average_confidence_level": 0.93,
                            "most_active_evaluator": "expert-evaluator-001",
                        },
                    },
                    category=ExampleCategory.ANALYTICS,
                ),
                APIExample(
                    summary="Test Results Analysis",
                    description="Detailed A/B test results with statistical analysis",
                    value={
                        "test_id": "test-xyz789",
                        "test_name": "Customer Support Response Quality",
                        "status": "completed",
                        "completion_date": "2024-01-06T10:15:00Z",
                        "total_evaluations": 200,
                        "duration_days": 5.2,
                        "results": {
                            "provider_a": {
                                "name": "OpenAI GPT-4 Turbo",
                                "model": "gpt-4-turbo-preview",
                                "scores": {
                                    "helpfulness": {"mean": 0.847, "std": 0.123, "median": 0.85},
                                    "professionalism": {"mean": 0.891, "std": 0.098, "median": 0.9},
                                    "accuracy": {"mean": 0.823, "std": 0.145, "median": 0.82},
                                    "clarity": {"mean": 0.779, "std": 0.167, "median": 0.78},
                                },
                                "overall_score": 0.835,
                                "confidence_interval": [0.811, 0.859],
                                "total_evaluations": 200,
                            },
                            "provider_b": {
                                "name": "Claude-3 Sonnet",
                                "model": "claude-3-sonnet-20240229",
                                "scores": {
                                    "helpfulness": {"mean": 0.812, "std": 0.134, "median": 0.81},
                                    "professionalism": {
                                        "mean": 0.856,
                                        "std": 0.112,
                                        "median": 0.86,
                                    },
                                    "accuracy": {"mean": 0.834, "std": 0.128, "median": 0.83},
                                    "clarity": {"mean": 0.798, "std": 0.151, "median": 0.8},
                                },
                                "overall_score": 0.825,
                                "confidence_interval": [0.802, 0.848],
                                "total_evaluations": 200,
                            },
                        },
                        "statistical_analysis": {
                            "p_value": 0.0423,
                            "effect_size": 0.127,
                            "is_significant": True,
                            "confidence_level": 0.95,
                            "winner": "provider_a",
                            "improvement_percentage": 1.21,
                            "power_analysis": 0.82,
                        },
                        "recommendations": [
                            "Provider A (GPT-4 Turbo) shows statistically significant better performance",
                            "The improvement is small but consistent across most criteria",
                            "Consider using Provider A for customer support applications",
                        ],
                    },
                    category=ExampleCategory.ANALYTICS,
                ),
                APIExample(
                    summary="Performance Metrics",
                    description="System performance and usage analytics",
                    value={
                        "time_period": "last_7_days",
                        "api_metrics": {
                            "total_requests": 45672,
                            "average_response_time": 234,
                            "success_rate": 0.9978,
                            "error_rate": 0.0022,
                            "requests_per_second_peak": 127,
                        },
                        "cache_metrics": {
                            "hit_rate": 0.847,
                            "miss_rate": 0.153,
                            "total_keys": 8934,
                            "memory_usage_mb": 256,
                        },
                        "security_metrics": {
                            "blocked_requests": 1847,
                            "rate_limited_requests": 234,
                            "failed_authentications": 67,
                            "suspicious_ips": 23,
                        },
                        "top_endpoints": [
                            {
                                "endpoint": "/api/v1/tests/",
                                "requests": 8234,
                                "avg_response_time": 178,
                            },
                            {
                                "endpoint": "/api/v1/evaluations/submit",
                                "requests": 7891,
                                "avg_response_time": 245,
                            },
                            {
                                "endpoint": "/api/v1/analytics/dashboard",
                                "requests": 3456,
                                "avg_response_time": 134,
                            },
                        ],
                    },
                    category=ExampleCategory.ANALYTICS,
                ),
            ],
        }

    @staticmethod
    def get_validation_examples() -> Dict[str, Any]:
        """Get examples for validation errors and edge cases."""

        return {
            "validation_errors": [
                {
                    "scenario": "Missing required fields",
                    "request": {
                        "name": "Test Provider"
                        # Missing provider_type and config
                    },
                    "response": {
                        "detail": [
                            {
                                "loc": ["body", "provider_type"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            },
                            {
                                "loc": ["body", "config"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            },
                        ]
                    },
                },
                {
                    "scenario": "Invalid field values",
                    "request": {
                        "name": "Test Provider",
                        "provider_type": "invalid_provider",
                        "config": {"api_key": "test", "temperature": 2.5},  # Invalid temperature
                    },
                    "response": {
                        "error": "validation_error",
                        "message": "Invalid input data provided",
                        "details": {
                            "provider_type": "must be one of: openai, anthropic, google",
                            "temperature": "must be between 0.0 and 2.0",
                        },
                    },
                },
            ],
            "edge_cases": [
                {
                    "scenario": "Extremely long input text",
                    "description": "Input text exceeding maximum length limits",
                    "response": {
                        "error": "validation_error",
                        "message": "Input text exceeds maximum length of 10000 characters",
                    },
                },
                {
                    "scenario": "Test with identical providers",
                    "description": "A/B test using the same provider for both sides",
                    "response": {
                        "error": "validation_error",
                        "message": "Provider A and Provider B cannot be the same",
                    },
                },
            ],
        }
