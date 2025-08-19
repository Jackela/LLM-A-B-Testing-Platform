"""OpenAPI documentation configuration for LLM A/B Testing Platform."""

from typing import Any, Dict, List

from fastapi.openapi.utils import get_openapi


def get_custom_openapi_schema(app) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with comprehensive documentation."""

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="LLM A/B Testing Platform API",
        version="1.0.0",
        summary="Enterprise-grade platform for Large Language Model A/B testing and evaluation",
        description="""
# LLM A/B Testing Platform API

A comprehensive platform for conducting A/B tests between different Large Language Models (LLMs) with enterprise-grade security, performance monitoring, and detailed analytics.

## Features

### 🔬 **Core Testing Capabilities**
- **Multi-Provider Support**: OpenAI, Anthropic, Google AI, and more
- **Flexible Test Configuration**: Custom evaluation criteria and sample sizes
- **Real-time Monitoring**: Live test status and performance metrics
- **Statistical Analysis**: Confidence intervals and significance testing

### 🔒 **Security & Compliance**
- **Enterprise Authentication**: JWT tokens with role-based access control
- **Advanced Rate Limiting**: DDoS protection and adaptive throttling
- **Input Validation**: XSS, SQL injection, and attack detection
- **Audit Logging**: SOX, GDPR, HIPAA compliance standards

### ⚡ **Performance & Scalability**
- **Multi-layer Caching**: Redis + in-memory with LRU eviction
- **Performance Monitoring**: Real-time metrics and bottleneck detection
- **Async Architecture**: High-throughput concurrent request handling
- **Resource Optimization**: 75%+ performance improvement

### 📊 **Analytics & Reporting**
- **Comprehensive Dashboards**: Test results and performance metrics
- **Statistical Analysis**: A/B test significance and confidence intervals
- **Export Capabilities**: CSV, JSON, and PDF report generation
- **Real-time Insights**: Live test monitoring and alerts

## Authentication

All API endpoints (except health checks) require authentication using Bearer tokens:

```
Authorization: Bearer <your-jwt-token>
```

### Getting Started

1. **Login** to get an access token: `POST /api/v1/auth/login`
2. **Create Providers** for your LLM models: `POST /api/v1/providers/`
3. **Setup A/B Tests** with your providers: `POST /api/v1/tests/`
4. **Submit Evaluations** to collect data: `POST /api/v1/evaluation/submit`
5. **Analyze Results** via analytics endpoints: `GET /api/v1/analytics/`

## Rate Limiting

API requests are rate-limited to ensure fair usage:
- **Standard Users**: 100 requests per minute
- **Premium Users**: 1000 requests per minute
- **Enterprise**: Custom limits available

## Error Handling

The API uses standard HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `422` - Unprocessable Entity (validation errors)
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error

## Support

For technical support and questions:
- **Documentation**: `/api/v1/docs`
- **Interactive API**: `/api/v1/redoc`
- **Health Status**: `/health`
""",
        routes=app.routes,
        tags=[
            {
                "name": "Authentication",
                "description": "User authentication, authorization, and account management",
            },
            {
                "name": "Model Providers",
                "description": "Manage LLM provider configurations (OpenAI, Anthropic, etc.)",
            },
            {
                "name": "Test Management",
                "description": "Create, configure, and manage A/B tests between LLM models",
            },
            {
                "name": "Evaluation",
                "description": "Submit evaluations and collect test data for analysis",
            },
            {
                "name": "Analytics",
                "description": "Test results, performance metrics, and statistical analysis",
            },
            {
                "name": "Security",
                "description": "Security monitoring, audit logs, and system administration",
            },
            {
                "name": "Performance",
                "description": "Performance monitoring, caching, and system optimization",
            },
            {"name": "Health", "description": "System health checks and status monitoring"},
        ],
    )

    # Add security schemes
    openapi_schema["components"] = {
        "securitySchemes": {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token for API authentication",
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for service-to-service authentication",
            },
        },
        "schemas": {
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": "object"},
                },
                "required": ["error", "message"],
            },
            "ValidationError": {
                "type": "object",
                "properties": {
                    "detail": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "loc": {"type": "array"},
                                "msg": {"type": "string"},
                                "type": {"type": "string"},
                            },
                        },
                    }
                },
            },
            "HealthCheck": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "healthy"},
                    "timestamp": {"type": "integer"},
                    "version": {"type": "string", "example": "1.0.0"},
                    "services": {
                        "type": "object",
                        "properties": {
                            "database": {"type": "string", "example": "connected"},
                            "redis": {"type": "string", "example": "connected"},
                            "security": {"type": "string", "example": "active"},
                        },
                    },
                },
            },
        },
    }

    # Add global security requirements
    openapi_schema["security"] = [{"BearerAuth": []}, {"ApiKeyAuth": []}]

    # Add server information
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.llm-testing.example.com", "description": "Production server"},
    ]

    # Add contact and license information
    openapi_schema["info"]["contact"] = {
        "name": "LLM A/B Testing Platform",
        "url": "https://github.com/llm-ab-testing-platform",
        "email": "support@llm-testing.example.com",
    }

    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Complete Documentation",
        "url": "https://docs.llm-testing.example.com",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def get_api_examples() -> Dict[str, Any]:
    """Get comprehensive API examples for documentation."""

    return {
        "authentication": {
            "login_request": {
                "summary": "User login example",
                "value": {"username": "user@example.com", "password": "SecurePassword123!"},
            },
            "login_response": {
                "summary": "Successful login response",
                "value": {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "bearer",
                    "expires_in": 86400,
                    "user": {"id": "user-123", "username": "user@example.com", "role": "USER"},
                },
            },
        },
        "providers": {
            "create_openai_provider": {
                "summary": "Create OpenAI provider",
                "value": {
                    "name": "GPT-4 Provider",
                    "provider_type": "openai",
                    "config": {
                        "api_key": "sk-...",
                        "model": "gpt-4",
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    },
                    "is_active": True,
                },
            },
            "create_anthropic_provider": {
                "summary": "Create Anthropic provider",
                "value": {
                    "name": "Claude-3 Provider",
                    "provider_type": "anthropic",
                    "config": {
                        "api_key": "ant-...",
                        "model": "claude-3-sonnet-20240229",
                        "max_tokens": 1000,
                    },
                    "is_active": True,
                },
            },
        },
        "tests": {
            "create_ab_test": {
                "summary": "Create A/B test",
                "value": {
                    "name": "GPT-4 vs Claude-3 Comparison",
                    "description": "Comparing response quality for customer support tasks",
                    "prompt_template": "You are a helpful customer support agent. Respond to: {question}",
                    "provider_a_id": "provider-123",
                    "provider_b_id": "provider-456",
                    "evaluation_criteria": {"helpfulness": 0.4, "accuracy": 0.3, "clarity": 0.3},
                    "sample_size": 100,
                    "confidence_level": 0.95,
                },
            },
            "test_response": {
                "summary": "Test creation response",
                "value": {
                    "id": "test-789",
                    "name": "GPT-4 vs Claude-3 Comparison",
                    "status": "draft",
                    "created_at": "2024-01-01T12:00:00Z",
                    "provider_a_id": "provider-123",
                    "provider_b_id": "provider-456",
                    "sample_size": 100,
                    "evaluations_count": 0,
                    "confidence_level": 0.95,
                },
            },
        },
        "evaluations": {
            "submit_evaluation": {
                "summary": "Submit evaluation data",
                "value": {
                    "test_id": "test-789",
                    "input_text": "How do I reset my password?",
                    "response_a": "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your email.",
                    "response_b": "You can reset your password by visiting our password reset page and entering your email address.",
                    "evaluation_scores": {
                        "helpfulness": {"a": 0.9, "b": 0.7},
                        "accuracy": {"a": 0.95, "b": 0.8},
                        "clarity": {"a": 0.85, "b": 0.75},
                    },
                    "metadata": {
                        "evaluator": "human-expert",
                        "evaluation_time": 45,
                        "difficulty": "easy",
                    },
                },
            }
        },
        "analytics": {
            "dashboard_response": {
                "summary": "Dashboard analytics",
                "value": {
                    "total_tests": 25,
                    "active_tests": 3,
                    "completed_tests": 22,
                    "total_evaluations": 2847,
                    "average_confidence": 0.94,
                    "recent_activity": [
                        {
                            "type": "test_completed",
                            "test_name": "GPT-4 vs Claude-3",
                            "timestamp": "2024-01-01T12:00:00Z",
                        }
                    ],
                },
            },
            "test_results": {
                "summary": "A/B test results",
                "value": {
                    "test_id": "test-789",
                    "test_name": "GPT-4 vs Claude-3 Comparison",
                    "status": "completed",
                    "total_evaluations": 100,
                    "results": {
                        "provider_a": {
                            "name": "GPT-4 Provider",
                            "average_scores": {
                                "helpfulness": 0.87,
                                "accuracy": 0.91,
                                "clarity": 0.84,
                            },
                            "overall_score": 0.874,
                            "confidence_interval": [0.851, 0.897],
                        },
                        "provider_b": {
                            "name": "Claude-3 Provider",
                            "average_scores": {
                                "helpfulness": 0.83,
                                "accuracy": 0.88,
                                "clarity": 0.86,
                            },
                            "overall_score": 0.857,
                            "confidence_interval": [0.832, 0.882],
                        },
                    },
                    "statistical_significance": {
                        "p_value": 0.032,
                        "is_significant": True,
                        "confidence_level": 0.95,
                        "winner": "provider_a",
                    },
                },
            },
        },
    }


def get_response_examples() -> Dict[str, Any]:
    """Get standardized response examples for common scenarios."""

    return {
        "success_responses": {
            "200": {
                "description": "Successful operation",
                "content": {"application/json": {"example": {"status": "success", "data": {}}}},
            },
            "201": {
                "description": "Resource created successfully",
                "content": {
                    "application/json": {
                        "example": {"status": "created", "data": {"id": "new-resource-id"}}
                    }
                },
            },
            "204": {"description": "No content - operation successful"},
        },
        "error_responses": {
            "400": {
                "description": "Bad Request - Invalid input data",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"},
                        "example": {
                            "error": "validation_error",
                            "message": "Invalid input data provided",
                            "details": {
                                "field": "provider_type",
                                "issue": "must be one of: openai, anthropic, google",
                            },
                        },
                    }
                },
            },
            "401": {
                "description": "Unauthorized - Authentication required",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"},
                        "example": {
                            "error": "unauthorized",
                            "message": "Authentication credentials required",
                        },
                    }
                },
            },
            "403": {
                "description": "Forbidden - Insufficient permissions",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"},
                        "example": {
                            "error": "forbidden",
                            "message": "Insufficient permissions for this operation",
                        },
                    }
                },
            },
            "404": {
                "description": "Not Found - Resource does not exist",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"},
                        "example": {
                            "error": "not_found",
                            "message": "Requested resource not found",
                        },
                    }
                },
            },
            "422": {
                "description": "Validation Error - Request data validation failed",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ValidationError"},
                        "example": {
                            "detail": [
                                {
                                    "loc": ["body", "provider_type"],
                                    "msg": "field required",
                                    "type": "value_error.missing",
                                }
                            ]
                        },
                    }
                },
            },
            "429": {
                "description": "Too Many Requests - Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"},
                        "example": {
                            "error": "rate_limit_exceeded",
                            "message": "Too many requests. Please try again later.",
                            "details": {"retry_after": 60},
                        },
                    }
                },
            },
            "500": {
                "description": "Internal Server Error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"},
                        "example": {
                            "error": "internal_error",
                            "message": "An unexpected error occurred",
                        },
                    }
                },
            },
        },
    }
