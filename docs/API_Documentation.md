# LLM A/B Testing Platform API

Version: 1.0.0  
Generated: 2024-08-14

## Overview

Enterprise-grade platform for Large Language Model A/B testing and evaluation with comprehensive security, performance monitoring, and detailed analytics.

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

## Base URLs

- Development: `http://localhost:8000`
- Production: `https://api.llm-testing.example.com`

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

## API Endpoints

### Authentication

#### `POST /api/v1/auth/login`

**User authentication and token generation**

Authenticate with username/email and password to receive a JWT access token.

**Request Body:**

```json
{
  "username": "user@company.com",
  "password": "SecurePassword123!"
}
```

**Responses:**

- `200`: Authentication successful
- `401`: Invalid credentials
- `422`: Validation error

**Response Example:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user-12345",
    "username": "user@company.com",
    "role": "USER",
    "permissions": ["CREATE_TEST", "READ_TEST"]
  }
}
```

---

#### `GET /api/v1/auth/me`

**Get current user profile**

Returns the profile information for the authenticated user.

**Headers:**
- `Authorization: Bearer <token>` (required)

**Responses:**

- `200`: User profile retrieved
- `401`: Invalid or expired token

---

### Model Providers

#### `POST /api/v1/providers/`

**Create a new model provider**

Register a new LLM provider configuration for A/B testing.

**Request Body:**

```json
{
  "name": "OpenAI GPT-4 Turbo",
  "provider_type": "openai",
  "config": {
    "api_key": "sk-proj-...",
    "model": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "is_active": true,
  "description": "Latest GPT-4 Turbo model"
}
```

**Responses:**

- `201`: Provider created successfully
- `400`: Invalid provider configuration
- `401`: Authentication required
- `422`: Validation errors

---

#### `GET /api/v1/providers/`

**List all providers**

Retrieve all model providers accessible to the current user.

**Query Parameters:**
- `is_active` (boolean, optional): Filter by active status
- `provider_type` (string, optional): Filter by provider type

**Responses:**

- `200`: List of providers
- `401`: Authentication required

---

#### `GET /api/v1/providers/{provider_id}`

**Get specific provider**

Retrieve details for a specific model provider.

**Parameters:**
- `provider_id` (path, string, required): Provider identifier

**Responses:**

- `200`: Provider details
- `404`: Provider not found
- `401`: Authentication required

---

### Test Management

#### `POST /api/v1/tests/`

**Create a new A/B test**

Set up a new A/B test between two model providers.

**Request Body:**

```json
{
  "name": "Customer Support Response Quality",
  "description": "Comparing GPT-4 vs Claude-3 for customer support",
  "prompt_template": "You are a helpful customer support agent. {question}",
  "provider_a_id": "provider-abc123",
  "provider_b_id": "provider-def456",
  "evaluation_criteria": {
    "helpfulness": 0.35,
    "accuracy": 0.25,
    "professionalism": 0.25,
    "clarity": 0.15
  },
  "sample_size": 200,
  "confidence_level": 0.95
}
```

**Validation Rules:**
- Evaluation criteria weights must sum to 1.0
- Sample size must be between 10 and 10,000
- Confidence level must be between 0.8 and 0.99
- Provider A and B must be different

**Responses:**

- `201`: Test created successfully
- `400`: Invalid test configuration
- `422`: Validation errors

---

#### `GET /api/v1/tests/`

**List all tests**

Retrieve all A/B tests accessible to the current user.

**Query Parameters:**
- `status` (string, optional): Filter by test status
- `created_by` (string, optional): Filter by creator

**Responses:**

- `200`: List of tests
- `401`: Authentication required

---

#### `POST /api/v1/tests/{test_id}/start`

**Start an A/B test**

Begin collecting evaluation data for the specified test.

**Parameters:**
- `test_id` (path, string, required): Test identifier

**Responses:**

- `200`: Test started successfully
- `400`: Test cannot be started (invalid status)
- `404`: Test not found

---

### Evaluation

#### `POST /api/v1/evaluation/submit`

**Submit evaluation data**

Submit evaluation scores for a specific test comparison.

**Request Body:**

```json
{
  "test_id": "test-xyz789",
  "input_text": "How do I reset my password?",
  "response_a": "To reset your password, go to the login page...",
  "response_b": "You can reset your password by visiting...",
  "evaluation_scores": {
    "helpfulness": {"a": 0.9, "b": 0.7},
    "accuracy": {"a": 0.95, "b": 0.8},
    "professionalism": {"a": 0.85, "b": 0.9},
    "clarity": {"a": 0.8, "b": 0.95}
  },
  "evaluator": "expert-evaluator-001",
  "metadata": {
    "difficulty": "medium",
    "category": "account_management"
  }
}
```

**Responses:**

- `201`: Evaluation submitted successfully
- `400`: Invalid evaluation data
- `404`: Test not found
- `422`: Validation errors

---

#### `GET /api/v1/evaluation/test/{test_id}/results`

**Get evaluation results**

Retrieve all evaluations and aggregated results for a test.

**Parameters:**
- `test_id` (path, string, required): Test identifier

**Responses:**

- `200`: Evaluation results
- `404`: Test not found

---

### Analytics

#### `GET /api/v1/analytics/dashboard`

**Get dashboard summary**

Retrieve high-level analytics and recent activity.

**Response Example:**

```json
{
  "summary": {
    "total_tests": 47,
    "active_tests": 5,
    "completed_tests": 39,
    "total_evaluations": 8429,
    "total_providers": 12
  },
  "recent_activity": [
    {
      "type": "test_completed",
      "test_name": "Email Response Quality",
      "timestamp": "2024-01-01T11:45:00Z",
      "result": "Provider A significantly better"
    }
  ],
  "performance_metrics": {
    "average_test_duration": 5.2,
    "average_evaluations_per_test": 179,
    "average_confidence_level": 0.93
  }
}
```

---

#### `GET /api/v1/analytics/test-results`

**Get comprehensive test results**

Retrieve detailed statistical analysis for completed tests.

**Query Parameters:**
- `test_id` (string, optional): Specific test ID
- `limit` (integer, optional): Number of results to return

**Response Example:**

```json
{
  "test_id": "test-xyz789",
  "test_name": "Customer Support Response Quality", 
  "status": "completed",
  "total_evaluations": 200,
  "results": {
    "provider_a": {
      "name": "OpenAI GPT-4 Turbo",
      "overall_score": 0.835,
      "confidence_interval": [0.811, 0.859],
      "scores": {
        "helpfulness": {"mean": 0.847, "std": 0.123},
        "accuracy": {"mean": 0.823, "std": 0.145}
      }
    },
    "provider_b": {
      "name": "Claude-3 Sonnet", 
      "overall_score": 0.825,
      "confidence_interval": [0.802, 0.848]
    }
  },
  "statistical_analysis": {
    "p_value": 0.0423,
    "is_significant": true,
    "winner": "provider_a",
    "improvement_percentage": 1.21
  }
}
```

---

### Security

#### `GET /api/v1/security/status`

**Get security system status**

Retrieve security monitoring and system health information.

**Responses:**

- `200`: Security status retrieved
- `403`: Admin access required

---

### Performance

#### `GET /api/v1/performance/status`

**Get performance metrics**

Retrieve system performance and caching statistics.

**Response Example:**

```json
{
  "api_metrics": {
    "total_requests": 45672,
    "average_response_time": 234,
    "success_rate": 0.9978
  },
  "cache_metrics": {
    "hit_rate": 0.847,
    "memory_usage_mb": 256
  },
  "security_metrics": {
    "blocked_requests": 1847,
    "rate_limited_requests": 234
  }
}
```

---

### Health

#### `GET /health`

**System health check**

Basic health check endpoint (no authentication required).

**Response:**

```json
{
  "status": "healthy",
  "timestamp": 1640995200,
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "redis": "connected", 
    "security": "active"
  }
}
```

---

## Data Models

### Provider Configuration

```json
{
  "api_key": "string",
  "model": "string",
  "temperature": 0.7,
  "max_tokens": 2000,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

### Test Configuration

```json
{
  "name": "string",
  "description": "string",
  "prompt_template": "string with {variables}",
  "provider_a_id": "string",
  "provider_b_id": "string", 
  "evaluation_criteria": {
    "criterion_name": 0.0
  },
  "sample_size": 100,
  "confidence_level": 0.95
}
```

### Evaluation Scores

```json
{
  "criterion_name": {
    "a": 0.0,
    "b": 0.0
  }
}
```

## Error Handling

The API uses standard HTTP status codes and returns consistent error responses:

### Error Response Format

```json
{
  "error": "error_type",
  "message": "Human-readable error message",
  "details": {
    "additional": "context information"
  }
}
```

### Status Codes

| Code | Description | Action |
|------|-------------|--------|
| 200 | Success | Continue |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Check request format and data |
| 401 | Unauthorized | Provide valid authentication |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Verify resource exists |
| 422 | Validation Error | Fix validation issues |
| 429 | Rate Limited | Reduce request rate |
| 500 | Server Error | Contact support |

### Rate Limiting

API requests are rate-limited:
- **Standard Users**: 100 requests per minute
- **Premium Users**: 1000 requests per minute
- **Enterprise**: Custom limits

When rate limited, the response includes a `Retry-After` header indicating when to retry.

## SDK Examples

### Python

```python
import httpx
import asyncio

class LLMTestingClient:
    def __init__(self, token: str, base_url: str = "http://localhost:8000"):
        self.token = token
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    async def create_provider(self, provider_data: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/providers/",
                json=provider_data,
                headers=self.headers
            )
            return response.json()
    
    async def create_test(self, test_data: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/tests/",
                json=test_data,
                headers=self.headers
            )
            return response.json()
    
    async def submit_evaluation(self, evaluation_data: dict):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/evaluation/submit",
                json=evaluation_data,
                headers=self.headers
            )
            return response.json()

# Usage example
async def main():
    client = LLMTestingClient("your-token-here")
    
    # Create provider
    provider = await client.create_provider({
        "name": "GPT-4 Provider",
        "provider_type": "openai",
        "config": {
            "api_key": "sk-...",
            "model": "gpt-4",
            "temperature": 0.7
        },
        "is_active": True
    })
    
    print(f"Created provider: {provider['id']}")

asyncio.run(main())
```

### JavaScript

```javascript
class LLMTestingClient {
    constructor(token, baseUrl = 'http://localhost:8000') {
        this.token = token;
        this.baseUrl = baseUrl;
        this.headers = { 
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async createProvider(providerData) {
        const response = await fetch(`${this.baseUrl}/api/v1/providers/`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(providerData)
        });
        return response.json();
    }
    
    async createTest(testData) {
        const response = await fetch(`${this.baseUrl}/api/v1/tests/`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(testData)
        });
        return response.json();
    }
    
    async submitEvaluation(evaluationData) {
        const response = await fetch(`${this.baseUrl}/api/v1/evaluation/submit`, {
            method: 'POST', 
            headers: this.headers,
            body: JSON.stringify(evaluationData)
        });
        return response.json();
    }
}

// Usage example
const client = new LLMTestingClient('your-token-here');

client.createProvider({
    name: 'Claude-3 Provider',
    provider_type: 'anthropic',
    config: {
        api_key: 'ant-...',
        model: 'claude-3-sonnet',
        max_tokens: 2000
    },
    is_active: true
}).then(provider => {
    console.log('Created provider:', provider.id);
});
```

### cURL Examples

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "user@example.com", "password": "password"}'

# Create Provider
curl -X POST http://localhost:8000/api/v1/providers/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPT-4 Provider",
    "provider_type": "openai",
    "config": {
      "api_key": "sk-...",
      "model": "gpt-4",
      "temperature": 0.7
    },
    "is_active": true
  }'

# Create A/B Test
curl -X POST http://localhost:8000/api/v1/tests/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Comparison",
    "description": "Comparing two models",
    "prompt_template": "Answer: {question}",
    "provider_a_id": "provider-1",
    "provider_b_id": "provider-2",
    "evaluation_criteria": {"accuracy": 1.0},
    "sample_size": 100,
    "confidence_level": 0.95
  }'

# Submit Evaluation
curl -X POST http://localhost:8000/api/v1/evaluation/submit \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "test_id": "test-123",
    "input_text": "What is AI?",
    "response_a": "AI is artificial intelligence",
    "response_b": "AI stands for artificial intelligence",
    "evaluation_scores": {
      "accuracy": {"a": 0.9, "b": 0.8}
    }
  }'
```

## Support

- **Interactive Documentation**: `/api/v1/docs`
- **Alternative Documentation**: `/api/v1/redoc`
- **Health Status**: `/health`
- **OpenAPI Schema**: `/openapi.json`

For technical support and questions:
- **GitHub**: https://github.com/llm-ab-testing-platform
- **Email**: support@llm-testing.example.com