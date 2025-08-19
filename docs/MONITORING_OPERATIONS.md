# Monitoring & Observability Operations Guide

## Overview

This guide covers the comprehensive monitoring, observability, and alerting system for the LLM A/B Testing Platform.

## Architecture Overview

### Monitoring Stack

1. **Metrics Collection**: Prometheus-compatible metrics
2. **Logging**: Structured JSON logging with OpenTelemetry
3. **Tracing**: Distributed tracing with Jaeger/OTLP
4. **Alerting**: Multi-channel alerts with correlation
5. **Health Checks**: Comprehensive system health monitoring
6. **Dashboards**: Real-time monitoring dashboards

### Data Flow

```
Application → Metrics Collector → Prometheus → Grafana
            → Structured Logger → Log Aggregation → Search/Analysis
            → Tracer → Jaeger → Trace Analysis
            → Health Checker → Alert Manager → Notifications
```

## Metrics & KPIs

### System Metrics

**CPU & Memory:**
- `system_cpu_usage_percent` - CPU usage by core
- `system_memory_usage_bytes` - Memory usage by type
- `system_memory_usage_percent` - Overall memory percentage
- `process_threads_total` - Number of process threads
- `process_open_fds_total` - Open file descriptors

**Disk & Network:**
- `system_disk_usage_bytes` - Disk usage by device and type
- `system_disk_usage_percent` - Disk usage percentage
- `system_network_bytes_total` - Network bytes transferred

### Application Metrics

**HTTP Requests:**
- `http_requests_total` - Total HTTP requests by method, endpoint, status
- `http_request_duration_seconds` - Request duration histogram

**Database:**
- `database_queries_total` - Total queries by operation and table
- `database_query_duration_seconds` - Query duration histogram
- `database_connections_active` - Active database connections

**Authentication:**
- `authentication_attempts_total` - Auth attempts by result and method
- `authentication_duration_seconds` - Auth duration histogram

**Errors & Security:**
- `application_errors_total` - Application errors by component and type
- `security_events_total` - Security events by type and severity
- `rate_limit_hits_total` - Rate limit violations

### Business Metrics

**Test Management:**
- `tests_created_total` - Total tests created by user role
- `tests_active_total` - Currently active tests
- `test_duration_hours` - Test duration histogram

**Model Providers:**
- `model_requests_total` - Model requests by provider, model, status
- `model_response_time_seconds` - Model response time
- `model_tokens_total` - Token usage by provider and type

**User Activity:**
- `user_actions_total` - User actions by type and role
- `active_users_total` - Currently active users

### Performance Thresholds

**Critical Thresholds:**
- CPU usage > 90% for 5 minutes
- Memory usage > 95% for 2 minutes
- Disk usage > 95%
- Error rate > 10% for 10 minutes
- Response time > 10 seconds

**Warning Thresholds:**
- CPU usage > 80% for 10 minutes
- Memory usage > 85% for 5 minutes
- Disk usage > 90%
- Error rate > 5% for 15 minutes
- Response time > 5 seconds

## Logging System

### Log Levels & Categories

**Log Levels:**
- `DEBUG`: Detailed diagnostic information
- `INFO`: General application flow
- `WARNING`: Potentially harmful situations
- `ERROR`: Error events that don't stop execution
- `CRITICAL`: Very severe error events

**Event Types:**
- `request`: HTTP request/response events
- `response`: API response events
- `error`: Error and exception events
- `security`: Security-related events
- `performance`: Performance metrics and timing
- `business`: Business logic events
- `system`: System and infrastructure events
- `audit`: Audit trail events

### Structured Logging Format

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "message": "User authentication successful",
  "event_type": "security",
  "context": {
    "correlation_id": "req_abc123",
    "user_id": "user_456",
    "session_id": "sess_789",
    "trace_id": "trace_def456",
    "span_id": "span_ghi789"
  },
  "metadata": {
    "module": "auth",
    "function": "authenticate_user",
    "line": 142,
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "duration_ms": 45.7
  }
}
```

### Log Management

**Log Files:**
- `app.log` - General application logs (rotated daily, 30 days retention)
- `app_errors.log` - Error logs only (rotated at 10MB, 5 files retention)
- `security.log` - Security events (rotated at 10MB, 10 files retention)

**Log Aggregation:**
- Centralized log collection
- Full-text search capabilities
- Real-time log streaming
- Correlation with traces and metrics

## Distributed Tracing

### Trace Configuration

```python
# OpenTelemetry configuration
from src.infrastructure.monitoring.tracing import setup_distributed_tracing, TraceConfig

config = TraceConfig(
    service_name="llm-ab-testing",
    service_version="1.0.0",
    environment="production",
    jaeger_endpoint="http://jaeger:14268/api/traces",
    sampling_ratio=0.1  # 10% sampling for production
)

tracing_manager = setup_distributed_tracing(config)
```

### Instrumentation

**Automatic Instrumentation:**
- FastAPI requests and responses
- SQLAlchemy database queries
- Redis cache operations
- HTTP client requests (httpx)
- Logger correlation

**Manual Instrumentation:**
```python
from src.infrastructure.monitoring.tracing import trace_operation

# Trace a function
@trace_function("business_operation", "internal")
async def complex_business_logic():
    pass

# Trace a code block
async with trace_async_operation("model_inference", "external") as span:
    span.set_attribute("model_provider", "openai")
    result = await call_model_api()
    span.set_attribute("tokens_used", result.token_count)
```

### Trace Analysis

**Key Spans to Monitor:**
- HTTP request processing
- Database query execution
- External API calls
- Authentication flows
- Business logic operations

**Performance Analysis:**
- Identify slow operations
- Find bottlenecks in request flows
- Analyze error propagation
- Monitor external service dependencies

## Health Monitoring

### Health Check Configuration

```python
from src.infrastructure.monitoring.health import get_health_checker

# Configure health checks
health_checker = get_health_checker(
    database_session_factory=get_db_session,
    redis_url="redis://localhost:6379",
    external_services={
        "openai": "https://api.openai.com",
        "anthropic": "https://api.anthropic.com"
    }
)

# Run health check
health_status = await health_checker.check_health()
print(f"System health: {health_status.status}")
```

### Health Endpoints

**Health Check URLs:**
- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive health status
- `GET /health/metrics` - Health metrics in Prometheus format

**Health Status Responses:**
```json
{
  "status": "healthy",
  "message": "All systems operational", 
  "timestamp": "2024-01-15T10:30:45.123Z",
  "uptime_seconds": 86400,
  "checks": [
    {
      "name": "database",
      "status": "healthy",
      "message": "Database connection OK",
      "duration_ms": 12.5,
      "details": {
        "query_time_ms": 8.2,
        "active_connections": 5
      }
    }
  ]
}
```

### Health Monitoring Setup

**Continuous Monitoring:**
```python
from src.infrastructure.monitoring.health import get_health_monitor

# Setup health monitor
health_monitor = get_health_monitor(
    check_interval=60,  # Check every minute
    alert_callback=health_status_changed
)

# Start monitoring
await health_monitor.start()
```

## Alerting System

### Alert Rules Configuration

**Critical Alerts:**
```yaml
- name: "database_connection_failed"
  description: "Database connection failure"
  condition: "database_health == 'unhealthy'"
  severity: "critical"
  channels: ["email", "slack", "webhook"]
  duration_minutes: 1

- name: "high_error_rate"
  description: "High error rate in API requests"
  condition: "error_rate > 0.1"
  severity: "high"
  channels: ["email", "slack"]
  duration_minutes: 10
```

### Multi-Channel Notifications

**Email Alerts:**
```python
# Email configuration
email_config = {
    'smtp_host': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'alerts@yourcompany.com',
    'password': 'your-app-password',
    'from_email': 'alerts@yourcompany.com',
    'to_emails': ['admin@yourcompany.com', 'oncall@yourcompany.com']
}
```

**Slack Integration:**
```python
# Slack webhook configuration
slack_config = {
    'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
    'channel': '#alerts',
    'username': 'AlertBot'
}
```

**Generic Webhooks:**
```python
# Webhook configuration for external systems
webhook_config = {
    'url': 'https://your-monitoring-system.com/api/alerts',
    'headers': {
        'Authorization': 'Bearer your-api-token',
        'Content-Type': 'application/json'
    }
}
```

### Alert Correlation & Noise Reduction

**Features:**
- Alert grouping by similarity
- Rate limiting per channel
- Escalation policies
- Automatic correlation of related events
- Suppression of duplicate alerts

**Configuration:**
```python
from src.infrastructure.monitoring.alerting import get_alert_manager

alert_manager = get_alert_manager()

# Trigger alert
await alert_manager.trigger_alert(
    "high_cpu_usage",
    "CPU usage is above 90%",
    labels={"component": "system", "instance": "web-1"},
    annotations={"cpu_percent": 95.2}
)
```

## Dashboard Configuration

### Grafana Dashboards

**System Overview Dashboard:**
- System resource utilization
- Application performance metrics
- Error rates and response times
- Active users and sessions

**Application Performance Dashboard:**
- Request throughput and latency
- Database query performance
- Cache hit rates
- External service response times

**Business Metrics Dashboard:**
- Test creation and completion rates
- Model provider performance
- User activity patterns
- Cost and usage analytics

**Security Dashboard:**
- Authentication metrics
- Security events timeline
- Rate limiting statistics
- Threat detection alerts

### Prometheus Queries

**Key Queries:**
```promql
# Average response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# Error rate
rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m])

# Database query performance
histogram_quantile(0.95, rate(database_query_duration_seconds_bucket[5m]))

# Active users
active_users_total
```

## Operational Procedures

### Daily Monitoring Tasks

**Morning Checklist (15 minutes):**
1. Review overnight alerts and incidents
2. Check system health dashboard
3. Verify backup completion status
4. Monitor resource utilization trends
5. Check external service status

**Ongoing Monitoring:**
- Watch alert notifications
- Investigate performance anomalies
- Monitor error rates and patterns
- Track business metric trends

**End-of-Day Review (10 minutes):**
1. Summary of alerts and incidents
2. Performance trend analysis
3. Resource capacity planning
4. Tomorrow's maintenance tasks

### Weekly Monitoring Tasks

**Monday**: Alert rule review and optimization
**Tuesday**: Dashboard updates and new metrics
**Wednesday**: Performance baseline updates
**Thursday**: Capacity planning and scaling
**Friday**: Monitoring system maintenance

### Monthly Monitoring Activities

- Monitoring system performance review
- Alert effectiveness analysis
- Dashboard usage analytics
- Monitoring tool updates and patches
- Capacity planning and forecasting

## Troubleshooting Guide

### Common Issues

**High CPU Usage:**
1. Check top processes in system metrics
2. Review application performance traces
3. Analyze database query patterns
4. Check for infinite loops or heavy computations
5. Scale resources if needed

**Memory Leaks:**
1. Monitor memory usage trends over time
2. Check for growing object counts in traces
3. Review connection pool configurations
4. Analyze garbage collection patterns
5. Restart services if necessary

**Database Performance:**
1. Check query execution times in traces
2. Review connection pool utilization
3. Analyze slow query logs
4. Check for missing indexes
5. Monitor database server resources

**External Service Issues:**
1. Check external service health status
2. Review API response times and errors
3. Verify authentication and authorization
4. Check rate limiting and quotas
5. Implement circuit breakers if needed

### Performance Optimization

**Application Level:**
- Optimize database queries
- Implement caching strategies
- Reduce external API calls
- Optimize serialization/deserialization
- Use connection pooling

**Infrastructure Level:**
- Scale horizontally with load balancers
- Optimize database configurations
- Implement CDN for static content
- Use caching layers (Redis)
- Monitor and tune garbage collection

## Monitoring Configuration

### Environment Variables

```bash
# Monitoring
LOG_LEVEL=INFO
LOG_DIR=/var/log/llm-ab-testing
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=60

# Tracing
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
OTLP_ENDPOINT=http://otel-collector:4317
TRACE_SAMPLING_RATIO=0.1

# Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/your/webhook
ALERT_FROM_EMAIL=alerts@yourcompany.com
ALERT_TO_EMAILS=admin@yourcompany.com,oncall@yourcompany.com
```

### Docker Compose Integration

```yaml
version: '3.8'
services:
  app:
    environment:
      - LOG_LEVEL=INFO
      - JAEGER_ENDPOINT=http://jaeger:14268/api/traces
      - METRICS_PORT=9090
    ports:
      - "9090:9090"  # Metrics endpoint

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
```

## Contact Information

**Monitoring Team:**
- Primary: monitoring@yourcompany.com
- On-Call: +1-XXX-XXX-XXXX
- Slack: #monitoring-alerts

**Escalation:**
1. Site Reliability Engineer
2. Engineering Manager
3. VP Engineering
4. CTO

## Additional Resources

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Tracing Guide](https://www.jaegertracing.io/docs/)
- [Alert Runbook Templates](./runbooks/)