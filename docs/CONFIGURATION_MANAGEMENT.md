# Configuration Management Guide

## 🎯 Configuration Strategy

The LLM A/B Testing Platform uses **environment-based configuration** with clear separation between development, testing, staging, and production environments.

**Core Principles**:
- **Environment Isolation**: Each environment has distinct configurations
- **Secret Security**: Sensitive data never in version control
- **Configuration Validation**: All configurations validated at startup
- **Hot Reloading**: Non-critical configs support runtime updates

## 📁 Configuration Structure

```
├── configs/                    # Environment configurations
│   ├── grafana/               # Monitoring dashboards
│   │   ├── dashboards/
│   │   └── datasources/
│   ├── nginx.conf             # Reverse proxy configuration
│   └── prometheus.yml         # Metrics collection
├── .env.example              # Environment template
├── .env.local                # Local development (gitignored)
├── .env.test                 # Test environment
├── .env.staging              # Staging environment
├── .env.production           # Production environment (encrypted)
├── docker-compose.yml        # Production services
├── docker-compose.dev.yml    # Development services
├── docker-compose.test.yml   # Test services
├── alembic.ini              # Database migration config
└── pyproject.toml           # Project and tool configuration
```

## 🔧 Environment Configuration

### Environment Variables

**Core Application Settings**:
```bash
# Application
APP_NAME="LLM A/B Testing Platform"
APP_VERSION="1.0.0"
DEBUG=false
ENVIRONMENT=production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=300

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501
DASHBOARD_TITLE="LLM A/B Testing Dashboard"
```

**Database Configuration**:
```bash
# PostgreSQL
DATABASE_URL=postgresql://username:password@host:port/database
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_TIMEOUT=30
SQLALCHEMY_ECHO=false

# Redis
REDIS_URL=redis://host:port/db
REDIS_POOL_SIZE=10
REDIS_TIMEOUT=5
```

**External Services**:
```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
JAEGER_URL=http://localhost:14268

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/llm-testing.log
```

### Environment-Specific Configurations

#### Development (.env.local)
```bash
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_testing_dev
REDIS_URL=redis://localhost:6379/0
SQLALCHEMY_ECHO=true

# Mock external services for development
MOCK_LLM_PROVIDERS=true
ENABLE_PERFORMANCE_MONITORING=false
```

#### Testing (.env.test)
```bash
TESTING=true
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/llm_testing_test
REDIS_URL=redis://localhost:6379/1

# Test-specific settings
DISABLE_AUTHENTICATION=true
MOCK_EXTERNAL_SERVICES=true
TEST_TIMEOUT=300
COVERAGE_THRESHOLD=85
```

#### Staging (.env.staging)
```bash
ENVIRONMENT=staging
DATABASE_URL=postgresql://user:pass@staging-db:5432/llm_testing_staging
REDIS_URL=redis://staging-redis:6379/0

# Staging-specific settings
ENABLE_DEBUG_ENDPOINTS=true
ENABLE_PERFORMANCE_MONITORING=true
RATE_LIMIT_ENABLED=false
```

#### Production (.env.production)
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Production database (from secrets)
DATABASE_URL=${DATABASE_URL_SECRET}
REDIS_URL=${REDIS_URL_SECRET}

# Security settings
ENABLE_AUTHENTICATION=true
ENABLE_RATE_LIMITING=true
ENABLE_AUDIT_LOGGING=true
CORS_ORIGINS=https://yourdomain.com
```

## 🔒 Secret Management

### Secret Categories

**Level 1 - Critical Secrets**:
- Database credentials
- LLM provider API keys
- JWT signing keys
- Encryption keys

**Level 2 - Sensitive Configuration**:
- Third-party service URLs
- Monitoring credentials
- SSL certificates

**Level 3 - Environment Settings**:
- Feature flags
- Rate limits
- Cache settings

### Secret Storage Strategy

#### Development
```bash
# Use .env.local (gitignored)
OPENAI_API_KEY=sk-development-key
ANTHROPIC_API_KEY=sk-development-key
JWT_SECRET_KEY=dev-secret-key-not-for-production
```

#### Production
```bash
# Use environment variables from secure storage
export DATABASE_URL=$(vault kv get -field=url secret/database)
export OPENAI_API_KEY=$(vault kv get -field=key secret/openai)
export JWT_SECRET_KEY=$(vault kv get -field=key secret/jwt)
```

### Secret Validation
```python
# src/infrastructure/security/secrets_manager.py
class SecretsManager:
    def validate_required_secrets(self):
        """Validate all required secrets are present."""
        required_secrets = [
            "DATABASE_URL",
            "REDIS_URL", 
            "JWT_SECRET_KEY"
        ]
        
        missing = [s for s in required_secrets if not os.getenv(s)]
        if missing:
            raise ConfigurationError(f"Missing secrets: {missing}")
```

## ⚙️ Service Configuration

### Docker Compose Configuration

#### Development Services (docker-compose.dev.yml)
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: llm_testing_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./configs/grafana:/etc/grafana/provisioning

volumes:
  postgres_dev_data:
```

#### Test Services (docker-compose.test.yml)
```yaml
version: '3.8'
services:
  postgres-test:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: llm_testing_test
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"
    tmpfs:
      - /var/lib/postgresql/data  # In-memory for faster tests
      
  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    command: redis-server --save ""  # Disable persistence
```

#### Production Services (docker-compose.yml)
```yaml
version: '3.8'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      
  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    environment:
      - API_BASE_URL=http://api:8000
    depends_on:
      - api
      
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./configs/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
      - dashboard
```

### Database Configuration

#### Alembic Configuration (alembic.ini)
```ini
[alembic]
script_location = alembic
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = postgresql://postgres:postgres@localhost:5432/llm_testing

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = --line-length 100 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[logger_alembic]
level = INFO
handlers =
qualname = alembic
```

#### Database Connection Configuration
```python
# src/infrastructure/persistence/database.py
class DatabaseConfig:
    def __init__(self):
        self.url = os.getenv("DATABASE_URL")
        self.pool_size = int(os.getenv("DATABASE_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        self.pool_timeout = int(os.getenv("DATABASE_TIMEOUT", "30"))
        self.echo = os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true"
        
    def validate(self):
        if not self.url:
            raise ConfigurationError("DATABASE_URL is required")
        if not self.url.startswith("postgresql://"):
            raise ConfigurationError("Only PostgreSQL databases supported")
```

## 📊 Monitoring Configuration

### Prometheus Configuration (configs/prometheus.yml)
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'llm-testing-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'llm-testing-dashboard'
    static_configs:
      - targets: ['dashboard:8501']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard Configuration
```yaml
# configs/grafana/dashboards/dashboard.yml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

### Nginx Configuration (configs/nginx.conf)
```nginx
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }
    
    upstream dashboard_backend {
        server dashboard:8501;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # API routes
        location /api/ {
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Dashboard routes
        location / {
            proxy_pass http://dashboard_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

## 🔄 Configuration Management Workflow

### Development Workflow
1. **Copy template**: `cp .env.example .env.local`
2. **Set development values**: Edit `.env.local` with local settings
3. **Start services**: `make services-up`
4. **Validate configuration**: `make check-config`

### Deployment Workflow
1. **Environment preparation**: Create environment-specific `.env` file
2. **Secret injection**: Load secrets from secure storage
3. **Configuration validation**: Verify all required settings
4. **Service deployment**: Deploy with environment configuration
5. **Health verification**: Confirm services are healthy

### Configuration Updates
```bash
# Non-breaking configuration changes
1. Update configuration files
2. Test in development environment
3. Deploy to staging for validation
4. Roll out to production with monitoring

# Breaking configuration changes
1. Plan backward compatibility
2. Deploy configuration first
3. Deploy application changes
4. Remove deprecated configuration
```

## 🛠️ Configuration Utilities

### Configuration Validation Script
```python
# scripts/validate-config.py
#!/usr/bin/env python3
"""Validate environment configuration."""

import os
import sys
from urllib.parse import urlparse

def validate_database_url(url):
    """Validate database URL format and connectivity."""
    if not url:
        return False, "DATABASE_URL is required"
    
    parsed = urlparse(url)
    if parsed.scheme != 'postgresql':
        return False, "Only PostgreSQL databases supported"
    
    return True, "Database URL is valid"

def validate_redis_url(url):
    """Validate Redis URL format."""
    if not url:
        return False, "REDIS_URL is required"
    
    parsed = urlparse(url)
    if parsed.scheme != 'redis':
        return False, "Invalid Redis URL scheme"
    
    return True, "Redis URL is valid"

def main():
    """Main validation function."""
    errors = []
    
    # Validate required environment variables
    required_vars = [
        "DATABASE_URL",
        "REDIS_URL",
        "JWT_SECRET_KEY"
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate database configuration
    db_valid, db_msg = validate_database_url(os.getenv("DATABASE_URL"))
    if not db_valid:
        errors.append(db_msg)
    
    # Validate Redis configuration
    redis_valid, redis_msg = validate_redis_url(os.getenv("REDIS_URL"))
    if not redis_valid:
        errors.append(redis_msg)
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  ❌ {error}")
        sys.exit(1)
    else:
        print("✅ Configuration validation passed")

if __name__ == "__main__":
    main()
```

### Environment Setup Script
```bash
#!/bin/bash
# scripts/setup-environment.sh

set -euo pipefail

ENVIRONMENT=${1:-development}

echo "🔧 Setting up $ENVIRONMENT environment..."

# Copy appropriate environment file
if [ "$ENVIRONMENT" = "development" ]; then
    cp .env.example .env.local
    echo "📝 Created .env.local from template"
    echo "💡 Please edit .env.local with your local settings"
elif [ "$ENVIRONMENT" = "test" ]; then
    if [ ! -f .env.test ]; then
        echo "❌ .env.test file not found"
        exit 1
    fi
    ln -sf .env.test .env
    echo "🔗 Linked .env.test"
else
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Valid environments: development, test"
    exit 1
fi

# Validate configuration
python scripts/validate-config.py

echo "✅ Environment setup complete"
```

## 📚 Configuration Best Practices

### Security Guidelines
1. **Never commit secrets** to version control
2. **Use environment variables** for all configuration
3. **Validate configuration** at application startup
4. **Rotate secrets regularly** in production
5. **Use least privilege** for service accounts

### Maintainability Guidelines
1. **Document all configuration** options
2. **Provide sensible defaults** where possible
3. **Group related settings** logically
4. **Use consistent naming** conventions
5. **Version configuration** schemas

### Operational Guidelines
1. **Monitor configuration** drift between environments
2. **Test configuration** changes in staging first
3. **Plan rollback procedures** for configuration changes
4. **Automate configuration** validation
5. **Maintain configuration** documentation

---

## 🎯 Quick Reference

### Common Commands
```bash
# Environment setup
make setup-env ENV=development
make validate-config

# Service management
make services-up ENV=development
make services-down
make services-logs

# Configuration validation
python scripts/validate-config.py
make check-config

# Secret management
make encrypt-secrets
make decrypt-secrets ENV=production
```

### Environment Variables Priority
1. **Explicit environment variables** (highest priority)
2. **Environment-specific .env files** (.env.production)
3. **General .env file** (.env)
4. **Default values in code** (lowest priority)

*Configuration management evolves with deployment needs. Keep documentation synchronized with actual configuration files.*