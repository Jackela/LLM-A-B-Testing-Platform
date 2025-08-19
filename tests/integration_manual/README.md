# Manual Integration Tests

This directory contains **manual integration tests** that require specific setup or external services and are run manually rather than as part of the automated CI/CD pipeline.

## 🎯 Purpose

Manual integration tests serve to:
- Validate integration with external services that aren't easily mocked
- Test scenarios requiring manual setup or intervention
- Perform exploratory testing of complex workflows
- Validate security and access control mechanisms

## 📁 Test Organization

```
tests/integration_manual/
├── README.md                    # This file
├── test_integration_simple.py  # Basic integration validation
└── test_security_simple.py     # Security mechanism testing
```

## 🔒 Security Tests

### test_security_simple.py
Tests security mechanisms including:
- Authentication flows
- Authorization controls
- Input validation
- SQL injection prevention
- XSS protection
- Rate limiting
- Audit logging

**Manual Setup Required**:
- Valid authentication credentials
- Different user roles and permissions
- Security scanning tools
- Network access controls

## 🔗 Integration Tests

### test_integration_simple.py
Tests basic integration scenarios:
- Database connectivity
- External API integrations
- Service-to-service communication
- Configuration validation
- Health check endpoints

**Manual Setup Required**:
- Live database connections
- External service credentials
- Network connectivity
- Specific environment configurations

## 🚀 Running Manual Tests

### Prerequisites
```bash
# 1. Set up environment
export DATABASE_URL="postgresql://user:pass@prod-db:5432/database"
export REDIS_URL="redis://prod-redis:6379/0"

# 2. Configure credentials
export OPENAI_API_KEY="real-api-key"
export ANTHROPIC_API_KEY="real-api-key"

# 3. Set up security context
export TEST_USER_TOKEN="valid-jwt-token"
export ADMIN_USER_TOKEN="admin-jwt-token"
```

### Execute Tests
```bash
# Run all manual integration tests
cd tests/integration_manual
python -m pytest . -v

# Run security tests only
python -m pytest test_security_*.py -v

# Run with specific markers
python -m pytest -m "manual" -v
python -m pytest -m "security" -v
```

### Individual Test Execution
```bash
# Security testing
python -m pytest test_security_simple.py::test_authentication_flow -v

# Integration testing
python -m pytest test_integration_simple.py::test_external_api_connectivity -v
```

## ⚙️ Test Environment Setup

### Security Test Environment
```bash
# 1. User Management Setup
# Create test users with different roles
# - Regular user
# - Admin user
# - Limited permission user

# 2. Authentication Setup
# Configure JWT tokens for each user type
# Set up API key authentication
# Configure OAuth flows if applicable

# 3. Network Setup
# Configure firewall rules for testing
# Set up rate limiting rules
# Configure SSL/TLS certificates
```

### Integration Test Environment
```bash
# 1. Database Setup
# Create dedicated test database
# Configure read/write permissions
# Set up connection pooling

# 2. External Services
# Configure LLM provider accounts
# Set up monitoring service connections
# Configure message queue connections

# 3. Infrastructure
# Set up load balancers
# Configure reverse proxies
# Set up SSL termination
```

## 📋 Test Execution Checklist

### Pre-Test Setup
- [ ] **Environment Configuration**: All required environment variables set
- [ ] **Credentials**: Valid API keys and authentication tokens
- [ ] **Network Access**: Connectivity to external services confirmed
- [ ] **Database State**: Test database in known clean state
- [ ] **Service Dependencies**: All required services running and healthy

### During Test Execution
- [ ] **Monitor Resources**: Watch CPU, memory, and network usage
- [ ] **Log Analysis**: Monitor application and system logs
- [ ] **Error Handling**: Verify graceful error handling
- [ ] **Performance**: Monitor response times and throughput
- [ ] **Security**: Validate security controls are effective

### Post-Test Cleanup
- [ ] **Data Cleanup**: Remove or anonymize test data
- [ ] **Resource Cleanup**: Release external resources
- [ ] **Log Review**: Analyze logs for issues or improvements
- [ ] **Documentation**: Update test results and findings
- [ ] **Issue Tracking**: Log any issues discovered

## 🔍 Test Categories

### Authentication & Authorization
```python
# Example test structure
def test_jwt_token_validation():
    """Test JWT token validation with various scenarios."""
    
def test_role_based_access_control():
    """Test that users can only access authorized resources."""
    
def test_api_key_authentication():
    """Test API key authentication flows."""
```

### Input Validation & Security
```python
def test_sql_injection_prevention():
    """Test SQL injection attack prevention."""
    
def test_xss_protection():
    """Test cross-site scripting protection."""
    
def test_input_sanitization():
    """Test input validation and sanitization."""
```

### External Service Integration
```python
def test_llm_provider_connectivity():
    """Test connectivity to all configured LLM providers."""
    
def test_database_failover():
    """Test database failover scenarios."""
    
def test_monitoring_integration():
    """Test integration with monitoring services."""
```

## 📊 Test Results Documentation

### Test Execution Log Template
```markdown
# Manual Integration Test Execution

**Date**: YYYY-MM-DD
**Tester**: Name
**Environment**: staging/production-like
**Duration**: X minutes

## Test Results Summary
- **Total Tests**: X
- **Passed**: X
- **Failed**: X
- **Skipped**: X

## Failed Tests
1. **test_name**: Description of failure and investigation steps
2. **test_name**: Description of failure and next actions

## Performance Observations
- Response times within acceptable ranges
- No memory leaks observed
- Database performance stable

## Security Findings
- All authentication flows working correctly
- No security vulnerabilities detected
- Rate limiting functioning as expected

## Recommendations
1. Action item 1
2. Action item 2
3. Follow-up investigation needed
```

### Issue Tracking
Document any issues discovered during manual testing:
- **Security vulnerabilities**: Immediate escalation required
- **Performance issues**: Monitor and optimize if needed
- **Integration failures**: Work with external service providers
- **Configuration problems**: Update documentation and procedures

## 🛠️ Test Utilities

### Setup Scripts
```bash
# scripts/setup-manual-tests.sh
#!/bin/bash
# Set up environment for manual integration testing

echo "Setting up manual integration test environment..."

# Database setup
echo "Configuring test database..."
# Add database setup commands

# Credential validation
echo "Validating credentials..."
# Add credential validation

# Service health checks
echo "Checking service health..."
# Add health check commands

echo "Manual test environment ready!"
```

### Helper Functions
```python
# tests/integration_manual/helpers.py
def validate_environment():
    """Validate that all required environment variables are set."""
    
def setup_test_users():
    """Create test users with appropriate roles."""
    
def cleanup_test_data():
    """Clean up test data after manual test execution."""
    
def generate_test_report():
    """Generate summary report of manual test execution."""
```

## 📈 Maintenance & Updates

### Regular Reviews
- **Weekly**: Review test execution results and update procedures
- **Monthly**: Validate test coverage of manual scenarios
- **Quarterly**: Update test environments and credentials
- **Annually**: Review and update security test scenarios

### Process Improvements
- Automate manual steps where possible
- Improve test documentation and procedures
- Enhance error reporting and debugging
- Streamline environment setup processes

### Knowledge Sharing
- Document lessons learned from manual testing
- Share findings with development and operations teams
- Update security and integration best practices
- Train team members on manual testing procedures

---

## 📞 Support & Escalation

### For Security Issues
1. **Immediate**: Stop testing and escalate to security team
2. **Document**: Record details of security findings
3. **Remediate**: Work with development team on fixes
4. **Verify**: Re-test after remediation

### For Integration Issues
1. **Investigate**: Check logs and service status
2. **Coordinate**: Work with external service providers if needed
3. **Document**: Record integration issues and resolutions
4. **Prevent**: Update monitoring and alerting

### For Test Environment Issues
1. **Troubleshoot**: Check environment configuration
2. **Reset**: Restore environment to known good state
3. **Escalate**: Contact infrastructure team if needed
4. **Document**: Update setup procedures

*Manual integration tests are critical for validating real-world scenarios. Execute them carefully and document findings thoroughly.*