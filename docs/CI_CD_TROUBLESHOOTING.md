# CI/CD Troubleshooting Guide

This guide helps diagnose and fix common CI/CD pipeline issues in the LLM A/B Testing Platform.

## 🚨 Quick Fix Commands

```bash
# Local testing before pushing
make check-system
make test-enhanced
make test-ci

# Reset and retry
make clean
make setup
make test
```

## 📋 Common Issues and Solutions

### 1. Test Matrix Failures

**Symptoms**: Multiple test jobs failing across different Python versions
**Root Causes**: 
- Service connectivity issues
- Database migration failures
- Dependency conflicts

**Solutions**:
```bash
# Check services locally
make check-services

# Reset database
make db-reset

# Update dependencies
make update
```

### 2. PostgreSQL Connection Issues

**Error**: `pg_isready` fails or connection timeouts
**Solutions**:

```yaml
# Improved service configuration (already implemented)
services:
  postgres:
    image: postgres:15-alpine
    env:
      POSTGRES_PASSWORD: postgres
      POSTGRES_USER: postgres
      POSTGRES_DB: test_db
    options: >-
      --health-cmd "pg_isready -U postgres"
      --health-interval 10s
      --health-timeout 5s
      --health-retries 10
```

**Local debugging**:
```bash
# Test PostgreSQL locally
docker run --rm -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15-alpine
pg_isready -h localhost -p 5432 -U postgres
```

### 3. Redis Connection Issues

**Error**: Redis ping fails or timeouts
**Solutions**:

```bash
# Test Redis locally
docker run --rm -p 6379:6379 redis:7-alpine
redis-cli -h localhost -p 6379 ping
```

### 4. Poetry/Dependency Issues

**Error**: Poetry installation or dependency resolution fails
**Solutions**:

```bash
# Clear Poetry cache
poetry cache clear pypi --all
poetry cache clear _default_cache --all

# Reinstall dependencies
rm poetry.lock
poetry install --with dev
```

### 5. Alembic Migration Failures

**Error**: Database migration fails during setup
**Solutions**:

```bash
# Check migration status
poetry run alembic current
poetry run alembic check

# Reset and retry
poetry run alembic downgrade base
poetry run alembic upgrade head
```

### 6. Timeout Issues

**Error**: Tests timeout during execution
**Solutions**:

1. **Increase timeouts** (already implemented):
   ```yaml
   timeout-minutes: 120
   --timeout=1800
   ```

2. **Optimize slow tests**:
   ```bash
   # Find slow tests
   poetry run pytest --durations=20
   
   # Run only fast tests
   make test-fast
   ```

### 7. Coverage Failures

**Error**: Coverage below threshold
**Solutions**:

```bash
# Check coverage details
poetry run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Lower threshold temporarily
poetry run pytest --cov-fail-under=70
```

## 🔧 Debugging Strategies

### 1. Local Reproduction

```bash
# Reproduce CI environment locally
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/test_db
export REDIS_URL=redis://localhost:6379/0
export TESTING=true

# Run enhanced test suite
python scripts/test-runner.py --verbose
```

### 2. Service Health Checks

```bash
# Check all services
make status

# Start services if needed
make services-up

# Check logs
make services-logs
```

### 3. Step-by-Step Debugging

```bash
# 1. Environment check
make check-system

# 2. Install dependencies
make install

# 3. Start services
make services-up

# 4. Setup database
make setup-db

# 5. Run quality checks
make quality

# 6. Run tests incrementally
make test-unit
make test-integration
make test-e2e
```

## 📊 Monitoring and Alerts

### 1. GitHub Actions Monitoring

- **Workflow runs**: Monitor failure patterns
- **Artifact uploads**: Check for missing reports
- **Duration trends**: Watch for performance regressions

### 2. Error Pattern Analysis

```bash
# Common error patterns to watch for:
# - "ConnectionError"
# - "TimeoutError" 
# - "ImportError"
# - "ModuleNotFoundError"
# - "AssertionError"
```

### 3. Performance Metrics

```bash
# Track these metrics:
# - Test execution time
# - Service startup time
# - Coverage percentage
# - Security scan results
```

## 🚀 Optimization Tips

### 1. Parallel Execution

```bash
# Use parallel testing
make test-parallel

# Or with specific worker count
poetry run pytest -n 4
```

### 2. Smart Test Selection

```bash
# Run only changed files
poetry run pytest --lf  # last failed
poetry run pytest --ff  # failed first

# Test specific markers
poetry run pytest -m "not slow"
```

### 3. Caching Strategies

- **Poetry dependencies**: Cached by lock file hash
- **Docker layers**: Optimized layer ordering
- **Test results**: Incremental testing

## 🛠️ Advanced Troubleshooting

### 1. Matrix Strategy Issues

If you need to reduce matrix complexity:

```yaml
strategy:
  matrix:
    python-version: ['3.11']  # Reduce to single version
    # Remove test-type matrix for now
  fail-fast: true  # Stop on first failure
```

### 2. Resource Constraints

```yaml
# Reduce parallel jobs
jobs:
  test-suite:
    runs-on: ubuntu-latest
    # Remove matrix temporarily
```

### 3. Emergency Bypass

```yaml
# Temporary CI bypass for urgent fixes
on:
  push:
    branches: [main, develop]
    paths-ignore:
      - 'docs/**'
      - '*.md'
```

## 📞 Escalation Process

### Level 1: Self-Service
- Use this troubleshooting guide
- Check recent commit changes
- Run local reproduction steps

### Level 2: Team Review
- Create GitHub issue with:
  - Error logs
  - Reproduction steps
  - Environment details
- Tag relevant team members

### Level 3: Infrastructure
- Contact DevOps/Platform team
- Provide comprehensive error analysis
- Include timing and frequency data

## 🔍 Useful Commands

```bash
# GitHub CLI debugging
gh run list --limit 10
gh run view <run-id> --log
gh workflow list

# Local debugging
make info           # Project information
make status         # Environment status
make test-verbose   # Detailed test output
make coverage       # Coverage analysis

# Docker debugging
docker ps
docker logs <container-id>
docker exec -it <container-id> bash
```

## 📋 Checklist for CI/CD Fixes

- [ ] Services are healthy and accessible
- [ ] Dependencies are correctly installed
- [ ] Database migrations run successfully
- [ ] Environment variables are set correctly
- [ ] Test markers are properly configured
- [ ] Timeouts are appropriate for test complexity
- [ ] Error handling is comprehensive
- [ ] Artifacts are uploaded correctly
- [ ] Notifications work as expected

## 🎯 Success Metrics

A healthy CI/CD pipeline should have:
- ✅ **Success Rate**: >95% for main branch
- ⏱️ **Duration**: <15 minutes for full suite
- 📊 **Coverage**: >80% code coverage
- 🔒 **Security**: No critical vulnerabilities
- 📈 **Stability**: <5% flaky test rate

---

*This guide is updated based on actual pipeline issues and solutions. Keep it current as the project evolves.*