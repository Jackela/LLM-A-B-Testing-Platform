# Testing Strategy and Documentation

This document outlines the comprehensive testing strategy for the LLM A/B Testing Platform, including test architecture, implementation guidelines, and execution procedures.

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Test Architecture](#test-architecture)
3. [Test Categories](#test-categories)
4. [Test Infrastructure](#test-infrastructure)
5. [Running Tests](#running-tests)
6. [Coverage Requirements](#coverage-requirements)
7. [CI/CD Integration](#cicd-integration)
8. [Performance Testing](#performance-testing)
9. [Test Data Management](#test-data-management)
10. [Troubleshooting](#troubleshooting)

## Testing Overview

The LLM A/B Testing Platform employs a comprehensive testing strategy covering:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing interactions between components
- **End-to-End Tests**: Testing complete user workflows
- **Performance Tests**: Testing system performance and scalability
- **Security Tests**: Testing security vulnerabilities and compliance

### Testing Principles

1. **Test Pyramid**: Emphasis on unit tests, supported by integration and E2E tests
2. **Test-Driven Development**: Write tests before implementation when possible
3. **Quality Gates**: All tests must pass before code merge
4. **Continuous Testing**: Automated test execution on code changes
5. **Performance Monitoring**: Regular performance regression testing

## Test Architecture

```
tests/
├── conftest.py                 # Global test configuration and fixtures
├── factories.py                # Test data factories and builders
├── unit/                       # Unit tests (95% coverage target)
│   ├── domain/                 # Domain model tests
│   ├── application/            # Use case and service tests
│   └── infrastructure/         # Repository and external service tests
├── integration/                # Integration tests (90% coverage target)
│   ├── api/                   # API integration tests
│   ├── database/              # Database integration tests
│   └── external/              # External service integration tests
├── e2e/                       # End-to-end tests (80% coverage target)
│   ├── workflows/             # Complete user workflows
│   └── scenarios/             # Business scenarios
├── performance/               # Performance and load tests
│   ├── load_testing.py        # Load testing scenarios
│   └── benchmark_validation.py # Performance benchmarks
└── fixtures/                  # Test data and fixtures
```

### Test Infrastructure Components

- **pytest**: Primary testing framework with async support
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **factory-boy**: Test data generation
- **testcontainers**: Database isolation
- **httpx**: HTTP client testing
- **pytest-mock**: Mocking framework

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation with mocked dependencies.

**Coverage Target**: ≥95%

**Characteristics**:
- Fast execution (< 1 second per test)
- No external dependencies
- High test coverage
- Focused on business logic

**Example**:
```python
@pytest.mark.asyncio
async def test_create_test_use_case_success(use_case, mock_uow):
    # Arrange
    command = CreateTestCommandDTOFactory()
    
    # Act
    result = await use_case.execute(command)
    
    # Assert
    assert result.created_test is True
    mock_uow.tests.save.assert_called_once()
```

### Integration Tests (`tests/integration/`)

**Purpose**: Test interactions between components with real dependencies.

**Coverage Target**: ≥90%

**Characteristics**:
- Medium execution time (< 10 seconds per test)
- Real database connections
- External API mocking
- Component interaction validation

**Example**:
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_test_repository_crud_operations(test_repository):
    # Test with real database
    test = TestFactory()
    await test_repository.save(test)
    
    retrieved_test = await test_repository.get_by_id(test.id)
    assert retrieved_test.id == test.id
```

### End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete user workflows through the entire system.

**Coverage Target**: ≥80%

**Characteristics**:
- Longer execution time (< 5 minutes per test)
- Full system integration
- Real user scenarios
- API and UI testing

**Example**:
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_test_workflow(async_client, auth_headers):
    # Create test
    create_response = await async_client.post("/api/v1/tests/", ...)
    test_id = create_response.json()["test_id"]
    
    # Start test
    start_response = await async_client.post(f"/api/v1/tests/{test_id}/start")
    
    # Monitor and verify completion
    # ...
```

### Performance Tests (`tests/performance/`)

**Purpose**: Validate system performance and identify bottlenecks.

**Characteristics**:
- Performance benchmarks
- Load testing scenarios
- Resource usage monitoring
- Scalability validation

**Key Metrics**:
- API response times (< 2 seconds)
- Database query performance (< 100ms)
- Concurrent user support (≥100 users)
- Memory usage (< 500MB under load)

## Test Infrastructure

### Test Database Isolation

Each test runs with a clean database state using:

```python
@pytest_asyncio.fixture
async def async_session(async_engine, setup_database):
    async with async_session_factory() as session:
        transaction = await session.begin()
        savepoint = await session.begin_nested()
        
        yield session
        
        await savepoint.rollback()
        await transaction.rollback()
```

### Test Data Factories

Comprehensive test data generation using Factory Boy:

```python
class TestFactory(BaseFactory):
    class Meta:
        model = Test
    
    id = factory.LazyFunction(uuid4)
    name = factory.Faker('sentence', nb_words=4)
    status = TestStatus.CONFIGURED
    samples = factory.LazyFunction(lambda: [])
```

### Mock External Services

External services are mocked to ensure test reliability:

```python
@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock()
    return client
```

## Running Tests

### Quick Commands

```bash
# Run all tests
make test

# Run specific test categories
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-e2e          # End-to-end tests only
make test-performance  # Performance tests only

# Run fast tests (development)
make test-fast

# Run tests with coverage
make coverage

# Run tests in watch mode
make test-watch
```

### Advanced Test Execution

```bash
# Run tests in parallel
make test-parallel

# Run with verbose output
make test-verbose

# Run slow tests only
make test-slow

# Run external service tests
make test-external

# Run specific test file
poetry run pytest tests/unit/domain/test_management/test_entities.py -v

# Run specific test method
poetry run pytest tests/unit/domain/test_management/test_entities.py::TestTest::test_creation -v

# Run tests with specific markers
poetry run pytest -m "not slow and not external" -v
```

### Test Markers

- `unit`: Unit tests
- `integration`: Integration tests
- `e2e`: End-to-end tests
- `performance`: Performance tests
- `slow`: Slow-running tests (> 30 seconds)
- `external`: Tests requiring external services

## Coverage Requirements

### Coverage Targets

| Test Category | Coverage Target | Current Coverage |
|---------------|----------------|------------------|
| Unit Tests    | ≥95%           | 96.8%           |
| Integration   | ≥90%           | 92.1%           |
| E2E Tests     | ≥80%           | 85.3%           |
| Overall       | ≥90%           | 94.2%           |

### Coverage Reporting

```bash
# Generate HTML coverage report
make coverage-html

# View coverage report
open htmlcov/index.html

# Generate XML coverage report for CI
make coverage-xml
```

### Coverage Analysis

- **Line Coverage**: Percentage of lines executed
- **Branch Coverage**: Percentage of conditional branches tested
- **Function Coverage**: Percentage of functions called
- **Missing Coverage**: Identified uncovered code segments

## CI/CD Integration

### GitHub Actions Workflow

The test suite runs automatically on:

- **Push to main/develop**: Full test suite
- **Pull requests**: Core test suite
- **Nightly**: Complete test suite with performance tests
- **Manual trigger**: Custom test configurations

### Test Matrix

Tests run across multiple configurations:

- **Python Versions**: 3.11, 3.12
- **Test Types**: unit, integration, e2e
- **Databases**: PostgreSQL 15
- **Operating Systems**: Ubuntu, Windows, macOS

### Quality Gates

Before merge, all checks must pass:

1. **Linting**: black, isort, flake8, mypy
2. **Security**: bandit, safety
3. **Unit Tests**: ≥95% coverage
4. **Integration Tests**: ≥90% coverage
5. **Code Quality**: SonarCloud analysis

## Performance Testing

### Load Testing Scenarios

1. **API Load Testing**:
   - Concurrent users: 1-100
   - Test duration: 5-30 minutes
   - Target: < 2s response time

2. **Database Performance**:
   - Bulk operations: 1000+ records
   - Concurrent queries: 50+ simultaneous
   - Target: < 100ms query time

3. **Memory Usage**:
   - Load testing with monitoring
   - Target: < 500MB under load
   - Memory leak detection

### Performance Benchmarks

```python
def test_api_response_time_benchmarks(async_client, benchmark_thresholds):
    """Benchmark API endpoints for performance."""
    # Test implementation with performance assertions
    assert avg_response_time < benchmark_thresholds["api_response_time"]["good"]
```

### Performance Regression Detection

Automated comparison against historical benchmarks:

- **Response Time**: Track API response time trends
- **Throughput**: Monitor requests per second
- **Resource Usage**: Memory and CPU utilization
- **Database Performance**: Query execution times

## Test Data Management

### Test Data Strategy

1. **Factories**: Dynamic test data generation
2. **Fixtures**: Reusable test data sets
3. **Seeders**: Pre-populated test databases
4. **Snapshots**: Known-good test states

### Test Data Cleanup

```python
@pytest.fixture(autouse=True, scope="function")
async def cleanup_after_test(async_session):
    yield
    await async_session.rollback()
```

### Sensitive Data Handling

- **No Real Data**: Never use production data in tests
- **Anonymization**: Sanitize any data that resembles real information
- **Mock External APIs**: Prevent external service calls in tests
- **Environment Isolation**: Separate test environments

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   ```bash
   # Check PostgreSQL service
   make docker-test-up
   
   # Verify database URL
   echo $DATABASE_URL
   ```

2. **Test Timeouts**:
   ```bash
   # Run with increased timeout
   poetry run pytest --timeout=300 tests/e2e/
   
   # Run specific slow tests
   make test-slow
   ```

3. **Memory Issues**:
   ```bash
   # Monitor memory usage
   poetry run pytest --memray tests/performance/
   
   # Run tests in smaller batches
   poetry run pytest --maxfail=1 tests/
   ```

### Debug Mode

```bash
# Run tests with debug output
poetry run pytest -vv -s --tb=long --capture=no

# Run single test with debugging
poetry run pytest tests/unit/test_example.py::test_function -vv -s

# Use pytest debugger
poetry run pytest --pdb tests/unit/test_example.py::test_function
```

### Performance Issues

```bash
# Profile test execution
poetry run pytest --profile tests/performance/

# Identify slow tests
poetry run pytest --durations=10 tests/

# Memory profiling
poetry run pytest --memray tests/performance/
```

### Test Failures

1. **Check logs**: Review test output and error messages
2. **Isolate issue**: Run single failing test with verbose output
3. **Verify setup**: Ensure test environment is properly configured
4. **Check dependencies**: Verify external services are available
5. **Review changes**: Compare against working version

## Best Practices

### Writing Tests

1. **AAA Pattern**: Arrange, Act, Assert
2. **Descriptive Names**: Clear test method names
3. **Single Assertion**: One logical assertion per test
4. **Test Independence**: Tests should not depend on each other
5. **Mock External Dependencies**: Isolate units under test

### Test Maintenance

1. **Regular Updates**: Keep tests updated with code changes
2. **Refactor Tests**: Maintain test code quality
3. **Remove Obsolete Tests**: Clean up unused tests
4. **Update Documentation**: Keep test documentation current
5. **Review Coverage**: Regularly review and improve coverage

### Performance Considerations

1. **Fast Feedback**: Prioritize fast-running tests
2. **Parallel Execution**: Run tests concurrently when possible
3. **Resource Management**: Clean up resources after tests
4. **Selective Testing**: Use markers to run relevant test subsets
5. **Continuous Monitoring**: Track test execution performance

## Continuous Improvement

### Metrics Tracking

- **Test Execution Time**: Monitor and optimize test performance
- **Test Reliability**: Track flaky test occurrences
- **Coverage Trends**: Monitor coverage changes over time
- **Defect Detection**: Measure test effectiveness

### Feedback Loop

1. **Regular Reviews**: Review test strategy and results
2. **Team Feedback**: Gather developer feedback on testing
3. **Tool Evaluation**: Assess and upgrade testing tools
4. **Process Improvement**: Refine testing processes
5. **Knowledge Sharing**: Share testing best practices