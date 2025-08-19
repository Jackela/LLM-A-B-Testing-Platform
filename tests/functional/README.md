# Functional Tests

This directory contains **functional tests** that validate complete user workflows and business scenarios across the LLM A/B Testing Platform.

## 🎯 Purpose

Functional tests ensure that:
- Complete user workflows work end-to-end
- Business requirements are met
- Integration between components functions correctly
- Real-world scenarios are validated

## 📁 Test Organization

```
tests/functional/
├── README.md                           # This file
├── test_complete_arc_easy_dataset.py   # ARC Easy dataset processing
├── test_complete_dataset_evaluation.py # Full dataset evaluation workflows
├── test_complete_dataset_quick.py      # Quick dataset validation
├── test_complete_training_set.py       # Training set processing
├── test_dual_model_training_set.py     # Multi-model training scenarios
├── test_final_functional.py            # Final validation tests
├── test_functional_requirements.py     # Business requirement validation
├── test_minimal_functional.py          # Core functionality tests
├── test_multi_model_comparison.py      # Model comparison workflows
├── test_real_api_integration.py        # Live API integration tests
├── test_real_dataset_evaluation.py     # Real dataset processing
└── test_real_multi_model_comparison.py # Production-like comparisons
```

## 🧪 Test Categories

### Dataset Processing Tests
- **test_complete_arc_easy_dataset.py**: Validates ARC Easy dataset handling
- **test_complete_dataset_evaluation.py**: Full dataset evaluation workflows
- **test_real_dataset_evaluation.py**: Real-world dataset processing

### Model Comparison Tests  
- **test_dual_model_training_set.py**: Two-model comparison scenarios
- **test_multi_model_comparison.py**: Multiple model evaluations
- **test_real_multi_model_comparison.py**: Production comparison workflows

### Integration Tests
- **test_real_api_integration.py**: Live API connectivity and responses
- **test_functional_requirements.py**: Business requirement validation

### Validation Tests
- **test_final_functional.py**: Pre-release validation suite
- **test_minimal_functional.py**: Core functionality verification

## 🚀 Running Functional Tests

### All Functional Tests
```bash
# Run all functional tests
cd tests/functional
python -m pytest . -v

# Run with coverage
python -m pytest . -v --cov=src --cov-report=html
```

### Specific Test Categories
```bash
# Dataset processing tests
python -m pytest test_*dataset* -v

# Model comparison tests  
python -m pytest test_*model* -v

# Real API integration tests
python -m pytest test_real_* -v
```

### Individual Test Files
```bash
# Run specific test file
python -m pytest test_minimal_functional.py -v

# Run with detailed output
python -m pytest test_complete_dataset_evaluation.py -v -s
```

## ⚙️ Test Configuration

### Environment Requirements
```bash
# Required environment variables
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/test_db"
export REDIS_URL="redis://localhost:6379/0"
export TESTING=true

# Optional API keys for real integration tests
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Test Data Dependencies
- **Dataset files**: Tests may require specific dataset files in `data/processed/`
- **Sample data**: `test_sample.json` for quick validation
- **Mock responses**: Predefined responses for offline testing

## 📊 Test Execution Strategy

### Development Phase
- Run **minimal functional tests** for quick validation
- Use **mock APIs** to avoid external dependencies
- Focus on **core workflows**

### Pre-commit Phase
- Run **functional requirement tests**
- Validate **critical user paths**
- Ensure **backwards compatibility**

### Pre-release Phase
- Run **complete test suite** including real API tests
- Validate **all dataset processing** scenarios
- Test **production-like configurations**

## 🔧 Test Infrastructure

### Mock Services
Tests use mocked external services by default:
```python
# Automatic mocking for offline testing
@pytest.fixture(autouse=True)
def mock_llm_providers():
    """Mock LLM provider responses for consistent testing."""
    with patch('src.application.services.model_provider') as mock:
        yield mock
```

### Real Service Integration
Some tests require real services:
```python
# Real API integration tests
@pytest.mark.integration
@pytest.mark.requires_api_keys
def test_real_openai_integration():
    """Test against real OpenAI API."""
    # Requires actual API key
```

### Test Data Management
```python
# Test data fixtures
@pytest.fixture
def sample_dataset():
    """Load sample dataset for testing."""
    return load_test_data("test_sample.json")

@pytest.fixture  
def large_dataset():
    """Load large dataset for performance testing."""
    return load_test_data("arc_easy_subset.json")
```

## 📈 Test Maintenance

### Regular Reviews
- **Monthly**: Review test execution time and flakiness
- **Quarterly**: Validate test coverage of business requirements
- **Releases**: Update tests for new functionality

### Performance Monitoring
- Track test execution time trends
- Identify and optimize slow tests
- Monitor test reliability and failure rates

### Documentation Updates
- Keep test descriptions current with functionality
- Update environment setup instructions
- Maintain clear test categorization

## 🎯 Best Practices

### Test Design
- **Clear test names** that describe the scenario
- **Independent tests** that can run in any order
- **Appropriate mocking** to avoid external dependencies
- **Realistic test data** that reflects production scenarios

### Error Handling
- **Comprehensive assertions** with clear error messages
- **Graceful handling** of external service failures
- **Timeout management** for long-running operations
- **Resource cleanup** after test completion

### Maintainability
- **Shared fixtures** for common test setup
- **Helper functions** for repeated operations
- **Clear test organization** by functionality
- **Regular refactoring** to reduce duplication

---

## 📞 Support

For questions about functional tests:
1. Check test documentation and comments
2. Review related integration tests in `tests/integration/`
3. Consult team members familiar with specific workflows
4. Update documentation when adding new test scenarios

*Keep functional tests aligned with business requirements and user workflows.*