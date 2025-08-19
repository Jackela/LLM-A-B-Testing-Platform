# 🌊 Vibe Coding Specifications for AI Agents

## 🎯 Vibe Coding Philosophy

**Vibe Coding** = Specification-driven development that creates code optimized for AI agent collaboration, with maximum clarity, consistency, and extensibility.

### Core Principles
1. **AI-First Documentation**: Every specification written for AI comprehension
2. **Explicit Contracts**: Clear interfaces between components and agents  
3. **Test-Driven Specifications**: Specs include testable acceptance criteria
4. **Evolutionary Architecture**: Designed for continuous enhancement by AI agents
5. **Context-Rich Artifacts**: Maximum context preservation for future AI interactions

## 📋 Agent Task Templates

### Template 1: Domain Implementation Task
```yaml
# AGENT TASK: Domain Implementation
task_id: "domain_[context_name]"
agent_persona: "domain_architect" 
context: "[bounded_context_name]"

# SPECIFICATIONS
domain_focus: |
  Implement [domain_name] following DDD principles with complete TDD coverage.
  Focus on business logic, aggregate boundaries, and domain invariants.

# INPUT SPECIFICATIONS  
input_artifacts:
  - domain_model: "docs/DOMAIN_SPECIFICATIONS.md#[section]"
  - acceptance_criteria: "docs/ACCEPTANCE_CRITERIA.md#[section]" 
  - interface_contracts: "src/domain/[context]/interfaces/"

# OUTPUT SPECIFICATIONS
output_structure:
  primary_path: "src/domain/[context]/"
  structure: |
    entities/          # Aggregate roots and entities
    value_objects/     # Immutable value objects  
    repositories/      # Data access interfaces
    services/         # Domain services
    events/           # Domain events
    exceptions/       # Domain-specific exceptions
    
test_structure:
  test_path: "tests/unit/domain/[context]/"
  coverage_requirement: ">90%"
  test_types: ["unit", "integration", "property"]

# VALIDATION SPECIFICATIONS
quality_gates:
  - name: "Domain Model Completeness"
    check: "All entities have proper aggregate boundaries"
    validation: "pytest tests/architecture/test_domain_boundaries.py"
    
  - name: "Business Rules Coverage"
    check: "All business rules have corresponding tests"
    validation: "pytest tests/unit/domain/[context]/ --cov-fail-under=90"
    
  - name: "Interface Segregation"
    check: "Clean interfaces for dependent layers"
    validation: "mypy src/domain/[context]/ --strict"

# SUCCESS CRITERIA
deliverables:
  - "Complete domain model with all aggregates"
  - "Repository interfaces for data access"
  - "Domain services for complex business logic"  
  - "Comprehensive test suite with >90% coverage"
  - "Clear interface contracts for application layer"
```

### Template 2: Application Service Task
```yaml
# AGENT TASK: Application Service Implementation
task_id: "application_[service_name]"
agent_persona: "application_architect"
context: "[use_case_context]"

# SPECIFICATIONS
service_focus: |
  Implement application services that orchestrate domain objects to fulfill use cases.
  Focus on transaction boundaries, error handling, and external integration points.

# INPUT SPECIFICATIONS
dependencies:
  - domain_models: "src/domain/[context]/"
  - interface_definitions: "Domain repository and service interfaces"
  - use_case_specifications: "docs/USE_CASES.md#[section]"

# OUTPUT SPECIFICATIONS  
output_structure:
  primary_path: "src/application/services/[context]/"
  structure: |
    use_cases/         # Use case implementations
    services/          # Application services
    handlers/          # Command/query handlers
    dto/              # Data transfer objects
    mappers/          # Domain-DTO mapping
    
integration_points:
  - "Domain repositories (dependency injection)"
  - "External APIs (through adapters)"
  - "Message queue (async operations)"
  - "Transaction management (database)"

# VALIDATION SPECIFICATIONS
quality_gates:
  - name: "Use Case Coverage"
    check: "All specified use cases implemented"
    validation: "pytest tests/integration/use_cases/ -v"
    
  - name: "Transaction Integrity"  
    check: "Proper transaction boundaries and rollback"
    validation: "pytest tests/integration/transactions/"
    
  - name: "Error Handling"
    check: "Graceful error handling and user feedback"
    validation: "pytest tests/integration/error_scenarios/"
```

### Template 3: Infrastructure Implementation Task
```yaml
# AGENT TASK: Infrastructure Implementation  
task_id: "infrastructure_[component_name]"
agent_persona: "infrastructure_architect"
context: "[infrastructure_context]"

# SPECIFICATIONS
infrastructure_focus: |
  Implement infrastructure concerns following clean architecture principles.
  Focus on external integrations, data persistence, and cross-cutting concerns.

# OUTPUT SPECIFICATIONS
output_structure:
  primary_path: "src/infrastructure/[component]/"
  structure: |
    adapters/          # External system adapters
    repositories/      # Concrete repository implementations  
    clients/          # API clients and connectors
    config/           # Configuration management
    monitoring/       # Logging, metrics, health checks
    
# VALIDATION SPECIFICATIONS
quality_gates:
  - name: "Integration Reliability"
    check: "Robust error handling and retry logic"
    validation: "pytest tests/integration/external_apis/"
    
  - name: "Performance Requirements"
    check: "Meets specified SLAs"
    validation: "pytest tests/performance/ --benchmark"
    
  - name: "Configuration Validation" 
    check: "Type-safe configuration with defaults"
    validation: "pytest tests/integration/config/"
```

## 🔧 AI Agent Communication Protocol

### Agent Handoff Specification
```yaml
# HANDOFF PROTOCOL
handoff_format:
  from_agent: "[previous_agent_id]"
  to_agent: "[next_agent_id]"
  
  completion_artifacts:
    - path: "[output_path]"
      type: "[code|tests|docs|config]"
      status: "VALIDATED"
      interface_contracts: "[exported_interfaces]"
      
  validation_results:
    quality_gates: "[all_passed]"
    test_coverage: "[percentage]"
    performance_metrics: "[if_applicable]"
    
  next_agent_context:
    available_interfaces: "[interface_list]"
    integration_points: "[integration_specifications]"
    constraints: "[any_limitations]"

# DEPENDENCY RESOLUTION
dependency_matrix:
  agent_id: "[current_agent]"
  requires: ["[list_of_prerequisite_agents]"]
  provides: ["[list_of_interface_exports]"] 
  blocks: ["[list_of_dependent_agents]"]
```

### Code Style & Standards

```yaml
# CODE STANDARDS FOR AI AGENTS
code_style:
  formatting: "black --line-length 100"
  import_sorting: "isort --profile black"
  type_checking: "mypy --strict"
  linting: "flake8 --max-line-length 100"
  
  docstring_style: "Google style docstrings"
  naming_conventions:
    classes: "PascalCase"
    functions: "snake_case"
    constants: "UPPER_SNAKE_CASE"
    files: "snake_case"
    
  async_patterns: "Use async/await for I/O operations"
  error_handling: "Explicit exception types, no bare except"
  logging: "Structured logging with context"

# TESTING STANDARDS
testing_standards:
  test_naming: "test_[what_is_being_tested]_[expected_outcome]"
  test_structure: "Arrange-Act-Assert (AAA) pattern" 
  mock_usage: "Mock external dependencies, not domain logic"
  fixture_usage: "pytest fixtures for test data setup"
  
  coverage_targets:
    domain: ">90%"
    application: ">85%" 
    infrastructure: ">80%"
    presentation: ">75%"
```

## 📊 AI Agent Success Metrics

### Quality Metrics per Agent
```yaml
domain_agents:
  business_logic_coverage: ">95%"
  invariant_enforcement: "100%"
  aggregate_boundary_clarity: "Architectural tests pass"
  
application_agents:
  use_case_completion: "100%"
  integration_reliability: ">99%"
  transaction_integrity: "100%"
  
infrastructure_agents:
  external_integration_reliability: ">99.5%"
  performance_sla_compliance: "100%"
  configuration_validation: "100%"
  
presentation_agents:
  api_documentation_completeness: "100%"
  user_experience_validation: "Manual testing checklist"
  accessibility_compliance: "WCAG 2.1 AA"
```

### Inter-Agent Coordination Metrics
```yaml
coordination_metrics:
  interface_compatibility: "100% - No breaking changes"
  dependency_resolution_time: "<5 minutes per handoff"
  integration_test_pass_rate: ">99%"
  cross_boundary_transaction_integrity: "100%"
```

## 🚀 Agent Execution Environment

### Required Tools & Environment
```yaml
agent_environment:
  python_version: "3.11+"
  package_manager: "poetry"
  testing_framework: "pytest + factory_boy"
  type_checker: "mypy" 
  code_formatter: "black + isort"
  
  development_database: "SQLite (local)"
  test_database: "In-memory SQLite"
  message_queue: "Redis (containerized)"
  
  external_apis:
    openai: "OPENAI_API_KEY environment variable"
    anthropic: "ANTHROPIC_API_KEY environment variable"  
    gemini: "GEMINI_API_KEY environment variable"

# VALIDATION ENVIRONMENT  
validation_tools:
  architecture_tests: "pytest + custom architecture validators"
  contract_tests: "pytest + httpx for API contracts"
  performance_tests: "pytest-benchmark"
  security_tests: "bandit + safety"
```

### Agent Context Sharing
```yaml
context_sharing:
  project_state: "Git repository with all previous agent outputs"
  documentation: "All specs, designs, and decisions in docs/"
  test_results: "Historical test results for regression detection"
  performance_baselines: "Benchmark results for performance validation"
  
shared_artifacts:
  domain_interfaces: "src/domain/*/interfaces/"
  application_contracts: "src/application/contracts/"
  infrastructure_adapters: "src/infrastructure/adapters/"
  test_fixtures: "tests/fixtures/"
```

This vibe coding specification ensures AI agents can work efficiently in parallel while maintaining code quality, architectural integrity, and seamless integration across the entire LLM A/B Testing Platform.