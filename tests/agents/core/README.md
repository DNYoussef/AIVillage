# Agent Core Test Suite

Comprehensive behavioral test suite for the AIVillage agent system following connascence principles and Test-Driven Development practices.

## Overview

This test suite validates agent behaviors and contracts rather than implementation details, ensuring tests remain stable during refactoring while providing confidence in system correctness.

## Test Organization

```
tests/agents/core/
├── behavioral/           # Contract tests for observable behaviors
│   └── test_agent_contracts.py
├── integration/          # Component interaction tests
│   └── test_component_interactions.py
├── properties/           # Property-based invariant tests
│   └── test_agent_invariants.py
├── performance/          # Performance validation tests
│   └── test_performance_validation.py
├── isolation/            # Test isolation validation
│   └── test_test_isolation.py
├── validation/           # Connascence compliance validation
│   └── test_connascence_compliance.py
├── fixtures/             # Reusable test data and utilities
│   ├── test_builders.py
│   └── conftest.py
└── README.md
```

## Testing Principles

### 1. Behavioral Testing

Tests focus on **what** agents do rather than **how** they do it:

```python
# ✅ Good: Tests behavior
async def test_agent_processes_tasks_successfully(agent):
    task = create_test_task(content="test")
    result = await agent.process_task(task)
    assert result["status"] == "success"

# ❌ Bad: Tests implementation
async def test_agent_calls_process_specialized_task(agent):
    assert hasattr(agent, '_process_specialized_task')
```

### 2. Connascence Management

Following connascence principles to minimize coupling:

**Static Connascence (weakest to strongest):**
- **Name (CoN)**: Use descriptive names
- **Type (CoT)**: Use type hints
- **Meaning (CoM)**: Avoid magic values
- **Position (CoP)**: Use keyword arguments
- **Algorithm (CoA)**: Extract shared logic

**Dynamic Connascence:**
- **Execution (CoE)**: Use context managers
- **Timing (CoTg)**: Avoid order dependencies
- **Value (CoV)**: Use dependency injection
- **Identity (CoI)**: Avoid global state

### 3. Test Data Builders

Use the Builder pattern for consistent test data:

```python
# ✅ Good: Using builders
task = (TaskInterfaceBuilder()
        .with_type("test")
        .with_content("test content")
        .with_priority(5)
        .build())

# ❌ Bad: Direct construction
task = TaskInterface("id", "test", "content", 5, None, {}, {}, datetime.now())
```

### 4. Property-Based Testing

Validate invariants across many inputs:

```python
@given(st.text(min_size=1, max_size=100))
async def test_agent_handles_all_text_inputs(agent, text_input):
    task = create_test_task(content=text_input)
    result = await agent.process_task(task)
    assert result is not None
    assert "status" in result
```

## Test Categories

### Behavioral Contracts (`behavioral/`)

Validates core agent behaviors:
- Lifecycle management (initialization, shutdown)
- Task processing contracts
- Message handling patterns
- Memory and reflection systems
- State consistency

### Component Integration (`integration/`)

Tests how components work together:
- Communication with state manager
- Task processing affects memory systems
- Configuration applies to all components
- MCP tools integration
- Error propagation and recovery

### System Properties (`properties/`)

Property-based tests for invariants:
- State transitions are deterministic
- Message ordering preserved
- Capabilities are idempotent
- Metrics are monotonic
- Agent ID immutability

### Performance Validation (`performance/`)

Ensures performance characteristics:
- Task processing latency bounds
- Memory usage constraints
- Throughput requirements
- Resource cleanup efficiency
- Performance regression detection

### Test Isolation (`isolation/`)

Validates test isolation:
- Agent instances are isolated
- External service mocks are isolated
- No shared mutable state
- Concurrent test safety
- Connascence violation prevention

### Connascence Compliance (`validation/`)

Meta-tests for test quality:
- No excessive positional parameters
- Minimal magic values
- No algorithm duplication
- Proper builder usage
- Interface abstraction compliance

## Running Tests

### All Tests
```bash
pytest tests/agents/core/
```

### By Category
```bash
pytest tests/agents/core/behavioral/     # Behavioral contracts
pytest tests/agents/core/integration/   # Integration tests
pytest tests/agents/core/properties/    # Property-based tests
pytest tests/agents/core/performance/   # Performance tests
pytest tests/agents/core/isolation/     # Isolation tests
pytest tests/agents/core/validation/    # Compliance tests
```

### By Markers
```bash
pytest -m behavioral     # Behavioral contract tests
pytest -m integration    # Integration tests
pytest -m property       # Property-based tests
pytest -m "not slow"     # Exclude slow tests
```

### With Coverage
```bash
pytest tests/agents/core/ --cov=packages.agents.core --cov-report=html
```

## Test Utilities

### Builders (`fixtures/test_builders.py`)

Reusable builders for test data:
- `AgentMetadataBuilder` - Create agent metadata
- `TaskInterfaceBuilder` - Create tasks
- `MessageInterfaceBuilder` - Create messages
- `QuietStarReflectionBuilder` - Create reflections
- `GeometricSelfStateBuilder` - Create state snapshots
- `TestScenarioBuilder` - Create complete scenarios

### Fixtures (`fixtures/conftest.py`)

Shared test fixtures:
- `mock_agent` - Initialized test agent
- `agent_factory` - Create multiple agents
- `performance_monitor` - Performance measurement
- `test_isolation` - Isolation utilities

### Quick Examples

```python
# Create a test agent
agent = await agent_factory("TestAgent")

# Create test data
task = priority_task(8).with_content("urgent task").build()
message = urgent_message().with_receiver(agent.agent_id).build()
reflection = learning_reflection().with_context("test learning").build()

# Create scenarios
scenario = create_test_scenario("multi_agent",
                               agent_count=3,
                               task_count=5)
```

## Best Practices

### 1. Test Naming
Use descriptive names that explain the behavior:

```python
# ✅ Good
async def test_agent_rejects_unsupported_task_types(agent):

# ❌ Bad
async def test_task_handling(agent):
```

### 2. Arrange-Act-Assert
Structure tests clearly:

```python
async def test_agent_records_task_completion_reflection(agent):
    # Given: Agent ready to process task
    task = create_test_task(content="reflection test")
    initial_reflections = len(agent.personal_journal)

    # When: Task is processed
    result = await agent.process_task(task)

    # Then: Reflection should be recorded
    assert result["status"] == "success"
    assert len(agent.personal_journal) > initial_reflections
```

### 3. Use Fixtures for Setup
Avoid repetitive setup code:

```python
@pytest.fixture
async def configured_agent(agent_factory):
    agent = await agent_factory("ConfiguredAgent")
    agent.adas_config["adaptation_rate"] = 0.2
    return agent

async def test_with_configured_agent(configured_agent):
    # Test uses pre-configured agent
    pass
```

### 4. Test Edge Cases
Include boundary conditions:

```python
@pytest.mark.parametrize("content", [
    "",  # Empty content
    "x" * 10000,  # Very large content
    None,  # None content
    "special\nchars\t\"'",  # Special characters
])
async def test_agent_handles_edge_case_content(agent, content):
    task = create_test_task(content=content)
    result = await agent.process_task(task)
    assert result is not None
```

### 5. Mock External Dependencies
Isolate units under test:

```python
async def test_agent_handles_rag_service_failure(agent):
    # Given: RAG service that fails
    agent.rag_client.query.side_effect = Exception("Service unavailable")

    # When: Agent attempts to use RAG
    result = await agent.query_group_memory("test query")

    # Then: Should handle gracefully
    assert result["status"] == "error"
    assert "unavailable" in result["message"]
```

## Quality Metrics

The test suite tracks several quality metrics:

- **Coverage**: >90% line coverage on core agent code
- **Performance**: All tests complete in <10 seconds
- **Isolation**: Zero shared state between tests
- **Connascence**: <5 violations per 100 functions
- **Maintainability**: Tests remain stable during refactoring

## Contributing

When adding new tests:

1. Choose the appropriate category
2. Use existing builders and fixtures
3. Follow naming conventions
4. Include docstrings for test classes
5. Add property-based tests for new invariants
6. Ensure tests are isolated and deterministic
7. Run connascence validation

## Troubleshooting

### Common Issues

**Tests fail intermittently:**
- Check for shared state between tests
- Verify proper cleanup in fixtures
- Look for timing dependencies

**Performance tests slow:**
- Use `pytest -m "not slow"` to skip
- Check for resource leaks
- Profile with `--profile-svg`

**Connascence violations:**
- Run `pytest tests/agents/core/validation/`
- Use builders instead of direct constructors
- Extract magic values to constants
- Use keyword arguments for complex calls

### Debugging

```bash
# Verbose output
pytest tests/agents/core/ -v

# Stop on first failure
pytest tests/agents/core/ -x

# Run specific test
pytest tests/agents/core/behavioral/test_agent_contracts.py::TestAgentContracts::test_agent_responds_to_initialization

# Profile performance
pytest tests/agents/core/performance/ --profile-svg
```
