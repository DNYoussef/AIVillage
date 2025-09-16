# Quiet-STaR Test Suite

Comprehensive test suite for the Quiet-STaR (Self-Taught Reasoning) implementation, achieving >85% code coverage with robust validation across all components.

## Overview

This test suite implements comprehensive testing for the Quiet-STaR reasoning enhancement algorithms based on "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" by Zelikman et al. (2024).

## Test Structure

### 🧪 Test Categories

#### Unit Tests
- **ThoughtGenerator**: Token-wise parallel sampling, nucleus sampling, coherence filtering
- **CoherenceScorer**: Multi-criteria scoring (semantic, syntactic, predictive utility)
- **MixingHead**: Neural mixing network, gradient flow, weight conservation
- **ThoughtInjector**: Injection point identification, difficulty scoring, attention analysis
- **OptimizationStrategies**: Curriculum learning, regularization, adaptive sampling

#### Integration Tests
- **End-to-End Pipeline**: Complete thought generation and mixing workflow
- **Component Interaction**: Inter-component data flow and compatibility
- **Training Simulation**: Multi-epoch training with optimization strategies

#### Performance Tests
- **Speed Benchmarks**: Thought generation latency and throughput
- **Memory Efficiency**: Memory usage scaling with batch size and sequence length
- **Reasoning Improvement**: Metrics for measuring reasoning enhancement

#### Property-Based Tests
- **Algorithm Invariants**: Mathematical properties that must hold
- **Coherence Score Bounds**: Scores always in [0, 1] range
- **Mixing Weight Conservation**: Weights sum to 1.0
- **Thought Length Consistency**: Generated thoughts match configured length

#### Contract Tests
- **Input Validation**: Proper handling of edge cases and invalid inputs
- **Output Guarantees**: Shape, type, and range validation
- **Gradient Flow**: Proper backpropagation through all components

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m performance    # Performance benchmarks
pytest -m property       # Property-based tests
pytest -m contract       # Contract tests
```

### Advanced Usage

```bash
# Run with detailed coverage report
pytest --cov-report=html --cov-report=term-missing

# Run performance benchmarks
pytest -m performance --benchmark-only

# Run in parallel (faster)
pytest -n auto

# Run with verbose output
pytest -v --tb=long

# Skip slow tests
pytest --fast

# GPU tests only (if CUDA available)
pytest -m gpu

# Generate test report
pytest --html=report.html --self-contained-html
```

## Test Configuration

### Files
- `pytest.ini`: Main pytest configuration
- `.coveragerc`: Coverage measurement settings
- `conftest.py`: Shared fixtures and utilities
- `requirements-test.txt`: Test dependencies

### Coverage Requirements
- **Minimum Coverage**: 85%
- **Branch Coverage**: Enabled
- **Missing Lines**: Reported
- **Exclusions**: Mock models, demo code, abstract methods

## Test Data and Fixtures

### Shared Fixtures
- `test_config`: Standard Quiet-STaR configuration
- `mock_language_model`: Full mock language model
- `fast_mock_model`: Lightweight model for performance tests
- `sample_input_data`: Predefined test inputs
- `performance_monitor`: Memory and timing utilities

### Test Data Generation
- **Input Patterns**: Repetitive, diverse, boundary cases
- **Logit Distributions**: Normal, uniform, peaked
- **Attention Patterns**: Focused, dispersed, random

## Performance Benchmarks

### Target Metrics
- **Thought Generation**: <2s for batch_size=4, seq_len=20
- **Memory Usage**: <1GB growth per test run
- **Reasoning Improvement**: Measurable entropy reduction

### Benchmark Categories
1. **Latency Tests**: Single forward pass timing
2. **Throughput Tests**: Batch processing efficiency
3. **Memory Tests**: Peak memory usage tracking
4. **Scaling Tests**: Performance vs. input size

## Quality Gates

### Automated Checks
✅ All tests pass
✅ Coverage >85%
✅ No critical performance regressions
✅ Property invariants hold
✅ Contract specifications met

### Manual Validation
- Algorithm correctness review
- Performance characteristic analysis
- Edge case handling verification

## Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Quiet-STaR Tests
  run: |
    pip install -r requirements-test.txt
    pytest --cov-fail-under=85 --junit-xml=test-results.xml
```

### Coverage Reporting
- HTML reports in `htmlcov/`
- XML reports for CI integration
- Terminal summary with missing lines

## Troubleshooting

### Common Issues

#### CUDA Tests Failing
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Skip GPU tests if needed
pytest -m "not gpu"
```

#### Memory Issues
```bash
# Run with smaller batch sizes
pytest --tb=short -x  # Stop on first failure

# Monitor memory usage
pytest --profile-mem
```

#### Slow Tests
```bash
# Skip slow tests
pytest --fast

# Run specific test file
pytest test_quietstar.py::TestThoughtGenerator
```

### Debug Mode
```bash
# Run with Python debugger
pytest --pdb

# Verbose output with timing
pytest -v --durations=10

# Show all output (including prints)
pytest -s
```

## Test Development

### Adding New Tests

1. **Choose Category**: Unit, integration, performance, property, or contract
2. **Add Marker**: Use appropriate pytest marker
3. **Use Fixtures**: Leverage existing fixtures for consistency
4. **Follow Naming**: `test_<component>_<behavior>`

### Example Test Structure
```python
@pytest.mark.unit
def test_thought_generator_shape(thought_generator, mock_model, sample_input_data):
    """Test ThoughtGenerator output shapes."""
    result = thought_generator.generate_thoughts(
        sample_input_data['medium']['input_ids'],
        sample_input_data['medium']['attention_mask'],
        mock_model,
        sample_input_data['medium']['position']
    )

    assert result['thoughts'].shape[0] == 2  # batch_size
    assert result['thoughts'].shape[1] == 8  # num_thoughts
    assert result['thoughts'].shape[2] == 4  # thought_length
```

### Performance Test Guidelines
- Use `performance_monitor` fixture
- Set realistic thresholds
- Test multiple scenarios
- Include memory profiling

## Integration with SPEK Platform

This test suite integrates with the broader SPEK Enhanced Development Platform:

- **Quality Gates**: Enforces 85% coverage requirement
- **NASA POT10 Compliance**: Rigorous testing standards
- **Defense Industry Ready**: Comprehensive validation
- **CI/CD Integration**: Automated quality checks

## Documentation

- Algorithm details: `../../../../../src/quiet_star/algorithms.py`
- SPEK methodology: `../../../../../docs/S-R-P-E-K-METHODOLOGY.md`
- Quality standards: `../../../../../docs/NASA-POT10-COMPLIANCE-STRATEGIES.md`

## Support

For issues with the test suite:
1. Check this README for common solutions
2. Review test output for specific error details
3. Run individual test files to isolate issues
4. Use debug mode for detailed investigation

---

**Remember**: Tests are the safety net that enables confident development. Every component change should be validated through this comprehensive test suite.