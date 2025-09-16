# Contributing to BitNet Phase 4

Welcome to BitNet Phase 4! We're excited to have you contribute to this revolutionary 1-bit neural network optimization platform. This guide will help you get started with contributing effectively.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [NASA POT10 Compliance](#nasa-pot10-compliance)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## Code of Conduct

### Our Commitment

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to adhere to our standards of conduct:

- **Be respectful**: Treat all community members with respect and professionalism
- **Be inclusive**: Welcome newcomers and diverse perspectives
- **Be collaborative**: Work together constructively and support each other
- **Be professional**: Maintain high standards in all interactions

### Enforcement

Violations of our code of conduct should be reported to conduct@agentforge.dev. All reports will be handled confidentially and appropriately.

## Getting Started

### Prerequisites

Before contributing, ensure you have:

```bash
# Required software
python >= 3.8
pytorch >= 2.0
cuda >= 11.8 (for GPU development)
git >= 2.30

# Development tools
pytest >= 7.0
black >= 23.0
flake8 >= 6.0
mypy >= 1.0
```

### Initial Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/bitnet-phase4.git
   cd bitnet-phase4
   ```

2. **Set Up Development Environment**
   ```bash
   # Create conda environment
   conda create -n bitnet-dev python=3.10
   conda activate bitnet-dev

   # Install PyTorch with CUDA
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

   # Install development dependencies
   pip install -e ".[dev]"
   ```

3. **Verify Setup**
   ```bash
   # Run tests
   pytest tests/ -v

   # Check code quality
   black --check src/ tests/
   flake8 src/ tests/
   mypy src/

   # Verify BitNet import
   python -c "from src.ml.bitnet import create_bitnet_model; print('Setup successful!')"
   ```

### Development Environment Configuration

Create a `.env` file for development:

```bash
# Copy template
cp .env.example .env.dev

# Configure for development
BITNET_DEV_MODE=true
BITNET_LOG_LEVEL=DEBUG
BITNET_ENABLE_PROFILING=true
CUDA_VISIBLE_DEVICES=0  # Adjust for your setup
```

## Development Workflow

### Branch Strategy

We use **GitFlow** with feature branches:

```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### Branch Naming Convention

- **Features**: `feature/short-description`
- **Bug fixes**: `fix/bug-description`
- **Documentation**: `docs/update-description`
- **Performance**: `perf/optimization-description`
- **Refactor**: `refactor/component-description`

### Commit Message Standards

We follow **Conventional Commits** specification:

```bash
# Format: type(scope): description
feat(optimization): add custom CUDA kernels for 1-bit operations
fix(validation): resolve memory leak in performance profiler
docs(api): add comprehensive OpenAPI documentation
perf(memory): optimize tensor allocation for 8x reduction
test(integration): add Phase 3 integration test suite
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `perf`: Performance improvements
- `test`: Testing additions/changes
- `refactor`: Code restructuring
- `style`: Formatting changes
- `chore`: Build/tooling changes

## Coding Standards

### Python Code Style

We enforce strict coding standards for NASA POT10 compliance:

#### 1. Code Formatting

```python
# Use Black for consistent formatting
black src/ tests/

# Configuration in pyproject.toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
```

#### 2. Type Annotations

**All functions must have type annotations:**

```python
from typing import Dict, List, Optional, Tuple, Union
import torch

def optimize_bitnet_model(
    model: torch.nn.Module,
    optimization_level: str = "production"
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """
    Optimize BitNet model with comprehensive validation.

    Args:
        model: PyTorch model to optimize
        optimization_level: One of 'development', 'balanced', 'production'

    Returns:
        Tuple of (optimized_model, optimization_statistics)

    Raises:
        ValueError: If optimization_level is invalid
    """
    # Implementation with type safety
    stats: Dict[str, float] = {}
    return model, stats
```

#### 3. NASA POT10 Compliance Requirements

**Function Length Limit:**
```python
# ✅ COMPLIANT: Functions ≤60 lines
def compliant_function() -> bool:
    """Short, focused function."""
    # Implementation within 60 lines
    return True

# ❌ NON-COMPLIANT: Functions >60 lines
def overly_long_function():
    # This would violate NASA POT10 Rule 4
    pass
```

**Assertion Requirements:**
```python
# ✅ COMPLIANT: ≥2 assertions per function
def validate_input(tensor: torch.Tensor) -> torch.Tensor:
    """Validate input tensor with required assertions."""
    # Assertion 1: Type validation
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"

    # Assertion 2: Shape validation
    assert len(tensor.shape) >= 2, f"Expected ≥2D tensor, got {len(tensor.shape)}D"

    # Assertion 3: Value range validation
    assert tensor.numel() > 0, "Tensor cannot be empty"

    return tensor
```

**Loop Bounds:**
```python
# ✅ COMPLIANT: Fixed loop bounds
def process_layers(model: torch.nn.Module) -> None:
    """Process model layers with bounded iteration."""
    max_layers = 50  # Fixed upper bound

    for i, layer in enumerate(model.modules()):
        if i >= max_layers:  # Safety bound
            break
        # Process layer...
```

#### 4. Error Handling

```python
def robust_function(input_data: Any) -> Optional[torch.Tensor]:
    """Example of proper error handling."""
    try:
        # Validate inputs
        if input_data is None:
            logger.warning("Input data is None, returning empty result")
            return None

        # Process with assertions
        assert isinstance(input_data, (torch.Tensor, np.ndarray)), \
               f"Invalid input type: {type(input_data)}"

        # Main logic
        result = process_input(input_data)

        # Validate output
        assert result is not None, "Processing failed to produce result"
        return result

    except AssertionError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None
```

### Documentation Standards

#### 1. Docstring Format

```python
def create_bitnet_model(config_dict: Optional[Dict[str, Any]] = None) -> BitNetModel:
    """
    Create BitNet model with specified configuration.

    Creates a 1-bit quantized neural network model with comprehensive optimization
    capabilities, achieving 8x memory reduction and 2-4x inference speedup while
    maintaining <10% accuracy degradation.

    Args:
        config_dict: Optional configuration overrides. If None, uses default
                    base configuration with production optimization profile.
                    Keys can include:
                    - model_size: One of 'tiny', 'small', 'base', 'large', 'xlarge'
                    - optimization_profile: One of 'development', 'production', 'inference'
                    - compliance_level: One of 'standard', 'enhanced', 'defense_grade'

    Returns:
        BitNetModel: Initialized model with specified configuration. The model
                    includes quantized linear layers, optimized attention mechanism,
                    and NASA POT10 compliance features.

    Raises:
        ValueError: If config_dict contains invalid configuration parameters
        RuntimeError: If model initialization fails due to insufficient resources

    Examples:
        Create basic model:
        >>> model = create_bitnet_model()
        >>> stats = model.get_model_stats()
        >>> print(f"Parameters: {stats['total_parameters_millions']:.1f}M")

        Create large production model:
        >>> config = {'model_size': 'large', 'optimization_profile': 'production'}
        >>> model = create_bitnet_model(config)
        >>> memory_info = model.get_memory_footprint()
        >>> print(f"Memory: {memory_info['model_memory_mb']:.1f} MB")

    Note:
        This function is the primary entry point for BitNet model creation.
        For advanced configuration, use BitNetConfig class directly.

    See Also:
        BitNetConfig: Comprehensive configuration management
        optimize_bitnet_model: Apply optimization strategies
        validate_bitnet_performance: Performance validation
    """
```

#### 2. Code Comments

```python
class BitNetLinear(nn.Module):
    """BitNet Linear Layer with 1-bit weight quantization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with 1-bit quantization and scaling."""

        # Step 1: Apply straight-through estimator for quantization
        # This enables gradient flow while quantizing to {-1, +1}
        weight_quantized = StraightThroughEstimator.apply(self.weight)

        # Step 2: Compute scaling factor from original weights
        # Uses L1 norm as per BitNet paper for optimal scaling
        alpha = torch.mean(torch.abs(self.weight))
        self.alpha = alpha.detach()  # Store for monitoring

        # Step 3: Scaled binary linear transformation
        # Formula: Y = alpha * (X @ W_q) where W_q ∈ {-1, +1}
        output = F.linear(x, weight_quantized, self.bias) * alpha

        # Step 4: NASA compliance monitoring
        if self.training:
            # Track quantization error for compliance validation
            error = torch.mean(torch.abs(self.weight - weight_quantized))
            self.quantization_error = error.detach()

        return output
```

## Testing Guidelines

### Test Structure

We maintain comprehensive test coverage (target: ≥92%):

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_bitnet_model.py
│   ├── test_optimization.py
│   └── test_validation.py
├── integration/            # Integration tests across components
│   ├── test_phase_integration.py
│   ├── test_end_to_end.py
│   └── test_api_endpoints.py
├── performance/            # Performance and benchmark tests
│   ├── test_memory_benchmarks.py
│   ├── test_speed_benchmarks.py
│   └── test_accuracy_validation.py
├── compliance/             # NASA POT10 compliance tests
│   ├── test_nasa_compliance.py
│   ├── test_security_validation.py
│   └── test_audit_trails.py
└── fixtures/               # Test data and fixtures
    ├── model_configs.py
    ├── test_data.py
    └── benchmark_baselines.py
```

### Writing Tests

#### 1. Unit Tests

```python
import pytest
import torch
from src.ml.bitnet import BitNetModel, BitNetConfig, ModelSize

class TestBitNetModel:
    """Test suite for BitNet model functionality."""

    @pytest.fixture
    def base_config(self) -> BitNetConfig:
        """Standard configuration for testing."""
        return BitNetConfig(
            model_size=ModelSize.SMALL,  # Use small for faster tests
            optimization_profile="development"
        )

    @pytest.fixture
    def test_model(self, base_config: BitNetConfig) -> BitNetModel:
        """Test model instance."""
        return BitNetModel(base_config)

    def test_model_creation(self, test_model: BitNetModel):
        """Test basic model creation and properties."""
        # NASA Compliance: Multiple assertions required
        assert test_model is not None, "Model creation failed"
        assert hasattr(test_model, 'forward'), "Model missing forward method"
        assert len(list(test_model.parameters())) > 0, "Model has no parameters"

        # Verify architecture
        stats = test_model.get_model_stats()
        assert stats['total_parameters_millions'] > 0, "Invalid parameter count"
        assert stats['quantized_parameters_millions'] > 0, "No quantized parameters"

    def test_forward_pass(self, test_model: BitNetModel):
        """Test forward pass functionality."""
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Test forward pass
        with torch.no_grad():
            outputs = test_model(input_ids)

        # NASA Compliance: Comprehensive validation
        assert 'logits' in outputs, "Missing logits in output"
        assert outputs['logits'].shape == (batch_size, seq_len, test_model.config.architecture.vocab_size), \
               f"Incorrect output shape: {outputs['logits'].shape}"
        assert not torch.isnan(outputs['logits']).any(), "NaN values in output"
        assert torch.isfinite(outputs['logits']).all(), "Non-finite values in output"

    @pytest.mark.performance
    def test_memory_usage(self, test_model: BitNetModel):
        """Test memory usage is within expected bounds."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            test_model = test_model.to(device)

            # Measure memory usage
            torch.cuda.reset_peak_memory_stats()
            input_ids = torch.randint(0, 1000, (4, 128)).to(device)

            with torch.no_grad():
                _ = test_model(input_ids)

            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

            # Verify memory efficiency (should be <100MB for small model)
            assert peak_memory < 100, f"Memory usage too high: {peak_memory:.1f} MB"
```

#### 2. Integration Tests

```python
class TestPhaseIntegration:
    """Test integration with other Agent Forge phases."""

    def test_phase2_evomerge_integration(self):
        """Test integration with Phase 2 EvoMerge."""
        # Create BitNet model with EvoMerge integration
        config = BitNetConfig()
        config.phase_integration.evomerge_integration = True

        model = BitNetModel(config)

        # Simulate EvoMerge integration
        # (In real scenario, this would load actual EvoMerge weights)
        original_weights = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                original_weights[name] = param.data.clone()

        # Verify integration capability
        assert config.phase_integration.evomerge_integration, "EvoMerge integration not enabled"
        assert len(original_weights) > 0, "No weights found for integration"

    def test_phase3_quietstar_integration(self):
        """Test integration with Phase 3 Quiet-STaR."""
        config = BitNetConfig()
        config.phase_integration.quiet_star_integration = True

        model = BitNetModel(config)

        # Test thought vector integration
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        thought_vectors = torch.randn(batch_size, seq_len, config.architecture.hidden_size)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, thought_vectors=thought_vectors)

        # Verify thought integration
        assert 'logits' in outputs, "Missing logits with thought integration"
        assert outputs['logits'].shape[0] == batch_size, "Incorrect batch size in output"
```

#### 3. Performance Tests

```python
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_memory_reduction_target(self):
        """Test that 8x memory reduction target is achieved."""
        from src.ml.bitnet import validate_bitnet_performance

        # Create optimized model
        model = create_bitnet_model({'model_size': 'base', 'optimization_profile': 'production'})
        test_inputs = [(torch.randint(0, 50000, (4, 128)),) for _ in range(5)]

        # Run validation
        results = validate_bitnet_performance(model, test_inputs, create_baseline=True)

        # Verify targets
        memory_reduction = results['memory_validation']['memory_reduction_factor']
        assert memory_reduction >= 8.0, f"Memory reduction {memory_reduction:.1f}x below 8x target"

    def test_inference_speed_target(self):
        """Test that 2-4x speedup target is achieved."""
        from src.ml.bitnet.profiling import create_speed_profiler

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_bitnet_model({'optimization_profile': 'inference'})
        model = model.to(device)

        # Create profiler and test
        profiler = create_speed_profiler(device, "comprehensive")

        def input_generator(batch_size=1):
            return (torch.randint(0, 50000, (batch_size, 128)).to(device),)

        results = profiler.comprehensive_speed_analysis(model, input_generator, "test_model")
        speedup = results["speed_validation"]["speedup_ratio"]

        assert speedup >= 2.0, f"Speedup {speedup:.1f}x below 2x minimum target"
        # Note: 4x target is optimal, 2x is minimum for passing
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests only
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ -v --benchmark # Performance tests
pytest tests/compliance/ -v              # NASA compliance tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only

# Run specific test files
pytest tests/unit/test_bitnet_model.py -v
pytest tests/integration/test_phase_integration.py -v
```

## NASA POT10 Compliance

### Compliance Requirements

All code contributions must meet NASA POT10 standards:

#### Rule 1: Avoid Complex Flow Constructs
```python
# ✅ ALLOWED
if condition:
    return early_result

# ❌ FORBIDDEN
goto label  # Never use goto
```

#### Rule 2: All Loops Must Have Fixed Bounds
```python
# ✅ COMPLIANT
max_iterations = 1000
for i in range(max_iterations):
    if termination_condition():
        break

# ❌ NON-COMPLIANT
while True:  # Unbounded loop
    pass
```

#### Rule 3: Avoid Heap Allocation After Initialization
```python
# ✅ COMPLIANT: Pre-allocate during initialization
class MemoryOptimizer:
    def __init__(self, max_size: int):
        self.buffer = torch.empty(max_size, device=self.device)  # Pre-allocate

    def process(self, data: torch.Tensor) -> torch.Tensor:
        # Use pre-allocated buffer, no new allocations
        return self.buffer[:data.size(0)]

# ❌ NON-COMPLIANT: Runtime allocation
def process_data(data: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(data)  # Runtime allocation
    return result
```

#### Rule 4: Functions ≤60 Lines
```python
# ✅ COMPLIANT: Function within limit
def validate_tensor_input(tensor: torch.Tensor) -> bool:
    """Validate input tensor (within 60 lines)."""
    # Assertion 1
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
    # Assertion 2
    assert tensor.numel() > 0, "Tensor cannot be empty"
    # Implementation within bounds
    return True

# ❌ NON-COMPLIANT: Function >60 lines - must be split
```

#### Rule 5: ≥2 Assertions Per Function
```python
# ✅ COMPLIANT: Multiple assertions
def optimize_model(model: torch.nn.Module, level: str) -> torch.nn.Module:
    """Optimize model with required assertions."""
    # Assertion 1: Input validation
    assert model is not None, "Model cannot be None"
    # Assertion 2: Parameter validation
    assert level in ["development", "production"], f"Invalid level: {level}"
    # Assertion 3: State validation
    assert len(list(model.parameters())) > 0, "Model has no parameters"

    # Implementation...
    return model
```

### Compliance Validation

```bash
# Run compliance checker
python scripts/nasa_pot10_validator.py src/

# Generate compliance report
python scripts/generate_compliance_report.py --output compliance-report.json

# Check specific rules
python scripts/check_function_length.py src/ml/bitnet/
python scripts/check_assertions.py src/ml/bitnet/
```

## Documentation

### Documentation Types

1. **API Documentation**: Comprehensive docstrings for all public functions
2. **User Guide**: Getting started and usage examples
3. **Technical Documentation**: Architecture and implementation details
4. **Compliance Documentation**: NASA POT10 requirements and validation

### Writing Documentation

#### 1. API Documentation

```python
def validate_bitnet_performance(
    model: BitNetModel,
    test_inputs: List[Tuple[torch.Tensor, ...]],
    create_baseline: bool = False
) -> Dict[str, Any]:
    """
    Validate BitNet model performance against targets.

    Comprehensive validation framework that tests memory reduction (8x target),
    inference speedup (2-4x target), accuracy preservation (<10% degradation),
    and NASA POT10 compliance (95% score). Results include executive summary
    with production readiness assessment.

    Args:
        model: BitNet model to validate. Must be properly initialized with
               quantized layers and optimization applied.
        test_inputs: List of input tuples for testing. Each tuple contains
                    model inputs (input_ids, attention_mask, etc.). Recommended
                    to include various batch sizes and sequence lengths.
        create_baseline: Whether to create full-precision baseline for comparison.
                        If True, creates equivalent full-precision model for
                        accurate performance measurement. Defaults to False for
                        faster validation.

    Returns:
        Comprehensive validation results dictionary containing:
        - final_report: Executive summary with production readiness status
        - memory_validation: Memory usage analysis and 8x target verification
        - speed_validation: Inference speed analysis and 2-4x target verification
        - accuracy_validation: Accuracy preservation and <10% degradation check
        - nasa_pot10_compliance: Compliance score and defense industry readiness

        Structure:
        {
            'final_report': {
                'executive_summary': {
                    'production_ready': bool,
                    'targets_achieved': bool,
                    'overall_status': 'PASSED' | 'FAILED'
                },
                'detailed_metrics': {
                    'memory_reduction_achieved': float,
                    'speedup_achieved': float,
                    'accuracy_preservation': float,
                    'nasa_compliance_score': float
                }
            },
            'memory_validation': {...},
            'speed_validation': {...},
            'accuracy_validation': {...},
            'nasa_pot10_compliance': {...}
        }

    Raises:
        ValueError: If model is None or test_inputs is empty
        RuntimeError: If validation fails due to insufficient resources
        AssertionError: If model fails basic sanity checks

    Examples:
        Basic validation:
        >>> model = create_bitnet_model({'model_size': 'base'})
        >>> test_inputs = [(torch.randint(0, 1000, (4, 128)),) for _ in range(5)]
        >>> results = validate_bitnet_performance(model, test_inputs)
        >>> print(f"Ready: {results['final_report']['executive_summary']['production_ready']}")

        Comprehensive validation with baseline:
        >>> results = validate_bitnet_performance(model, test_inputs, create_baseline=True)
        >>> metrics = results['final_report']['detailed_metrics']
        >>> print(f"Memory: {metrics['memory_reduction_achieved']:.1f}x")
        >>> print(f"Speed: {metrics['speedup_achieved']:.1f}x")
        >>> print(f"Accuracy: {metrics['accuracy_preservation']:.1%}")

    Note:
        Validation with create_baseline=True is more accurate but significantly
        slower as it creates and tests a full-precision equivalent model.
        For development, use create_baseline=False for faster iteration.

    See Also:
        create_bitnet_model: Create models for validation
        optimize_bitnet_model: Apply optimizations before validation
        BitNetPerformanceValidator: Lower-level validation interface
    """
```

#### 2. README Updates

When adding new features, update relevant README sections:

```markdown
## New Feature: Custom Optimization Workflows

BitNet Phase 4 now supports custom optimization workflows for specialized use cases:

### Usage

```python
from src.ml.bitnet.optimization import CustomOptimizer

# Create custom optimizer
optimizer = CustomOptimizer(device, strategy="selective")

# Apply custom optimization
model = optimizer.optimize_with_strategy(model, "attention_first")
```

### Benefits
- Flexible optimization strategies
- Domain-specific optimizations
- Advanced performance tuning
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build API documentation
sphinx-build -b html docs/ docs/_build/

# Build and serve locally
cd docs/
python -m http.server 8080
# Visit http://localhost:8080
```

## Pull Request Process

### PR Checklist

Before submitting a pull request, ensure:

- [ ] **Code Quality**
  - [ ] Code follows style guidelines (Black, Flake8)
  - [ ] Type annotations added for all functions
  - [ ] NASA POT10 compliance validated
  - [ ] No lint errors or warnings

- [ ] **Testing**
  - [ ] All existing tests pass
  - [ ] New tests added for new functionality
  - [ ] Test coverage maintained (≥92%)
  - [ ] Performance tests pass if applicable

- [ ] **Documentation**
  - [ ] Docstrings added for all public functions
  - [ ] README updated if needed
  - [ ] CHANGELOG updated with changes
  - [ ] Examples provided for new features

- [ ] **Compliance**
  - [ ] NASA POT10 compliance maintained
  - [ ] Security scan passes
  - [ ] No critical or high security issues
  - [ ] Audit trail compatibility verified

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)
- [ ] Manual testing completed

## Performance Impact
- Memory usage: [Increased/Decreased/No change]
- Inference speed: [Faster/Slower/No change]
- Training speed: [Faster/Slower/No change]

## Breaking Changes
List any breaking changes and migration steps needed.

## Related Issues
Fixes #[issue_number]
Related to #[issue_number]

## Screenshots/Benchmarks
If applicable, add screenshots or benchmark results.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] NASA POT10 compliance maintained
```

### Review Process

1. **Automated Checks**: All CI/CD checks must pass
2. **Code Review**: At least one reviewer approval required
3. **Performance Review**: Performance impact assessed
4. **Compliance Review**: NASA POT10 compliance verified
5. **Final Approval**: Maintainer approval for merge

### Merge Requirements

- All CI/CD checks passing
- At least one approved review
- No merge conflicts
- Up-to-date with main branch
- NASA POT10 compliance score ≥95%

## Issue Guidelines

### Issue Types

We use issue templates for consistent reporting:

#### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. See error

**Expected Behavior**
What should have happened.

**Actual Behavior**
What actually happened.

**Environment**
- Python version:
- PyTorch version:
- CUDA version:
- OS:
- Hardware:

**Additional Context**
Screenshots, logs, or other context.

**NASA Compliance Impact**
Does this affect POT10 compliance?
```

#### Feature Request Template

```markdown
**Feature Summary**
Brief description of the feature.

**Motivation**
Why is this feature needed?

**Detailed Description**
Comprehensive description of the feature.

**Proposed Implementation**
Ideas for implementation approach.

**Performance Considerations**
Expected impact on memory/speed.

**Compliance Considerations**
NASA POT10 compliance implications.

**Alternatives Considered**
Other approaches considered.
```

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `performance`: Performance-related issues
- `compliance`: NASA POT10 compliance issues
- `security`: Security-related issues
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `priority:high`: High priority
- `priority:medium`: Medium priority
- `priority:low`: Low priority

## Community

### Communication Channels

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Email**: support@agentforge.dev for sensitive issues

### Getting Help

1. **Documentation**: Check comprehensive documentation first
2. **Search Issues**: Look for existing solutions
3. **Ask Questions**: Use GitHub Discussions for questions
4. **Report Bugs**: Use issue templates for bug reports

### Recognition

We recognize contributors through:

- **Contributors List**: All contributors listed in README
- **Release Notes**: Contributions highlighted in releases
- **Special Recognition**: Outstanding contributions featured

### Professional Support

For enterprise users requiring professional support:
- **Email**: enterprise@agentforge.dev
- **Training**: Custom training programs available
- **Consulting**: Architecture and implementation consulting

---

Thank you for contributing to BitNet Phase 4! Your contributions help advance the state of 1-bit neural network optimization and make AI more efficient and accessible.

**Questions?** Reach out via GitHub Discussions or support@agentforge.dev