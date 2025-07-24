# Known Test Issues

This document tracks known test failures that are accepted as current limitations and serve as architectural canaries.

## BitNet Compression Test
**Test**: `tests/compression/test_stage1.py::test_bitnet_quantization_accuracy`
**Status**: XFAIL (Expected Failure)
**Issue**: To be created

### Description
The BitNet 1.58-bit quantization test fails due to architectural limitations in the current compression pipeline implementation.

### Root Cause
- **Missing BitNet Implementation**: The current compression system lacks the specialized 1.58-bit quantization algorithm required for BitNet
- **Quantization Infrastructure**: The test expects specific quantization methods that are not yet implemented
- **Model Architecture Mismatch**: Current models may not support the extreme quantization levels required by BitNet

### Impact
- BitNet compression features are not available
- Tests serve as canaries for compression system improvements
- Performance benchmarks cannot include BitNet comparisons

### Workarounds
- Use standard 8-bit or 16-bit quantization instead
- Implement custom quantization for specific use cases
- Monitor for future BitNet library availability

### Resolution Plan
- [ ] **Step 1**: Research BitNet quantization algorithms and requirements
- [ ] **Step 2**: Design BitNet integration architecture
- [ ] **Step 3**: Implement 1.58-bit quantization support
- [ ] **Step 4**: Add BitNet model loading and conversion
- [ ] **Step 5**: Update compression pipeline to support BitNet
- [ ] **Target**: Q2 2025 (pending resource allocation)

---

## EvoMerge Architectural Issues
**Tests**: `tests/evo_merge/test_*.py`
**Status**: Collection errors (Cannot import)
**Issue**: To be created

### Description
EvoMerge tests cannot be collected due to fundamental architectural dependencies that are not currently available in the test environment.

### Root Cause
- **Missing EvoMerge Core**: The `evo_merge` module expects specific agent architecture components
- **Dependency Chain Issues**: Complex dependency tree involving agent systems, model loaders, and memory management
- **Architecture Evolution**: The agent architecture has evolved beyond what EvoMerge tests expect
- **Import Path Problems**: Module structure changes have broken the import paths

### Impact
- EvoMerge functionality cannot be tested
- Agent merging and evolution features are untested
- Integration tests for agent composition are missing

### Canary Usage
These tests serve as canaries for:
- **Architectural Changes**: When agent system architecture changes significantly
- **Model Loading Infrastructure**: Changes to how models are loaded and managed
- **Memory Management**: Improvements to memory handling that might enable EvoMerge
- **Dependency Resolution**: Updates to the dependency system that might fix imports

### Technical Details
```python
# Expected architecture that's missing:
from agent_forge.evomerge import EvoMergeEngine
from agent_forge.memory import MemoryManager
from agents.base import BaseAgent

# Current status: ImportError on module collection
```

### Resolution Plan
- [ ] **Step 1**: Audit current agent architecture vs EvoMerge expectations
- [ ] **Step 2**: Identify minimal changes needed for EvoMerge compatibility
- [ ] **Step 3**: Create EvoMerge architecture compatibility layer
- [ ] **Step 4**: Implement missing agent composition interfaces
- [ ] **Step 5**: Re-enable EvoMerge tests with proper dependencies
- [ ] **Target**: Q3 2025 (major architecture work required)

---

## Test Health Monitoring

### Canary Test Pattern
Both BitNet and EvoMerge tests follow the "canary test" pattern:

```python
@pytest.mark.canary
@pytest.mark.xfail(reason="Architectural limitation - monitors for changes")
def test_architectural_feature():
    """Canary test - fails until architecture supports feature X"""
    # This test is expected to fail but serves as an indicator
    # that something fundamental has changed when it starts passing
    pass
```

### Monitoring Integration
The test monitoring system tracks these canary tests specifically:

- **Unexpected Passes**: Alert when canary tests start passing (architecture may have changed)
- **Changed Failure Modes**: Alert when canary tests fail differently than expected
- **Dependency Changes**: Monitor for new dependencies that might enable canary features

### Adding New Canary Tests
When adding architectural features that don't yet exist:

1. Create the test with expected implementation
2. Mark with `@pytest.mark.canary`
3. Mark with `@pytest.mark.xfail(reason="...")`
4. Document in this file
5. Create GitHub issue for tracking

### Review Schedule
This document should be reviewed quarterly to:
- Update resolution plans and timelines
- Remove resolved issues
- Add new architectural limitations discovered
- Update canary test monitoring rules

---

*Last Updated: 2025-01-23*
*Next Review: 2025-04-23*
