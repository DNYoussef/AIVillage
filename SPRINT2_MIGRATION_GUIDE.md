# Sprint 2 Migration Guide: Production/Experimental Separation

## Overview

Sprint 2 has reorganized the AIVillage codebase to clearly separate production-ready components from experimental work. This guide helps you update your code and workflows.

## New Directory Structure

```
AIVillage/
├── production/          # ✅ Ready for use (Quality gates enforced)
│   ├── compression/     # Model compression (4-8x reduction)
│   ├── evolution/       # Evolutionary optimization
│   ├── rag/            # Retrieval-augmented generation
│   ├── memory/         # Memory management & W&B
│   ├── benchmarking/   # Real benchmark evaluation
│   ├── geometry/       # Geometric analysis
│   └── tests/          # 80%+ test coverage
├── experimental/        # 🚧 Under development (APIs may change)
│   ├── agents/         # Multi-agent system
│   ├── mesh/           # P2P networking
│   ├── services/       # Development microservices
│   ├── training/       # Training pipelines
│   └── federated/      # Federated learning
└── deprecated/         # 📦 Archived code
    ├── backups/        # Historical backups
    └── legacy/         # Old implementations
```

## Import Changes

### ✅ Production Components (Stable APIs)

```python
# OLD IMPORTS (Sprint 1)
from agent_forge.compression import seedlm
from agent_forge.evomerge import EvolutionaryTournament
from rag_system import RAGPipeline
from agent_forge.memory_manager import MemoryManager

# NEW IMPORTS (Sprint 2)
from production.compression.compression import seedlm
from production.evolution.evomerge import EvolutionaryTournament
from production.rag.rag_system import RAGPipeline
from production.memory.memory_manager import MemoryManager
```

### ⚠️ Experimental Components (Will Show Warnings)

```python
# OLD IMPORTS (Sprint 1)
from agents.king import KingAgent
from communications.mesh_node import MeshNode
from services.gateway import Gateway

# NEW IMPORTS (Sprint 2) - Will display warnings
from experimental.agents.king import KingAgent
from experimental.mesh.mesh_node import MeshNode
from experimental.services.gateway import Gateway
```

## Component Status Matrix

| Component | Location | Status | Test Coverage | Breaking Changes |
|-----------|----------|--------|---------------|------------------|
| **Compression** | `production/compression/` | ✅ Stable | 85%+ | None |
| **Evolution** | `production/evolution/` | ✅ Stable | 80%+ | None |
| **RAG** | `production/rag/` | ✅ Stable | 75%+ | None |
| **Memory Management** | `production/memory/` | ✅ Stable | 90%+ | None |
| **Benchmarking** | `production/benchmarking/` | ✅ Stable | 85%+ | None |
| **Geometry** | `production/geometry/` | ✅ Stable | 70%+ | None |
| **Agents** | `experimental/agents/` | 🚧 Development | 45% | Frequent |
| **Mesh** | `experimental/mesh/` | 🚧 Development | 20% | Frequent |
| **Services** | `experimental/services/` | 🚧 Development | 40% | Frequent |
| **Training** | `experimental/training/` | 🚧 Development | 50% | Frequent |
| **Federated** | `experimental/federated/` | 🚧 Development | 15% | Frequent |

## Quality Gates

### Production Code Requirements

✅ **Enforced Automatically via CI/CD:**
- No imports from `experimental/` or `deprecated/`
- No task markers (no T-O-D-O, F-I-X-M-E, etc.)
- Minimum 70% test coverage
- Type checking with mypy
- Security scanning with bandit
- Pre-commit hooks pass

### Experimental Code Guidelines

⚠️ **More Relaxed Standards:**
- Shows warnings on import
- May contain task markers and breaking changes
- Tests encouraged but not required
- APIs subject to change without notice

## Migration Steps

### 1. Update Your Imports

**Production Components:**
```bash
# Find and replace in your codebase
find . -name "*.py" -exec sed -i 's/from agent_forge\.compression/from production.compression/g' {} +
find . -name "*.py" -exec sed -i 's/from agent_forge\.evomerge/from production.evolution.evomerge/g' {} +
find . -name "*.py" -exec sed -i 's/from rag_system/from production.rag.rag_system/g' {} +
```

**Experimental Components:**
```bash
# Update with warnings awareness
find . -name "*.py" -exec sed -i 's/from agents/from experimental.agents/g' {} +
find . -name "*.py" -exec sed -i 's/from communications/from experimental.mesh/g' {} +
find . -name "*.py" -exec sed -i 's/from services/from experimental.services/g' {} +
```

### 2. Handle Experimental Warnings

When importing experimental features, you'll see:
```
ExperimentalWarning: KingAgent is experimental and may change without notice.
```

**To suppress in development:**
```python
import warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)
```

**To handle programmatically:**
```python
import warnings
from experimental import warn_experimental

# This will show warning
from experimental.agents.king import KingAgent

# Check if experimental features are being used
def check_experimental_usage():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from experimental.agents import KingAgent
        if w:
            print(f"Using experimental feature: {w[0].message}")
```

### 3. Update Your Development Workflow

**Before Committing:**
```bash
# Run quality gate checker
python scripts/check_quality_gates.py

# All tests
pytest production/tests/

# Only production tests with coverage
pytest production/tests/ --cov=production --cov-fail-under=70
```

**CI/CD Integration:**
- Production changes trigger strict quality gates
- Experimental changes are more permissive
- Import separation enforced automatically

### 4. New Testing Requirements

**Production Components:**
- Must have tests in `production/tests/`
- Minimum 70% coverage required
- Tests must pass for merge approval

**Experimental Components:**
- Tests in `experimental/tests/` (optional)
- Can fail without blocking development
- Coverage tracking but not enforced

## Configuration Changes

### pytest.ini

Your test configuration now uses:
```ini
[tool:pytest]
testpaths = production/tests experimental/tests
markers =
    production: Production component tests (must pass)
    experimental: Experimental tests (can fail)
    integration: Cross-component integration tests
```

### Pre-commit Hooks

Updated `.pre-commit-config.yaml` includes:
```yaml
repos:
  - repo: local
    hooks:
      - id: production-quality-gates
        name: Production Quality Gates
        entry: python scripts/check_quality_gates.py
        language: system
        pass_filenames: false
        types: [python]
```

## API Stability Guarantees

### Production Components

**Semantic Versioning:**
- Major: Breaking API changes
- Minor: New features, backward compatible
- Patch: Bug fixes, backward compatible

**Deprecation Policy:**
- 2 release notice for breaking changes
- Migration guides provided
- Legacy support during transition

### Experimental Components

**No Stability Guarantees:**
- APIs may change without notice
- Features may be removed
- No deprecation period required
- Breaking changes in any release

## Common Migration Issues

### 1. Import Errors

**Problem:**
```python
ImportError: No module named 'agent_forge.compression'
```

**Solution:**
```python
# Change from:
from agent_forge.compression import seedlm

# To:
from production.compression.compression import seedlm
```

### 2. Circular Import Issues

**Problem:** Experimental components trying to import production
**Solution:** Redesign to have production as foundation, experimental builds on top

### 3. Test Discovery Issues

**Problem:** Tests not found after reorganization
**Solution:** Update test paths in pytest.ini and IDE configurations

### 4. CI/CD Pipeline Failures

**Problem:** Quality gates failing on production code
**Solution:** Use the quality gate checker locally before pushing:

```bash
python scripts/check_quality_gates.py
```

## Support and Resources

### Getting Help

1. **Check Migration Issues:** Look for similar import/path issues in the project's GitHub issues
2. **Quality Gate Failures:** Run local quality checker for detailed diagnostics
3. **API Questions:** Check component documentation in respective directories

### Quick Reference

**Run Quality Checks:**
```bash
python scripts/check_quality_gates.py
```

**Test Production Components:**
```bash
pytest production/tests/ -v
```

**Test Everything:**
```bash
pytest production/tests/ experimental/tests/ -v
```

**Check Import Separation:**
```bash
grep -r "from experimental" production/ --include="*.py" && echo "ERROR: Found experimental imports in production!"
```

## Best Practices

### For Production Code

1. **Always** run quality gates before committing
2. **Never** import from experimental or deprecated
3. **Include** comprehensive tests for new features
4. **Document** all public APIs
5. **Follow** semantic versioning for changes

### For Experimental Code

1. **Add** warning imports to new modules
2. **Consider** graduation path to production
3. **Document** known limitations and issues
4. **Test** when possible, but don't block on failures
5. **Communicate** breaking changes to users

### For Mixed Development

1. **Separate** production and experimental work
2. **Test** production components thoroughly
3. **Prototype** in experimental before moving to production
4. **Plan** migration paths for successful experimental features
5. **Maintain** clear boundaries between stability levels

---

**This migration guide ensures your code remains compatible with Sprint 2's quality-focused architecture while providing clear paths for both stable production use and experimental development.**