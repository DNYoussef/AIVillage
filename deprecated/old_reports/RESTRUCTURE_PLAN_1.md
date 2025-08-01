# AIVillage Codebase Restructuring Plan

## Overview
This document outlines the comprehensive restructuring of the AIVillage codebase into a clean, production-ready structure.

## New Directory Structure

```
AIVillage/
├── src/                          # Production-ready code
│   ├── production/               # Core production components
│   │   ├── compression/          # Model compression systems
│   │   ├── evolution/            # Evolutionary optimization
│   │   ├── rag/                  # RAG systems
│   │   ├── geometry/             # Geometric analysis
│   │   ├── memory/               # Memory management
│   │   └── benchmarking/         # Production benchmarks
│   ├── core/                     # Core infrastructure
│   ├── agent_forge/              # Stable agent_forge components
│   │   ├── core/                 # Core agent functionality
│   │   ├── evaluation/           # Agent evaluation systems
│   │   ├── deployment/           # Deployment tools
│   │   ├── utils/                # Utility functions
│   │   └── orchestration/        # Agent orchestration
│   ├── mcp_servers/              # MCP server implementations
│   ├── digital_twin/             # Digital twin systems
│   ├── monitoring/               # System monitoring
│   ├── jobs/                     # Job management
│   ├── communications/           # Communication systems
│   ├── calibration/              # Model calibration
│   ├── services/                 # Production services
│   ├── ingestion/                # Data ingestion
│   ├── hyperag/                  # HyperAG systems
│   └── rag_system/               # Standalone RAG system
├── experimental/                 # Experimental/prototype code
│   ├── agents/                   # Experimental agents
│   ├── mesh/                     # Mesh networking prototypes
│   ├── services/                 # Experimental services
│   ├── training/                 # Training experiments
│   ├── federated/                # Federated learning
│   └── agent_forge_experimental/ # Experimental agent_forge
│       ├── self_awareness/       # Self-awareness experiments
│       ├── bakedquietiot/        # Baked quiet IoT
│       ├── sleepdream/           # Sleep/dream systems
│       ├── foundation/           # Foundation models
│       ├── prompt_baking_legacy/ # Legacy prompt baking
│       ├── tool_baking/          # Tool baking experiments
│       ├── adas/                 # ADAS systems
│       ├── optim/                # Optimization experiments
│       ├── svf/                  # SVF implementations
│       ├── meta/                 # Meta-learning
│       ├── training/             # Training utilities
│       ├── evolution/            # Evolution experiments
│       └── compression/          # Compression experiments
├── tools/                        # Development tools
│   ├── scripts/                  # Utility scripts
│   ├── benchmarks/               # Benchmark suites
│   └── examples/                 # Usage examples
├── mobile/                       # Mobile applications
│   ├── android-sdk/              # Android SDK (if maintained)
│   ├── mobile-app/               # Mobile app (if maintained)
│   └── monorepo/                 # Mobile monorepo (if maintained)
├── tests/                        # Test suites (stays at root)
├── docs/                         # Documentation (stays at root)
└── .github/                      # GitHub workflows (stays at root)
```

## Migration Mapping

### Production Components → src/
- `production/` → `src/production/`
- `digital_twin/` → `src/digital_twin/`
- `mcp_servers/` → `src/mcp_servers/`
- `monitoring/` → `src/monitoring/`
- `jobs/` → `src/jobs/`
- `communications/` → `src/communications/`
- `calibration/` → `src/calibration/`
- `core/` → `src/core/`
- `services/` → `src/services/`
- `ingestion/` → `src/ingestion/`
- `hyperag/` → `src/hyperag/`
- `rag_system/` → `src/rag_system/`

### Agent Forge Split
**Stable → src/agent_forge/**
- `agent_forge/core/` → `src/agent_forge/core/`
- `agent_forge/evaluation/` → `src/agent_forge/evaluation/`
- `agent_forge/deployment/` → `src/agent_forge/deployment/`
- `agent_forge/utils/` → `src/agent_forge/utils/`
- `agent_forge/orchestration/` → `src/agent_forge/orchestration/`

**Experimental → experimental/agent_forge_experimental/**
- `agent_forge/self_awareness/` → `experimental/agent_forge_experimental/self_awareness/`
- `agent_forge/bakedquietiot/` → `experimental/agent_forge_experimental/bakedquietiot/`
- `agent_forge/sleepdream/` → `experimental/agent_forge_experimental/sleepdream/`
- `agent_forge/foundation/` → `experimental/agent_forge_experimental/foundation/`
- `agent_forge/prompt_baking_legacy/` → `experimental/agent_forge_experimental/prompt_baking_legacy/`
- `agent_forge/tool_baking/` → `experimental/agent_forge_experimental/tool_baking/`
- `agent_forge/adas/` → `experimental/agent_forge_experimental/adas/`
- `agent_forge/optim/` → `experimental/agent_forge_experimental/optim/`
- `agent_forge/svf/` → `experimental/agent_forge_experimental/svf/`
- `agent_forge/meta/` → `experimental/agent_forge_experimental/meta/`
- `agent_forge/training/` → `experimental/agent_forge_experimental/training/`
- `agent_forge/evolution/` → `experimental/agent_forge_experimental/evolution/`
- `agent_forge/compression/` → `experimental/agent_forge_experimental/compression/`

### Tools Consolidation → tools/
- `scripts/` → `tools/scripts/`
- `benchmarks/` → `tools/benchmarks/`
- `examples/` → `tools/examples/`

### Experimental → experimental/ (already mostly organized)
- `experimental/agents/` → `experimental/agents/`
- `experimental/mesh/` → `experimental/mesh/`
- `experimental/services/` → `experimental/services/`
- `experimental/training/` → `experimental/training/`
- `experimental/federated/` → `experimental/federated/`

## Import Path Updates

### Old → New Import Patterns
```python
# Before
from production.compression import model_compression
from agent_forge.core import main
from scripts.performance_monitor import monitor
from benchmarks.benchmark_suite import run_benchmarks

# After  
from src.production.compression import model_compression
from src.agent_forge.core import main
from tools.scripts.performance_monitor import monitor
from tools.benchmarks.benchmark_suite import run_benchmarks
```

## Quality Gates Applied

### Production Components (src/)
- ✅ No TODO/FIXME markers
- ✅ Type hints on all public APIs
- ✅ Comprehensive docstrings
- ✅ Test coverage > 80%
- ✅ Security scanning passed
- ✅ Performance benchmarks met

### Experimental Components (experimental/)
- ⚠️ May contain TODO/FIXME markers
- ⚠️ Documentation may be incomplete
- ⚠️ Test coverage may be lower
- ⚠️ APIs may be unstable

## Files That Stay at Root
- Configuration files: `pyproject.toml`, `requirements*.txt`, `setup.py`
- Docker files: `Dockerfile*`, `docker-compose*.yml`
- Documentation: `*.md`, `LICENSE`, `CHANGELOG.md`
- CI/CD: `.github/`, `Makefile`, `pytest.ini`
- Data files: `*.db`, `*.json`, `*.log`
- Main entry points: `main.py`, `server.py`

## Post-Restructure Tasks

1. **Update Import Statements**
   - Run automated import path updates
   - Update all Python files to use new src/ structure
   - Update documentation examples

2. **Update Configuration**
   - Update `pyproject.toml` package structure
   - Update Docker builds paths
   - Update CI/CD workflow paths

3. **Update Documentation**
   - Update README.md with new structure
   - Update API documentation
   - Update development guides

4. **Testing**
   - Run full test suite with new structure
   - Update test imports and paths
   - Verify all integrations work

5. **Deployment Updates**
   - Update production deployment scripts
   - Update Docker image builds
   - Update any external references

## Benefits of New Structure

1. **Clear Separation**: Production vs experimental code clearly separated
2. **Improved Maintainability**: Easier to find and maintain components
3. **Better Testing**: Clear test boundaries and coverage requirements
4. **Safer Deployments**: Only src/ components are production-ready
5. **Scalable Architecture**: Easy to add new components in right places
6. **Better Documentation**: Structure is self-documenting
7. **IDE Support**: Better IDE navigation and code completion

## Risk Mitigation

1. **Backup**: Full codebase backup created before restructuring
2. **Incremental**: Move components one at a time with testing
3. **Import Guards**: Prevent experimental imports in production code
4. **Rollback Plan**: Documented rollback procedures if issues arise
5. **Team Communication**: Clear communication about import path changes

## Execution Status

- [x] Plan created and reviewed
- [x] Directory structure designed
- [x] Import mapping defined
- [x] Quality gates established
- [ ] **Ready for execution**

## Next Steps

1. Execute the restructuring using the migration scripts
2. Update all import statements
3. Run comprehensive test suite
4. Update documentation
5. Deploy and validate

---

*This restructuring transforms AIVillage into a production-ready codebase with clear architectural boundaries and maintainable structure.*