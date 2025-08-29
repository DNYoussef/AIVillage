# AI Village File Organization Strategy

## Folder Architecture Overview

### Core vs Src Directory Structure

**`core/` Directory - Domain Logic Layer:**
- **Purpose**: Business domain logic and core system components
- **Pattern**: Domain-driven design with bounded contexts
- **Contains**:
  - `agent-forge/` - Agent orchestration and training systems
  - `agents/` - Agent implementations and coordination systems
  - `decentralized_architecture/` - Unified system architectures
  - `domain/` - Business entities, services, and policies
  - `hyperrag/` - RAG system core implementations
  - `rag/` - Knowledge retrieval and synthesis systems

**`src/` Directory - Application Layer:**
- **Purpose**: Application interfaces and deployment packages
- **Pattern**: Clean architecture adapters and interfaces
- **Contains**:
  - `agent_forge/` - Training interfaces and validation systems
  - `data/journals/` - Application data and session management
  - `interfaces/` - External interface adapters
  - Village integration packages and deployment artifacts

### Architectural Principles

1. **Dependency Direction**: `src/` → `core/` (never reverse)
2. **Separation of Concerns**: Domain logic isolated from application concerns
3. **Interface Segregation**: Clean boundaries between layers
4. **Single Responsibility**: Each directory has one clear purpose

## Loose Files Consolidation Plan

### Python Files to Move:

**Debug/Development Scripts** → `scripts/debug/`:
- `debug_controller_imports.py`
- `test_agent_integration.py`
- `test_agent_system.py`
- `test_backend_integration.py`
- `test_memory_simple.py`
- `test_parameter_analysis.py`
- `test_simple_agent_system.py`

**Integration/Demo Scripts** → `examples/`:
- `demo_integration.py`
- `start_backend_services.py`

**Build Configuration** → Keep in root (standard):
- `setup.py` - Legacy build compatibility

**Documentation to Organize**:

**Project Status Reports** → `docs/status/`:
- `AGENT_6_MISSION_COMPLETE.md`
- `COGNATE_25M_EVOMERGE_COMPLETE_RESULTS.md`
- `COGNATE_CONSOLIDATION_SUMMARY.md`
- `COGNATE_VALIDATION_SUMMARY.md`
- `CONSOLIDATION_COMPLETE_SUMMARY.md`
- `CONSOLIDATION_SUMMARY.md`
- `PERFORMANCE_MISSION_COMPLETE.md`
- `PRECOMMIT_OPTIMIZATION_SUMMARY.md`
- `UNIFIED_REFINER_STATUS.md`

**Implementation Guides** → `docs/guides/`:
- `README_AGENT_FORGE_CONSOLIDATED.md`
- `START_AGENT_FORGE.md`

**Security Reports** → `docs/security/`:
- `security-integration-report.md`

**Keep in Root** (Standard project files):
- `README.md` - Main project documentation
- `CHANGELOG.md` - Version history
- `CLAUDE.md` - Development configuration
- `CONTRIBUTING.md` - Contribution guidelines
- `TABLE_OF_CONTENTS.md` - Navigation aid

### File Movement Implementation

```bash
# Create target directories
mkdir -p scripts/debug
mkdir -p docs/status
mkdir -p docs/guides
mkdir -p docs/security/reports

# Move debug/test scripts
mv debug_controller_imports.py scripts/debug/
mv test_*.py scripts/debug/

# Move integration examples
mv demo_integration.py examples/
mv start_backend_services.py examples/

# Move documentation
mv *_COMPLETE*.md docs/status/
mv *_SUMMARY.md docs/status/
mv README_AGENT_FORGE_CONSOLIDATED.md docs/guides/
mv START_AGENT_FORGE.md docs/guides/
mv security-integration-report.md docs/security/reports/
```

### Benefits of This Organization

1. **Clean Root Directory**: Only essential project files remain
2. **Logical Grouping**: Related files organized by purpose
3. **Improved Navigation**: Clear directory structure for developers
4. **Maintainability**: Easier to find and update related files
5. **Documentation Structure**: Status reports and guides properly categorized

### Post-Consolidation Tasks

1. **Update Import Paths**: Fix any broken imports after moves
2. **Update Documentation References**: Fix links to moved files
3. **Validate System Integrity**: Run tests to ensure no functionality broken
4. **Update CI/CD Paths**: Adjust any hardcoded paths in workflows

---

*This organization strategy follows clean architecture principles while maintaining the existing domain-driven structure of the codebase.*
