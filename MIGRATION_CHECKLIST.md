# Main.py Consolidation Migration Checklist

## Pre-Migration Tasks
- [x] Audit all main.py files
- [x] Create entry point mapping document
- [x] Generate checksums for verification
- [ ] Create legacy backup directory
- [ ] Backup all current main.py files

## File Structure Changes

### 1. Create Legacy Backup
```bash
mkdir legacy_mains
cp main.py legacy_mains/main.py.root
cp agent_forge/main.py legacy_mains/main.py.agent_forge
cp agent_forge/core/main.py legacy_mains/main.py.agent_forge_core
cp agents/king/main.py legacy_mains/main.py.king
cp rag_system/main.py legacy_mains/main.py.rag
```

### 2. Create New Structure
- [ ] Create unified root main.py
- [ ] Create service-specific entry points
- [ ] Update __init__.py files for module imports

### 3. Import Updates Required

#### Files to Update:
- [ ] Any scripts importing from old main.py locations
- [ ] Docker configurations
- [ ] CI/CD pipeline configurations
- [ ] Documentation references
- [ ] README.md usage examples

#### Import Patterns to Search:
```bash
# Search for imports
grep -r "from main import" .
grep -r "import main" .
grep -r "main\.py" .
```

## CLI Interface Design

### Unified CLI Structure
```bash
# New unified interface
python main.py --mode agent-forge --action [train|deploy|test]
python main.py --mode king --action [run|analyze|plan]
python main.py --mode rag --action [query|index|evaluate]
python main.py --mode core --action [util|config|status]

# Service-specific usage
python -m agent_forge.main --action train
python -m agents.king.main --action run
```

## Testing Requirements

### Entry Point Tests
- [ ] Test unified CLI with all modes
- [ ] Test service-specific entry points
- [ ] Test backward compatibility
- [ ] Test error handling
- [ ] Test help/usage output

### Test Commands
```bash
# Test unified entry point
python main.py --help
python main.py --mode agent-forge --help
python main.py --mode king --help
python main.py --mode rag --help

# Test service-specific
python -m agent_forge.main --help
python -m agents.king.main --help
python -m rag_system.main --help
```

## Documentation Updates

### Files to Update
- [ ] README.md - Usage examples
- [ ] docs/ENTRY_POINTS.md - New entry point guide
- [ ] docs/QUICK_START.md - Updated quick start
- [ ] CONTRIBUTING.md - Development setup

## Rollback Plan
- [ ] Keep legacy_mains/ directory for 30 days
- [ ] Document rollback procedure
- [ ] Test rollback process
- [ ] Create rollback script

## Validation Steps
- [ ] All entry points work correctly
- [ ] No broken imports
- [ ] Documentation is accurate
- [ ] CI/CD pipelines pass
- [ ] Docker builds work
- [ ] All tests pass
