# AIVillage Core Directory Cleanup Plan

## Context
- Trust Score: 42% (documentation vs reality)
- Major Issue: Stub implementations presented as complete
- Working Features: Compression, Evolution, Basic RAG
- Problematic: Self-evolving system, Agent specialization

## Root Directory Analysis Results

### Python Files Found (6 total):
1. `main.py` - Unified entry point (207 lines) ✅ KEEP
2. `memory_constrained_evolution.py` - Evolution runner (230 lines) ➡️ MOVE TO SCRIPTS
3. `server.py` - Dev server with warnings (381 lines) ⚠️ KEEP WITH FIXES
4. `setup.py` - Package setup (89 lines) ✅ KEEP
5. `test_dashboard.py` - Test dashboard (165 lines) ➡️ MOVE TO TESTS
6. `test_real_fixes.py` - Test fixes (225 lines) ➡️ MOVE TO TESTS

### Markdown Files Found (16 total):
- Multiple analysis reports (recent)
- Historic documentation (MAGI, etc.)
- Core docs (README, CONTRIBUTING, etc.)

## Cleanup Actions

### Files to Move to scripts/
| File | Reason | Note |
|------|--------|------|
| memory_constrained_evolution.py | Entry point script | Historic evolution runner |

### Files to Move to tests/
| File | Reason | Note |
|------|--------|------|
| test_dashboard.py | Test file | Agent Forge test dashboard |
| test_real_fixes.py | Test file | Implementation fix tests |

### Files to Keep in Root
| File | Reason | Action |
|------|--------|--------|
| main.py | Primary entry point | Keep as-is |
| server.py | Dev server | Add clearer warnings |
| setup.py | Package setup | Keep as-is |

### Special Handling: server.py
- Analysis confirmed: Has DEV_MODE warnings
- Currently in docs as production example
- Action: Enhance warnings, update docs

## Critical Issues Found

### 1. Self-Evolving System (from analysis)
- Location: `agents/unified_base_agent.py:791`
- Status: Stub implementation only
- Trust Issue: Documented as working
- Action: Move related files to stubs/ directory

### 2. Server.py Discrepancy
- Documentation shows as quick start
- Code has "DEVELOPMENT ONLY" warnings
- Reality: Not production-ready
- Action: Fix documentation, enhance warnings

### 3. Stub Implementations (29 files found)
- Many NotImplementedError instances
- TODO comments throughout
- Action: Organize stubs properly

## Import Dependencies

### Files Importing Root Scripts
```bash
# Will check who imports the files we're moving
grep -r "import memory_constrained_evolution\|from memory_constrained_evolution" . --include="*.py"
grep -r "import test_dashboard\|from test_dashboard" . --include="*.py"
grep -r "import test_real_fixes\|from test_real_fixes" . --include="*.py"
```

## Backup Strategy
- Create timestamped backup before any moves
- Preserve git history where possible
- Test after each move

## Post-Cleanup Actions
1. Update import statements
2. Fix documentation references
3. Test all entry points
4. Update CI/CD paths
5. Create new README for moved directories

## Implementation Order
1. Create backup
2. Create target directories (scripts/, tests/, stubs/)
3. Move files using git mv to preserve history
4. Fix imports
5. Test functionality
6. Update documentation

## Risk Mitigation
- All moves preserve files (no deletions)
- Git history maintained
- Comprehensive backup created
- Test after each change
