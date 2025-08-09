# AIVillage Linting Issues - Priority Fix Plan

## Executive Summary

**Total Issues Discovered**: 267,124+ across entire codebase
**In Core Directory Alone**: 5,297 issues 
**Fixable Issues**: ~50-60% can be auto-fixed with `--fix` options

## Issue Categories by Priority

### 🔴 CRITICAL - Security & Functionality (Fix Immediately)

1. **Security Issues (S-series)**
   - Hardcoded credentials/secrets
   - SQL injection vulnerabilities
   - Path traversal issues
   - Insecure random generators

2. **Logic Errors (F-series)**
   - Undefined variables
   - Unreachable code
   - Import errors
   - Syntax errors

3. **Type Safety (ANN, TC series)**
   - Missing critical type annotations
   - Type checking failures
   - API contract violations

### 🟡 HIGH PRIORITY - Code Quality (Fix Next)

1. **Performance Issues (PERF-series)**
   - Memory leaks
   - Inefficient loops
   - Database query issues

2. **Error Handling (BLE, TRY series)**
   - Bare except clauses
   - Poor exception handling
   - Missing error context

3. **Documentation (D-series)**
   - Missing docstrings in public APIs
   - Inadequate documentation

### 🟢 MEDIUM PRIORITY - Style & Consistency

1. **Code Style (E, W series)**
   - Line length violations (3,060+ fixable)
   - Whitespace issues
   - Quote consistency

2. **Import Organization (I-series)**
   - Import sorting
   - Unused imports
   - Import conventions

3. **Code Simplification (SIM, UP series)**
   - Modernization opportunities
   - Simplifiable expressions

### ⚪ LOW PRIORITY - Aesthetic

1. **Naming Conventions (N-series)**
   - Variable naming style
   - Function naming conventions

2. **Comments & Formatting**
   - Comment style
   - Blank line consistency

## Auto-Fix Strategy

### Phase 1: Safe Auto-Fixes (Run First)
```bash
# These are safe to run automatically
ruff check . --fix --select E,W,F541,I,UP,SIM
python -m black .
python -m isort .
```

**Estimated Fixes**: ~150,000+ issues (mostly formatting)

### Phase 2: Moderate Risk Auto-Fixes
```bash
# Review before applying
ruff check . --fix --select F,UP,SIM,PIE,PYI
```

**Estimated Fixes**: ~50,000+ issues

### Phase 3: Manual Review Required
- Security issues (S-series)
- Documentation issues (D-series) 
- Complex refactoring (PLR-series)
- Type annotation issues (ANN-series)

## Implementation Plan

### Week 1: Critical Security & Functionality
1. **Day 1-2**: Fix all F-series (logic errors)
2. **Day 3-4**: Address S-series (security issues)  
3. **Day 5**: Resolve import errors and undefined variables

### Week 2: Auto-Fixes & Style
1. **Day 1**: Run safe auto-fixes (E,W,I series)
2. **Day 2**: Apply Black and isort formatting
3. **Day 3-4**: Review and apply moderate auto-fixes
4. **Day 5**: Test and validate changes

### Week 3: Code Quality
1. **Day 1-2**: Add critical type annotations
2. **Day 3-4**: Improve error handling
3. **Day 5**: Add missing docstrings for public APIs

## Tools Configuration Standardization

### Immediate Actions Needed:

1. **Resolve Line Length Conflict**:
   - Standardize on 88 characters (Black default)
   - Update `.flake8` and `.isort.cfg` to match
   - Update `pyproject.toml` if needed

2. **Update Ruff Configuration**:
   - Remove deprecated rules (ANN101, ANN102)
   - Configure fix-safe rules for auto-fixing
   - Set appropriate ignore patterns

3. **Pre-commit Integration**:
   - Fix pre-commit environment issues
   - Add ruff to pre-commit hooks
   - Configure staged-files-only linting

## Risk Assessment

### Low Risk (Can Auto-Fix Safely):
- Formatting issues (E,W series): ~80,000+ issues
- Import sorting (I series): ~20,000+ issues  
- Quote consistency (Q series): ~10,000+ issues

### Medium Risk (Review Recommended):
- Code modernization (UP series): ~15,000+ issues
- Simplification (SIM series): ~8,000+ issues
- Unused variables/imports (F series subset): ~5,000+ issues

### High Risk (Manual Review Required):
- Security issues (S series): ~2,000+ issues
- Type annotations (ANN series): ~25,000+ issues
- Complex refactoring (PLR series): ~10,000+ issues

## Success Metrics

### Target Reductions:
- **Week 1**: 50,000+ critical issues resolved
- **Week 2**: 150,000+ style issues auto-fixed  
- **Week 3**: 25,000+ code quality improvements

### Final Goals:
- **Critical Issues**: 0 remaining
- **Total Issues**: <10,000 (96% reduction)
- **Auto-fixable Issues**: 0 remaining
- **Pre-commit Compliance**: 100% passing

## Next Steps

1. **Immediate**: Run comprehensive linting with proper issue counting
2. **This Week**: Execute Phase 1 auto-fixes
3. **Next Week**: Manual security and critical issue review
4. **Ongoing**: Implement pre-commit enforcement

## File-Specific Hotspots

Based on sample from `src/core`, major issue concentrations:

1. **src/core/security/**: Security and style issues
2. **src/core/compression/**: Type annotations and documentation  
3. **src/core/communication.py**: Missing docstrings
4. **src/core/chat_engine.py**: Magic numbers and commented code

These directories should receive priority attention during manual review phases.

---

**Generated**: 2025-08-09  
**Total Estimated Fix Time**: 3-4 weeks with dedicated focus  
**Automation Level**: ~70% can be auto-fixed safely