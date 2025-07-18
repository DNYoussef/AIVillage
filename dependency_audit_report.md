# Dependency Audit Report

## Executive Summary

This report analyzes dependency versions across all services and requirements files in the AIVillage project to identify conflicts, inconsistencies, and security vulnerabilities. The analysis covers Python packages, JavaScript/Node.js dependencies, and workspace configurations.

## Key Findings

### 🔴 Critical Issues

1. **FastAPI Version Conflicts**: Multiple conflicting versions across services
2. **Pydantic Version Inconsistencies**: Different versions could cause serialization issues
3. **Unpinned Dependencies**: Many services have unpinned versions causing potential instability
4. **Security Vulnerabilities**: Outdated packages with known security issues

### 🟡 Moderate Issues

1. **Testing Framework Inconsistencies**: Different pytest versions across services
2. **HTTP Client Library Conflicts**: HTTPX version mismatches
3. **Missing Dependency Declarations**: Some services missing explicit dependencies

## Detailed Analysis

### Python Dependencies

#### 1. FastAPI Version Conflicts

**Main Project (pyproject.toml):**
- Production: `fastapi>=0.95.1`
- Development: `fastapi==0.116.0` (with testclient)

**Services:**
- Gateway: `fastapi` (unpinned)
- Twin: `fastapi` (unpinned)
- Communications: `fastapi==0.104.1`

**⚠️ Issue**: Different FastAPI versions can cause API incompatibilities and serialization issues.

#### 2. Pydantic Version Inconsistencies

**Main Project:** `pydantic>=2.8.2`
**Services:**
- Twin: `pydantic` (unpinned)
- Communications: `pydantic==2.5.0`
- Agent Forge EvoMerge: `pydantic>=1.8.2`

**⚠️ Issue**: Pydantic v1 vs v2 are incompatible. The EvoMerge component uses v1 while others use v2.

#### 3. Core Library Conflicts

**HTTPX Versions:**
- Main (dev): `httpx==0.28.1`
- Gateway: `httpx` (unpinned)
- Twin: `httpx` (unpinned)
- Tests: `httpx` (unpinned)

**Uvicorn Versions:**
- Main: `uvicorn>=0.22.0`
- Gateway: `uvicorn` (unpinned)
- Twin: `uvicorn` (unpinned)
- Communications: `uvicorn==0.24.0`

#### 4. ML Framework Conflicts

**PyTorch Versions:**
- Main: `torch>=2.3.0`
- Agent Forge: `torch==2.3.0`
- EvoMerge: `torch>=1.9.0`

**Transformers Versions:**
- Main: `transformers>=4.41.1`
- Agent Forge: `transformers==4.41.2`
- EvoMerge: `transformers>=4.11.0`

#### 5. Testing Framework Inconsistencies

**Pytest Versions:**
- Main (dev): `pytest` (unpinned)
- Communications: `pytest==7.4.3`
- EvoMerge: `pytest>=6.2.0`
- Tests: `pytest` (unpinned)

**Pytest-asyncio Versions:**
- Main (dev): `pytest-asyncio==1.0.0`
- Communications: `pytest-asyncio==0.21.1`

### JavaScript/Node.js Dependencies

#### 1. TypeScript Versions

**Monorepo Root:** `typescript^5.0.0`
**Web App:** `typescript^5.0.0`
**UI Kit:** `typescript^5.0.0`

✅ **Good**: Consistent TypeScript versions across all packages.

#### 2. React Ecosystem

**React Versions:**
- Web App: `react^18.0.0`
- UI Kit: `react^18.0.0`

✅ **Good**: Consistent React versions.

#### 3. Development Tools

**ESLint Versions:**
- Root: `eslint^8.0.0`
- Web App: `eslint^8.0.0`
- UI Kit: `eslint^8.0.0`

✅ **Good**: Consistent ESLint versions.

## Security Vulnerability Analysis

### High-Risk Packages

1. **requests==2.31.0** (Communications)
   - Known security vulnerabilities in older versions
   - Should be updated to `>=2.32.3`

2. **Unpinned Dependencies**
   - Many services have unpinned versions that could introduce security issues
   - Potential for supply chain attacks

### Outdated Packages

1. **FastAPI 0.104.1** (Communications)
   - Missing recent security patches
   - Should be updated to latest stable version

2. **Pydantic 2.5.0** (Communications)
   - Older version missing recent fixes
   - Should be updated to `>=2.8.2`

## Recommendations

### Immediate Actions Required

1. **Standardize FastAPI Version**
   ```
   Recommended: fastapi>=0.116.0
   ```

2. **Fix Pydantic Version Conflicts**
   - Update EvoMerge to use Pydantic v2
   - Standardize on `pydantic>=2.8.2`

3. **Pin Critical Dependencies**
   - Pin FastAPI, Pydantic, and HTTPX versions in all services
   - Use version ranges for better compatibility

4. **Update Security-Critical Packages**
   - Update requests to `>=2.32.3`
   - Update all unpinned dependencies

### Long-term Improvements

1. **Implement Dependency Management Strategy**
   - Use Poetry for all Python projects
   - Implement lock files for reproducible builds

2. **Add Dependency Scanning**
   - Integrate tools like Safety, Bandit, or Snyk
   - Add automated security scanning to CI/CD

3. **Standardize Testing Dependencies**
   - Use consistent pytest versions across all services
   - Standardize on pytest-asyncio version

4. **Create Shared Requirements**
   - Create shared requirements files for common dependencies
   - Use inheritance to reduce duplication

## Proposed Standard Versions

### Python Core Dependencies
```
fastapi>=0.116.0
pydantic>=2.8.2
httpx>=0.28.1
uvicorn>=0.24.0
pytest>=7.4.3
pytest-asyncio>=0.21.1
requests>=2.32.3
```

### ML/AI Dependencies
```
torch>=2.3.0
transformers>=4.41.2
accelerate>=0.29.3
```

### JavaScript/Node.js Dependencies
```
typescript: ^5.0.0
react: ^18.0.0
eslint: ^8.0.0
```

## Implementation Plan

1. **Phase 1**: Fix critical security vulnerabilities
2. **Phase 2**: Standardize FastAPI and Pydantic versions
3. **Phase 3**: Implement dependency management tooling
4. **Phase 4**: Add automated security scanning

## Conclusion

The AIVillage project has significant dependency management issues that pose security risks and could cause runtime conflicts. Immediate action is required to standardize versions and fix security vulnerabilities, particularly in the FastAPI and Pydantic ecosystems.

Priority should be given to:
1. Resolving Pydantic v1/v2 conflicts
2. Standardizing FastAPI versions
3. Updating security-critical packages
4. Implementing consistent dependency management practices
