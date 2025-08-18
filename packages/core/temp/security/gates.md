# Security Gates Documentation

This document describes the automated security gates that will cause Pull Request failures and prevent merging to the main branch.

## Overview

Security gates are automated checks that run on every Pull Request to ensure production security readiness. These gates implement the security requirements identified in the AI Village Integration Readiness audit.

## Security Gates That Will Fail PRs

### 1. HTTP URLs in Production Code

**What it checks:**
- Scans all production source code (`src/production/`) for hardcoded HTTP URLs
- Validates production configuration files for insecure HTTP endpoints
- Checks Docker Compose files for external HTTP URLs

**Failure conditions:**
- Any `"http://` or `'http://` URLs found in production Python files
- HTTP URLs in `config/aivillage_config_production.yaml` (excluding localhost)
- External HTTP URLs in docker-compose files (internal service communication is allowed)

**How to fix:**
- Replace HTTP URLs with HTTPS equivalents
- Use environment-conditional logic for development vs production URLs
- For production configs, ensure all external services use HTTPS

**Example of acceptable code:**
```python
# ✅ Acceptable - environment conditional
if os.getenv("AIVILLAGE_ENV") == "production":
    base_url = "https://secure.example.com"
else:
    base_url = "http://localhost:8000"  # Development only

# ✅ Acceptable - security validation code
if url.startswith("http://"):
    raise HTTPSecurityError("Use HTTPS in production")
```

**Example of failing code:**
```python
# ❌ Will fail CI
API_ENDPOINT = "http://insecure.example.com/api"  # Hardcoded HTTP URL

# ❌ Will fail CI
config = {
    "external_service": "http://api.service.com"  # HTTP in production config
}
```

### 2. Unsafe Pickle Usage

**What it checks:**
- Scans source code for `pickle.loads()` and `pickle.load()` usage
- Excludes security-related files that are allowed to reference pickle for documentation

**Failure conditions:**
- Any `pickle.loads(` or `pickle.load(` found in source code
- Missing security annotations for legitimate pickle usage

**How to fix:**
- Replace `pickle.loads()` with `SecureSerializer.loads()` from `src.core.security.secure_serialization`
- Replace `pickle.dumps()` with `SecureSerializer.dumps()`
- Add `# SECURITY: safe usage` comment for legitimate cases

**Example of fix:**
```python
# ❌ Will fail CI
import pickle
data = pickle.loads(serialized_data)

# ✅ Acceptable
from src.core.security.secure_serialization import secure_loads
data = secure_loads(serialized_data)
```

### 3. Bandit Security Issues

**What it checks:**
- Runs Bandit security scanner on production code
- Fails on HIGH severity security issues
- Fails if more than 3 MEDIUM severity issues are found

**Failure conditions:**
- Any HIGH severity security vulnerabilities
- More than 3 MEDIUM severity issues in production code

**Common issues Bandit detects:**
- Hardcoded passwords or secrets
- Use of `shell=True` in subprocess calls
- Weak cryptographic functions
- SQL injection vulnerabilities
- XML parsing vulnerabilities

**How to fix:**
- Move secrets to environment variables
- Use secure subprocess calls without `shell=True`
- Replace weak crypto functions (MD5, SHA1) with SHA256+
- Use parameterized queries for database operations
- Add `# nosec` comment for false positives with justification

### 4. Critical Ruff Issues

**What it checks:**
- Runs Ruff linter with error-level checks
- Focuses on syntax errors, undefined names, and import issues

**Failure conditions:**
- Syntax errors (E9xx codes)
- Undefined name usage (F63x, F7xx codes)
- Import-related errors (F82x codes)

**How to fix:**
- Fix Python syntax errors
- Import all used modules and functions
- Remove unused imports
- Fix undefined variable references

### 5. Production Configuration Validation

**What it checks:**
- Runs existing HTTP security validation tests
- Validates production environment configuration

**Failure conditions:**
- Production config validation tests fail
- Environment variables contain insecure HTTP URLs

**How to fix:**
- Ensure `AIVILLAGE_ENV=production` works without HTTP URLs
- Update production configuration files to use HTTPS
- Set secure defaults for production environment variables

## CI/CD Integration

The security gates are implemented in `.github/workflows/security-gates.yml` and run automatically on:

- Every Pull Request to `main` or `dev` branches
- Every push to `main` or `dev` branches

### Workflow Steps

1. **Setup**: Install Python, dependencies, and security tools
2. **HTTP URL Check**: Scan for insecure HTTP URLs
3. **Pickle Usage Check**: Scan for unsafe pickle usage
4. **Bandit Scan**: Run security vulnerability analysis
5. **Ruff Check**: Run critical code quality checks
6. **MyPy Check**: Type check security-critical modules (warnings only)
7. **Config Validation**: Run production configuration tests

### Required Dependencies

The CI workflow automatically installs:
- `bandit`: Security vulnerability scanner
- `ruff`: Fast Python linter
- `mypy`: Static type checker
- `portalocker`: Cross-platform file locking
- `pytest`: Test runner
- `pyyaml`: YAML configuration parsing

## Bypassing Security Gates (Emergency Use Only)

In exceptional circumstances, security gates can be bypassed with proper justification:

### Temporary Bypass Methods

1. **Add security annotations:**
```python
dangerous_code()  # SECURITY: safe usage - justified reason here
```

2. **Use nosec for Bandit:**
```python
subprocess.call(cmd, shell=True)  # nosec B602 - validated input only
```

3. **Skip specific tests:**
```python
@unittest.skip("Security exception approved by security team - ticket #123")
def test_that_would_fail():
    pass
```

### Approval Process

Any security gate bypass must:
1. Have clear justification documented in code
2. Include reference to security team approval
3. Be temporary with follow-up ticket to fix properly
4. Be reviewed by security team before merge

## Monitoring and Alerts

Security gate failures trigger:
- Immediate PR status check failure
- Notification to development team
- Security team alert for repeated failures
- Metrics collection for security posture tracking

## Regular Updates

Security gates are updated regularly to:
- Add new vulnerability patterns
- Update security scanning tools
- Incorporate lessons learned from security incidents
- Align with evolving security best practices

## Testing Security Gates Locally

Before submitting a PR, developers can run security checks locally:

```bash
# Install security tools
pip install bandit ruff mypy portalocker pytest pyyaml

# Check for HTTP URLs
grep -r --include="*.py" --exclude-dir="test*" '"http://' src/production/

# Check for pickle usage
grep -r --include="*.py" 'pickle\.loads' src/

# Run Bandit security scan
bandit -r src/production/ -l -i -f txt

# Run Ruff critical checks
ruff check src/production/ --select E9,F63,F7,F82

# Run security tests
python -m pytest tmp/security/ -v
```

## Contact and Support

For questions about security gates:
- Security Team: security@aivillage.dev
- Development Team: dev@aivillage.dev
- Documentation: https://github.com/AIVillage/security-docs

**Remember: Security gates exist to protect production systems and user data. They should be respected and properly addressed rather than bypassed.**
