# AIVillage Security Test Suite

Comprehensive security testing framework for AIVillage with behavioral testing approach and connascence-compliant security validation.

## Overview

This security test suite validates all security implementations deployed in AIVillage, including:

- **Vulnerability Reporting Workflow** - Tests SECURITY.md process and escalation procedures
- **GitHub Security Templates** - Validates issue/PR security integration and threat modeling
- **Dependency Auditing Pipeline** - Comprehensive SCA scanning across ~2,927 dependencies
- **SBOM Generation & Signing** - Cryptographic artifact integrity and supply chain security
- **Admin Interface Security** - Localhost binding, MFA, and access controls
- **Security Boundaries** - Connascence-compliant security coupling and isolation
- **GrokFast ML Security** - ML optimization security and model integrity validation
- **Attack Prevention** - Negative testing for injection, XSS, path traversal, and other attacks
- **Performance Overhead** - Security control performance impact validation
- **Governance Compliance** - GDPR, COPPA, FERPA, OWASP Top 10 framework adherence

## Architecture

### Test Categories

```
tests/security/
├── unit/                    # Individual security component tests
│   ├── test_vulnerability_reporting.py    # SECURITY.md workflow
│   ├── test_security_templates.py         # GitHub integration
│   ├── test_dependency_auditing.py        # SCA pipeline
│   ├── test_sbom_generation.py            # SBOM & signing
│   ├── test_admin_security.py             # Admin interfaces
│   ├── test_boundary_security.py          # Security boundaries
│   └── test_grokfast_security.py          # ML security
├── integration/             # End-to-end workflow tests
│   └── test_security_workflows.py         # Cross-component integration
├── performance/             # Security overhead validation
│   └── test_security_overhead.py          # Performance impact tests
├── compliance/              # Governance framework tests
│   └── test_governance_framework.py       # GDPR/COPPA/FERPA/OWASP
├── negative/                # Attack prevention tests
│   └── test_attack_prevention.py          # Security control effectiveness
└── run_security_tests.py   # Comprehensive test runner
```

### Testing Principles

**Behavioral Testing Approach**
- Tests security *contracts* and *guarantees*, not internal implementation
- Validates security *behavior* under various conditions
- Ensures security controls *prevent* malicious actions
- Focuses on *outcomes* rather than implementation details

**Connascence Compliance**
- Security tests avoid coupling violations
- Strong connascence kept local within test modules
- Weak connascence used for cross-component integration
- Security boundaries tested without implementation coupling

**Security-First Design**
- All tests validate security properties and guarantees
- Negative testing ensures attack prevention effectiveness
- Performance tests ensure security overhead remains acceptable
- Compliance tests validate regulatory framework adherence

## Quick Start

### Run All Security Tests

```bash
# Run complete security test suite
python tests/security/run_security_tests.py

# Generate HTML report
python tests/security/run_security_tests.py --report-format html --output-dir reports/

# Verbose output with detailed results
python tests/security/run_security_tests.py --verbose
```

### Run Specific Test Categories

```bash
# Unit tests only
python tests/security/run_security_tests.py --category unit

# Attack prevention tests
python tests/security/run_security_tests.py --category negative

# Performance validation
python tests/security/run_security_tests.py --category performance

# Compliance validation
python tests/security/run_security_tests.py --category compliance
```

### Fast Test Execution

```bash
# Quick validation (fail-fast mode)
python tests/security/run_security_tests.py --fast

# CI/CD integration
python tests/security/run_security_tests.py --fast --report-format json
```

## Test Categories Detail

### 1. Unit Tests (`tests/security/unit/`)

Individual security component validation with behavioral testing approach.

**Vulnerability Reporting (`test_vulnerability_reporting.py`)**
- Tests SECURITY.md workflow implementation
- Validates vulnerability classification and escalation
- Ensures proper SLA compliance (15min critical, 1hr high)
- Verifies audit logging and confidentiality

**Security Templates (`test_security_templates.py`)**
- Validates GitHub issue/PR security integration
- Tests threat modeling workflow automation
- Ensures security checklist completeness
- Validates security automation workflow detection

**Dependency Auditing (`test_dependency_auditing.py`)**
- Tests comprehensive SCA scanning across ecosystems
- Validates vulnerability detection and risk assessment
- Ensures ~2,927 dependency scalability
- Tests security gate integration thresholds

**SBOM Generation (`test_sbom_generation.py`)**
- Validates SBOM completeness and structure
- Tests cryptographic signing workflow
- Ensures artifact integrity verification
- Validates supply chain security properties

**Admin Security (`test_admin_security.py`)**
- Tests localhost-only binding enforcement
- Validates MFA requirement implementation
- Ensures session security and timeout handling
- Tests privilege escalation prevention

**Security Boundaries (`test_boundary_security.py`)**
- Validates security level enforcement
- Tests connascence-compliant coupling patterns
- Ensures cross-boundary security controls
- Validates trust boundary communication

**GrokFast ML Security (`test_grokfast_security.py`)**
- Tests ML optimization security controls
- Validates model integrity verification
- Ensures gradient validation and attack prevention
- Tests ML model access control enforcement

### 2. Integration Tests (`tests/security/integration/`)

End-to-end security workflow validation.

**Security Workflows (`test_security_workflows.py`)**
- Tests vulnerability-to-resolution complete workflow
- Validates cross-component security integration
- Ensures SBOM generation triggers from dependency scans
- Tests admin approval integration for critical vulnerabilities

### 3. Performance Tests (`tests/security/performance/`)

Security overhead and scalability validation.

**Security Overhead (`test_security_overhead.py`)**
- Authentication latency benchmarking (≤100ms threshold)
- Authorization check performance (≤50ms threshold)
- Cryptographic throughput validation (≥100 Mbps)
- Audit logging overhead measurement (≤5% impact)
- Concurrent security operation scalability

### 4. Compliance Tests (`tests/security/compliance/`)

Governance framework and regulatory compliance validation.

**Governance Framework (`test_governance_framework.py`)**
- **GDPR Compliance**: Data processing lawfulness, minimization, erasure rights
- **COPPA Compliance**: Parental consent, child data collection limits
- **FERPA Compliance**: Educational records protection, access controls
- **OWASP Top 10**: Injection prevention, authentication, sensitive data protection
- Multi-framework compliance scoring and recommendations

### 5. Negative Tests (`tests/security/negative/`)

Attack prevention and security boundary enforcement.

**Attack Prevention (`test_attack_prevention.py`)**
- **SQL Injection**: Input sanitization and parameterized query validation
- **XSS Prevention**: Output encoding and content sanitization
- **Command Injection**: Input validation and command execution protection
- **Path Traversal**: Directory traversal attack prevention
- **DoS Attacks**: Rate limiting and resource exhaustion protection
- **Data Exfiltration**: Suspicious activity detection and prevention
- **Privilege Escalation**: Role-based access control enforcement

## Security Test Results Interpretation

### Security Score Calculation

```
Security Score = (Successful Tests / Total Tests) × Weight Factor

Weight Factors by Assessment Level:
- EXCELLENT (≥95%): 95 + (rate-95) × 0.5  [Max ~97.5]
- GOOD (≥90%): rate × 1.0                  [90-95]
- ACCEPTABLE (≥80%): rate × 0.9            [72-85.5]
- NEEDS_IMPROVEMENT (≥70%): rate × 0.8     [56-72]
- CRITICAL (<70%): rate × 0.7              [<56]
```

### Risk Level Determination

- **LOW**: Excellent/Good assessment levels
- **MEDIUM**: Acceptable assessment level  
- **HIGH**: Needs improvement assessment level
- **CRITICAL**: Critical assessment level or security score <56

### Compliance Status

- **FULLY_COMPLIANT**: ≥95% test success rate
- **SUBSTANTIALLY_COMPLIANT**: ≥85% test success rate
- **PARTIALLY_COMPLIANT**: ≥70% test success rate
- **NON_COMPLIANT**: <70% test success rate

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Security Tests
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements.txt
      
      - name: Run security tests
        run: |
          python tests/security/run_security_tests.py \
            --fast \
            --report-format json \
            --output-dir reports/security
      
      - name: Upload security report
        uses: actions/upload-artifact@v3
        with:
          name: security-test-report
          path: reports/security/
```

### Security Gate Integration

```bash
# Security gate validation
python tests/security/run_security_tests.py --fast
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "❌ Security tests failed - blocking deployment"
    exit 1
else
    echo "✅ Security tests passed - deployment approved"
fi
```

## Test Development Guidelines

### Writing Security Tests

1. **Focus on Behavior**: Test security *contracts* and *guarantees*
2. **Avoid Implementation Coupling**: Test *what* not *how*
3. **Use Descriptive Names**: Test names should explain security requirement
4. **Validate Security Properties**: Assert security guarantees are maintained
5. **Include Negative Cases**: Test that security controls *prevent* attacks

### Example Test Structure

```python
def test_security_contract_description(self):
    \"\"\"
    Security Contract: [Clear statement of security requirement]
    Tests [specific security behavior] without coupling to implementation.
    \"\"\"
    # Arrange - Set up security context
    security_context = create_security_context()
    
    # Act - Perform security-relevant operation
    result = security_context.perform_operation()
    
    # Assert - Verify security guarantees
    self.assertTrue(result["security_validated"],
                   "Security validation must succeed")
    self.assertIn("audit_logged", result,
                 "Security operations must be audit logged")
```

### Connascence Guidelines for Security Tests

1. **Keep Strong Connascence Local**: Within same test class/method only
2. **Use Weak Connascence**: Name-based coupling across components
3. **Avoid Position Coupling**: Use keyword arguments for multi-parameter calls
4. **Eliminate Magic Values**: Use constants/enums for security levels
5. **Minimize Algorithm Coupling**: Test behavior, not implementation algorithms

## Performance Benchmarks

### Expected Performance Thresholds

| Security Operation | Threshold | Test Category |
|-------------------|-----------|---------------|
| Authentication | ≤100ms | Performance |
| Authorization | ≤50ms | Performance |
| Encryption Throughput | ≥100 Mbps | Performance |
| Vulnerability Scan | ≤30s per ecosystem | Performance |
| SBOM Generation | ≤10s for 100 components | Performance |
| Audit Logging Overhead | ≤5% performance impact | Performance |

### Scalability Validation

- **Concurrent Operations**: 20+ threads with 95%+ success rate
- **Large Dependency Sets**: ~2,927 dependencies scanned efficiently
- **High Request Rates**: DoS protection at 100+ requests/minute threshold
- **Memory Efficiency**: Security operations <50MB memory increase

## Security Implementation Status

Based on comprehensive testing, the following security implementations are validated:

### ✅ Fully Implemented & Tested

- **Vulnerability Reporting**: Complete SECURITY.md workflow with SLA compliance
- **Issue/PR Security**: GitHub template integration with threat modeling
- **Dependency Auditing**: SCA scanning across Python/JavaScript/Rust ecosystems
- **SBOM & Signing**: Cryptographic artifact integrity with verification
- **Admin Interface**: Localhost binding with MFA enforcement
- **Security Boundaries**: Connascence-compliant access controls
- **ML Security**: GrokFast optimization with model integrity validation
- **Attack Prevention**: Comprehensive injection/XSS/traversal/DoS protection
- **Performance**: Security overhead within acceptable limits
- **Compliance**: GDPR/COPPA/FERPA/OWASP framework adherence

### 📈 Test Coverage Metrics

- **Total Security Tests**: 80+ comprehensive behavioral tests
- **Security Categories**: 5 categories (unit/integration/performance/compliance/negative)
- **Attack Vectors**: 12+ attack types with prevention validation
- **Compliance Frameworks**: 4 major frameworks (GDPR/COPPA/FERPA/OWASP)
- **Performance Benchmarks**: 6 critical security operation thresholds
- **Dependencies Covered**: ~2,927 dependencies across 3 ecosystems

## Troubleshooting

### Common Issues

**Test Import Errors**
```bash
# Ensure Python path includes project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
python tests/security/run_security_tests.py
```

**Performance Test Failures**
- Check system load during test execution
- Verify no resource-intensive processes running
- Consider adjusting performance thresholds for test environment

**Compliance Test Failures**
- Review system configuration for compliance requirements
- Ensure all required security features are implemented
- Check compliance rule validation logic

### Debug Mode

```bash
# Enable verbose debugging
python tests/security/run_security_tests.py --verbose --category unit

# Run specific test module
python -m pytest tests/security/unit/test_vulnerability_reporting.py -v
```

## Contributing

When adding new security tests:

1. Follow behavioral testing principles
2. Ensure connascence compliance
3. Add performance benchmarks for new security operations
4. Update compliance tests for new regulatory requirements
5. Include negative testing for new security controls

## Security Contact

For security-related questions about the test suite:
- **Security Team**: security@aivillage.dev
- **Test Issues**: Create issue with `security-tests` label
- **Urgent Security**: Follow SECURITY.md vulnerability reporting process

---

**Generated by AIVillage Security Test Suite v1.0**  
**Last Updated**: 2025-08-29  
**Maintained by**: AIVillage Security Team