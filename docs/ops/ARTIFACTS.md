# AIVillage Operational Artifacts Index

## Overview

This document provides a comprehensive index of all operational artifacts collected by the AIVillage CI/CD pipeline and operational monitoring systems. These artifacts provide visibility into system health, security posture, performance characteristics, and compliance status.

**Collection System**: `scripts/operational/collect_artifacts.py`
**Configuration**: `config/artifacts_collection.json`
**Validation**: `scripts/operational/validate_artifacts.py`
**GitHub Workflow**: `.github/workflows/artifacts-collection.yml`
**Makefile Targets**: `make artifacts`, `make artifacts-validate`

## Artifact Categories

### 📊 Coverage Artifacts

**Purpose**: Track test coverage and code quality metrics to ensure comprehensive testing.

| Artifact | Location | Format | Description |
|----------|----------|---------|-------------|
| Coverage XML | `artifacts/coverage/coverage.xml` | XML | Machine-readable coverage report |
| Coverage HTML | `artifacts/coverage/htmlcov/` | HTML | Human-readable coverage dashboard |
| Coverage JSON | `artifacts/coverage/coverage.json` | JSON | Programmatic coverage data |
| Pytest Coverage | `artifacts/coverage/pytest-coverage.xml` | XML | Pytest-specific coverage report |
| Pytest HTML | `artifacts/coverage/pytest-htmlcov/` | HTML | Pytest HTML coverage report |

**Thresholds**:
- Minimum Coverage: 80%
- Warning Threshold: <85%
- Target Coverage: >90%

**Usage**:
```bash
# Generate coverage artifacts
make test-coverage

# View HTML report
open artifacts/coverage/htmlcov/index.html

# Parse XML programmatically
python -c "import xml.etree.ElementTree as ET; print(ET.parse('artifacts/coverage/coverage.xml').getroot().get('line-rate'))"
```

### 🔒 Security Artifacts

**Purpose**: Identify security vulnerabilities, code quality issues, and compliance violations.

| Artifact | Location | Format | Tool | Description |
|----------|----------|---------|------|-------------|
| Bandit SAST | `artifacts/security/bandit-report.json` | JSON | Bandit | Static Application Security Testing |
| Safety Deps | `artifacts/security/safety-report.json` | JSON | Safety | Dependency vulnerability scanning |
| Semgrep SAST | `artifacts/security/semgrep-report.json` | JSON | Semgrep | Advanced static analysis |

**Severity Levels**:
- **HIGH**: Critical security issues requiring immediate attention
- **MEDIUM**: Important issues to address in next sprint
- **LOW**: Minor issues for future improvement

**Alert Thresholds**:
- HIGH severity issues: 0 (block deployment)
- MEDIUM severity issues: <5 (warning)
- Dependency vulnerabilities: <3 (warning)

**Usage**:
```bash
# Collect security artifacts
make artifacts-security

# Parse Bandit results
jq '.results[] | select(.issue_severity == "HIGH")' artifacts/security/bandit-report.json

# Check dependency vulnerabilities
jq '.vulnerabilities | length' artifacts/security/safety-report.json
```

### 📋 SBOM (Software Bill of Materials) Artifacts

**Purpose**: Maintain comprehensive inventory of software components for compliance and supply chain security.

| Artifact | Location | Format | Description |
|----------|----------|---------|-------------|
| SPDX SBOM | `artifacts/sbom/aivillage-sbom.spdx` | SPDX | Industry-standard SBOM format |
| CycloneDX SBOM | `artifacts/sbom/aivillage-sbom.json` | JSON | CycloneDX format SBOM |
| Basic SBOM | `artifacts/sbom/basic-sbom.txt` | Text | Simple pip freeze output |
| Pip Audit | `artifacts/sbom/pip-audit-sbom.json` | JSON | Pip-audit generated SBOM |

**Compliance Standards**:
- **NTIA SBOM Guidelines**: Minimum elements covered
- **Executive Order 14028**: Software supply chain security
- **NIST SSDF**: Secure Software Development Framework

**Usage**:
```bash
# Generate SBOM
make artifacts-sbom

# Validate SBOM completeness
python -c "
import json
with open('artifacts/sbom/aivillage-sbom.json') as f:
    sbom = json.load(f)
print(f'Components: {len(sbom.get(\"components\", []))}')
"

# Check for vulnerable dependencies
pip-audit --format=json --output=audit-results.json
```

### ⚡ Performance Artifacts

**Purpose**: Monitor system performance, identify regressions, and optimize resource usage.

| Artifact | Location | Format | Description |
|----------|----------|---------|-------------|
| Benchmark Results | `artifacts/performance/benchmark-results.json` | JSON | Pytest-benchmark output |
| Memory Profile | `artifacts/performance/memory-profile.txt` | Text | Memory usage profiling |
| Load Test Results | `artifacts/performance/load-test.json` | JSON | Load testing metrics |
| Performance Metrics | `artifacts/performance/metrics.json` | JSON | Custom performance metrics |

**Key Metrics**:
- **Execution Time**: Function/endpoint response times
- **Memory Usage**: Peak and average memory consumption
- **Throughput**: Requests/operations per second
- **Resource Utilization**: CPU, memory, disk, network

**Regression Detection**:
- Threshold: >10% performance degradation
- Baseline: Rolling 30-day average
- Alert on: >2 standard deviations from baseline

**Usage**:
```bash
# Run performance benchmarks
make artifacts-performance

# Analyze benchmark results
python -c "
import json
with open('artifacts/performance/benchmark-results.json') as f:
    data = json.load(f)
benchmarks = data.get('benchmarks', [])
avg_time = sum(b['stats']['mean'] for b in benchmarks) / len(benchmarks)
print(f'Average execution time: {avg_time:.4f}s')
"
```

### 📊 Quality Artifacts

**Purpose**: Assess code quality, maintainability, and technical debt.

| Artifact | Location | Format | Tool | Description |
|----------|----------|---------|------|-------------|
| Ruff Linting | `artifacts/quality/ruff-report.json` | JSON | Ruff | Fast Python linter results |
| MyPy Types | `artifacts/quality/mypy-report.txt` | Text | MyPy | Type checking results |
| Complexity | `artifacts/quality/complexity-report.json` | JSON | Radon | Cyclomatic complexity analysis |
| Hotspots | `artifacts/quality/hotspots-report.json` | JSON | Custom | Git churn × complexity analysis |

**Quality Metrics**:
- **Code Coverage**: >80% target
- **Cyclomatic Complexity**: <10 per function
- **Type Coverage**: >70% functions typed
- **Linting Issues**: <100 total violations

**Hotspot Analysis**:
- **High Risk**: Churn >20 + Complexity >15
- **Medium Risk**: Churn >10 + Complexity >10
- **Refactoring Priority**: Risk score × File importance

**Usage**:
```bash
# Collect quality artifacts
make artifacts-quality

# View hotspots summary
jq '.[] | select(.risk_level == "HIGH") | {file: .file_path, risk: .hotspot_score}' artifacts/quality/hotspots-report.json

# Check complexity trends
jq '.[] | select(.complexity > 10) | {function: .name, complexity: .complexity}' artifacts/quality/complexity-report.json
```

### 🐳 Container Security Artifacts

**Purpose**: Scan container images and configurations for vulnerabilities and misconfigurations.

| Artifact | Location | Format | Tool | Description |
|----------|----------|---------|------|-------------|
| Trivy Scan | `artifacts/containers/trivy-report.json` | JSON | Trivy | Container vulnerability scan |
| Grype Scan | `artifacts/containers/grype-report.json` | JSON | Grype | Container vulnerability analysis |
| Docker Config | `artifacts/containers/docker-config-scan.json` | JSON | Custom | Dockerfile security analysis |

**Vulnerability Severities**:
- **CRITICAL**: Immediate patching required
- **HIGH**: Patch within 7 days
- **MEDIUM**: Patch within 30 days
- **LOW**: Patch when convenient

**Container Best Practices**:
- Use minimal base images
- Run as non-root user
- Scan for known vulnerabilities
- Regular base image updates

**Usage**:
```bash
# Scan containers
make artifacts-container

# Critical vulnerabilities check
trivy image --severity CRITICAL,HIGH aivillage:latest

# Parse Trivy results
jq '.Results[].Vulnerabilities[] | select(.Severity == "CRITICAL")' artifacts/containers/trivy-report.json
```

### 📋 Compliance Artifacts

**Purpose**: Demonstrate compliance with regulatory frameworks and internal policies.

| Artifact | Location | Format | Framework | Description |
|----------|----------|---------|-----------|-------------|
| GDPR Report | `artifacts/compliance/gdpr-compliance.json` | JSON | GDPR | Privacy compliance assessment |
| SOC2 Report | `artifacts/compliance/soc2-compliance.json` | JSON | SOC2 | Trust services criteria |
| PCI Report | `artifacts/compliance/pci-compliance.json` | JSON | PCI DSS | Payment security compliance |
| Security Controls | `artifacts/compliance/security-controls.json` | JSON | Custom | Internal security controls |

**Compliance Frameworks**:
- **GDPR**: Data privacy and protection
- **SOC2 Type II**: Security, availability, integrity
- **PCI DSS**: Payment card data security
- **NIST CSF**: Cybersecurity framework

**Audit Trail**:
- All compliance artifacts timestamped
- Change tracking via Git history
- Evidence collection automated
- Regular compliance reviews

**Usage**:
```bash
# Generate compliance reports
make artifacts-compliance

# Check GDPR compliance status
jq '.checks | to_entries[] | select(.value != "implemented")' artifacts/compliance/gdpr-compliance.json

# SOC2 control effectiveness
jq '.trust_services_criteria | to_entries[] | select(.value == "partial")' artifacts/compliance/soc2-compliance.json
```

### 📊 Collection Reports

**Purpose**: Metadata and status information about the artifacts collection process.

| Artifact | Location | Format | Description |
|----------|----------|---------|-------------|
| Collection Report | `artifacts/reports/collection_report_*.json` | JSON | Collection status and metadata |
| Validation Report | `artifacts/reports/validation-report.json` | JSON | Artifact validation results |
| Collection Metrics | `artifacts/reports/collection-metrics.json` | JSON | Performance metrics for collection |

**Report Contents**:
- Collection timestamp and duration
- Success/failure rates by category
- File sizes and checksums
- Validation results and thresholds

## CI/CD Integration

### GitHub Actions Workflow

The artifacts collection is integrated into the CI/CD pipeline via GitHub Actions:

```yaml
# .github/workflows/artifacts-collection.yml
name: Operational Artifacts Collection
on:
  push: [main, develop]
  pull_request: [main]
  schedule: ['0 2 * * *']  # Daily at 2 AM UTC
```

**Workflow Steps**:
1. **Setup**: Install dependencies and tools
2. **Test**: Run test suite with coverage
3. **Collect**: Execute artifacts collection
4. **Upload**: Store artifacts in GitHub Actions
5. **Notify**: Send status notifications
6. **Validate**: Check artifact completeness

### Makefile Integration

Artifacts collection is available via convenient Makefile targets:

```bash
# Collect all artifacts
make artifacts

# Individual categories
make artifacts-security    # Security scanning only
make artifacts-sbom       # SBOM generation only
make artifacts-performance # Performance benchmarks only
make artifacts-quality    # Code quality analysis only

# Validation
make artifacts-validate   # Validate collected artifacts
make artifacts-all       # Collect all + validate
```

### Local Development

For local development and testing:

```bash
# Install additional tools
pip install bandit safety semgrep trivy grype

# Run collection locally
python scripts/operational/collect_artifacts.py \
  --output-dir local-artifacts \
  --config config/artifacts_collection.json \
  --verbose

# Validate results
python scripts/operational/validate_artifacts.py \
  --artifacts-dir local-artifacts \
  --strict
```

## Retention and Storage

### Retention Policy

- **GitHub Actions**: 30 days (configurable)
- **Local Artifacts**: 7 days (configurable via `retention_days`)
- **Critical Security**: 90 days minimum
- **Compliance Reports**: 1 year minimum

### Storage Locations

- **CI/CD**: GitHub Actions artifact storage
- **Local**: `artifacts/` directory (gitignored)
- **Production**: S3/Azure Blob/GCS buckets
- **Long-term**: Compliance archival system

### Cleanup Process

Automatic cleanup prevents storage bloat:

```python
# Automatic cleanup in collect_artifacts.py
retention_days = config.get("retention_days", 30)
cutoff_time = time.time() - (retention_days * 24 * 3600)

for file_path in self.output_dir.rglob("*"):
    if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
        file_path.unlink()
```

## Monitoring and Alerting

### Collection Health

Monitor the health of the artifacts collection system:

- **Success Rate**: >95% artifacts collected successfully
- **Collection Time**: <30 minutes total duration
- **Artifact Size**: <500MB total size warning
- **Validation**: All critical artifacts present

### Alert Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Success Rate | <95% | <90% |
| Collection Duration | >20min | >30min |
| Critical Security Issues | >0 | >3 |
| Coverage Percentage | <80% | <70% |
| SBOM Components | <50 | <25 |

### Notification Channels

- **Console**: Local development feedback
- **GitHub Actions**: PR/commit status checks
- **Slack/Email**: Production alerts (configurable)
- **Dashboard**: Real-time monitoring (future)

## API and Programmatic Access

### Parsing Artifacts

Common patterns for programmatic artifact access:

#### Coverage Analysis
```python
import xml.etree.ElementTree as ET

# Parse coverage XML
tree = ET.parse('artifacts/coverage/coverage.xml')
coverage_pct = float(tree.getroot().get('line-rate')) * 100
print(f"Coverage: {coverage_pct:.1f}%")
```

#### Security Issues
```python
import json

# Parse Bandit results
with open('artifacts/security/bandit-report.json') as f:
    bandit = json.load(f)

high_issues = [r for r in bandit['results'] if r['issue_severity'] == 'HIGH']
print(f"High severity issues: {len(high_issues)}")
```

#### Performance Trends
```python
import json

# Analyze benchmark results
with open('artifacts/performance/benchmark-results.json') as f:
    benchmarks = json.load(f)

for benchmark in benchmarks['benchmarks']:
    name = benchmark['name']
    mean_time = benchmark['stats']['mean']
    print(f"{name}: {mean_time:.4f}s")
```

### Integration Libraries

For advanced integration, use these patterns:

```python
from pathlib import Path
import json

class ArtifactsReader:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = Path(artifacts_dir)

    def get_coverage_percentage(self):
        """Extract coverage percentage from XML report."""
        import xml.etree.ElementTree as ET
        xml_file = self.artifacts_dir / "coverage/coverage.xml"
        if xml_file.exists():
            tree = ET.parse(xml_file)
            return float(tree.getroot().get('line-rate', 0)) * 100
        return 0.0

    def get_security_issues(self, severity="HIGH"):
        """Get security issues by severity."""
        bandit_file = self.artifacts_dir / "security/bandit-report.json"
        if bandit_file.exists():
            with open(bandit_file) as f:
                data = json.load(f)
            return [r for r in data.get('results', [])
                   if r.get('issue_severity') == severity]
        return []

    def get_performance_baseline(self):
        """Calculate performance baseline from benchmarks."""
        bench_file = self.artifacts_dir / "performance/benchmark-results.json"
        if bench_file.exists():
            with open(bench_file) as f:
                data = json.load(f)
            benchmarks = data.get('benchmarks', [])
            if benchmarks:
                total_time = sum(b['stats']['mean'] for b in benchmarks)
                return total_time / len(benchmarks)
        return 0.0
```

## Best Practices

### Collection Guidelines

1. **Parallel Execution**: Use `--parallel` for faster collection
2. **Error Handling**: Use `|| true` for non-critical tools
3. **Resource Limits**: Set timeouts for long-running analyses
4. **Incremental**: Only collect changed artifacts when possible
5. **Validation**: Always validate collected artifacts

### Tool Configuration

1. **Bandit**: Configure exclusions in `pyproject.toml`
2. **Safety**: Use `.safety-policy.json` for vulnerability policies
3. **Coverage**: Set thresholds in `pyproject.toml`
4. **MyPy**: Use `mypy.ini` for type checking configuration
5. **Ruff**: Configure rules in `pyproject.toml`

### Security Considerations

1. **Sensitive Data**: Never collect credentials or secrets
2. **Access Control**: Restrict artifact access appropriately
3. **Encryption**: Encrypt artifacts in transit and at rest
4. **Audit Trail**: Maintain logs of artifact access
5. **Compliance**: Follow data retention regulations

### Performance Optimization

1. **Parallel Collection**: Run independent tools concurrently
2. **Incremental Updates**: Only regenerate changed artifacts
3. **Compression**: Use compression for large artifacts
4. **Caching**: Cache tool downloads and intermediate results
5. **Resource Monitoring**: Monitor collection resource usage

## Troubleshooting

### Common Issues

#### Collection Failures
- **Tool Missing**: Install required tools (bandit, safety, etc.)
- **Permission Errors**: Check file/directory permissions
- **Timeout Issues**: Increase timeout values in configuration
- **Memory Exhaustion**: Run collection in smaller batches

#### Validation Errors
- **Missing Artifacts**: Check tool execution success
- **Format Errors**: Verify tool output formats
- **Threshold Violations**: Review quality/security thresholds
- **Corruption**: Validate file checksums

#### Performance Issues
- **Slow Collection**: Enable parallel execution
- **Large Artifacts**: Increase artifact size limits
- **Tool Hangs**: Set appropriate timeouts
- **Resource Contention**: Run during off-peak hours

### Debug Commands

```bash
# Verbose collection
python scripts/operational/collect_artifacts.py --verbose --output-dir debug-artifacts

# Individual tool testing
bandit -r packages/ -f json
safety check --json
pytest tests/benchmarks/ --benchmark-json=test.json

# Validation debugging
python scripts/operational/validate_artifacts.py --artifacts-dir debug-artifacts --strict

# Check tool availability
which bandit safety mypy ruff trivy grype
```

### Log Analysis

Check logs for collection issues:

```bash
# Collection logs
tail -f artifacts_collection.log

# CI logs
gh run view --log  # GitHub CLI

# Local debugging
python -m pdb scripts/operational/collect_artifacts.py
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-08-19 | Initial operational artifacts system |
| 1.1 | TBD | Enhanced performance monitoring |
| 1.2 | TBD | Additional compliance frameworks |

---

This artifacts index provides comprehensive documentation for the AIVillage operational artifacts system. For questions or issues, please refer to the troubleshooting section or create a GitHub issue.
