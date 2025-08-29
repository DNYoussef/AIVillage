# AIVillage Security Architecture Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing the security architecture patterns designed for AIVillage. The implementation focuses on secure boundaries, localhost-only admin interfaces, threat modeling integration, and comprehensive SBOM/artifact signing.

## Quick Start

### Prerequisites
- Python 3.9+
- FastAPI
- Cryptography library
- Access to vulnerability databases (OSV, NVD)

### Installation

```bash
# Install security dependencies
pip install fastapi uvicorn cryptography pydantic requests semver toml

# Create directory structure
mkdir -p src/security/{boundaries,admin,threat_modeling,sbom}
mkdir -p logs keys data/threat_models
```

## Implementation Phases

## Phase 1: Security Boundaries (Week 1-2)

### 1.1 Deploy Secure Boundary Contracts

**File**: `src/security/boundaries/secure_boundary_contracts.py`

**Key Implementation Steps:**

1. **Initialize Security Services**:
```python
# Example integration with existing auth system
from src.security.boundaries.secure_boundary_contracts import (
    SecurityBoundaryFactory, SecurityContext, SecurityLevel
)

# Create security service implementations
class AIVillageAuthService:
    async def validate_token(self, token: str) -> bool:
        # Integrate with your existing JWT validation
        pass
    
    async def validate_mfa(self, context: SecurityContext) -> bool:
        # Integrate with your MFA system (TOTP, SMS, etc.)
        pass

# Initialize factory
auth_service = AIVillageAuthService()
authz_service = AIVillageAuthzService()
audit_service = AIVillageAuditService()
threat_service = AIVillageThreatService()

factory = SecurityBoundaryFactory(
    auth_service, authz_service, audit_service, threat_service
)
```

2. **Apply to Existing Components**:
```python
# Wrap existing admin endpoints
admin_boundary = factory.create_admin_boundary()

@app.get("/admin/users")
async def get_users(request: Request):
    context = extract_security_context(request)
    
    return await admin_boundary.execute_secured(
        context,
        lambda: existing_get_users_logic()
    )
```

### 1.2 Fix Admin Interface Bindings

**CRITICAL FIXES REQUIRED:**

1. **Identify All Admin Interfaces**:
```bash
# Find all admin servers binding to 0.0.0.0
grep -r "0\.0\.0\.0" . --include="*.py" | grep -i admin
```

2. **Apply Fixes**:
```python
# BEFORE (INSECURE)
uvicorn.run(app, host="0.0.0.0", port=3006)

# AFTER (SECURE)
uvicorn.run(app, host="127.0.0.1", port=3006)
```

3. **Validate Changes**:
```bash
# Test that admin interface is not accessible externally
curl http://YOUR_EXTERNAL_IP:3006  # Should fail
curl http://127.0.0.1:3006         # Should work from localhost
```

### 1.3 Update Admin Servers

**File**: Replace existing admin servers with `src/security/admin/localhost_only_server.py`

**Migration Steps:**

1. **Backup Existing Admin Servers**:
```bash
cp infrastructure/gateway/admin_server.py infrastructure/gateway/admin_server.py.backup
```

2. **Update Server Implementation**:
```python
# Use the new secure admin server
from src.security.admin.localhost_only_server import SecureAdminServer

# Initialize with security factory
server = SecureAdminServer(
    security_boundary_factory=factory,
    bind_interface="127.0.0.1",  # NEVER 0.0.0.0
    port=3006,
    debug=False  # Set to False in production
)

await server.start_server()
```

3. **Configure CORS Properly**:
```python
# NEVER use wildcard for admin interfaces
allow_origins=[
    "http://127.0.0.1:3000",
    "https://127.0.0.1:3000",
    "http://localhost:3000", 
    "https://localhost:3000"
]
```

## Phase 2: Threat Model Integration (Week 3-4)

### 2.1 Deploy Threat Analysis System

**File**: `src/security/threat_modeling/development_integration.py`

1. **Initialize Threat Database**:
```python
from src.security.threat_modeling.development_integration import (
    ThreatDatabase, ThreatAnalyzer, GitHubIntegration
)

# Initialize components
threat_db = ThreatDatabase("data/threat_models")
analyzer = ThreatAnalyzer(threat_db)
github_integration = GitHubIntegration(analyzer)
```

2. **Create GitHub Webhook Handler**:
```python
@app.post("/webhooks/github/pr")
async def handle_pr_webhook(request: Request):
    payload = await request.json()
    
    if payload.get("action") in ["opened", "synchronize"]:
        # Analyze PR for security impact
        analysis = await github_integration.analyze_pull_request(
            payload["pull_request"]
        )
        
        # Post comment to PR
        comment = github_integration.format_github_comment(analysis)
        # Use GitHub API to post comment
        
        # Add labels if high risk
        if analysis["overall_risk"] in ["critical", "high"]:
            # Add security review labels
            pass
    
    return {"status": "processed"}
```

3. **Integrate with CI/CD Pipeline**:
```yaml
# .github/workflows/security-analysis.yml
name: Security Analysis
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  threat-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Threat Analysis
        run: |
          python -m src.security.threat_modeling.development_integration \
            --pr-number ${{ github.event.number }} \
            --output threat-analysis.json
      
      - name: Upload Analysis
        uses: actions/upload-artifact@v3
        with:
          name: threat-analysis
          path: threat-analysis.json
```

### 2.2 Configure Threat Patterns

**File**: `data/threat_models/threat_patterns.yaml`

```yaml
admin_interface_exposure:
  category: elevation_of_privilege
  indicators:
    - "0.0.0.0"
    - "bind.*0.0.0.0"
    - "host.*0.0.0.0"
  description: "Admin interface exposed to all network interfaces"
  impact: critical
  likelihood: high
  mitigation: "Bind admin interfaces to localhost only (127.0.0.1)"

cors_wildcard:
  category: tampering
  indicators:
    - "allow_origins.*\\*"
    - "cors.*\\*"
  description: "CORS configuration allows all origins"
  impact: high
  likelihood: medium
  mitigation: "Specify explicit allowed origins"

hardcoded_secrets:
  category: information_disclosure
  indicators:
    - "password.*="
    - "secret.*="
    - "token.*="
    - "key.*="
  description: "Potential hardcoded credentials in code"
  impact: critical
  likelihood: medium
  mitigation: "Use environment variables or secure credential management"
```

## Phase 3: SBOM & Artifact Signing (Week 5-6)

### 3.1 Deploy SBOM Generation

**File**: `src/security/sbom/enhanced_sbom_generator.py`

1. **Generate Initial SBOM**:
```bash
# Generate SBOM for the project
python -m src.security.sbom.enhanced_sbom_generator \
    . \
    --output artifacts/aivillage-sbom.json
```

2. **Integrate with Build Process**:
```python
# build.py
from src.security.sbom.enhanced_sbom_generator import EnhancedSBOMGenerator

async def build_with_sbom():
    # Run normal build process
    build_artifacts = await run_build()
    
    # Generate SBOM
    generator = EnhancedSBOMGenerator()
    sbom = await generator.generate_comprehensive_sbom(
        Path("."),
        include_vulnerabilities=True,
        include_licenses=True
    )
    
    # Save and sign SBOM
    sbom_path = await generator.save_sbom(
        sbom,
        Path("artifacts/aivillage-sbom.json"),
        sign=True
    )
    
    return build_artifacts, sbom_path
```

3. **Automate Vulnerability Scanning**:
```yaml
# .github/workflows/sbom-scan.yml
name: SBOM and Vulnerability Scan
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  push:
    branches: [main]

jobs:
  sbom-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Generate SBOM
        run: |
          python -m src.security.sbom.enhanced_sbom_generator \
            . \
            --output sbom/aivillage-sbom.json
      
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom/
      
      - name: Check for Critical Vulnerabilities
        run: |
          python -c "
          import json
          with open('sbom/aivillage-sbom.json') as f:
              sbom = json.load(f)
          
          critical = sbom['security_analysis']['vulnerability_summary']['critical_vulnerabilities']
          high = sbom['security_analysis']['vulnerability_summary']['high_vulnerabilities']
          
          if critical > 0:
              print(f'❌ {critical} critical vulnerabilities found!')
              exit(1)
          elif high > 5:
              print(f'⚠️  {high} high-severity vulnerabilities found!')
              exit(1)
          else:
              print('✅ No critical vulnerabilities found')
          "
```

### 3.2 Implement Artifact Signing

1. **Generate Signing Keys**:
```bash
# Create keys directory
mkdir -p keys/

# Generate key pair (done automatically by ArtifactSigner)
python -c "
from src.security.sbom.enhanced_sbom_generator import ArtifactSigner
signer = ArtifactSigner()
print('Signing keys generated in keys/ directory')
"
```

2. **Sign Release Artifacts**:
```python
# release.py
from src.security.sbom.enhanced_sbom_generator import ArtifactSigner

def sign_release_artifacts(artifact_paths: List[Path]):
    signer = ArtifactSigner("keys/signing_private.pem")
    
    signed_artifacts = []
    for artifact_path in artifact_paths:
        signature_data = signer.sign_file(
            artifact_path,
            metadata={
                "release_version": "1.0.0",
                "build_timestamp": datetime.utcnow().isoformat(),
                "build_environment": "secure-ci"
            }
        )
        signed_artifacts.append(signature_data)
    
    return signed_artifacts
```

3. **Verify Artifacts**:
```python
# verify.py
def verify_release_artifacts(artifact_paths: List[Path]):
    signer = ArtifactSigner()
    
    for artifact_path in artifact_paths:
        result = signer.verify_file(artifact_path)
        
        if result["verified"]:
            print(f"✅ {artifact_path} - signature valid")
        else:
            print(f"❌ {artifact_path} - {result['error']}")
            return False
    
    return True
```

## Phase 4: Production Deployment

### 4.1 Environment Configuration

**Development (`config/security/dev.yaml`)**:
```yaml
security:
  admin_binding: "127.0.0.1"
  debug_mode: true
  mfa_required: false
  session_timeout_minutes: 60
  audit_logging: true

threat_modeling:
  github_integration: true
  auto_analysis: true
  comment_on_prs: true

sbom:
  vulnerability_scanning: true
  daily_scans: true
  sign_artifacts: false
```

**Production (`config/security/prod.yaml`)**:
```yaml
security:
  admin_binding: "127.0.0.1"  # NEVER change this
  debug_mode: false
  mfa_required: true
  session_timeout_minutes: 30
  audit_logging: true
  audit_retention_days: 365

threat_modeling:
  github_integration: true
  auto_analysis: true
  comment_on_prs: true
  block_high_risk_prs: true

sbom:
  vulnerability_scanning: true
  daily_scans: true
  sign_artifacts: true
  vulnerability_threshold:
    critical: 0
    high: 2
```

### 4.2 Docker Configuration

**Dockerfile.secure-admin**:
```dockerfile
FROM python:3.11-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 aivillage
USER aivillage

WORKDIR /app

# Copy security components
COPY src/security/ ./src/security/
COPY requirements-security.txt .

RUN pip install -r requirements-security.txt

# Bind to localhost only - never expose admin externally
EXPOSE 127.0.0.1:3006

CMD ["python", "-m", "src.security.admin.localhost_only_server"]
```

**docker-compose.security.yml**:
```yaml
version: '3.8'

services:
  secure-admin:
    build:
      context: .
      dockerfile: Dockerfile.secure-admin
    ports:
      - "127.0.0.1:3006:3006"  # Localhost only
    volumes:
      - ./logs:/app/logs
      - ./keys:/app/keys:ro
    environment:
      - SECURITY_CONFIG=/app/config/security/prod.yaml
    networks:
      - internal

networks:
  internal:
    driver: bridge
    internal: true  # No external access
```

### 4.3 Monitoring and Alerting

**Prometheus Metrics**:
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Security metrics
security_violations = Counter(
    'security_violations_total',
    'Total security violations',
    ['violation_type', 'component']
)

admin_access_attempts = Counter(
    'admin_access_attempts_total',
    'Admin access attempts',
    ['source_ip', 'result']
)

threat_analysis_duration = Histogram(
    'threat_analysis_duration_seconds',
    'Time spent on threat analysis'
)

vulnerability_count = Gauge(
    'vulnerabilities_total',
    'Total vulnerabilities found',
    ['severity']
)

# Usage in security components
def record_security_violation(violation_type: str, component: str):
    security_violations.labels(
        violation_type=violation_type,
        component=component
    ).inc()
```

**Grafana Dashboard** (`monitoring/security-dashboard.json`):
```json
{
  "dashboard": {
    "title": "AIVillage Security Dashboard",
    "panels": [
      {
        "title": "Security Violations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(security_violations_total[5m])",
            "legendFormat": "{{violation_type}}"
          }
        ]
      },
      {
        "title": "Admin Access Attempts",
        "type": "graph", 
        "targets": [
          {
            "expr": "rate(admin_access_attempts_total[5m])",
            "legendFormat": "{{result}}"
          }
        ]
      },
      {
        "title": "Vulnerability Counts",
        "type": "stat",
        "targets": [
          {
            "expr": "vulnerabilities_total",
            "legendFormat": "{{severity}}"
          }
        ]
      }
    ]
  }
}
```

## Validation and Testing

### Security Testing Checklist

- [ ] **Admin Interface Security**
  - [ ] Admin interfaces bind to 127.0.0.1 only
  - [ ] External access attempts are blocked
  - [ ] MFA is required and functional
  - [ ] Session management works correctly
  - [ ] Audit logging captures all admin actions

- [ ] **Boundary Security**
  - [ ] Security boundaries enforce access controls
  - [ ] Connascence violations are detected
  - [ ] Dependency injection works correctly
  - [ ] Error handling doesn't leak information

- [ ] **Threat Modeling**
  - [ ] GitHub integration posts PR comments
  - [ ] High-risk patterns are detected
  - [ ] Security labels are applied correctly
  - [ ] Analysis results are accurate

- [ ] **SBOM and Signing**
  - [ ] SBOM includes all components
  - [ ] Vulnerability scanning is complete
  - [ ] Artifacts are signed correctly
  - [ ] Signature verification works

### Penetration Testing

Run these tests to validate security:

```bash
# Test 1: Admin interface external access
curl -v http://YOUR_EXTERNAL_IP:3006/admin/status
# Expected: Connection refused or timeout

# Test 2: Admin interface localhost access
curl -v http://127.0.0.1:3006/health
# Expected: 200 OK

# Test 3: CORS validation
curl -H "Origin: https://evil.com" http://127.0.0.1:3006/admin/status
# Expected: CORS error

# Test 4: Security headers
curl -I http://127.0.0.1:3006/health
# Expected: Security headers present (X-Frame-Options, etc.)

# Test 5: Threat model analysis
# Create PR with security issues and verify comment is posted

# Test 6: SBOM generation
python -m src.security.sbom.enhanced_sbom_generator . --output test-sbom.json
# Expected: SBOM file created with vulnerability data

# Test 7: Artifact signing
echo "test data" > test-artifact.txt
python -c "
from src.security.sbom.enhanced_sbom_generator import ArtifactSigner
signer = ArtifactSigner()
signer.sign_file(Path('test-artifact.txt'))
result = signer.verify_file(Path('test-artifact.txt'))
print('Verification result:', result)
"
# Expected: Verification successful
```

## Troubleshooting

### Common Issues

1. **Admin Interface Still Accessible Externally**
   - Check all server configurations for `0.0.0.0` bindings
   - Verify firewall rules block external access
   - Ensure Docker port mappings use `127.0.0.1:`

2. **MFA Not Working**
   - Verify MFA service integration
   - Check session management configuration
   - Validate TOTP/SMS service endpoints

3. **Threat Analysis Not Triggering**
   - Verify GitHub webhook configuration
   - Check webhook endpoint accessibility
   - Validate GitHub API permissions

4. **Vulnerability Scanning Failing**
   - Check internet connectivity to OSV/NVD APIs
   - Verify API rate limits not exceeded
   - Ensure component parsing is correct

5. **Artifact Signing Errors**
   - Verify signing keys exist and are readable
   - Check file permissions on key files
   - Ensure cryptography library is installed

### Support and Maintenance

- **Log Locations**: `logs/admin_audit.log`, `logs/security_violations.log`
- **Configuration**: `config/security/`
- **Keys**: `keys/` (secure, backup regularly)
- **Data**: `data/threat_models/`

For security incidents or questions, consult the threat model documentation and security team protocols.

## Next Steps

After completing this implementation:

1. **Security Review**: Conduct comprehensive security review with external auditors
2. **Penetration Testing**: Engage security firm for thorough penetration testing
3. **Compliance Assessment**: Validate against required compliance frameworks
4. **Documentation**: Update security documentation and incident response procedures
5. **Training**: Train development team on security boundary patterns and threat modeling

The security architecture provides a strong foundation for AIVillage's distributed AI platform while maintaining developer productivity and system performance.