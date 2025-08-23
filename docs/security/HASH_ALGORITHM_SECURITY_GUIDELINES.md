# Hash Algorithm Security Guidelines

## Overview
This document outlines the secure hash algorithm usage guidelines for the AIVillage project, following industry best practices and addressing CWE-327 (Use of Broken or Risky Cryptographic Algorithm) vulnerabilities.

## Security-Critical Hash Algorithm Requirements

### ✅ APPROVED Algorithms
- **SHA-256** (Primary): Use for all security contexts including:
  - Data integrity verification in security contexts
  - Digital signatures and certificates
  - Password salt generation
  - Secure ID generation for PII/PHI data
  - Cryptographic nonces and tokens
  - Security audit trail generation

- **SHA-3 Family** (Alternative): For specialized security requirements
- **BLAKE2b/BLAKE2s** (Alternative): For high-performance security contexts

### ❌ PROHIBITED Algorithms (Security Contexts)
- **MD5**: Cryptographically broken, vulnerable to collision attacks
- **SHA-1**: Deprecated, vulnerable to collision attacks
- **MD4**: Severely broken, should never be used

### ⚠️ CONDITIONAL Usage (Non-Security Contexts)
MD5 and SHA-1 may ONLY be used in non-security contexts with explicit documentation:

```python
# CORRECT: Non-security usage with clear intent
import hashlib
hash_value = hashlib.md5(data, usedforsecurity=False).hexdigest()  # Used for deterministic seeding only

# INCORRECT: Security usage
import hashlib
security_hash = hashlib.md5(sensitive_data).hexdigest()  # ❌ CWE-327 violation
```

## Implementation Guidelines

### Security Context Examples
Replace MD5/SHA-1 with SHA-256 in these scenarios:

```python
# File integrity in security contexts
file_hash = hashlib.sha256(file_content).hexdigest()

# Secure ID generation for PII/PHI
location_id = f"secure_{hashlib.sha256(f'{db_path}:{table}:{column}'.encode()).hexdigest()[:16]}"

# Security audit trails
audit_id = f"audit_{int(time.time())}_{hashlib.sha256(f'{event_type}{location_id}'.encode()).hexdigest()[:8]}"

# Digital twin validation IDs
validation_id = f"val_{hashlib.sha256(f'{student_id}_{content_hash}_{timestamp}'.encode()).hexdigest()[:12]}"
```

### Non-Security Context Examples
Use `usedforsecurity=False` parameter when MD5 is used for non-cryptographic purposes:

```python
# Deterministic seeding for embeddings
text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()  # OK: Used for seeding
seed = int(text_hash[:8], 16)

# File integrity for non-security metadata
file_info["md5"] = hashlib.md5(content, usedforsecurity=False).hexdigest()  # OK: Legacy compatibility
```

## Migration Strategy

### Phase 1: Critical Security Contexts ✅ COMPLETED
- [x] PII/PHI management system (`pii_phi_manager.py`)
- [x] Digital twin security validation (`shield_validator.py`)
- [x] File upload security systems (`secure_file_upload.py`)

### Phase 2: Non-Security Contexts ✅ COMPLETED
- [x] Add `usedforsecurity=False` to existing MD5 usage in:
  - Vector embedding generation (deterministic seeding)
  - File integrity metadata (legacy compatibility)

### Phase 3: Documentation and Training
- [x] Create security guidelines documentation
- [ ] Update code review checklists
- [ ] Add automated security scanning rules

## Code Review Checklist

When reviewing code changes, verify:

- [ ] No new MD5/SHA-1 usage in security contexts
- [ ] SHA-256 used for all cryptographic operations
- [ ] Non-security MD5 usage includes `usedforsecurity=False`
- [ ] Clear comments explaining non-security hash usage
- [ ] Security-sensitive IDs use SHA-256 with appropriate truncation

## Automated Security Scanning

### Bandit Configuration
Add these rules to prevent CWE-327 violations:

```yaml
# .bandit
tests:
  - B324  # Use of insecure hash function (MD5/SHA1)

skips:
  - B324  # Skip only when usedforsecurity=False is used
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: hash-security-check
      name: Hash Algorithm Security Check
      entry: python scripts/check_hash_security.py
      language: system
      files: '\.py$'
```

## Emergency Response

If a security vulnerability related to weak hash algorithms is discovered:

1. **Immediate**: Identify all affected systems
2. **Assessment**: Evaluate impact on data integrity and security
3. **Mitigation**: Deploy secure hash replacements
4. **Validation**: Run comprehensive security scans
5. **Documentation**: Update security incident logs

## Compliance Requirements

### Regulatory Alignment
- **NIST SP 800-57**: Cryptographic key management standards
- **GDPR Article 32**: Technical and organizational security measures
- **HIPAA Security Rule**: Administrative, physical, and technical safeguards
- **SOX Section 404**: Internal controls over financial reporting

### Audit Trail Requirements
All security-related hash operations must:
- Use approved algorithms (SHA-256 minimum)
- Generate audit logs with secure timestamps
- Maintain integrity verification capabilities
- Support forensic analysis requirements

## Contact Information

**Security Team**: security@aivillage.local
**Compliance Officer**: compliance@aivillage.local
**Emergency Contact**: security-incident@aivillage.local

---

**Document Version**: 1.0
**Last Updated**: 2024-01-23
**Next Review**: 2024-04-23
**Classification**: Internal Use
