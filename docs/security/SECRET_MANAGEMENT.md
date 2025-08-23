# Secret Management Guide

## Overview

This document outlines the secret management procedures for AIVillage, including externalization, rotation, and emergency procedures following the security remediation of hardcoded secrets.

## Critical Security Context

**Date**: 2025-08-23
**Issue**: 120+ hardcoded secrets discovered during automation consolidation
**Status**: Remediated - All secrets externalized to environment variables
**Impact**: Critical production security vulnerability resolved

## Secret Categories

### 1. Database Credentials
- **PostgreSQL**: `AIVILLAGE_POSTGRES_PASSWORD`
- **Redis**: `AIVILLAGE_REDIS_PASSWORD`
- **Neo4j**: `AIVILLAGE_NEO4J_PASSWORD`
- **Rotation**: Every 90 days

### 2. Application Security
- **JWT Secret**: `AIVILLAGE_HYPERAG_JWT_SECRET`
- **Gateway API Key**: `AIVILLAGE_GATEWAY_API_KEY`
- **Gateway Secret**: `AIVILLAGE_GATEWAY_SECRET_KEY`
- **Rotation**: Every 60 days

### 3. External API Keys
- **OpenAI**: `AIVILLAGE_OPENAI_API_KEY`
- **Anthropic**: `AIVILLAGE_ANTHROPIC_API_KEY`
- **Rotation**: As required by providers or security policy

### 4. Infrastructure
- **Grafana Admin**: `AIVILLAGE_GRAFANA_PASSWORD`
- **TLS Certificates**: `AIVILLAGE_TLS_SECRET_NAME`
- **Rotation**: Before certificate expiration

### 5. Integration Secrets
- **Cogment Metrics DB**: `AIVILLAGE_COGMENT_METRICS_DB_URL`
- **Rotation**: Every 90 days

## Environment Configuration

### Production Environment
```bash
# Never commit actual production secrets!
# Use your organization's secrets management system
cp devops/deployment/.env.production.example devops/deployment/.env.production
# Edit .env.production with actual values
```

### Staging Environment
```bash
cp devops/deployment/.env.staging.example devops/deployment/.env.staging
# Use test credentials only
```

### Development Environment
```bash
cp devops/deployment/.env.development.example devops/deployment/.env.development
# Use personal development API keys
```

## Deployment Integration

### Kubernetes Secrets
Secrets are loaded into Kubernetes using the externalized format:

```yaml
# devops/deployment/k8s/secrets.yaml
stringData:
  POSTGRES_PASSWORD: "${AIVILLAGE_POSTGRES_PASSWORD}"
  REDIS_PASSWORD: "${AIVILLAGE_REDIS_PASSWORD}"
  # ... other secrets
```

### Helm Values
Production and staging values reference environment variables:

```yaml
# devops/deployment/helm/aivillage/values-production.yaml
secrets:
  postgresPassword: "${AIVILLAGE_POSTGRES_PASSWORD}"
  redisPassword: "${AIVILLAGE_REDIS_PASSWORD}"
  # ... other secrets
```

## Secret Generation Guidelines

### Strong Password Requirements
- **Minimum length**: 32 characters for production
- **Character set**: Alphanumeric + special characters
- **No dictionary words**: Use cryptographically secure random generation

### JWT Secrets
- **Minimum length**: 256 bits (32 bytes)
- **Generation**: `openssl rand -hex 32`
- **Encoding**: Base64 or hex encoding acceptable

### API Keys
- **Format**: Provider-specific (OpenAI: `sk-...`, Anthropic: `sk-ant-...`)
- **Source**: Official provider dashboards only
- **Scope**: Minimum required permissions

## Secret Rotation Procedures

### Automated Rotation (Recommended)
1. Use HashiCorp Vault or AWS Secrets Manager
2. Configure automatic rotation schedules
3. Update Kubernetes secrets automatically
4. Verify deployment health post-rotation

### Manual Rotation
1. Generate new secret using approved methods
2. Update environment variables in deployment system
3. Deploy updated configuration
4. Verify all services are healthy
5. Deactivate old secret after verification period

### Emergency Rotation
1. **Immediately** rotate the compromised secret
2. Update all environments simultaneously
3. Force restart all affected services
4. Monitor logs for authentication failures
5. Investigate potential security breach

## Validation and Testing

### Pre-deployment Validation
```bash
# Run secret validation script
python scripts/validate_secret_externalization.py

# Expected output: "SUCCESS: All hardcoded secrets have been externalized!"
```

### Configuration Testing
```bash
# Test Kubernetes secret loading
kubectl apply --dry-run=client -f devops/deployment/k8s/secrets.yaml

# Test Helm template rendering
helm template aivillage devops/deployment/helm/aivillage/ \
  --values devops/deployment/helm/aivillage/values-production.yaml
```

### Detect-secrets Scanning
```bash
# Scan for any remaining hardcoded secrets
detect-secrets scan --all-files --baseline .secrets.baseline

# Should return no violations
```

## Security Best Practices

### Access Control
- **Principle of Least Privilege**: Grant minimum required access
- **Role-based Access**: Use IAM roles/service accounts
- **Audit Logging**: Monitor all secret access
- **Multi-factor Authentication**: Required for secret management systems

### Storage Security
- **Never commit secrets to version control**
- **Use encrypted storage at rest**
- **Encrypt secrets in transit**
- **Regular security audits of secret stores**

### Development Guidelines
- **Use development/test secrets only for non-production**
- **Personal API keys acceptable for local development**
- **Never use production secrets in development/testing**
- **Regular rotation of development secrets**

## Monitoring and Alerting

### Secret Access Monitoring
- Log all secret retrieval operations
- Alert on unusual access patterns
- Monitor for failed authentication attempts
- Track secret age and rotation compliance

### Health Checks
- Verify secret loading in application startup
- Test database connectivity with credentials
- Validate API key functionality
- Monitor certificate expiration dates

## Incident Response

### Suspected Secret Compromise
1. **Immediately rotate** all potentially affected secrets
2. **Lock down access** to secret management systems
3. **Review access logs** for unauthorized usage
4. **Update incident response documentation**
5. **Conduct post-incident review**

### Service Outage Due to Secrets
1. **Verify secret validity** and accessibility
2. **Check environment variable configuration**
3. **Validate Kubernetes secret mounting**
4. **Review recent deployments and changes**
5. **Escalate to security team if compromise suspected**

## Compliance and Auditing

### Regular Audits
- **Monthly**: Verify secret rotation compliance
- **Quarterly**: Review access permissions
- **Annually**: Full security assessment

### Documentation Requirements
- **Secret inventory**: Maintain current list of all secrets
- **Rotation schedules**: Document planned rotation dates
- **Access logs**: Preserve for compliance requirements
- **Incident reports**: Document all security-related incidents

## Tools and Resources

### Validation Tools
- `scripts/validate_secret_externalization.py` - Validate externalization
- `detect-secrets` - Scan for hardcoded secrets
- `helm template` - Test configuration rendering

### Secret Management Systems
- **HashiCorp Vault** - Enterprise secret management
- **AWS Secrets Manager** - Cloud-native solution
- **Azure Key Vault** - Microsoft cloud solution
- **Kubernetes Secrets** - Basic cluster-level secrets

### Generation Tools
```bash
# Strong password generation
openssl rand -base64 32

# JWT secret generation
openssl rand -hex 32

# API key validation
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

## Remediation Summary

The following files were modified during secret externalization:

1. **`devops/deployment/k8s/secrets.yaml`**
   - Lines 9, 10, 11, 14, 17, 18, 21: Externalized 7 hardcoded secrets
   - All secrets now use `${AIVILLAGE_*}` environment variable format

2. **`devops/deployment/helm/aivillage/values-production.yaml`**
   - Line 29: TLS secret name externalized
   - Lines 228-234: All 7 production secrets externalized

3. **`devops/deployment/helm/aivillage/values-staging.yaml`**
   - Line 27: Staging TLS secret name externalized

4. **`config/cogment/deployment_config.yaml`**
   - Line 165: Database credentials externalized

5. **`core/gateway/config.yaml`**
   - Lines 18, 19: Gateway security keys externalized

## Contact Information

**Security Team**: security@aivillage.com
**DevOps Team**: devops@aivillage.com
**Emergency Contact**: security-emergency@aivillage.com

---

**Last Updated**: 2025-08-23
**Next Review**: 2025-11-23
**Version**: 1.0 (Post-Remediation)
