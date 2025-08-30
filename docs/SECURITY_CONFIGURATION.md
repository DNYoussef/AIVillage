# Security Configuration Guide

## Overview
This guide covers the security configuration requirements for AIVillage, focusing on proper secret management and environment variable setup.

## Critical Environment Variables

### Required Security Variables

All production deployments **MUST** set the following environment variables:

```bash
# JWT Secret (REQUIRED - minimum 32 characters)
JWT_SECRET=your_secure_jwt_secret_here

# Application Secret Key (REQUIRED - minimum 32 characters)
SECRET_KEY=your_secure_app_secret_here

# API Gateway Key (REQUIRED)
API_KEY=your_secure_api_key_here
```

### Generating Secure Values

Use the following methods to generate cryptographically secure secrets:

```bash
# For JWT_SECRET and SECRET_KEY (32+ character hex)
openssl rand -hex 32

# For API_KEY (16+ character hex)
openssl rand -hex 16

# Alternative using Python
python -c "import secrets; print(secrets.token_hex(32))"
```

## Security Validation

The application performs automatic validation on startup:

1. **Presence Check**: All required environment variables must be set
2. **Insecure Default Detection**: Prevents use of known insecure default values
3. **Length Validation**: Ensures minimum security requirements

### Blocked Insecure Values

The following values are automatically rejected:
- `dev-secret-change-in-production`
- `CHANGE_IN_PRODUCTION_USE_SECURE_SECRET_KEY`
- `dev-gateway-key-12345`
- `dev-secret-key-change-in-production`
- `changeme`, `default`, `secret`

## Configuration Files

### Services Configuration (config/services.yaml)
Uses environment variable substitution:
```yaml
security:
  secret_key: "${SECRET_KEY}"
```

### RBAC Configuration (config/security/rbac.json)
Uses environment variable substitution:
```json
{
  "rbac": {
    "jwt_secret": "${JWT_SECRET}"
  }
}
```

## Production Checklist

Before deploying to production:

- [ ] Set all required environment variables
- [ ] Use cryptographically secure random values
- [ ] Verify no hardcoded secrets remain in code
- [ ] Test application startup with production configuration
- [ ] Ensure environment variables are properly secured in deployment environment

## Security Best Practices

1. **Never commit secrets to version control**
2. **Use different secrets for different environments**
3. **Rotate secrets regularly**
4. **Store secrets securely in your deployment platform**
5. **Monitor for secret exposure**

## Troubleshooting

### Common Errors

**Error**: `JWT_SECRET environment variable is required`
**Solution**: Set the JWT_SECRET environment variable with a secure value

**Error**: `Insecure default JWT secret detected`
**Solution**: Replace any default/example values with secure random values

**Error**: Application fails to start
**Solution**: Check all required environment variables are set and valid