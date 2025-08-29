---
name: 🐛 Bug Report
about: Report a bug to help us improve AIVillage
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Bug Description
<!-- A clear and concise description of the bug -->

## 🛡️ Security Assessment
<!-- IMPORTANT: Consider security implications before reporting -->

### Potential Security Impact
- [ ] **No Security Impact** - Pure functionality bug
- [ ] **Low Security Risk** - Minor information disclosure
- [ ] **Medium Security Risk** - Potential for privilege escalation
- [ ] **High Security Risk** - Data exposure or system compromise
- [ ] **Unknown** - Needs security team evaluation

<!-- ⚠️ If Medium/High security risk, consider using private security reporting instead -->
<!-- See SECURITY.md for responsible disclosure process -->

### Threat Modeling Quick Check
- [ ] **Authentication Bypass** - Does this affect login/access control?
- [ ] **Data Exposure** - Could this reveal sensitive information?
- [ ] **Injection Vector** - Could this lead to code/command injection?
- [ ] **Denial of Service** - Could this crash or overwhelm the system?
- [ ] **Cryptographic Issue** - Does this affect encryption/signatures?
- [ ] **Network Vulnerability** - Could this expose network communications?

## 🔄 Steps to Reproduce
<!-- Steps to reproduce the behavior -->
1. Go to '...'
2. Run command '...'
3. Click on '...'
4. See error

## ✅ Expected Behavior
<!-- What you expected to happen -->

## ❌ Actual Behavior
<!-- What actually happened -->

## 📱 Environment
<!-- Please complete the following information -->
- **OS**: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- **Python Version**: [e.g., 3.11.5]
- **AIVillage Version**: [e.g., 0.5.1]
- **Installation Method**: [e.g., pip, git clone, Docker]
- **Network Configuration**: [e.g., behind firewall, VPN, direct connection]
- **Container Runtime**: [e.g., Docker 24.0.6, Podman 4.7.0]

## 📋 Component
<!-- Which component is affected? -->
- [ ] **Core System** - Central application logic
- [ ] **Agent System** - AI agent training and deployment
- [ ] **P2P Communication** - Distributed networking
- [ ] **RAG System** - Retrieval-augmented generation
- [ ] **Mobile Client** - iOS/Android applications
- [ ] **Web Interface** - Browser frontend
- [ ] **CLI Tools** - Command-line utilities
- [ ] **API Gateway** - REST/GraphQL endpoints
- [ ] **Authentication** - Login/access control
- [ ] **Database** - Data persistence layer
- [ ] **Monitoring** - Logging/observability
- [ ] **Documentation** - Docs and guides
- [ ] **CI/CD Pipeline** - Build/deployment system

## 🎯 Risk Assessment

### Business Impact
- [ ] **Critical** - System unusable, data loss risk
- [ ] **High** - Major functionality broken
- [ ] **Medium** - Feature partially broken
- [ ] **Low** - Minor inconvenience

### User Impact
- [ ] **All Users** - Platform-wide impact
- [ ] **Specific User Groups** - [specify which groups]
- [ ] **Administrators Only** - Admin interface issues
- [ ] **Developers Only** - Development environment issues

### Data Sensitivity
- [ ] **Public Data** - No sensitive information involved
- [ ] **Internal Data** - Non-sensitive internal information
- [ ] **User Data** - Personal but non-sensitive user information
- [ ] **Sensitive Data** - PII, financial, health, or confidential data

## 🔍 Additional Context
<!-- Any additional context, logs, or screenshots -->

### Error Logs
```
<!-- Paste relevant error logs here -->
<!-- IMPORTANT: Remove any sensitive information (passwords, tokens, PII) -->
```

### Configuration
```yaml
# Paste relevant configuration (remove sensitive data)
# Replace sensitive values with [REDACTED] or similar
```

### Screenshots/Recording
<!-- Add screenshots or screen recordings if applicable -->
<!-- Blur or redact any sensitive information -->

## 🧪 Tests Failing
<!-- If this affects tests -->
- [ ] Unit tests failing
- [ ] Integration tests failing
- [ ] Security tests failing
- [ ] Performance tests failing  
- [ ] CI/CD pipeline failing

## 🔒 Security Considerations

### Input Validation
- [ ] Does this involve user input that isn't properly validated?
- [ ] Could malicious input exploit this bug?
- [ ] Are there any injection possibilities?

### Access Control
- [ ] Does this affect user permissions or roles?
- [ ] Could this lead to unauthorized access?
- [ ] Are admin functions exposed inappropriately?

### Data Handling
- [ ] Does this involve sensitive data processing?
- [ ] Could this lead to data leakage?
- [ ] Are encryption/decryption operations affected?

### Network Security
- [ ] Does this affect network communications?
- [ ] Could this expose internal services?
- [ ] Are there man-in-the-middle attack vectors?

## 💡 Possible Solution
<!-- If you have ideas on how to fix it -->

### Root Cause Hypothesis
<!-- What do you think is causing this issue? -->

### Suggested Approach
- [ ] **Code Fix** - Modify application logic
- [ ] **Configuration Change** - Update settings/parameters
- [ ] **Infrastructure Change** - Modify deployment/environment
- [ ] **Documentation Update** - Clarify usage/behavior
- [ ] **Security Patch** - Address security vulnerability

### Testing Approach
<!-- How should the fix be validated? -->
- [ ] Unit tests to cover the bug scenario
- [ ] Integration tests for component interaction
- [ ] Security tests to prevent regression
- [ ] Performance tests to ensure no degradation
- [ ] Manual testing scenarios

## 🔄 Regression Check
<!-- Has this worked before? -->
- [ ] **New Feature** - Never worked as expected
- [ ] **Recent Regression** - Worked in version: [specify]
- [ ] **Long-standing Issue** - Been broken for a while
- [ ] **Environment-specific** - Only fails in certain conditions

## 🔗 Related Issues
<!-- Link to related issues if any -->
- Related to #
- Duplicate of #
- Depends on #
- Blocks #

## 📊 Monitoring & Alerting
<!-- How can we detect this issue in production? -->
- [ ] **Add Logging** - Improve error logging/tracing
- [ ] **Add Metrics** - Track relevant performance indicators
- [ ] **Add Alerts** - Monitor for this error condition
- [ ] **Health Checks** - Include in system health monitoring

---

**Triage Team Internal Use Only:**
- **Priority**: [ ] P0-Critical [ ] P1-High [ ] P2-Medium [ ] P3-Low
- **Security Review Needed**: [ ] Yes [ ] No
- **Assigned Team**: 
- **Estimated Effort**: [ ] XS (<1d) [ ] S (1-3d) [ ] M (1w) [ ] L (2-4w) [ ] XL (1m+)