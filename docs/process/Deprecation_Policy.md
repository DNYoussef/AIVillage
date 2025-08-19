# AIVillage Deprecation Policy

## Overview

This document outlines the deprecation policy for the AIVillage project, including APIs, components, configurations, and other interfaces. Our goal is to balance innovation with stability, providing clear migration paths while minimizing disruption to users and developers.

## Deprecation Philosophy

### Core Principles

1. **Predictability**: Deprecated features follow a clear, time-bound sunset schedule
2. **Communication**: Deprecations are announced early with clear migration guidance
3. **Support**: Deprecated features receive maintenance during the sunset period
4. **Migration**: Clear upgrade paths and tooling are provided
5. **Gradual**: Changes are phased to minimize impact on users

### Deprecation vs. Removal

- **Deprecation**: Feature still works but is marked for future removal
- **Removal**: Feature is completely eliminated from the codebase
- **Migration Period**: Time between deprecation announcement and removal

## Deprecation Categories

### 🔌 Public APIs

**Scope**: REST endpoints, GraphQL schemas, client SDKs, public libraries

**Timeline**:
- **Announcement**: 6 months before removal
- **Migration Period**: 6 months minimum
- **Removal**: After migration period expires

**Process**:
1. Add deprecation headers to HTTP responses
2. Update OpenAPI specification with deprecation markers
3. Add console warnings to client SDKs
4. Provide migration documentation

**Example Headers**:
```http
Sunset: Wed, 19 Feb 2026 23:59:59 GMT
Deprecation: true
Link: <https://docs.aivillage.com/api/migration/v1-to-v2>; rel="deprecation"; title="Migration Guide"
```

### 🏗️ Internal Components

**Scope**: Internal services, libraries, data formats, configuration schemas

**Timeline**:
- **Announcement**: 3 months before removal
- **Migration Period**: 3 months minimum
- **Removal**: After internal migration complete

**Process**:
1. Add deprecation warnings to logs
2. Create internal migration tickets
3. Update internal documentation
4. Coordinate team migration efforts

### 📱 Client Features

**Scope**: Mobile app features, CLI commands, UI components

**Timeline**:
- **Announcement**: 4 months before removal
- **Migration Period**: 4 months minimum
- **Removal**: With next major version

**Process**:
1. Show in-app deprecation notices
2. Update user documentation
3. Provide feature alternatives
4. Gather user feedback

### ⚙️ Configuration & Infrastructure

**Scope**: Config file formats, environment variables, deployment methods

**Timeline**:
- **Announcement**: 6 months before removal
- **Migration Period**: 6 months minimum
- **Removal**: With infrastructure upgrade

**Process**:
1. Support both old and new formats
2. Log warnings for old format usage
3. Provide migration scripts/tools
4. Update deployment documentation

## Semantic Versioning & Deprecation

We follow [Semantic Versioning](https://semver.org/) principles:

### Major Version Changes (X.0.0)
- **Can remove** previously deprecated features
- **Can introduce** breaking changes
- **Must provide** comprehensive migration guide
- **Timeline**: Minimum 12 months between major versions

### Minor Version Changes (1.X.0)
- **Can deprecate** existing features
- **Cannot remove** existing features
- **Must maintain** backward compatibility
- **Can add** new features and capabilities

### Patch Version Changes (1.1.X)
- **Cannot deprecate** or remove features
- **Must maintain** full compatibility
- **Only for** bug fixes and security updates

## Deprecation Announcement Process

### 1. Internal Review
- [ ] Engineering team reviews deprecation proposal
- [ ] Impact assessment completed
- [ ] Migration path defined
- [ ] Timeline established
- [ ] Alternative solutions identified

### 2. Public Announcement
- [ ] Deprecation notice published in CHANGELOG.md
- [ ] Documentation updated with migration guide
- [ ] Blog post or announcement (for major deprecations)
- [ ] GitHub issue created for tracking
- [ ] Team communications sent

### 3. Implementation
- [ ] Deprecation warnings implemented in code
- [ ] Headers/metadata added to APIs
- [ ] Monitoring added for deprecated feature usage
- [ ] Migration tools developed (if needed)
- [ ] Support documentation updated

### 4. Migration Support
- [ ] Migration guide published and tested
- [ ] Support team trained on migration process
- [ ] Community feedback channels monitored
- [ ] Migration tooling validated
- [ ] User success tracked

## Deprecation Warning Implementation

### API Endpoints

```python
# Example: API deprecation warning
from flask import Flask, jsonify, request
import warnings
from datetime import datetime

def deprecated_endpoint():
    # Add deprecation headers
    response = jsonify({"status": "deprecated", "message": "Use /v2/endpoint instead"})
    response.headers['Sunset'] = 'Wed, 19 Feb 2026 23:59:59 GMT'
    response.headers['Deprecation'] = 'true'
    response.headers['Link'] = '<https://docs.aivillage.com/migration/v1-v2>; rel="deprecation"'

    # Log deprecation usage
    logger.warning(f"Deprecated API endpoint accessed: {request.path} by {request.remote_addr}")

    return response
```

### Python Libraries

```python
# Example: Function deprecation
import warnings
from functools import wraps

def deprecated(func):
    """Decorator to mark functions as deprecated."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in v2.0. "
            f"Use {func.__name__.replace('old_', 'new_')} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper

@deprecated
def old_function():
    """This function is deprecated."""
    pass
```

### Configuration Files

```yaml
# Example: Configuration deprecation
# config.yaml
api:
  # DEPRECATED: 'legacy_endpoint' will be removed in v2.0
  # Use 'endpoint_url' instead
  legacy_endpoint: "https://api.old.com"  # Remove after 2026-02-19
  endpoint_url: "https://api.new.com"     # New preferred option
```

### CLI Commands

```python
# Example: CLI deprecation warning
import click

@click.command()
@click.option('--old-flag', is_flag=True, hidden=True,
              help='DEPRECATED: Use --new-flag instead')
@click.option('--new-flag', is_flag=True,
              help='New improved flag')
def my_command(old_flag, new_flag):
    if old_flag:
        click.echo(
            click.style(
                "WARNING: --old-flag is deprecated and will be removed in v2.0. "
                "Use --new-flag instead.",
                fg='yellow'
            ),
            err=True
        )
        new_flag = True  # Auto-migrate

    # Command implementation
```

## Migration Documentation Template

For each deprecation, create documentation following this template:

```markdown
# Migration Guide: [Feature Name] Deprecation

## Summary
- **Feature**: [Deprecated feature name]
- **Deprecated in**: v1.5.0 (August 19, 2025)
- **Removal planned**: v2.0.0 (February 19, 2026)
- **Migration deadline**: February 19, 2026

## What's Changing
[Detailed explanation of what's being deprecated and why]

## Migration Steps
1. [Step-by-step migration instructions]
2. [Include code examples]
3. [Mention any gotchas or edge cases]

## Before (Deprecated)
```python
# Old deprecated usage
old_api_call()
```

## After (Recommended)
```python
# New recommended usage
new_api_call()
```

## Automated Migration
[If migration tools are available]

## Timeline
- **August 19, 2025**: Deprecation warning added
- **November 19, 2025**: Migration reminder notices
- **February 19, 2026**: Feature removed

## Getting Help
[Support channels and resources]
```

## Monitoring Deprecated Features

### Usage Tracking
- Log all deprecated feature usage
- Track unique users/applications affected
- Monitor migration progress over time
- Generate weekly deprecation reports

### Metrics Dashboard
Track key metrics:
- **Usage Decline**: % reduction in deprecated feature usage
- **Migration Progress**: % of users successfully migrated
- **Support Tickets**: Issues related to deprecated features
- **Timeline Adherence**: On-schedule vs. delayed removals

### Alerts
Set up monitoring for:
- Sudden increases in deprecated feature usage
- Migration deadline approaching with high usage
- New applications using deprecated features
- Migration errors or failures

## Communication Channels

### Internal Communication
- **Engineering Team**: Slack #deprecations channel
- **Product Team**: Monthly deprecation review meetings
- **Support Team**: Deprecation training and FAQ updates
- **Documentation Team**: Migration guide reviews

### External Communication
- **Developers**: Email announcements, GitHub issues
- **Users**: In-app notifications, blog posts
- **Partners**: Direct communication for major changes
- **Community**: Discord, forums, social media

## Deprecation Schedule Management

### Current Deprecations

| Feature | Deprecated | Removal Planned | Status | Migration Guide |
|---------|------------|----------------|---------|-----------------|
| Legacy Agent API v1 | 2025-08-19 | 2026-02-19 | Active | [Migration Guide](../api/agent-api-v1-v2.md) |
| Old Config Format | 2025-08-19 | 2026-02-19 | Active | [Config Migration](../config/legacy-format.md) |

### Upcoming Removals

| Feature | Removal Date | Days Remaining | Action Required |
|---------|--------------|----------------|-----------------|
| Legacy Agent API v1 | 2026-02-19 | 183 | Migrate to v2 API |
| Old Config Format | 2026-02-19 | 183 | Update config files |

## Exception Handling

### Emergency Deprecations
For security vulnerabilities or critical issues:
- **Immediate announcement** with 30-day minimum notice
- **Expedited timeline** with enhanced support
- **Direct communication** to affected users
- **Priority migration assistance**

### Extension Requests
Users can request timeline extensions for:
- **Complex migrations** requiring significant development
- **Enterprise integrations** with long procurement cycles
- **External dependencies** blocking migration
- **Business-critical applications** needing additional time

Extension criteria:
- Valid technical or business justification
- Committed migration timeline
- Security and maintenance considerations
- Impact on other users and project roadmap

## Success Metrics

Deprecation policy effectiveness measured by:

### Process Metrics
- **Timeline Adherence**: % of deprecations completed on schedule
- **Migration Success**: % of users successfully migrated
- **Support Burden**: Reduction in support tickets over time
- **Developer Satisfaction**: Survey feedback on deprecation process

### Quality Metrics
- **Breaking Changes**: Reduction in unplanned breaking changes
- **Migration Issues**: Bugs found in migration process
- **Documentation Quality**: Feedback on migration guides
- **Tool Effectiveness**: Success rate of automated migration tools

## Policy Evolution

This policy will be reviewed and updated:
- **Quarterly**: Based on team feedback and lessons learned
- **After major releases**: To incorporate deprecation experiences
- **Community feedback**: User suggestions for improvement
- **Industry changes**: Alignment with ecosystem best practices

---

## FAQ

**Q: How do I know if a feature I'm using is deprecated?**
A: Check for deprecation warnings in logs, HTTP headers, or in-app notifications. Monitor our CHANGELOG.md and GitHub issues.

**Q: Can deprecation timelines be extended?**
A: Yes, see the Exception Handling section above for extension criteria and process.

**Q: What if I can't migrate before the removal deadline?**
A: Contact our support team immediately. We can discuss extension options or provide migration assistance.

**Q: How do I request a feature deprecation?**
A: Create a GitHub issue with the deprecation proposal template and follow our internal review process.

**Q: Will deprecated features receive bug fixes?**
A: Yes, security fixes and critical bugs will be addressed during the migration period, but no new features will be added.

---

**Remember**: Deprecation is not punishment—it's evolution. We deprecate features to improve the platform while providing safe migration paths for our users.

---

*This Deprecation Policy is a living document. Feedback and suggestions for improvement are welcome through GitHub issues or team discussions.*

**Last Updated**: August 19, 2025
**Version**: 1.0
**Next Review**: November 19, 2025
