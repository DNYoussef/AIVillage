# HypeRAG Permission System

## 6. Permission System Design

### Role-Based Access Control (RBAC)

The HypeRAG permission system implements a comprehensive RBAC model with granular operation control.

### Core Roles

| Role | Permissions | Description |
|------|-------------|-------------|
| **king** | READ, WRITE, GRAPH_MODIFY, REPAIR_APPROVE, ADAPTER_MANAGE | Full system access, can modify graph structure and approve repairs |
| **sage** | READ, WRITE, GRAPH_MODIFY, ADAPTER_USE | Strategic knowledge management, can modify graph but not approve repairs |
| **magi** | READ, WRITE_CODE_DOCS, ADAPTER_USE | Development-focused, limited write to code documentation |
| **watcher** | READ, MONITOR | Read-only access with monitoring capabilities |
| **external** | READ_LIMITED | Limited read access to public knowledge |
| **guardian** | READ, GATE_OVERRIDE, REPAIR_APPROVE, POLICY_MANAGE | Safety validation and override capabilities |
| **innovator** | READ, REPAIR_PROPOSE | Can propose repairs but not apply them |
| **admin** | ALL | System administration, full access |

### Permission Types

```yaml
Permissions:
  # Read Operations
  READ:
    - Query knowledge base
    - View graph structure
    - Access embeddings
    - Read audit logs

  READ_LIMITED:
    - Query public knowledge only
    - No access to embeddings
    - No graph visualization

  # Write Operations
  WRITE:
    - Add knowledge to Hippo-Index
    - Create hyperedges
    - Update confidence scores
    - Add documents

  WRITE_CODE_DOCS:
    - Add code documentation
    - Update technical specs
    - Limited domain: code

  # Graph Operations
  GRAPH_MODIFY:
    - Alter graph structure
    - Delete nodes/edges
    - Bulk operations
    - Version control

  # Repair Operations
  REPAIR_PROPOSE:
    - Submit repair proposals
    - Run GDC validation
    - Suggest improvements

  REPAIR_APPROVE:
    - Approve repair proposals
    - Override quarantine
    - Direct graph fixes

  # Adapter Operations
  ADAPTER_USE:
    - Activate adapters
    - Switch domains
    - Personalization

  ADAPTER_MANAGE:
    - Upload new adapters
    - Revoke adapters
    - Sign adapters

  # System Operations
  GATE_OVERRIDE:
    - Override Guardian decisions
    - Force apply/reject
    - Emergency controls

  POLICY_MANAGE:
    - Update safety policies
    - Modify thresholds
    - Configure rules

  MONITOR:
    - View system metrics
    - Access performance data
    - Read health status
```

### Operation Matrix

| Operation | king | sage | magi | watcher | external | guardian | innovator |
|-----------|------|------|------|---------|----------|----------|-----------|
| Query (normal) | ✓ | ✓ | ✓ | ✓ | ✓* | ✓ | ✓ |
| Query (creative) | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Add knowledge | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Add code docs | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Modify graph | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Propose repair | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Approve repair | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Use adapter | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |
| Upload adapter | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Override gate | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ |
| View metrics | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |

*Limited to public knowledge

### Dynamic Permissions

```yaml
DynamicPermissions:
  ContextBased:
    - User's own data: WRITE allowed
    - Quarantined items: READ requires approval
    - High-risk operations: Require 2FA

  TimeBased:
    - Business hours: Full permissions
    - After hours: Restricted writes
    - Maintenance window: Admin only

  ResourceBased:
    - CPU quota per role
    - Memory limits
    - Query complexity bounds

  GeoBased:
    - Region restrictions
    - Data sovereignty rules
    - Compliance requirements
```

### Permission Enforcement Architecture

```yaml
EnforcementLayers:
  MCPCore:
    - Initial auth check
    - Role extraction from JWT
    - Basic permission validation

  GuardianGate:
    - Operation-specific checks
    - Policy evaluation
    - Risk assessment

  ResourceLevel:
    - Node/edge ownership
    - Domain boundaries
    - Adapter signatures

  AuditTrail:
    - All operations logged
    - Permission decisions
    - Override reasons
```

### Policy Configuration

```yaml
# policies/permissions.yaml
policies:
  default_deny: true

  rules:
    - name: "King full access"
      role: king
      permissions: ["*"]

    - name: "Sage knowledge management"
      role: sage
      permissions:
        - "READ"
        - "WRITE"
        - "GRAPH_MODIFY"
        - "ADAPTER_USE"
      restrictions:
        - no_repair_approval
        - no_adapter_upload

    - name: "Magi code focus"
      role: magi
      permissions:
        - "READ"
        - "WRITE_CODE_DOCS"
        - "ADAPTER_USE"
      conditions:
        - domain: "code"
        - file_types: [".py", ".js", ".md"]

    - name: "External limited read"
      role: external
      permissions:
        - "READ_LIMITED"
      conditions:
        - public_only: true
        - rate_limit: 10/hour
        - no_embeddings: true

    - name: "Guardian safety override"
      role: guardian
      permissions:
        - "READ"
        - "GATE_OVERRIDE"
        - "REPAIR_APPROVE"
        - "POLICY_MANAGE"
      audit:
        - log_all_overrides
        - notify_admin
```

### Delegation and Impersonation

```yaml
Delegation:
  AllowedDelegations:
    king:
      - can_delegate_to: [sage, magi]
      - permissions: [READ, WRITE]
      - duration: 24h

    sage:
      - can_delegate_to: [magi]
      - permissions: [READ]
      - duration: 8h

  Impersonation:
    admin:
      - can_impersonate: ["*"]
      - requires: audit_reason
      - max_duration: 1h

    guardian:
      - can_impersonate: [king, sage]
      - for_operations: [REPAIR_APPROVE]
      - requires: emergency_flag
```

### Multi-Factor Authentication

```yaml
MFARequirements:
  Operations:
    GRAPH_MODIFY:
      - roles: [sage]
      - method: TOTP

    REPAIR_APPROVE:
      - roles: [king, guardian]
      - method: FIDO2

    ADAPTER_MANAGE:
      - roles: ["*"]
      - method: SMS

  Escalation:
    - Failed attempts: 3
    - Lock duration: 15m
    - Admin notification: true
```

### Permission Inheritance

```yaml
Inheritance:
  Hierarchy:
    admin:
      inherits: []
      grants_all: true

    king:
      inherits: [sage]
      additional: [REPAIR_APPROVE, ADAPTER_MANAGE]

    sage:
      inherits: [magi, watcher]
      additional: [GRAPH_MODIFY]

    magi:
      inherits: [watcher]
      additional: [WRITE_CODE_DOCS]

    watcher:
      inherits: [external]
      additional: [MONITOR]
```

### Audit Requirements

```yaml
AuditLog:
  Required:
    - All write operations
    - Permission denials
    - Override decisions
    - Role changes
    - Failed auth attempts

  Format:
    timestamp: ISO8601
    actor: user/agent ID
    role: active role
    operation: attempted operation
    resource: affected resource
    result: success/denied
    reason: denial reason
    ip_address: source IP
    session_id: tracking ID

  Retention:
    standard: 90 days
    security: 1 year
    compliance: 7 years
```

### API Key Permissions

```yaml
APIKeys:
  Types:
    Development:
      - prefix: "hrag_dev_"
      - permissions: [READ, MONITOR]
      - rate_limit: 100/hour

    Production:
      - prefix: "hrag_prod_"
      - permissions: [READ, WRITE]
      - rate_limit: 1000/hour

    Enterprise:
      - prefix: "hrag_ent_"
      - permissions: custom
      - rate_limit: custom

  Management:
    - Rotation required: 90 days
    - Revocation: immediate
    - Scope limiting: per domain
```

### Emergency Access

```yaml
EmergencyAccess:
  BreakGlass:
    authorized_roles: [admin]
    requires:
      - Two-person rule
      - Audit reason
      - Time limit: 4h
      - Notification: all admins

    grants:
      - All permissions
      - Bypass Guardian
      - Direct DB access

    post_action:
      - Full audit review
      - Permission reset
      - Security scan
```

### Compliance Mappings

```yaml
Compliance:
  GDPR:
    READ: "Data access"
    WRITE: "Data processing"
    GRAPH_MODIFY: "Data alteration"

  HIPAA:
    READ_LIMITED: "Minimum necessary"
    AUDIT: "Access logging"

  SOC2:
    POLICY_MANAGE: "Change control"
    MONITOR: "Continuous monitoring"
```

## Integration with Components

### Guardian Gate Integration
- Every operation passes through permission check
- Guardian can override with GATE_OVERRIDE permission
- Policy violations logged separately

### Adapter System Integration
- Adapters inherit user permissions
- Signed adapters have embedded permission requirements
- Runtime permission verification

### Audit Trail Integration
- Permission decisions create audit entries
- Failed attempts trigger alerts
- Compliance reports generated monthly
