# AIVillage Backup and Restore Runbook

## Table of Contents

1. [Overview](#overview)
2. [Emergency Response](#emergency-response)
3. [Restore Procedures](#restore-procedures)
4. [Verification and Validation](#verification-and-validation)
5. [Troubleshooting](#troubleshooting)
6. [Recovery Scenarios](#recovery-scenarios)
7. [Best Practices](#best-practices)

## Overview

This runbook provides step-by-step procedures for backing up and restoring AIVillage systems. It covers emergency response, planned maintenance, disaster recovery, and partial system restoration.

### System Components Covered

- **RBAC System**: User authentication, roles, permissions, tenant management
- **Tenant Data**: Multi-tenant isolated data and configurations
- **Agent Systems**: All 23 specialized agents and configurations
- **RAG Collections**: Vector databases, embeddings, knowledge graphs
- **P2P Networks**: BitChat and BetaNet communication configurations
- **AI Models**: Agent Forge models, trained weights, inference engines
- **System Configuration**: Deployment configs, environment settings
- **System Logs**: Application logs, audit trails, system metrics

### Backup Types Available

- **Full System Backup**: Complete system state including all components
- **Incremental Backup**: Changes since last full backup (6-hour intervals)
- **Tenant Backup**: Single tenant data isolation for compliance
- **Component Backup**: Specific component restoration for targeted fixes
- **Emergency Backup**: Immediate backup triggered by alerts or manual intervention

## Emergency Response

### Immediate Response Checklist

When a system failure or data loss is detected:

1. **Stop the Bleeding**
   ```bash
   # Stop services to prevent further damage
   sudo systemctl stop aivillage-*

   # Isolate affected systems
   sudo ufw deny incoming
   ```

2. **Assess Damage**
   ```bash
   # Check backup status
   python -m packages.core.backup.backup_cli list --limit 10

   # Verify last successful backup
   python -m packages.core.backup.backup_cli info <backup_id>
   ```

3. **Determine Recovery Strategy**
   - **Total System Loss**: Full disaster recovery
   - **Component Failure**: Component-specific restore
   - **Data Corruption**: Point-in-time recovery
   - **Security Incident**: Clean restore with security hardening

4. **Execute Recovery Plan**
   Follow appropriate procedure below based on damage assessment.

### Emergency Backup (Before Recovery)

Always create an emergency backup of current state before restore:

```bash
# Trigger emergency backup
python -m packages.core.backup.backup_cli emergency

# Monitor backup progress
python -m packages.core.backup.backup_cli scheduler --format table
```

## Restore Procedures

### Full System Restore

**Use Case**: Complete system failure, disaster recovery, major corruption

**Prerequisites**:
- System access (root/admin privileges)
- Network connectivity
- Sufficient disk space (2x backup size)
- Valid backup with `COMPLETED` status

**Procedure**:

1. **Identify Target Backup**
   ```bash
   # List available backups
   python -m packages.core.backup.backup_cli list --type full --limit 10

   # Get backup details
   python -m packages.core.backup.backup_cli info backup_full_20250819_020000_abc123
   ```

2. **Prepare System**
   ```bash
   # Stop all AIVillage services
   sudo systemctl stop aivillage-api aivillage-agents aivillage-rag

   # Verify disk space
   df -h /var/lib/aivillage

   # Create rollback point (automatic)
   # This is done automatically unless --no-rollback is specified
   ```

3. **Execute Restore**
   ```bash
   # Dry run first (recommended)
   python -m packages.core.backup.backup_cli restore backup_full_20250819_020000_abc123 \
     --type full --strategy replace --dry-run

   # Execute full restore
   python -m packages.core.backup.backup_cli restore backup_full_20250819_020000_abc123 \
     --type full --strategy replace
   ```

4. **Monitor Progress**
   The restore command will show real-time progress. For long-running restores:
   ```bash
   # Check restore status
   python -m packages.core.backup.backup_cli list-restores --limit 5
   ```

5. **Verify Restoration**
   ```bash
   # Check component verification results
   # These are shown automatically at restore completion

   # Manual verification
   sudo systemctl start aivillage-api
   curl -f http://localhost:8000/health

   # Verify tenant access
   curl -f http://localhost:8000/api/v1/tenants \
     -H "Authorization: Bearer <token>"
   ```

**Expected Duration**: 15-45 minutes depending on backup size

**Rollback Procedure** (if restore fails):
```bash
# Rollback is automatic on failure
# Manual rollback if needed:
python -c "
from packages.core.backup.restore_manager import RestoreManager
from packages.core.backup.backup_manager import BackupManager
import asyncio

async def rollback():
    backup_mgr = BackupManager()
    restore_mgr = RestoreManager(backup_mgr)

    # Find failed restore
    restores = await restore_mgr.list_restores(limit=1)
    if restores and restores[0].rollback_point_created:
        await restore_mgr._rollback_restore(restores[0])
        print('Rollback completed')

asyncio.run(rollback())
"
```

### Component-Specific Restore

**Use Case**: Single component failure, configuration corruption, targeted recovery

**Common Components**:
- `rbac_system` - Authentication and tenant management
- `agents` - Agent configurations and states
- `rag_collections` - Vector databases and knowledge
- `configurations` - System configuration files
- `models` - AI model weights and metadata

**Procedure**:

1. **Identify Component and Backup**
   ```bash
   # List backups containing the component
   python -m packages.core.backup.backup_cli list --limit 20

   # Verify component exists in backup
   python -m packages.core.backup.backup_cli info backup_full_20250819_020000_abc123
   ```

2. **Choose Restore Strategy**
   - `replace` - Replace existing component completely
   - `merge` - Merge with existing data (where supported)
   - `side_by_side` - Restore alongside existing (for comparison)
   - `test_restore` - Restore to test location only

3. **Execute Component Restore**
   ```bash
   # Example: Restore agent configurations
   python -m packages.core.backup.backup_cli restore backup_full_20250819_020000_abc123 \
     --type component --component agents --strategy replace

   # Example: Restore RAG collections with merge
   python -m packages.core.backup.backup_cli restore backup_full_20250819_020000_abc123 \
     --type component --component rag_collections --strategy merge
   ```

4. **Restart Affected Services**
   ```bash
   # Restart specific services
   sudo systemctl restart aivillage-agents  # for agents component
   sudo systemctl restart aivillage-rag     # for rag_collections component
   ```

**Expected Duration**: 2-10 minutes depending on component size

### Tenant-Specific Restore

**Use Case**: Single tenant data corruption, compliance restoration, tenant isolation

**Procedure**:

1. **Identify Tenant and Backup**
   ```bash
   # List backups with tenant data
   python -m packages.core.backup.backup_cli list --type full --limit 10

   # Verify tenant exists in backup
   python -m packages.core.backup.backup_cli info backup_full_20250819_020000_abc123
   ```

2. **Execute Tenant Restore**
   ```bash
   # Restore specific tenant
   python -m packages.core.backup.backup_cli restore backup_full_20250819_020000_abc123 \
     --type tenant --tenant tenant_healthcare_001 --strategy replace

   # Restore with merge (preserve recent changes)
   python -m packages.core.backup.backup_cli restore backup_full_20250819_020000_abc123 \
     --type tenant --tenant tenant_healthcare_001 --strategy merge
   ```

3. **Verify Tenant Access**
   ```bash
   # Test tenant authentication
   curl -f http://localhost:8000/api/v1/auth/login \
     -d '{"tenant_id":"tenant_healthcare_001","username":"user@example.com","password":"***"}'

   # Verify tenant data access
   curl -f http://localhost:8000/api/v1/tenants/tenant_healthcare_001/data \
     -H "Authorization: Bearer <token>"
   ```

**Expected Duration**: 5-15 minutes depending on tenant data size

### Point-in-Time Recovery

**Use Case**: Revert to specific time due to data corruption, user error, or security incident

**Procedure**:

1. **Identify Recovery Point**
   ```bash
   # List backups by date
   python -m packages.core.backup.backup_cli list --limit 50 --format table

   # Find backup closest to desired recovery time
   # Full backups: Daily at 02:00 UTC
   # Incremental backups: Every 6 hours
   ```

2. **Plan Recovery Strategy**
   ```bash
   # For recovery within last 24 hours: Use incremental
   python -m packages.core.backup.backup_cli info backup_incr_20250819_140000_def456

   # For recovery beyond 24 hours: Use full backup
   python -m packages.core.backup.backup_cli info backup_full_20250818_020000_ghi789
   ```

3. **Execute Point-in-Time Restore**
   ```bash
   # Create emergency backup of current state
   python -m packages.core.backup.backup_cli emergency

   # Restore to specific point in time
   python -m packages.core.backup.backup_cli restore backup_incr_20250819_140000_def456 \
     --type full --strategy replace
   ```

**Expected Duration**: 10-30 minutes

## Verification and Validation

### Automated Verification

The restore system automatically performs verification checks:

- **Database Integrity**: Verifies databases are accessible and not corrupted
- **File Existence**: Confirms critical configuration files exist
- **Service Health**: Checks that services can start successfully
- **Tenant Isolation**: Validates multi-tenant data separation

### Manual Verification Checklist

After any restore operation:

1. **Service Health**
   ```bash
   # Check all services are running
   sudo systemctl status aivillage-*

   # Verify API endpoints
   curl -f http://localhost:8000/health
   curl -f http://localhost:8000/api/v1/health
   ```

2. **Authentication System**
   ```bash
   # Test authentication
   curl -X POST http://localhost:8000/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"***"}'
   ```

3. **Tenant Access**
   ```bash
   # Verify each tenant can authenticate
   for tenant in tenant_healthcare_001 tenant_finance_002; do
     echo "Testing $tenant"
     curl -f http://localhost:8000/api/v1/tenants/$tenant/health
   done
   ```

4. **Agent Systems**
   ```bash
   # Verify agents are responsive
   curl -f http://localhost:8000/api/v1/agents/king/status
   curl -f http://localhost:8000/api/v1/agents/magi/status
   ```

5. **RAG System**
   ```bash
   # Test knowledge retrieval
   curl -X POST http://localhost:8000/api/v1/rag/query \
     -H "Content-Type: application/json" \
     -d '{"query":"test query","tenant_id":"tenant_healthcare_001"}'
   ```

6. **P2P Networks**
   ```bash
   # Check P2P connectivity
   curl -f http://localhost:8000/api/v1/p2p/status

   # Verify BitChat mesh
   curl -f http://localhost:8000/api/v1/p2p/bitchat/peers
   ```

### Performance Verification

After restore, verify system performance:

1. **Response Times**
   ```bash
   # API response times should be < 500ms
   time curl -f http://localhost:8000/api/v1/health
   ```

2. **Database Performance**
   ```bash
   # Database queries should complete quickly
   python -c "
   import time
   import sqlite3
   start = time.time()
   conn = sqlite3.connect('data/rbac.db')
   cursor = conn.cursor()
   cursor.execute('SELECT COUNT(*) FROM users')
   result = cursor.fetchone()
   conn.close()
   print(f'Query took {time.time() - start:.2f}s, result: {result[0]}')
   "
   ```

3. **Memory Usage**
   ```bash
   # Check for memory leaks
   ps aux | grep aivillage | awk '{sum+=$6} END {print "Total Memory: " sum/1024 " MB"}'
   ```

## Troubleshooting

### Common Issues and Solutions

#### 1. Restore Fails with "Insufficient Disk Space"

**Problem**: Not enough space for backup extraction
**Solution**:
```bash
# Check available space
df -h

# Clean up old logs
find /var/log/aivillage -name "*.log" -mtime +7 -delete

# Clean up old backup files (keep last 10)
python -c "
from packages.core.backup.backup_manager import BackupManager
import asyncio

async def cleanup():
    mgr = BackupManager()
    backups = await mgr.list_backups(limit=100)
    for backup in backups[10:]:  # Keep newest 10
        try:
            await mgr._delete_backup(backup.backup_id)
            print(f'Deleted {backup.backup_id}')
        except Exception as e:
            print(f'Failed to delete {backup.backup_id}: {e}')

asyncio.run(cleanup())
"
```

#### 2. Restore Hangs or Takes Too Long

**Problem**: Restore process appears stuck
**Solution**:
```bash
# Check restore progress in another terminal
python -m packages.core.backup.backup_cli list-restores --limit 1 --format json

# Check system resources
top -p $(pgrep -f "backup\|restore")
iotop -p $(pgrep -f "backup\|restore")

# If truly stuck, kill and restart
pkill -f "backup\|restore"
# Then retry with smaller component restore
```

#### 3. Services Won't Start After Restore

**Problem**: AIVillage services fail to start after restoration
**Solution**:
```bash
# Check service logs
sudo journalctl -u aivillage-api -f
sudo journalctl -u aivillage-agents -f

# Common fixes:
# 1. Fix file permissions
sudo chown -R aivillage:aivillage /var/lib/aivillage
sudo chmod -R 755 /var/lib/aivillage

# 2. Verify database connections
python -c "
import sqlite3
try:
    conn = sqlite3.connect('data/rbac.db')
    print('RBAC DB: OK')
    conn.close()
except Exception as e:
    print(f'RBAC DB: ERROR - {e}')
"

# 3. Check configuration files
python -c "
import json
with open('config/production.json') as f:
    config = json.load(f)
    print('Config loaded successfully')
"
```

#### 4. Partial Data Loss After Restore

**Problem**: Some data missing after restore
**Solution**:
```bash
# Check backup contents
python -m packages.core.backup.backup_cli info <backup_id>

# Verify component inclusion
# Missing data might be in excluded components

# Try component-specific restore
python -m packages.core.backup.backup_cli restore <backup_id> \
  --type component --component <missing_component> --strategy merge
```

#### 5. Authentication Fails After Restore

**Problem**: Cannot login after RBAC system restore
**Solution**:
```bash
# Verify RBAC database integrity
sqlite3 data/rbac.db "SELECT COUNT(*) FROM users;"
sqlite3 data/rbac.db "SELECT COUNT(*) FROM roles;"

# Reset admin password if needed
python -c "
from packages.core.security.rbac_manager import RBACManager
import asyncio

async def reset_admin():
    rbac = RBACManager()
    success = await rbac.reset_admin_password('new_secure_password')
    print(f'Admin password reset: {success}')

asyncio.run(reset_admin())
"

# Verify tenant configurations
sqlite3 data/tenants.db "SELECT tenant_id, status FROM tenants;"
```

### Recovery from Failed Restore

If a restore operation fails and system is in an inconsistent state:

1. **Check for Automatic Rollback**
   ```bash
   python -m packages.core.backup.backup_cli list-restores --limit 1 --format json
   # Look for status: "rolled_back"
   ```

2. **Manual Rollback** (if needed)
   ```bash
   # Find the restore operation
   python -m packages.core.backup.backup_cli list-restores --limit 5

   # Trigger manual rollback
   python -c "
   from packages.core.backup.restore_manager import RestoreManager
   from packages.core.backup.backup_manager import BackupManager
   import asyncio

   async def rollback():
       backup_mgr = BackupManager()
       restore_mgr = RestoreManager(backup_mgr)

       # Get specific restore
       restore = await restore_mgr.get_restore_status('<restore_id>')
       if restore and restore.rollback_point_created:
           await restore_mgr._rollback_restore(restore)
           print('Rollback completed')
       else:
           print('No rollback point available')

   asyncio.run(rollback())
   "
   ```

3. **Emergency Recovery**
   If rollback fails, restore from known-good emergency backup:
   ```bash
   # Find emergency backup (created before restore attempt)
   python -m packages.core.backup.backup_cli list --limit 10 | grep emergency

   # Restore emergency backup
   python -m packages.core.backup.backup_cli restore <emergency_backup_id> \
     --type full --strategy replace --no-rollback
   ```

## Recovery Scenarios

### Scenario 1: Database Corruption

**Symptoms**: Database connection errors, data inconsistencies
**Recovery**:
```bash
# 1. Stop services
sudo systemctl stop aivillage-*

# 2. Backup corrupt databases
mkdir /tmp/corrupt_backup
cp data/*.db /tmp/corrupt_backup/

# 3. Restore database components only
python -m packages.core.backup.backup_cli restore <backup_id> \
  --type component --component rbac_system --strategy replace

# 4. Verify database integrity
sqlite3 data/rbac.db "PRAGMA integrity_check;"

# 5. Start services
sudo systemctl start aivillage-*
```

### Scenario 2: Configuration File Corruption

**Symptoms**: Services won't start, configuration errors
**Recovery**:
```bash
# 1. Backup current configs
mkdir /tmp/config_backup
cp -r config/ /tmp/config_backup/

# 2. Restore configurations
python -m packages.core.backup.backup_cli restore <backup_id> \
  --type component --component configurations --strategy replace

# 3. Verify configurations
python -c "
import json
for config_file in ['config/production.json', 'config/agents.json']:
    try:
        with open(config_file) as f:
            json.load(f)
        print(f'{config_file}: OK')
    except Exception as e:
        print(f'{config_file}: ERROR - {e}')
"

# 4. Restart services
sudo systemctl restart aivillage-*
```

### Scenario 3: Agent System Failure

**Symptoms**: Agents not responding, agent errors in logs
**Recovery**:
```bash
# 1. Check agent status
curl http://localhost:8000/api/v1/agents/status

# 2. Restore agent configurations and states
python -m packages.core.backup.backup_cli restore <backup_id> \
  --type component --component agents --strategy replace

# 3. Restart agent services
sudo systemctl restart aivillage-agents

# 4. Verify agents are responding
for agent in king magi oracle sage; do
    echo "Testing $agent"
    curl http://localhost:8000/api/v1/agents/$agent/ping
done
```

### Scenario 4: Complete System Failure

**Symptoms**: No services responding, hardware failure, disaster
**Recovery**:
```bash
# 1. Deploy new infrastructure
# (Follow deployment documentation)

# 2. Restore complete system
python -m packages.core.backup.backup_cli restore <latest_backup> \
  --type full --strategy replace

# 3. Update network configurations
# (Update IP addresses, hostnames as needed)

# 4. Comprehensive verification
./scripts/verify_system_health.sh
```

### Scenario 5: Security Incident

**Symptoms**: Unauthorized access detected, data breach suspected
**Recovery**:
```bash
# 1. Isolate system
sudo ufw deny incoming

# 2. Create forensic backup of current state
python -m packages.core.backup.backup_cli emergency

# 3. Restore from clean backup (before incident)
python -m packages.core.backup.backup_cli restore <clean_backup> \
  --type full --strategy replace

# 4. Update all credentials
python scripts/rotate_all_credentials.py

# 5. Apply security patches
sudo apt update && sudo apt upgrade -y

# 6. Re-enable access with enhanced monitoring
sudo ufw allow 22/tcp
sudo ufw allow 8000/tcp
sudo ufw --force enable
```

## Best Practices

### Backup Strategy

1. **Regular Schedule**
   - Full backups: Daily at 02:00 UTC
   - Incremental backups: Every 6 hours
   - Tenant backups: Weekly per compliance requirements
   - Configuration backups: Every 4 hours

2. **Retention Policy**
   - Full backups: 30 days
   - Incremental backups: 7 days
   - Tenant backups: Per compliance (90+ days)
   - Emergency backups: 14 days

3. **Monitoring**
   ```bash
   # Set up automated backup monitoring
   python -m packages.core.backup.backup_cli scheduler --format json \
     > /var/log/aivillage/backup_status.json

   # Alert on backup failures
   if ! python -m packages.core.backup.backup_cli list --limit 1 --format json \
        | jq -e '.backups[0].status == "completed"'; then
       echo "ALERT: Latest backup failed" | mail -s "Backup Alert" admin@company.com
   fi
   ```

### Restore Strategy

1. **Test Restores Monthly**
   ```bash
   # Automated monthly restore test
   backup_id=$(python -m packages.core.backup.backup_cli list --limit 1 --format json \
               | jq -r '.backups[0].backup_id')

   python -m packages.core.backup.backup_cli restore $backup_id \
     --type component --component configurations --strategy test_restore
   ```

2. **Document Recovery Times**
   - Full system restore: 15-45 minutes
   - Component restore: 2-10 minutes
   - Tenant restore: 5-15 minutes
   - Emergency backup: 5-10 minutes

3. **Maintain Recovery Infrastructure**
   - Test restore environment monthly
   - Keep recovery procedures updated
   - Train team on restore procedures
   - Maintain emergency contact information

### Security Considerations

1. **Backup Encryption**
   - All backups are encrypted at rest
   - Encryption keys stored in secure vault
   - Regular key rotation (90 days)

2. **Access Control**
   - Backup operations require admin privileges
   - Restore operations are logged and audited
   - Emergency procedures have approval workflows

3. **Data Retention Compliance**
   - Healthcare tenants: 7-year retention
   - Financial tenants: 5-year retention
   - Standard tenants: 3-year retention
   - Automated cleanup respects retention requirements

### Monitoring and Alerting

1. **Backup Health Monitoring**
   ```bash
   # Daily backup health check
   python -c "
   from packages.core.backup.backup_scheduler import BackupScheduler
   import asyncio

   async def health_check():
       # Get scheduler status
       jobs = scheduler.get_all_jobs_status()

       # Check for failed jobs
       failed_jobs = [j for j in jobs if j['consecutive_failures'] > 0]
       if failed_jobs:
           print(f'ALERT: {len(failed_jobs)} backup jobs failing')

       # Check backup age
       backups = await backup_manager.list_backups(limit=1)
       if backups:
           age_hours = (datetime.utcnow() - backups[0].created_at).total_seconds() / 3600
           if age_hours > 26:  # Should have daily backup
               print(f'ALERT: Last backup is {age_hours:.1f} hours old')

   asyncio.run(health_check())
   "
   ```

2. **Restore Testing Automation**
   ```bash
   # Weekly restore test
   #!/bin/bash

   # Get latest backup
   backup_id=$(python -m packages.core.backup.backup_cli list --limit 1 --format json \
               | jq -r '.backups[0].backup_id')

   # Test restore in isolated environment
   if python -m packages.core.backup.backup_cli restore $backup_id \
        --type component --component configurations --strategy test_restore; then
       echo "Weekly restore test: PASSED"
   else
       echo "Weekly restore test: FAILED" | mail -s "Restore Test Alert" admin@company.com
   fi
   ```

---

## Emergency Contacts

- **Primary On-Call**: [Your on-call system]
- **Backup Administrator**: [Backup admin contact]
- **Infrastructure Team**: [Infrastructure contact]
- **Security Team**: [Security team contact]

## Documentation Updates

This runbook should be reviewed and updated:
- After any restore procedure
- Monthly during backup testing
- After system architecture changes
- Following any incidents or outages

Last Updated: 2025-08-19
Version: 1.0
Next Review: 2025-09-19
