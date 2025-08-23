# 🔄 Emergency Rollback Procedures
## AIVillage System Recovery Guide

**Document Version**: 1.0  
**Last Updated**: August 23, 2025  
**Emergency Contact**: System Administrator  
**Recovery Time Objective (RTO)**: < 15 minutes  

⚠️ **CRITICAL**: Use these procedures only in production emergencies

---

## 🚨 When to Execute Rollback

### **Immediate Rollback Required:**
- System performance degraded >50% from baseline
- Critical component failures affecting user experience  
- Security vulnerabilities discovered in new components
- Data corruption or loss detected
- Cascade failures across multiple systems

### **Rollback Not Required:**
- Individual component issues (use component isolation instead)
- Performance issues <25% degradation from baseline
- Non-critical feature failures
- Minor configuration issues

---

## 🎯 Rollback Decision Matrix

| Severity | User Impact | Downtime | Recovery Action |
|----------|-------------|-----------|-----------------|
| **Critical** | >50% users affected | >5 minutes | IMMEDIATE ROLLBACK |
| **High** | 10-50% users affected | 1-5 minutes | Component isolation first, then rollback if needed |
| **Medium** | <10% users affected | <1 minute | Fix in place, schedule rollback if needed |
| **Low** | No user impact | None | Monitor and fix in next maintenance window |

---

## 🔧 Pre-Rollback Checklist

### **Before Starting Rollback** (2 minutes):

1. **☐ Assess Impact**
   - Identify affected components
   - Estimate user impact percentage
   - Document current system state

2. **☐ Notify Stakeholders**
   - Alert system administrators
   - Notify development team
   - Update status page if applicable

3. **☐ Backup Current State**
   - Export current configuration
   - Save recent logs for post-incident analysis
   - Document timeline of events

4. **☐ Validate Rollback Readiness**
   - Confirm backup files exist
   - Verify rollback procedures tested
   - Check rollback dependencies available

---

## 🚀 Rollback Execution Procedures

### **Phase 1: Component Isolation** (2-3 minutes)

#### **Isolate Failed Components:**
```bash
# Stop failed services
systemctl stop aivillage-gateway      # If gateway issues
systemctl stop aivillage-p2p          # If P2P issues  
systemctl stop aivillage-agents       # If agent issues
systemctl stop aivillage-rag          # If knowledge issues

# Verify isolation successful
systemctl status aivillage-*
```

#### **Enable Maintenance Mode:**
```bash
# Create maintenance page
echo "System maintenance in progress" > /var/www/html/maintenance.html

# Route traffic to maintenance page
nginx -s reload
```

### **Phase 2: File System Rollback** (5-7 minutes)

#### **Restore from Backup Files:**
```bash
# Navigate to system directory
cd /opt/aivillage

# Restore all backup files (created during deployment)
find . -name "*.backup" -exec sh -c 'mv "$1" "${1%.backup}"' _ {} \;

# Verify restoration
find . -name "*.backup" | wc -l  # Should show 0 if all restored
```

#### **Restore Specific Components:**
```bash
# Gateway rollback
cp core/gateway/server.py.backup core/gateway/server.py

# Agent system rollback  
rm -rf packages/core/legacy/error_handling.py
cp core/agents/cognative_nexus_controller.py.backup core/agents/cognative_nexus_controller.py

# P2P system rollback
cp core/p2p/mesh_protocol.py.backup core/p2p/mesh_protocol.py

# HyperRAG rollback
cp core/rag/hyper_rag.py.backup core/rag/hyper_rag.py
```

### **Phase 3: Configuration Rollback** (2-3 minutes)

#### **Restore System Configuration:**
```bash
# Restore environment variables
cp config/.env.backup config/.env

# Restore service configurations
cp config/systemd/*.backup /etc/systemd/system/
systemctl daemon-reload

# Restore database connections (if applicable)
cp config/database_config.json.backup config/database_config.json
```

#### **Restore Import Paths:**
```bash
# Run import restoration script
python scripts/restore_imports.py --from-backup

# Verify import consistency
python -c "import sys; sys.path.append('.'); from core.gateway.server import app; print('Gateway imports OK')"
```

### **Phase 4: Service Restoration** (3-5 minutes)

#### **Restart Services in Order:**
```bash
# 1. Start core gateway first
systemctl start aivillage-gateway
systemctl status aivillage-gateway

# 2. Start P2P network
systemctl start aivillage-p2p  
systemctl status aivillage-p2p

# 3. Start agent system
systemctl start aivillage-agents
systemctl status aivillage-agents

# 4. Start knowledge system (if stable)
systemctl start aivillage-rag
systemctl status aivillage-rag
```

#### **Health Check Validation:**
```bash
# Gateway health check
curl -f http://localhost:8000/health || echo "Gateway health check failed"

# P2P connectivity test
python scripts/test_p2p_connection.py

# Agent system test
python scripts/test_agent_response.py

# End-to-end test
python scripts/test_full_system.py
```

---

## 🧪 Rollback Validation

### **System Health Verification** (2 minutes):

#### **Performance Baseline Check:**
```python
# Run quick performance validation
python -c "
import time
import sys
sys.path.append('.')

# Test gateway
from core.gateway.server import GatewayConfig
start = time.perf_counter()
config = GatewayConfig()
gateway_time = (time.perf_counter() - start) * 1000
print(f'Gateway: {gateway_time:.2f}ms (should be <100ms)')

# Test P2P
from core.p2p.mesh_protocol import UnifiedMeshProtocol
start = time.perf_counter()  
mesh = UnifiedMeshProtocol(node_id='rollback_test')
p2p_time = (time.perf_counter() - start) * 1000
print(f'P2P: {p2p_time:.2f}ms (should be <100ms)')

print('Rollback validation complete')
"
```

#### **User Experience Test:**
```bash
# Simulate user request
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test rollback system", "user_id": "rollback_test"}'

# Check response time (should be <2s)
time curl -f http://localhost:8000/health
```

### **Success Criteria:**
- ✅ All services start successfully
- ✅ Health checks return OK
- ✅ Response times within baseline (Gateway <100ms, P2P <100ms)  
- ✅ No error logs in past 5 minutes
- ✅ User requests processed successfully

---

## 📊 Post-Rollback Actions

### **Immediate Actions** (15 minutes):

1. **☐ System Monitoring**
   - Monitor system for 15 minutes minimum
   - Watch for error patterns or performance issues
   - Verify user experience restored

2. **☐ Stakeholder Communication**
   - Update status page: "System restored"
   - Notify team of successful rollback
   - Provide preliminary incident summary

3. **☐ Log Collection**
   - Export logs from incident period
   - Save rollback execution logs
   - Document rollback timeline

### **Follow-up Actions** (24 hours):

1. **☐ Root Cause Analysis**
   - Analyze what caused the need for rollback
   - Identify prevention measures
   - Update deployment procedures if needed

2. **☐ Rollback Procedure Review**
   - Document what worked well in rollback
   - Identify rollback procedure improvements
   - Update rollback documentation

3. **☐ System Hardening**
   - Implement additional monitoring
   - Add health checks to prevent future issues
   - Consider circuit breakers for problematic components

---

## 🔍 Troubleshooting Common Rollback Issues

### **Issue 1: Backup Files Missing**
**Symptoms**: `*.backup` files not found  
**Cause**: Backup process not run during deployment  
**Solution**:
```bash
# Use git to restore previous version
git checkout HEAD~1 -- core/gateway/server.py
git checkout HEAD~1 -- core/p2p/mesh_protocol.py
# Continue with service restart
```

### **Issue 2: Services Won't Start**  
**Symptoms**: `systemctl start` fails  
**Cause**: Configuration file corruption  
**Solution**:
```bash
# Reset to default configuration
cp config/defaults/* config/
systemctl daemon-reload
systemctl start aivillage-*
```

### **Issue 3: Import Errors After Rollback**
**Symptoms**: Python import failures  
**Cause**: Import path changes not fully reverted  
**Solution**:
```bash
# Clear Python cache
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Restart Python processes
systemctl restart aivillage-*
```

### **Issue 4: Database Connection Issues**
**Symptoms**: Database connectivity failures  
**Cause**: Database migration not reverted  
**Solution**:
```bash
# Check database status
systemctl status postgresql

# Restore database backup if needed
pg_restore -d aivillage /backups/aivillage_backup.sql
```

---

## ⚡ Emergency Contacts

### **Escalation Chain:**

#### **Level 1: System Administrator** (Primary)
- **Contact**: System Admin  
- **Response Time**: < 5 minutes
- **Scope**: All rollback procedures, service management

#### **Level 2: Development Team Lead** (Secondary)  
- **Contact**: Dev Team Lead
- **Response Time**: < 15 minutes  
- **Scope**: Code issues, complex system problems

#### **Level 3: On-Call Engineer** (Emergency)
- **Contact**: On-call Engineer
- **Response Time**: < 30 minutes
- **Scope**: Critical system failures, data recovery

### **Emergency Procedures:**
```bash
# Page on-call engineer
curl -X POST https://pager.service.com/incidents \
  -H "Authorization: Bearer $PAGER_TOKEN" \
  -d '{"message": "AIVillage system rollback required", "severity": "critical"}'

# Send emergency notification  
echo "EMERGENCY: AIVillage rollback in progress" | mail -s "CRITICAL: System Rollback" team@aivillage.com
```

---

## 📚 Reference Documentation

### **Related Documents:**
- **System Architecture**: `docs/architecture/ARCHITECTURE.md`
- **Deployment Procedures**: `docs/deployment/PRODUCTION_GUIDE.md`
- **Monitoring Guide**: `docs/monitoring/SYSTEM_HEALTH.md`
- **Incident Response**: `docs/operations/INCIDENT_RESPONSE.md`

### **Test Rollback Procedures:**
```bash
# Run rollback simulation (safe)
python scripts/simulate_rollback.py --dry-run

# Validate rollback readiness
python scripts/validate_rollback_capability.py
```

---

## ✅ Rollback Checklist Summary

### **Pre-Rollback** (2 min):
- [ ] Impact assessment completed
- [ ] Stakeholders notified  
- [ ] Current state backed up
- [ ] Rollback readiness confirmed

### **Rollback Execution** (10-15 min):
- [ ] Components isolated
- [ ] Files restored from backup
- [ ] Configuration reverted
- [ ] Services restarted in order
- [ ] Health checks passing

### **Post-Rollback** (15+ min):
- [ ] System monitoring active
- [ ] User experience validated
- [ ] Stakeholders updated
- [ ] Incident documentation started

### **Follow-up** (24 hrs):
- [ ] Root cause analysis completed
- [ ] Rollback procedure reviewed
- [ ] System hardening implemented

---

**Document Maintained By**: Operations Team  
**Review Frequency**: Quarterly or after each rollback execution  
**Last Tested**: System deployment (August 23, 2025)  
**Next Review**: November 2025