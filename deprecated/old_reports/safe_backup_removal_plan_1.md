# Safe .Backup Files Removal Plan
## Phased Approach with Verification and Rollback

**Plan Created:** 2025-07-18 12:34:15 UTC
**Based on:** cleanup_report.md risk assessment

---

## Phase 1: Pre-Removal Safety Setup (5 minutes)

### 1.1 Create Safety Backup
```bash
# Create a compressed backup of all .backup files before removal
cd c:/Users/17175/Desktop/AIVillage
mkdir -p safety_archive/$(date +%Y%m%d_%H%M%S)
find . -name "*.backup*" -type f -exec cp --parents {} safety_archive/$(date +%Y%m%d_%H%M%S)/ \;
tar -czf safety_archive/backup_files_$(date +%Y%m%d_%H%M%S).tar.gz safety_archive/$(date +%Y%m%d_%H%M%S)/
```

### 1.2 Verification Script Setup
```bash
# Create verification script
cat > verify_before_removal.sh << 'EOF'
#!/bin/bash
echo "=== BACKUP VERIFICATION REPORT ==="
echo "Date: $(date)"
echo "Total .backup files found:"
find . -name "*.backup*" -type f | wc -l
echo ""
echo "Files by category:"
echo "Test files: $(find . -name "*.backup*" -path "*/tests/*" | wc -l)"
echo "Migration files: $(find . -name "*migrate*.backup*" | wc -l)"
echo "Source code: $(find . -name "*.backup*" -not -path "*/tests/*" -not -name "*migrate*" | wc -l)"
echo ""
echo "File sizes:"
find . -name "*.backup*" -type f -exec ls -lh {} \; | awk '{print $5 " " $9}'
EOF
chmod +x verify_before_removal.sh
```

---

## Phase 2: Safe Removal - LOW RISK (5 minutes)

### 2.1 Remove Test & Migration Backups (9 files)
```bash
# Execute safe removal for LOW risk files
echo "Removing LOW risk test and migration backups..."

# Test backups
rm -f tests/test_king_agent.py.backup
rm -f tests/test_king_agent_simple.py.backup
rm -f tests/test_layer_sequence.py.backup
rm -f tests/test_protocol.py.backup
rm -f tests/agents/test_evidence_flow.py.backup
rm -f agents/king/tests/test_integration.py.backup
rm -f agents/king/tests/test_king_agent.py.backup

# Migration script backup
rm -f scripts/migrate_error_handling.py.backup

# Verification
echo "LOW risk files removed. Remaining .backup files:"
find . -name "*.backup*" -type f | wc -l
```

### 2.2 Immediate Validation
```bash
# Quick system check
python -c "import sys; print('Python imports working')"
# Run a basic test to ensure no functionality broken
python -m pytest tests/test_king_agent.py -v || echo "Tests may need adjustment"
```

---

## Phase 3: MEDIUM RISK Review (10 minutes)

### 3.1 Configuration Backup Review
```bash
# Review configuration backups before removal
echo "Reviewing MEDIUM risk configuration backups..."

# Check if originals exist and are newer
for file in communications/community_hub.py.backup communications/protocol.py.backup rag_system/main.py.backup; do
    original="${file%.backup}"
    if [ -f "$original" ]; then
        echo "✓ Original exists: $original"
        echo "  Original modified: $(stat -c %y $original)"
        echo "  Backup modified: $(stat -c %y $file)"
        if [ "$original" -nt "$file" ]; then
            echo "  ✓ Original is newer - safe to remove backup"
        else
            echo "  ⚠ Original is older - review needed"
        fi
    else
        echo "⚠ Original missing: $original - manual review required"
    fi
done
```

### 3.2 Archive Configuration Backups
```bash
# Create archive for medium risk files
mkdir -p archive/config_backups_$(date +%Y%m%d)
mv communications/*.backup archive/config_backups_$(date +%Y%m%d)/
mv rag_system/main.py.backup archive/config_backups_$(date +%Y%m%d)/
echo "Configuration backups archived to: archive/config_backups_$(date +%Y%m%d)/"
```

---

## Phase 4: HIGH RISK Source Code Review (15-30 minutes)

### 4.1 Systematic Review Process
```bash
# Create review checklist
cat > high_risk_review.md << 'EOF'
# HIGH RISK Source Code Backup Review

## Review Process for Each File:
1. Compare backup with current file
2. Check git history for changes
3. Verify no active development branches use backup
4. Document any significant differences

## Files to Review (24 total):
- main.py.backup
- agents/orchestration.py.backup
- agents/unified_base_agent.py.backup
- [continue for all 24 high-risk files...]

## Decision Matrix:
- If current file is newer AND no significant differences → REMOVE
- If current file is newer BUT has significant differences → ARCHIVE
- If current file is missing → MANUAL REVIEW REQUIRED
EOF
```

### 4.2 Automated Comparison Script
```bash
# Create comparison script
cat > compare_backups.sh << 'EOF'
#!/bin/bash
echo "=== BACKUP COMPARISON REPORT ==="
for backup in $(find . -name "*.backup" -not -path "*/tests/*" -not -path "*/archive/*"); do
    original="${backup%.backup}"
    echo "Checking: $backup"

    if [ ! -f "$original" ]; then
        echo "  ⚠ MISSING: $original"
        continue
    fi

    if cmp -s "$original" "$backup"; then
        echo "  ✓ IDENTICAL: Safe to remove"
    else
        echo "  ⚠ DIFFERENT: Review required"
        echo "    Diff size: $(diff "$original" "$backup" | wc -l) lines"
    fi
done
EOF
chmod +x compare_backups.sh
```

---

## Phase 5: Final Validation (5 minutes)

### 5.1 System Integrity Check
```bash
# Comprehensive system check
echo "=== FINAL VALIDATION ==="

# Check for any remaining .backup files
remaining=$(find . -name "*.backup*" -type f | wc -l)
echo "Remaining .backup files: $remaining"

# Run critical system tests
python -c "
import os
import sys
try:
    # Test core imports
    from agents.unified_base_agent import UnifiedBaseAgent
    from core.communication import CommunicationProtocol
    print('✓ Core system imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

# Verify no broken symlinks or missing dependencies
find . -name "*.py" -exec python -m py_compile {} \; 2>/dev/null || echo "Some Python files have syntax issues"
```

---

## Rollback Procedures

### Emergency Rollback
```bash
# If issues arise, restore from safety archive
cd c:/Users/17175/Desktop/AIVillage
latest_archive=$(ls -t safety_archive/*.tar.gz | head -1)
if [ -n "$latest_archive" ]; then
    tar -xzf "$latest_archive" -C .
    echo "Restored from: $latest_archive"
else
    echo "No safety archive found for rollback"
fi
```

### Selective Restore
```bash
# Restore specific files if needed
restore_file="agents/unified_base_agent.py.backup"
if [ -f "safety_archive/*/$restore_file" ]; then
    cp safety_archive/*/$restore_file $restore_file
    echo "Restored: $restore_file"
fi
```

---

## Execution Timeline

| Phase | Duration | Action | Verification |
|-------|----------|--------|--------------|
| **Phase 1** | 5 min | Safety setup | Archive created |
| **Phase 2** | 5 min | Remove LOW risk | 9 files removed |
| **Phase 3** | 10 min | Archive MEDIUM risk | 4 files archived |
| **Phase 4** | 15-30 min | Review HIGH risk | 24 files reviewed |
| **Phase 5** | 5 min | Final validation | System integrity check |

**Total Estimated Time:** 40-55 minutes

---

## Safety Checklist

- [ ] Safety backup created in `safety_archive/`
- [ ] Verification scripts created and executable
- [ ] All team members notified of cleanup window
- [ ] Rollback procedures tested
- [ ] System functionality verified after each phase
- [ ] Documentation updated with removal log

---

## Quick Start Commands

```bash
# Run entire plan
./verify_before_removal.sh
./safe_backup_removal_plan.sh

# Or step by step:
# Phase 1: Setup
mkdir -p safety_archive && find . -name "*.backup*" -exec cp --parents {} safety_archive/ \;

# Phase 2: Safe removal
rm -f tests/*.backup* scripts/migrate_error_handling.py.backup

# Phase 3: Archive configs
mkdir -p archive/config_backups_$(date +%Y%m%d)
mv communications/*.backup archive/config_backups_$(date +%Y%m%d)/
```

**Emergency Contact:** If any issues arise, restore from safety_archive immediately.
