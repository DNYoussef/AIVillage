# Documentation Cleanup Plan

## Current State Analysis

### Total Documentation Files: 218 markdown files

### Categories:
1. **Root Level Clutter** (50+ files) - Reports, summaries, and temporary docs
2. **Duplicate README versions** - README.md, README_backup.md, README_final.md, README_updated.md
3. **Sprint/Report Files** - Various *_REPORT.md, *_SUMMARY.md files
4. **Deprecated Documentation** - Already partially archived but needs completion
5. **Tool-specific Documentation** - Scattered across various directories
6. **Hidden Directories** - .claude/, .cleanup_backups/, etc. with duplicate content

## Cleanup Strategy

### 1. Root Level Documentation
**Action**: Move to appropriate subdirectories or archive

**Keep at Root**:
- README.md (primary, updated version)
- LICENSE
- CHANGELOG.md
- CONTRIBUTING.md
- STYLE_GUIDE.md

**Archive to deprecated/old_reports/**:
- All *_REPORT.md files
- All *_SUMMARY.md files
- All *_DASHBOARD.md files
- All SPRINT*_*.md files

**Move to docs/**:
- HONEST_STATUS.md → docs/project_status.md
- DEPENDENCY_MIGRATION.md → docs/guides/dependency_migration.md
- All technical guides and plans

### 2. Consolidate docs/ Directory

**New Structure**:
```
docs/
├── README.md (documentation index)
├── architecture/
│   ├── system_overview.md
│   ├── architecture.md
│   ├── component_design.md
│   └── adr/ (architecture decision records)
├── guides/
│   ├── getting_started.md
│   ├── installation.md
│   ├── deployment.md
│   ├── testing.md
│   └── contribution.md
├── api/
│   ├── rest_api.md
│   ├── python_api.md
│   └── mcp_api.md
├── components/
│   ├── agent_forge.md
│   ├── compression.md
│   ├── evolution.md
│   ├── mesh_network.md
│   └── rag_system.md
├── development/
│   ├── roadmap.md
│   ├── changelog.md
│   └── known_issues.md
└── reference/
    ├── glossary.md
    ├── faq.md
    └── troubleshooting.md
```

### 3. Component-Specific Documentation

**Action**: Move to component directories

- agent_forge/README.md - Keep, ensure updated
- mcp_servers/hyperag/README.md - Keep, ensure updated
- experimental/*/README.md - Keep for each experimental component
- production/*/README.md - Add where missing

### 4. Hidden Directory Cleanup

**Delete Entirely**:
- .claude_analysis/
- .claude_cleanup/
- .cleanup_analysis/
- .cleanup_backups/
- .test_repair_backup/

**Keep but Consolidate**:
- .claude/ → Move agent definitions to docs/agents/
- .github/ - Keep as is (GitHub specific)

### 5. Archive Structure

```
deprecated/
├── README.md (explains archive)
├── archived_claims/ (misleading docs)
├── old_reports/ (historical reports)
├── old_guides/ (outdated guides)
└── sprint_archives/ (sprint-specific docs)
```

## Execution Steps

1. **Backup Current State**
2. **Create New Directory Structure**
3. **Move/Archive Root Level Files**
4. **Consolidate docs/ Directory**
5. **Clean Hidden Directories**
6. **Update Cross-References**
7. **Create Documentation Index**
8. **Verify No Broken Links**

## Success Criteria

- No more than 10 files at root level
- Clear documentation hierarchy in docs/
- All reports archived appropriately
- No duplicate documentation
- Component-specific docs with components
- Clean, navigable structure
