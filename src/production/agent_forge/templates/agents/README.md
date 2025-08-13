# DEPRECATED: Agent Template Subdirectory

⚠️ **This subdirectory has been consolidated for repository cleanup.**

## What Changed

The agent templates in this subdirectory contained simplified versions of the comprehensive templates available in the parent directory. To eliminate duplication and maintain a single source of truth, all agent templates have been consolidated.

## Migration Guide

### Old Path (Deprecated):
```
src/production/agent_forge/templates/agents/{name}.json
```

### New Path (Canonical):
```
src/production/agent_forge/templates/{name}_template.json
```

## Template Format Differences

**Old Format (agents/*.json):**
```json
{
  "name": "King Agent",
  "role": "Coordinates overall system strategy.",
  "default_params": {}
}
```

**New Format (*_template.json):**
```json
{
  "agent_id": "king",
  "specification": {
    "name": "King",
    "description": "Task orchestration and job scheduling leader",
    "primary_capabilities": ["task_orchestration", "resource_allocation", "decision_making"],
    "secondary_capabilities": ["strategic_planning", "conflict_resolution"],
    "behavioral_traits": {
      "leadership_style": "collaborative",
      "decision_speed": "balanced",
      "delegation_preference": "high"
    },
    "resource_requirements": {
      "cpu": "high",
      "memory": "medium",
      "network": "high",
      "storage": "low"
    }
  }
}
```

## Benefits

1. **Single Source of Truth**: One comprehensive template per agent
2. **Rich Specifications**: Full capability and behavioral trait definitions
3. **Resource Planning**: Detailed resource requirement specifications
4. **Consistency**: Standardized format across all agent types
5. **Reduced Duplication**: 288 lines of code savings

## Available Agent Templates

All 18 agent types are available in the parent directory:
- auditor_template.json
- curator_template.json
- ensemble_template.json
- gardener_template.json
- king_template.json
- legal_template.json
- magi_template.json
- maker_template.json
- medic_template.json
- navigator_template.json
- oracle_template.json
- polyglot_template.json
- sage_template.json
- shaman_template.json
- strategist_template.json
- sustainer_template.json
- sword_shield_template.json
- tutor_template.json

## Backup Location

Original simplified templates have been preserved at:
```
deprecated/agent_templates/agents_subdirectory_backup/
```

---
*Consolidated on 2025-01-26 as part of repository bloat cleanup initiative*
