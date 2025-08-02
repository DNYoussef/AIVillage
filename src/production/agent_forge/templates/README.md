# Agent Templates

This directory contains JSON templates for the 18 agents in the production
Agent Forge system. Each file follows the `<agent>_template.json` naming
pattern and describes the agent's capabilities, behaviors, and resource needs.

## Usage

1. **Create or modify a template** in this directory using the naming pattern
   `<agent>_template.json`.
2. **Register the agent** by adding its identifier to `master_config.json`
   under `agent_types`.
3. **Instantiate agents** through `AgentFactory`, which automatically loads
   templates from this directory.

### Template Structure

```json
{
  "agent_id": "king",
  "specification": {
    "name": "King",
    "description": "Task orchestration and job scheduling leader",
    "primary_capabilities": [
      "task_orchestration",
      "resource_allocation",
      "decision_making"
    ],
    "secondary_capabilities": [
      "strategic_planning",
      "conflict_resolution"
    ],
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

## Related Code

Templates are consumed by `AgentFactory` in
`src/production/agent_forge/agent_factory.py` to create specialized agents
based on these specifications.
