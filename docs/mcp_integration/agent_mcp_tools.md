# MCP Integration - Agent MCP Tools

## Overview

This document provides a comprehensive reference for all Model Control Protocol (MCP) tools available to AIVillage's 23 specialized agents. These tools enable unified access to all AIVillage systems through standardized interfaces, ensuring consistent agent behavior and democratic governance.

## 🧠 Universal Agent Tools

All 23 specialized agents have access to these core MCP tools through the unified interface:

### Knowledge & Memory Tools

#### `hyperrag_query`
**Purpose**: Query the HyperRAG knowledge system
**Permission Level**: READ_ONLY
**Available to**: All agents

```python
# Agent usage example
result = await agent.mcp_client.call("hyperrag_query", {
    "query": "What are the current system performance metrics?",
    "context": "Need data for decision making",
    "mode": "comprehensive"  # fast, balanced, comprehensive
})

# Response format
{
    "request_id": "query-uuid-here",
    "status": "success",
    "mode_used": "comprehensive",
    "result": {
        "answer": "System performance shows 85% efficiency...",
        "confidence": 0.92,
        "reasoning_path": [...],
        "sources": [...]
    },
    "guardian_decision": {
        "action": "APPLY",
        "semantic_score": 0.9,
        "utility_score": 0.85,
        "safety_score": 0.95
    },
    "metadata": {"processing_time_ms": 145.2}
}
```

#### `hyperrag_memory`
**Purpose**: Store and retrieve agent memories
**Permission Level**: OPERATOR (write), READ_ONLY (read)
**Available to**: All agents

```python
# Store memory
result = await agent.mcp_client.call("hyperrag_memory", {
    "action": "store",
    "content": "Learned that gradient compression improves federated learning efficiency",
    "tags": ["learning", "optimization", "federated"],
    "belief": 0.85  # Confidence level
})

# Search memories
result = await agent.mcp_client.call("hyperrag_memory", {
    "action": "search",
    "content": "federated learning optimization",
    "limit": 10
})

# Response format
{
    "status": "stored",
    "item_id": "memory-uuid-here",
    "results": [
        {"id": "mem-1", "content": "...", "confidence": 0.85}
    ]
}
```

#### `trust_network_query`
**Purpose**: Query Bayesian trust networks for knowledge validation
**Permission Level**: READ_ONLY
**Available to**: All agents

```python
result = await agent.mcp_client.call("trust_network_query", {
    "query": "model compression techniques reliability",
    "trust_threshold": 0.7,
    "include_probabilistic_reasoning": True
})
```

#### `knowledge_elevation`
**Purpose**: Elevate local knowledge to global distributed RAG
**Permission Level**: OPERATOR
**Available to**: All agents

```python
result = await agent.mcp_client.call("knowledge_elevation", {
    "content": "Novel approach to model quantization discovered",
    "confidence": 0.9,
    "domain": "model_compression",
    "privacy_level": "fully_anonymized"
})
```

### Communication & Coordination Tools

#### `agent_communication`
**Purpose**: Inter-agent messaging via P2P protocols
**Permission Level**: OPERATOR
**Available to**: All agents

```python
result = await agent.mcp_client.call("agent_communication", {
    "action": "send_message",
    "target_agent": "magi",
    "message": {
        "type": "research_request",
        "content": "Need analysis of compression pipeline performance",
        "priority": "normal",
        "response_required": True
    },
    "transport_preference": ["bitchat", "betanet"]
})

# Receive messages
result = await agent.mcp_client.call("agent_communication", {
    "action": "receive_messages",
    "filter": {"from_agent": "sage", "type": "governance"}
})
```

#### `fog_coordination`
**Purpose**: Coordinate with fog compute nodes for resource allocation
**Permission Level**: COORDINATOR
**Available to**: Magi, Oracle, King, Coordinator agents

```python
result = await agent.mcp_client.call("fog_coordination", {
    "action": "request_resources",
    "resource_requirements": {
        "cpu_cores": 4,
        "memory_gb": 8,
        "gpu_required": False,
        "max_latency_ms": 100
    },
    "task_description": "Model inference for knowledge analysis"
})
```

#### `resource_request`
**Purpose**: Request computational resources from system
**Permission Level**: OPERATOR
**Available to**: All agents

```python
result = await agent.mcp_client.call("resource_request", {
    "agent_id": "sage",
    "cpu_cores": 2,
    "memory_mb": 4096,
    "gpu_required": False,
    "duration_minutes": 60,
    "priority": "normal"
})
```

#### `status_broadcast`
**Purpose**: Broadcast agent status updates to system
**Permission Level**: READ_ONLY
**Available to**: All agents

```python
result = await agent.mcp_client.call("status_broadcast", {
    "status": "active",
    "current_task": "analyzing system performance",
    "health_score": 0.95,
    "resource_usage": {"cpu": 0.3, "memory": 0.6},
    "capabilities": ["analysis", "research", "governance"]
})
```

### Governance & Democracy Tools

#### `governance_proposal`
**Purpose**: Create governance proposals for system changes
**Permission Level**: GOVERNANCE
**Available to**: Sage, Curator, King

```python
result = await agent.mcp_client.call("governance_proposal", {
    "title": "Update Knowledge Retention Policy",
    "description": "Extend knowledge retention from 30 days to 90 days for improved agent performance",
    "type": "major_change",  # minor_change, major_change, critical_change
    "impact_assessment": {
        "affected_systems": ["hyperrag", "digital_twin"],
        "risk_level": "low",
        "reversible": True,
        "estimated_implementation_time": "2 hours"
    },
    "proposed_changes": {
        "config_updates": {"retention_days": 90},
        "affected_components": ["rag_system", "memory_management"]
    }
})

# Response format
{
    "proposal_id": "prop-uuid-here",
    "status": "created",
    "required_votes": 2,
    "voting_deadline": "2025-08-19T10:00:00Z",
    "current_votes": {"for": [], "against": []},
    "proposal_details": {...}
}
```

#### `governance_vote`
**Purpose**: Vote on governance proposals
**Permission Level**: GOVERNANCE
**Available to**: Sage, Curator, King

```python
result = await agent.mcp_client.call("governance_vote", {
    "proposal_id": "prop-uuid-here",
    "vote": "approve",  # approve, reject
    "reasoning": "Knowledge retention extension will improve agent decision-making capabilities",
    "priority": "normal",
    "conditions": []  # Optional conditions for approval
})

# Response format
{
    "vote_recorded": True,
    "proposal_status": "approved",  # voting_ongoing, approved, rejected
    "final_result": {
        "votes_for": ["sage", "curator"],
        "votes_against": [],
        "total_votes": 2,
        "required_votes": 2,
        "status": "approved"
    }
}
```

#### `policy_query`
**Purpose**: Query current system policies and configurations
**Permission Level**: READ_ONLY
**Available to**: All agents

```python
result = await agent.mcp_client.call("policy_query", {
    "policy_category": "knowledge_management",  # all, governance, security, privacy
    "include_history": False,
    "format": "structured"
})
```

#### `compliance_audit`
**Purpose**: Trigger compliance audits and access audit trails
**Permission Level**: GOVERNANCE
**Available to**: Sage, Curator, King

```python
result = await agent.mcp_client.call("compliance_audit", {
    "audit_type": "privacy",  # privacy, security, governance, full
    "scope": ["digital_twins", "rag_system"],
    "time_range": {"start": "2025-08-01", "end": "2025-08-18"},
    "include_recommendations": True
})
```

### System Management Tools

#### `system_overview`
**Purpose**: Get comprehensive system status and health metrics
**Permission Level**: READ_ONLY
**Available to**: All agents

```python
result = await agent.mcp_client.call("system_overview", {
    "include_components": "all",  # all, core, optional
    "detail_level": "summary",  # basic, summary, detailed
    "include_alerts": True,
    "include_performance": True
})

# Response format
{
    "system_health": 0.87,
    "component_status": {
        "digital_twins": {"status": "operational", "health_score": 0.9},
        "meta_agents": {"status": "operational", "health_score": 0.85},
        "distributed_rag": {"status": "operational", "health_score": 0.8},
        "p2p_network": {"status": "operational", "health_score": 0.75},
        "fog_compute": {"status": "operational", "health_score": 0.7}
    },
    "active_alerts": 2,
    "governance_status": {
        "active_proposals": 1,
        "authorized_agents": ["sage", "curator", "king"]
    },
    "privacy_compliance": {
        "data_protection": True,
        "local_data_only": True,
        "differential_privacy": True
    }
}
```

#### `resource_allocation`
**Purpose**: Manage system resource allocation
**Permission Level**: COORDINATOR
**Available to**: Magi, Oracle, King, Coordinator agents

```python
result = await agent.mcp_client.call("resource_allocation", {
    "action": "optimize",  # view, allocate, optimize, release
    "target_system": "fog_compute",
    "optimization_criteria": ["performance", "efficiency", "battery"],
    "constraints": {"max_cpu_usage": 0.8, "preserve_battery": True}
})
```

#### `performance_metrics`
**Purpose**: Access detailed system performance data
**Permission Level**: READ_ONLY
**Available to**: All agents

```python
result = await agent.mcp_client.call("performance_metrics", {
    "metrics": ["latency", "throughput", "resource_usage", "error_rates"],
    "time_range": "last_24_hours",
    "granularity": "hourly",
    "components": ["hyperrag", "p2p_network", "digital_twins"]
})
```

#### `emergency_override`
**Purpose**: Emergency system controls and overrides
**Permission Level**: EMERGENCY
**Available to**: King agent only

```python
result = await agent.mcp_client.call("emergency_override", {
    "override_action": "emergency_shutdown",  # emergency_shutdown, force_restart, privacy_lockdown
    "target_system": "all_systems",  # all_systems, specific component
    "justification": "Critical security vulnerability detected requiring immediate system shutdown",
    "estimated_downtime": "30_minutes",
    "recovery_plan": "Apply security patches and restart with enhanced monitoring"
})

# Response format
{
    "success": True,
    "emergency_id": "emerg-uuid-here",
    "action_taken": "emergency_shutdown_initiated",
    "affected_systems": ["all"],
    "estimated_completion": "2025-08-18T14:30:00Z",
    "recovery_instructions": "..."
}
```

## 🤖 Specialized Agent Tool Extensions

Different agent types have access to specialized toolsets beyond the universal tools:

### King Agent (Coordination) - Governance Level: EMERGENCY

```python
# Additional tools exclusive to King Agent
specialized_tools = [
    "emergency_system_shutdown",     # Emergency system shutdown
    "king_override_vote",           # Override democratic decisions
    "crisis_management",            # Crisis response coordination
    "system_recovery",              # System recovery procedures
    "agent_authorization_management", # Manage agent permissions
    "emergency_resource_requisition" # Emergency resource allocation
]

# Example: Crisis management
result = await king_agent.mcp_client.call("crisis_management", {
    "crisis_type": "security_breach",
    "severity": "high",
    "affected_systems": ["rag_system", "p2p_network"],
    "immediate_actions": ["isolate_affected_systems", "enable_emergency_protocols"],
    "coordination_required": True
})
```

### Sage Agent (Research) - Governance Level: GOVERNANCE

```python
# Research and knowledge-focused tools
specialized_tools = [
    "research_knowledge_gap",       # Identify knowledge gaps
    "auto_research_update",         # Automated research updates
    "knowledge_synthesis",          # Synthesize research findings
    "fact_verification",            # Verify knowledge accuracy
    "research_proposal_creation",   # Create research proposals
    "literature_analysis"           # Analyze research literature
]

# Example: Research knowledge gap identification
result = await sage_agent.mcp_client.call("research_knowledge_gap", {
    "domain": "model_compression",
    "analysis_depth": "comprehensive",
    "include_recent_papers": True,
    "gap_types": ["technical", "methodological", "empirical"],
    "priority_threshold": 0.7
})
```

### Magi Agent (Analysis) - Governance Level: GOVERNANCE

```python
# Analysis and investigation tools
specialized_tools = [
    "deep_system_analysis",         # Comprehensive system analysis
    "pattern_recognition",          # Identify system patterns
    "predictive_modeling",          # Create predictive models
    "anomaly_detection",            # Detect system anomalies
    "performance_optimization",     # System optimization analysis
    "risk_assessment"               # Security and operational risk assessment
]

# Example: Deep system analysis
result = await magi_agent.mcp_client.call("deep_system_analysis", {
    "analysis_target": "compression_pipeline",
    "analysis_type": "performance_bottleneck",
    "include_predictions": True,
    "optimization_suggestions": True,
    "risk_factors": ["performance", "accuracy", "resource_usage"]
})
```

### Oracle Agent (Prediction) - Governance Level: COORDINATOR

```python
# Prediction and forecasting tools
specialized_tools = [
    "system_performance_prediction", # Predict system performance
    "resource_demand_forecasting",   # Forecast resource needs
    "failure_prediction",            # Predict potential failures
    "trend_analysis",                # Analyze system trends
    "capacity_planning",             # Plan system capacity
    "optimization_forecasting"       # Predict optimization outcomes
]

# Example: Resource demand forecasting
result = await oracle_agent.mcp_client.call("resource_demand_forecasting", {
    "forecast_horizon": "7_days",
    "resource_types": ["cpu", "memory", "network", "storage"],
    "include_peak_predictions": True,
    "confidence_intervals": True,
    "optimization_recommendations": True
})
```

### Curator Agent (Memory) - Governance Level: GOVERNANCE

```python
# Memory and knowledge management tools
specialized_tools = [
    "memory_optimization",          # Optimize memory usage
    "knowledge_organization",       # Organize knowledge structures
    "data_lifecycle_management",    # Manage data lifecycles
    "memory_garbage_collection",    # Clean up unused memories
    "knowledge_archival",           # Archive old knowledge
    "memory_integrity_check"        # Verify memory integrity
]

# Example: Knowledge organization
result = await curator_agent.mcp_client.call("knowledge_organization", {
    "organization_strategy": "semantic_clustering",
    "include_trust_scores": True,
    "merge_similar_knowledge": True,
    "create_knowledge_maps": True,
    "optimization_level": "balanced"
})
```

## 🔐 Security and Authentication

### JWT Token Structure

All MCP tool calls require valid JWT authentication:

```python
# JWT payload structure for agents
{
    "agent_id": "sage",
    "role": "governance",
    "governance_level": "GOVERNANCE",
    "permissions": [
        "hyperag:read",
        "hyperag:write",
        "hyperag:governance:vote",
        "hyperag:governance:propose"
    ],
    "exp": 1724073600,  # Expiration timestamp
    "aud": "mcp_aivillage"
}
```

### Permission Validation

Every tool call goes through permission validation:

```python
def validate_tool_permission(agent_id: str, tool_name: str, governance_level: str) -> bool:
    """Validate agent permission for MCP tool"""

    # Tool permission requirements
    tool_permissions = {
        "hyperrag_query": "READ_ONLY",
        "hyperrag_memory": "OPERATOR",
        "governance_proposal": "GOVERNANCE",
        "governance_vote": "GOVERNANCE",
        "emergency_override": "EMERGENCY",
        "resource_allocation": "COORDINATOR"
    }

    # Agent governance levels
    agent_levels = {
        "king": "EMERGENCY",
        "sage": "GOVERNANCE",
        "curator": "GOVERNANCE",
        "magi": "GOVERNANCE",
        "oracle": "COORDINATOR"
    }

    # Level hierarchy (higher numbers = more permissions)
    level_hierarchy = {
        "READ_ONLY": 1,
        "OPERATOR": 2,
        "COORDINATOR": 3,
        "GOVERNANCE": 4,
        "EMERGENCY": 5
    }

    required_level = tool_permissions.get(tool_name, "READ_ONLY")
    agent_level = agent_levels.get(agent_id, "READ_ONLY")

    return level_hierarchy[agent_level] >= level_hierarchy[required_level]
```

## 📊 Tool Usage Examples

### Multi-Agent Coordination Workflow

```python
async def multi_agent_research_workflow():
    """Example of multi-agent coordination using MCP tools"""

    # 1. Sage identifies research gap
    gap_analysis = await sage_agent.mcp_client.call("research_knowledge_gap", {
        "domain": "federated_learning_optimization",
        "priority_threshold": 0.8
    })

    # 2. Sage creates governance proposal for research initiative
    proposal = await sage_agent.mcp_client.call("governance_proposal", {
        "title": "Federated Learning Optimization Research Initiative",
        "description": f"Research gap identified: {gap_analysis['gaps'][0]['description']}",
        "type": "major_change",
        "resources_required": {"compute_hours": 100, "data_access": True}
    })

    # 3. Curator and King vote on proposal
    curator_vote = await curator_agent.mcp_client.call("governance_vote", {
        "proposal_id": proposal["proposal_id"],
        "vote": "approve",
        "reasoning": "Research aligns with knowledge management objectives"
    })

    king_vote = await king_agent.mcp_client.call("governance_vote", {
        "proposal_id": proposal["proposal_id"],
        "vote": "approve",
        "reasoning": "Approved for system improvement"
    })

    # 4. Magi performs deep analysis on approved research
    analysis = await magi_agent.mcp_client.call("deep_system_analysis", {
        "analysis_target": "federated_learning_performance",
        "analysis_type": "optimization_opportunity",
        "include_predictions": True
    })

    # 5. Oracle forecasts impact of proposed optimizations
    forecast = await oracle_agent.mcp_client.call("system_performance_prediction", {
        "scenario": "federated_learning_optimization_implemented",
        "prediction_horizon": "30_days",
        "metrics": ["throughput", "accuracy", "resource_usage"]
    })

    # 6. Sage synthesizes findings and updates knowledge base
    synthesis = await sage_agent.mcp_client.call("knowledge_synthesis", {
        "research_inputs": [analysis, forecast],
        "synthesis_type": "comprehensive_report",
        "include_recommendations": True
    })

    # 7. Store results in distributed knowledge base
    storage = await sage_agent.mcp_client.call("knowledge_elevation", {
        "content": synthesis["synthesis_report"],
        "confidence": synthesis["confidence_score"],
        "domain": "federated_learning",
        "privacy_level": "anonymized"
    })

    return {
        "workflow_status": "completed",
        "proposal_id": proposal["proposal_id"],
        "knowledge_item_id": storage["item_id"],
        "participants": ["sage", "curator", "king", "magi", "oracle"]
    }
```

### Emergency Response Workflow

```python
async def emergency_response_workflow():
    """Example emergency response using King agent override"""

    # 1. System anomaly detected by monitoring
    anomaly = await magi_agent.mcp_client.call("anomaly_detection", {
        "severity_threshold": "high",
        "components": ["all"],
        "immediate_analysis": True
    })

    # 2. Assess if emergency response needed
    if anomaly["severity"] == "critical" and "security" in anomaly["threat_types"]:

        # 3. King agent initiates emergency protocols
        emergency = await king_agent.mcp_client.call("emergency_override", {
            "override_action": "privacy_lockdown",
            "target_system": "digital_twins",
            "justification": f"Critical security anomaly detected: {anomaly['description']}",
            "estimated_downtime": "15_minutes"
        })

        # 4. King coordinates system recovery
        recovery = await king_agent.mcp_client.call("system_recovery", {
            "emergency_id": emergency["emergency_id"],
            "recovery_plan": "isolate_and_patch",
            "coordinate_with_agents": ["magi", "sage"],
            "validation_required": True
        })

        return {
            "emergency_handled": True,
            "emergency_id": emergency["emergency_id"],
            "recovery_status": recovery["status"]
        }

    return {"emergency_handled": False, "reason": "anomaly below emergency threshold"}
```

## 📚 Agent Tool Reference

### Complete Tool Matrix

| Tool Name | King | Sage | Curator | Magi | Oracle | Others | Permission Level |
|-----------|------|------|---------|------|--------|--------|------------------|
| hyperrag_query | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | READ_ONLY |
| hyperrag_memory | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | OPERATOR |
| governance_proposal | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | GOVERNANCE |
| governance_vote | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | GOVERNANCE |
| emergency_override | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | EMERGENCY |
| resource_allocation | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | COORDINATOR |
| system_overview | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | READ_ONLY |
| agent_communication | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | OPERATOR |
| performance_metrics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | READ_ONLY |

### Tool Categories

**Core Knowledge Tools**: `hyperrag_query`, `hyperrag_memory`, `trust_network_query`, `knowledge_elevation`

**Communication Tools**: `agent_communication`, `fog_coordination`, `resource_request`, `status_broadcast`

**Governance Tools**: `governance_proposal`, `governance_vote`, `policy_query`, `compliance_audit`

**System Management**: `system_overview`, `resource_allocation`, `performance_metrics`, `emergency_override`

**Specialized Tools**: Agent-specific tools based on role and governance level

---

This comprehensive tool reference enables all 23 AIVillage agents to interact effectively with the complete system through standardized MCP interfaces, ensuring consistent behavior and democratic governance across the entire platform.
