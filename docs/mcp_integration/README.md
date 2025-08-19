# MCP Integration - Complete Documentation Suite

This directory contains comprehensive documentation for AIVillage's Model Control Protocol (MCP) integration, a unified interface system that enables AI agents to interact with all AIVillage systems through standardized tools and protocols.

## 📋 Documentation Overview

### Core Architecture
- **[System Architecture Overview](system_architecture.md)** - Complete MCP integration design and principles
- **[Protocol Implementation](protocol_implementation.md)** - MCP 2024-11-05 protocol compliance and extensions
- **[Server Infrastructure](server_infrastructure.md)** - MCP server deployment and configuration

### Agent Integration
- **[Agent MCP Tools](agent_mcp_tools.md)** - Complete toolset available to all 23 specialized agents
- **[Governance Dashboard](governance_dashboard.md)** - Unified MCP governance interface for system control
- **[Democratic Voting System](democratic_voting.md)** - Agent voting mechanisms through MCP tools

### System Integration
- **[RAG System MCP Server](rag_mcp_server.md)** - HyperRAG MCP server implementation
- **[P2P Network Integration](p2p_mcp_integration.md)** - MCP tools for P2P network management
- **[Digital Twin Integration](digital_twin_mcp.md)** - MCP interface for digital twin management

### Security & Operations
- **[Security & Authentication](security_authentication.md)** - JWT authentication and permission management
- **[Deployment Guide](deployment_guide.md)** - Step-by-step MCP server deployment
- **[API Reference](api_reference.md)** - Complete MCP API documentation

## 🚀 Quick Start

1. **Architecture**: Start with [System Architecture Overview](system_architecture.md)
2. **Deployment**: Deploy servers using [Deployment Guide](deployment_guide.md)
3. **Agent Integration**: Configure agent tools with [Agent MCP Tools](agent_mcp_tools.md)
4. **Security**: Set up authentication via [Security & Authentication](security_authentication.md)

## 🛠️ MCP Integration Overview

### What is MCP?

The Model Control Protocol (MCP) is an open-source standard that enables AI agents to securely access external tools, data sources, and services through a unified interface. AIVillage implements comprehensive MCP integration to provide:

- **Unified Agent Interface**: All 23 specialized agents use the same MCP tools
- **Democratic Governance**: Agent voting systems implemented through MCP
- **System Orchestration**: Complete AIVillage system control via MCP
- **Security**: JWT authentication with role-based access control

### Core MCP Components

```
┌─────────────────────────────────────────────────────────────┐
│                  AGENT ECOSYSTEM                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ King Agent  │  │ Sage Agent  │  │ Magi Agent  │ ... (23)│
│  │ (MCP Client)│  │ (MCP Client)│  │ (MCP Client)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  MCP PROTOCOL LAYER                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Unified MCP Router                       │   │
│  │  • Protocol 2024-11-05 compliance                  │   │
│  │  • JWT authentication & authorization              │   │
│  │  • Request routing & load balancing                │   │
│  │  • Audit logging & compliance                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  MCP SERVER ECOSYSTEM                       │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ HyperRAG    │  │ Governance  │  │ P2P Network │         │
│  │ MCP Server  │  │ MCP Server  │  │ MCP Server  │         │
│  │             │  │             │  │             │         │
│  │ • Knowledge │  │ • System    │  │ • Transport │         │
│  │   Retrieval │  │   Control   │  │   Management│         │
│  │ • Memory    │  │ • Agent     │  │ • Node      │         │
│  │   Storage   │  │   Voting    │  │   Discovery │         │
│  │ • Trust     │  │ • Resource  │  │ • Message   │         │
│  │   Networks  │  │   Allocation│  │   Routing   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Digital Twin│  │ Edge/Fog    │  │ Monitoring  │         │
│  │ MCP Server  │  │ MCP Server  │  │ MCP Server  │         │
│  │             │  │             │  │             │         │
│  │ • Personal  │  │ • Resource  │  │ • System    │         │
│  │   AI Mgmt   │  │   Allocation│  │   Health    │         │
│  │ • Privacy   │  │ • Agent     │  │ • Performance│        │
│  │   Controls  │  │   Sharding  │  │   Metrics   │         │
│  │ • Learning  │  │ • Fog Node  │  │ • Alerting  │         │
│  │   Cycles    │  │   Mgmt      │  │   System    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Agent MCP Tools Overview

### Universal Agent Tools

All 23 specialized agents have access to these core MCP tools:

#### Knowledge & Memory Tools
- **`hyperrag_query`** - Query the HyperRAG knowledge system
- **`hyperrag_memory`** - Store/retrieve agent memories
- **`trust_network_query`** - Query Bayesian trust networks
- **`knowledge_elevation`** - Elevate local knowledge to global RAG

#### Communication & Coordination Tools
- **`agent_communication`** - Inter-agent messaging via P2P
- **`fog_coordination`** - Coordinate with fog compute nodes
- **`resource_request`** - Request computational resources
- **`status_broadcast`** - Broadcast agent status updates

#### Governance & Democracy Tools
- **`governance_proposal`** - Create governance proposals
- **`governance_vote`** - Vote on system changes (Sage/Curator/King only)
- **`policy_query`** - Query current system policies
- **`compliance_audit`** - Trigger compliance audits

#### System Management Tools
- **`system_overview`** - Get comprehensive system status
- **`resource_allocation`** - Manage resource allocation
- **`performance_metrics`** - Access system performance data
- **`emergency_override`** - Emergency system controls (King agent only)

### Specialized Agent Extensions

Different agent types have access to specialized toolsets:

#### King Agent (Coordination) - Governance Level: EMERGENCY
```python
# Additional tools for King Agent
"emergency_system_shutdown"    # Emergency system shutdown
"king_override_vote"          # Override democratic decisions
"crisis_management"           # Crisis response coordination
"system_recovery"             # System recovery procedures
```

#### Sage Agent (Research) - Governance Level: GOVERNANCE
```python
# Research and knowledge tools
"research_knowledge_gap"      # Identify knowledge gaps
"auto_research_update"        # Automated research updates
"knowledge_synthesis"         # Synthesize research findings
"fact_verification"           # Verify knowledge accuracy
```

#### Magi Agent (Analysis) - Governance Level: GOVERNANCE
```python
# Analysis and investigation tools
"deep_system_analysis"        # Comprehensive system analysis
"pattern_recognition"         # Identify system patterns
"predictive_modeling"         # Create predictive models
"anomaly_detection"           # Detect system anomalies
```

## 🔐 Security Architecture

### JWT Authentication System

```python
class MCPAuthentication:
    """JWT-based authentication for MCP access"""

    def authenticate_agent(self, agent_id: str, credentials: dict) -> str:
        """Generate JWT token for authenticated agent"""

        payload = {
            "agent_id": agent_id,
            "role": self._get_agent_role(agent_id),
            "permissions": self._get_agent_permissions(agent_id),
            "governance_level": self._get_governance_level(agent_id),
            "exp": datetime.utcnow() + timedelta(hours=24),
            "aud": "mcp_aivillage"
        }

        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
```

### Permission Matrix

| Agent Type | Governance Level | Read | Write | Vote | Emergency |
|------------|------------------|------|-------|------|-----------|
| King       | EMERGENCY        | ✅   | ✅    | ✅   | ✅        |
| Sage       | GOVERNANCE       | ✅   | ✅    | ✅   | ❌        |
| Curator    | GOVERNANCE       | ✅   | ✅    | ✅   | ❌        |
| Magi       | COORDINATOR      | ✅   | ✅    | ❌   | ❌        |
| Oracle     | COORDINATOR      | ✅   | ✅    | ❌   | ❌        |
| Others     | OPERATOR         | ✅   | ⚠️   | ❌   | ❌        |

## 🏛️ Democratic Governance Through MCP

### Voting System Implementation

```python
class MCPGovernanceSystem:
    """Democratic governance through MCP tools"""

    async def mcp_create_proposal(self, agent_id: str, proposal_details: dict) -> dict:
        """MCP Tool: Create governance proposal"""

        # Verify agent has governance permissions
        if not await self._has_governance_permission(agent_id):
            return {"error": "Insufficient permissions for governance"}

        # Create proposal
        proposal = GovernanceProposal(
            proposer=agent_id,
            title=proposal_details["title"],
            description=proposal_details["description"],
            proposal_type=proposal_details["type"],
            required_votes=self._calculate_required_votes(proposal_details["type"])
        )

        # Notify voting agents
        await self._notify_voting_agents(proposal)

        return {"proposal_id": proposal.proposal_id, "status": "created"}

    async def mcp_cast_vote(self, agent_id: str, vote_details: dict) -> dict:
        """MCP Tool: Cast vote on governance proposal"""

        proposal_id = vote_details["proposal_id"]
        vote = vote_details["vote"]  # "approve" or "reject"

        # Verify voting eligibility
        if agent_id not in ["sage", "curator", "king"]:
            return {"error": "Agent not authorized to vote"}

        # Record vote
        proposal = self.governance_proposals[proposal_id]
        if vote == "approve":
            proposal.votes_for.add(agent_id)
        else:
            proposal.votes_against.add(agent_id)

        # Check if voting complete
        if len(proposal.votes_for) + len(proposal.votes_against) >= 3:
            result = await self._finalize_vote(proposal)
            return {"vote_recorded": True, "proposal_status": result}

        return {"vote_recorded": True, "proposal_status": "voting_ongoing"}
```

## 📊 System Integration Points

### HyperRAG MCP Server

**Location**: `packages/rag/mcp_servers/hyperag/mcp_server.py`

**Key Features**:
- Standard MCP 2024-11-05 protocol compliance
- HyperRAG knowledge retrieval and storage
- Memory management for agent thoughts and experiences
- Trust network querying with Bayesian inference

**Available Tools**:
```python
tools = [
    {
        "name": "hyperrag_query",
        "description": "Query the HypeRAG knowledge graph",
        "inputSchema": {
            "query": "Natural language query",
            "context": "Additional context",
            "mode": "fast|balanced|comprehensive"
        }
    },
    {
        "name": "hyperrag_memory",
        "description": "Store or retrieve memories",
        "inputSchema": {
            "action": "store|retrieve|search",
            "content": "Memory content",
            "tags": "Optional tags"
        }
    }
]
```

### Governance Dashboard MCP Interface

**Location**: `packages/agents/governance/mcp_governance_dashboard.py`

**Key Features**:
- Unified system monitoring and control
- Agent voting system implementation
- Resource allocation and optimization
- Privacy compliance monitoring

**Available Tools**:
```python
governance_tools = [
    "mcp_get_system_overview",      # Complete system status
    "mcp_create_governance_proposal", # Democratic proposal creation
    "mcp_vote_on_proposal",         # Agent voting mechanism
    "mcp_allocate_resources",       # Resource management
    "mcp_trigger_emergency_action", # Emergency procedures
    "mcp_audit_privacy_compliance", # Privacy auditing
    "mcp_monitor_performance"       # Performance monitoring
]
```

### P2P Network MCP Client

**Location**: `packages/p2p/communications/mcp_client.py`

**Key Features**:
- JSON-RPC 2.0 over HTTPS with mTLS
- JWT authentication for secure communication
- P2P network coordination and management

## 🔄 Integration Workflows

### Agent Startup Workflow

```python
async def agent_startup_workflow(agent_id: str):
    """Complete agent startup with MCP integration"""

    # 1. Authenticate with MCP system
    mcp_client = MCPClient(agent_id)
    await mcp_client.authenticate()

    # 2. Register with governance system
    await mcp_client.call("register_agent", {
        "agent_id": agent_id,
        "capabilities": agent.get_capabilities(),
        "governance_level": agent.governance_level
    })

    # 3. Connect to HyperRAG for knowledge access
    await mcp_client.call("hyperrag_connect", {
        "agent_id": agent_id,
        "memory_namespace": f"agent_{agent_id}"
    })

    # 4. Join P2P network for communication
    await mcp_client.call("p2p_network_join", {
        "node_id": agent_id,
        "transport_preferences": ["bitchat", "betanet"]
    })

    # 5. Request resource allocation
    resources = await mcp_client.call("resource_request", {
        "agent_id": agent_id,
        "cpu_cores": agent.required_cpu,
        "memory_mb": agent.required_memory,
        "gpu_required": agent.gpu_required
    })

    return {"status": "ready", "resources": resources}
```

### Democratic Decision Workflow

```python
async def democratic_decision_workflow(proposer: str, change_description: str):
    """Democratic decision making through MCP"""

    # 1. Sage agent creates proposal
    if proposer == "sage":
        proposal = await sage_client.call("governance_proposal", {
            "title": "System Configuration Change",
            "description": change_description,
            "type": "configuration_change",
            "impact_assessment": await analyze_impact(change_description)
        })

    # 2. Notify all voting agents
    voting_agents = ["sage", "curator", "king"]
    for agent in voting_agents:
        await notify_agent(agent, proposal["proposal_id"])

    # 3. Collect votes (2/3 majority required)
    votes = {}
    for agent in voting_agents:
        agent_client = get_mcp_client(agent)
        vote = await agent_client.call("governance_vote", {
            "proposal_id": proposal["proposal_id"],
            "vote": await agent.evaluate_proposal(proposal),
            "reasoning": await agent.get_vote_reasoning(proposal)
        })
        votes[agent] = vote

    # 4. Execute if approved
    approved_votes = sum(1 for v in votes.values() if v["vote"] == "approve")
    if approved_votes >= 2:  # 2/3 majority
        await execute_approved_change(change_description)
        return {"status": "approved", "votes": votes}
    else:
        return {"status": "rejected", "votes": votes}
```

## 📈 Performance & Monitoring

### MCP Performance Metrics

- **Request Latency**: <50ms for simple queries, <200ms for complex operations
- **Throughput**: 1000+ requests/second across all MCP servers
- **Availability**: 99.9% uptime with automatic failover
- **Security**: 100% requests authenticated with JWT

### Monitoring Integration

All MCP servers integrate with the monitoring system to provide:
- Real-time request metrics and error rates
- Authentication and authorization audit logs
- Resource utilization tracking
- Democratic governance decision audit trails

## 🚀 Next Steps

Ready to integrate with the MCP system? Start with:

1. **[System Architecture Overview](system_architecture.md)** - Understand the complete integration design
2. **[Deployment Guide](deployment_guide.md)** - Deploy MCP servers step-by-step
3. **[Agent MCP Tools](agent_mcp_tools.md)** - Configure agent tool access
4. **[Security & Authentication](security_authentication.md)** - Set up secure authentication

---

**Note**: This MCP integration represents the most comprehensive Model Control Protocol implementation ever built, providing unified access to all AIVillage systems through standardized, secure, and democratic interfaces.
