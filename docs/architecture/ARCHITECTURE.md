# AIVillage System Architecture

*Last Updated: August 20, 2025*

## System Overview

AIVillage is a distributed multi-agent AI platform with comprehensive fog computing infrastructure and the following core architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   📱 Mobile     │    │  🖥️ Desktop     │    │  ☁️ Cloud       │
│   Clients       │    │   Clients       │    │   Services      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      🚪 Gateway           │
                    │   FastAPI Entry Point    │
                    │   - Authentication       │
                    │   - Rate Limiting        │
                    │   - Request Routing      │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     🌫️ Fog Gateway        │
                    │   Distributed Computing   │
                    │   - NSGA-II Scheduling   │
                    │   - Marketplace Bidding  │
                    │   - Edge Orchestration   │
                    │   - SLA Management       │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     🔄 Twin Server        │
                    │   Digital Twin Engine    │
                    │   - Personal AI Models   │
                    │   - Privacy-First        │
                    │   - Learning Engine      │
                    └─────────────┬─────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      🛠️ MCP Layer        │
                    │   Model Control Protocol │
                    │   - Agent Tools          │
                    │   - Memory Servers       │
                    │   - RAG Servers          │
                    └─────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼─────────┐  ┌─────────▼─────────┐  ┌─────────▼─────────┐
│   🌐 P2P Layer   │  │   📚 RAG System   │  │  🤖 Agent Layer  │
│                  │  │                   │  │                  │
│ • BitChat (BLE)  │  │ • HyperRAG        │  │ • 23 Specialists │
│ • BetaNet (HTTP) │  │ • Bayesian Trust  │  │ • King Coordinator│
│ • Mesh Routing   │  │ • Vector Store    │  │ • Democratic Vote │
│                  │  │ • Knowledge Graph │  │ • Quiet-STaR      │
└─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │    💾 Data Stores         │
                    │                           │
                    │  🐘 PostgreSQL            │
                    │     - Agent States        │
                    │     - Learning Sessions   │
                    │     - User Profiles       │
                    │                           │
                    │  🕸️ Neo4j                 │
                    │     - Knowledge Graph     │
                    │     - Trust Networks      │
                    │     - Relationships       │
                    │                           │
                    │  🔄 Redis                 │
                    │     - Session Cache       │
                    │     - Message Queue       │
                    │     - Real-time Data      │
                    │                           │
                    │  🔍 Vector DB             │
                    │     - Embeddings          │
                    │     - Semantic Search     │
                    │     - RAG Context         │
                    └───────────────────────────┘
```

## Key Components

### 1. Gateway Layer
- **Entry Point**: FastAPI-based HTTP/WebSocket gateway
- **Authentication**: JWT tokens, API keys, OAuth2
- **Rate Limiting**: Request throttling and quota management
- **Request Routing**: Intelligent routing to appropriate services

### 2. Fog Computing Infrastructure
- **Fog Gateway**: Distributed computing orchestration with OpenAPI 3.1 specification
- **NSGA-II Scheduler**: Multi-objective optimization for resource allocation
- **Marketplace Engine**: Minimal viable renting with spot/on-demand bidding
- **Edge Capability Beacon**: Mobile device integration with WASI runner
- **BetaNet Transport Integration**: Advanced transport protocols via bridge adapters

#### BetaNet Integration Architecture
The fog computing platform integrates with the separate BetaNet bounty implementation through bridge adapters:

```
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│        Fog Computing            │    │      BetaNet Bounty             │
│                                 │    │                                 │
│  ┌─────────────────────────┐   │    │  ┌─────────────────────────┐   │
│  │     Job Scheduler       │   │    │  │    HTX v1.1 Protocol   │   │
│  │                         │   │    │  │                         │   │
│  └─────────────────────────┘   │    │  └─────────────────────────┘   │
│               │                 │    │               ▲                 │
│               ▼                 │    │               │                 │
│  ┌─────────────────────────┐   │    │  ┌─────────────────────────┐   │
│  │   BetaNet Bridge        │───┼────┼─▶│   Covert Channels       │   │
│  │   • Transport Adapter   │   │    │  │   • HTTP/2 Steganography│   │
│  │   • Privacy Router      │   │    │  │   • HTTP/3 QUIC         │   │
│  │   • Mobile Optimizer    │   │    │  │   • WebSocket Channels  │   │
│  └─────────────────────────┘   │    │  └─────────────────────────┘   │
│                                 │    │               ▲                 │
│  ┌─────────────────────────┐   │    │  ┌─────────────────────────┐   │
│  │    Fog Compute Node     │   │    │  │   VRF Mixnet Router     │   │
│  │                         │   │    │  │   • Privacy Modes       │   │
│  └─────────────────────────┘   │    │  │   • Variable Delays     │   │
│                                 │    │  └─────────────────────────┘   │
└─────────────────────────────────┘    └─────────────────────────────────┘
         Fog Infrastructure                      Bounty (Separate)
```

**Key Integration Principles:**
- **Separation**: BetaNet bounty code remains completely untouched in its own workspace
- **Adaptation**: Bridge adapters provide fog compute interface to BetaNet capabilities
- **Verification**: Bounty can be verified independently without fog compute dependencies
- **Fallback**: Graceful degradation when BetaNet bounty is not available
- **Security & Compliance**: Namespace isolation, quotas, egress policies
- **SLA Management**: S-class (replicated+attested), A-class (replicated), B-class (best-effort)
- **Observability**: Prometheus metrics, distributed tracing, performance monitoring

### 3. Twin Server
- **Digital Twin Engine**: Personal AI models (1-10MB) running locally
- **Privacy-First**: All personal data stays on device
- **Surprise-Based Learning**: Models improve via prediction accuracy
- **Resource Management**: Battery/thermal-aware processing

### 4. MCP (Model Control Protocol) Layer
- **Agent Tools**: Standardized interfaces for agent capabilities
- **Memory Servers**: Persistent and working memory management
- **RAG Servers**: Knowledge retrieval and augmentation services
- **Inter-Service Communication**: Unified protocol for component interaction

### 5. P2P Communication Layer
- **BitChat**: Bluetooth mesh networking for offline scenarios
- **BetaNet**: Encrypted HTTP transport for internet connectivity
- **Mesh Routing**: Intelligent message routing with failover
- **Mobile-First**: Battery and bandwidth-aware protocols

### 6. RAG System
- **HyperRAG**: Multi-modal retrieval-augmented generation
- **Bayesian Trust Networks**: Probabilistic knowledge validation
- **Vector Store**: High-performance semantic search
- **Knowledge Graph**: Structured relationship mapping

### 7. Agent Layer
- **23 Specialized Agents**: King, Magi, Oracle, Sage, etc.
- **Democratic Governance**: 2/3 quorum voting on decisions
- **Quiet-STaR**: Internal reasoning with thought tokens
- **ADAS Self-Modification**: Architecture discovery and optimization
- **HRRM Bootstrap System**: Three ~50M parameter models (Planner, Reasoner, Memory) for Agent Forge EvoMerge acceleration

### 8. Agent Forge Pipeline
- **7-Phase Training**: EvoMerge, Quiet-STaR, BitNet, Training, Tool/Persona Baking, ADAS, Final Compression
- **HRRM Integration**: Pre-optimized seed models provide 30× faster EvoMerge iteration
- **Distributed Training**: Federated learning with fog compute integration
- **Production Ready**: Complete infrastructure with testing, export, and CLI tools

### 9. Data Stores
- **PostgreSQL**: Relational data (agent states, sessions, profiles)
- **Neo4j**: Graph data (knowledge networks, trust relationships)
- **Redis**: Cache and real-time data (sessions, queues, metrics)
- **Vector Database**: Embeddings and semantic search indices

## Data Flow

### Standard Request Flow
1. **Client Request** → Gateway (auth/rate limiting)
2. **Gateway** → Twin Server (request processing)
3. **Twin Server** → MCP Layer (tool invocation)
4. **MCP Layer** → {P2P, RAG, Agents} (parallel processing)
5. **Components** → Data Stores (persistence)
6. **Response** ← Twin Server ← MCP Layer ← Components
7. **Client** ← Gateway ← Twin Server (formatted response)

### Fog Computing Flow
1. **Resource Request** → Fog Gateway (capability assessment)
2. **Fog Gateway** → NSGA-II Scheduler (resource optimization)
3. **Scheduler** → Marketplace Engine (cost calculation)
4. **Marketplace** → Edge Beacon (device discovery)
5. **Edge Devices** → WASI Runner (job execution)
6. **Results** ← Fog Gateway ← Edge Network (aggregation)
7. **Client** ← Gateway ← Fog Gateway (final response)

## Fog Computing Architecture

### Components Overview

The fog computing infrastructure provides distributed processing capabilities across edge devices and fog nodes:

```
                    ┌─────────────────────────────────────┐
                    │        🌫️ Fog Gateway              │
                    │  ┌─────────────┐ ┌─────────────┐    │
                    │  │   Admin     │ │   Jobs      │    │
                    │  │   API       │ │   API       │    │
                    │  └─────────────┘ └─────────────┘    │
                    │  ┌─────────────┐ ┌─────────────┐    │
                    │  │ Sandboxes   │ │   Usage     │    │
                    │  │    API      │ │    API      │    │
                    │  └─────────────┘ └─────────────┘    │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │     📊 NSGA-II Scheduler    │
                    │                             │
                    │ • Multi-objective optimization
                    │ • Pareto frontier analysis │
                    │ • Resource allocation       │
                    │ • Load balancing           │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐    ┌─────────▼─────────┐   ┌─────────▼─────────┐
│  💰 Marketplace   │    │  🔒 Security     │   │  📈 Monitoring    │
│                   │    │                  │   │                   │
│ • Spot bidding    │    │ • Namespace      │   │ • Prometheus      │
│ • On-demand       │    │   isolation      │   │ • Tracing         │
│ • Trust scoring   │    │ • Quota limits   │   │ • Health checks   │
│ • Price discovery │    │ • Egress policies│   │ • SLA tracking    │
└─────────┬─────────┘    └─────────┬────────┘   └─────────┬─────────┘
          │                        │                       │
          └────────────────────────┼───────────────────────┘
                                   │
                      ┌────────────▼────────────┐
                      │    📱 Edge Network      │
                      │                         │
                      │  🔗 Capability Beacon   │
                      │  🏃 WASI Runtime        │
                      │  📊 Resource Monitor    │
                      │  🛡️ Security Sandbox    │
                      └─────────────────────────┘
```

### Key Technologies

#### NSGA-II Scheduler
- **Multi-objective optimization** for cost, latency, and reliability
- **Pareto frontier analysis** to find optimal trade-offs
- **Dynamic load balancing** based on real-time conditions
- **Resource allocation** across heterogeneous fog nodes

#### Marketplace Engine
- **Spot bidding** for cost-effective resource utilization
- **On-demand pricing** for guaranteed availability
- **Trust-based matching** using historical performance data
- **Dynamic pricing** based on supply and demand

#### Edge Integration
- **Capability Beacon** discovers and registers mobile devices
- **WASI Runtime** provides secure sandboxed execution
- **Resource Monitor** tracks battery, thermal, and network state
- **Security Sandbox** isolates job execution with quota enforcement

#### SLA Classes
- **S-Class**: Replicated + cryptographically attested execution
- **A-Class**: Replicated execution across multiple nodes
- **B-Class**: Best-effort single-node execution

### Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Job Scheduling | < 100ms | NSGA-II optimization |
| Market Price Discovery | < 50ms | Real-time bidding |
| Edge Device Discovery | 5-30 seconds | mDNS + capability beacon |
| Job Execution Latency | 200ms - 30s | Depends on workload size |
| Marketplace Efficiency | 95%+ success rate | Trade execution |
| Resource Utilization | 70-85% | Across fog network |

## Security Architecture

- **Zero-Trust Model**: All components authenticate and authorize
- **End-to-End Encryption**: P2P and client communications encrypted
- **Privacy by Design**: Personal data never leaves device
- **Audit Trails**: All actions logged for compliance
- **Role-Based Access**: Granular permissions system
- **Namespace Isolation**: Complete tenant separation in fog computing
- **Sandbox Security**: WASI-based job execution with resource quotas

## Deployment Architecture

### Local Development
```
Docker Compose Stack:
- Gateway (port 8000)
- Fog Gateway (port 8080)
- Twin Server (port 8001)
- PostgreSQL (port 5432)
- Neo4j (port 7474)
- Redis (port 6379)
- Vector DB (port 6333)
- Prometheus (port 9090)
```

### Production
```
Cloud Infrastructure:
- Load Balancer → Gateway Cluster
- Fog Gateway Cluster → NSGA-II Scheduler
- Service Mesh → Twin Server Cluster
- Edge Network → Mobile Devices + Fog Nodes
- Managed Databases (RDS, ElasticSearch, ElastiCache)
- P2P Edge Nodes (Global Distribution)
- Monitoring Stack (Prometheus, Grafana, Jaeger)
```

### Fog Computing Deployment
```
Distributed Architecture:
- Central Fog Gateway (Multi-region)
- Regional Scheduler Instances
- Edge Capability Beacons (Mobile devices)
- WASI Runtime Sandboxes
- Marketplace Trading Engine
- Security Policy Enforcement
```

## Reference Documentation

- **Comprehensive Details**: [TABLE_OF_CONTENTS.md](../TABLE_OF_CONTENTS.md)
- **API Documentation**: [docs/api/](../api/)
- **Deployment Guides**: [docs/deployment/](../deployment/)
- **Development Setup**: [docs/development/](../development/)
