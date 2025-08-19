# AIVillage System Architecture

*Last Updated: August 19, 2025*

## System Overview

AIVillage is a distributed multi-agent AI platform with the following core architecture:

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

### 2. Twin Server
- **Digital Twin Engine**: Personal AI models (1-10MB) running locally
- **Privacy-First**: All personal data stays on device
- **Surprise-Based Learning**: Models improve via prediction accuracy
- **Resource Management**: Battery/thermal-aware processing

### 3. MCP (Model Control Protocol) Layer
- **Agent Tools**: Standardized interfaces for agent capabilities
- **Memory Servers**: Persistent and working memory management
- **RAG Servers**: Knowledge retrieval and augmentation services
- **Inter-Service Communication**: Unified protocol for component interaction

### 4. P2P Communication Layer
- **BitChat**: Bluetooth mesh networking for offline scenarios
- **BetaNet**: Encrypted HTTP transport for internet connectivity
- **Mesh Routing**: Intelligent message routing with failover
- **Mobile-First**: Battery and bandwidth-aware protocols

### 5. RAG System
- **HyperRAG**: Multi-modal retrieval-augmented generation
- **Bayesian Trust Networks**: Probabilistic knowledge validation
- **Vector Store**: High-performance semantic search
- **Knowledge Graph**: Structured relationship mapping

### 6. Agent Layer
- **23 Specialized Agents**: King, Magi, Oracle, Sage, etc.
- **Democratic Governance**: 2/3 quorum voting on decisions
- **Quiet-STaR**: Internal reasoning with thought tokens
- **ADAS Self-Modification**: Architecture discovery and optimization

### 7. Data Stores
- **PostgreSQL**: Relational data (agent states, sessions, profiles)
- **Neo4j**: Graph data (knowledge networks, trust relationships)
- **Redis**: Cache and real-time data (sessions, queues, metrics)
- **Vector Database**: Embeddings and semantic search indices

## Data Flow

1. **Client Request** → Gateway (auth/rate limiting)
2. **Gateway** → Twin Server (request processing)
3. **Twin Server** → MCP Layer (tool invocation)
4. **MCP Layer** → {P2P, RAG, Agents} (parallel processing)
5. **Components** → Data Stores (persistence)
6. **Response** ← Twin Server ← MCP Layer ← Components
7. **Client** ← Gateway ← Twin Server (formatted response)

## Security Architecture

- **Zero-Trust Model**: All components authenticate and authorize
- **End-to-End Encryption**: P2P and client communications encrypted
- **Privacy by Design**: Personal data never leaves device
- **Audit Trails**: All actions logged for compliance
- **Role-Based Access**: Granular permissions system

## Deployment Architecture

### Local Development
```
Docker Compose Stack:
- Gateway (port 8000)
- Twin Server (port 8001)
- PostgreSQL (port 5432)
- Neo4j (port 7474)
- Redis (port 6379)
- Vector DB (port 6333)
```

### Production
```
Cloud Infrastructure:
- Load Balancer → Gateway Cluster
- Service Mesh → Twin Server Cluster
- Managed Databases (RDS, ElasticSearch, ElastiCache)
- P2P Edge Nodes (Global Distribution)
```

## Reference Documentation

- **Comprehensive Details**: [TABLE_OF_CONTENTS.md](../TABLE_OF_CONTENTS.md)
- **API Documentation**: [docs/api/](../api/)
- **Deployment Guides**: [docs/deployment/](../deployment/)
- **Development Setup**: [docs/development/](../development/)
