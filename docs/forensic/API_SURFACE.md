# AIVillage API Surface Mapping

This document provides a comprehensive mapping of all API surfaces, interfaces, CLI tools, and external integration points across the AIVillage distributed AI platform.

## Table of Contents
- [Overview](#overview)
- [REST API Endpoints](#rest-api-endpoints)
- [WebSocket APIs](#websocket-apis)
- [Public Class/Function Exports](#public-classfunctions-exports)
- [Command-Line Interfaces](#command-line-interfaces)
- [External Integration Points](#external-integration-points)
- [API Contracts & Schemas](#api-contracts--schemas)
- [Security & Authentication](#security--authentication)

## Overview

AIVillage exposes multiple API surfaces across several domains:
- **Agent Forge**: 7-phase model training pipeline
- **P2P/Fog Computing**: Distributed mesh networking
- **RAG System**: Advanced retrieval-augmented generation
- **Digital Twin**: Privacy-preserving personal AI
- **WebSocket Communication**: Real-time updates and messaging
- **CLI Tools**: System management and operations

## REST API Endpoints

### Core Gateway Endpoints (`infrastructure/gateway/`)

#### Health & Status
- `GET /health` - Health check and service dependencies
- `GET /healthz` - Kubernetes-style health endpoint
- `GET /status` - System status overview
- `GET /metrics` - System metrics and telemetry

#### Agent Forge Pipeline
- `POST /phases/cognate/start` - Start Cognate pretraining phase
- `POST /phases/evomerge/start` - Start evolutionary merge phase
- `POST /phases/quietstar/start` - Start QuietStar reasoning phase
- `POST /phases/bitnet/start` - Start BitNet quantization phase
- `POST /phases/forge-training/start` - Start forge training phase
- `POST /phases/tool-persona/start` - Start tool persona phase
- `POST /phases/adas/start` - Start ADAS optimization phase
- `POST /phases/final-compression/start` - Start final compression phase
- `GET /phases/status` - Get status of all pipeline phases
- `GET /phases/{phase_id}/status` - Get specific phase status
- `POST /pipeline/run-all` - Run complete 7-phase pipeline
- `POST /pipeline/reset` - Reset pipeline state

#### Model Management
- `GET /models` - List available models
- `POST /models/{model_id}/register` - Register new model
- `POST /models/{model_id}/load` - Load model into memory
- `DELETE /models/{model_id}/unload` - Unload model from memory
- `POST /models/export` - Export trained models

#### Chat & Query Processing
- `POST /chat` - Chat with AI agents
- `POST /v1/chat` - Versioned chat endpoint
- `POST /v1/query` - RAG query processing
- `POST /v1/upload` - Upload documents for indexing
- `GET /sessions` - List chat sessions
- `GET /sessions/{session_id}` - Get specific session
- `DELETE /sessions/{session_id}` - Delete session
- `POST /compare` - Compare model responses

### P2P/Fog Computing Endpoints

#### P2P Network Status
- `GET /api/p2p/status` - P2P network health and status
- `GET /api/p2p/peers` - List connected peers
- `GET /api/p2p/messages` - P2P message history

#### Fog Computing
- `GET /api/fog/nodes` - List fog computing nodes
- `GET /api/fog/resources` - Available fog resources
- `GET /api/fog/marketplace` - Fog marketplace listings
- `GET /api/fog/tokens` - Token balances and transactions

#### Proof System
- `POST /proofs/generate` - Generate cryptographic proofs
- `POST /proofs/verify` - Verify proofs
- `GET /proofs/status` - Proof verification status

### Digital Twin Endpoints

#### Profile Management
- `GET /profiles` - List digital twin profiles
- `POST /profiles` - Create new profile
- `GET /profiles/{profile_id}` - Get specific profile
- `PUT /profiles/{profile_id}` - Update profile
- `DELETE /profiles/{profile_id}` - Delete profile
- `GET /profiles/{profile_id}/export` - Export profile data

#### Metrics & Analytics
- `GET /metrics` - Get twin metrics
- `POST /metrics` - Store metrics data
- `GET /metrics/stats` - Aggregated statistics

### RAG System Endpoints

#### Document Management
- `POST /documents` - Add document to knowledge base
- `GET /documents/{doc_id}` - Get document by ID
- `POST /index/add` - Add content to search index

#### Query Processing
- `POST /query` - Process RAG queries
- `GET /metrics/performance` - RAG performance metrics

### Administrative Endpoints

#### System Management
- `GET /api/system-metrics` - System resource metrics
- `GET /api/service-status` - Service health status
- `GET /api/agent-metrics` - Agent performance metrics
- `GET /api/network-status` - Network connectivity status
- `GET /api/logs` - System logs

#### Security & Authentication
- `POST /auth/login` - User authentication
- `POST /auth/register` - User registration
- `POST /auth/refresh` - Refresh JWT token

#### Billing & Usage
- `GET /usage` - Usage statistics
- `POST /quotes` - Price quotes
- `GET /prices` - Service pricing
- `POST /invoices` - Generate invoice
- `GET /invoices/{invoice_id}` - Get specific invoice
- `GET /summary` - Billing summary

## WebSocket APIs

### Real-time Communication
- `WS /ws` - General WebSocket connection
- `WS /ws/{session_id}` - Session-specific WebSocket
- `WS /ws/p2p-fog` - P2P/Fog real-time updates
- `POST /broadcast` - Broadcast message to WebSocket clients
- `POST /broadcast/{channel}` - Channel-specific broadcast

### P2P Mesh Integration
- `WS /mesh/ws` - P2P mesh WebSocket bridge
- `POST /mesh/start` - Start mesh networking
- `POST /mesh/stop` - Stop mesh networking
- `GET /mesh/status` - Mesh network status
- `POST /mesh/send` - Send P2P message
- `GET /mesh/peers` - List mesh peers
- `POST /mesh/connect` - Connect to peer
- `GET /mesh/discovery` - Peer discovery

### DHT Operations
- `POST /dht/store` - Store data in DHT
- `GET /dht/get/{key}` - Retrieve DHT data

## Public Class/Functions Exports

### Core Modules (`core/`)

#### Agent System
```python
# core/agents/
class BaseAgent
class SpecializedAgent
class KingAgent
class MagiAgent
class SageAgent
class OracleAgent
class NavigatorAgent

# Key Functions
def create_agent(agent_type: str) -> BaseAgent
def register_agent(agent: BaseAgent) -> bool
def get_agent_capabilities(agent_id: str) -> List[str]
```

#### RAG System
```python
# core/rag/
class HyperRAG
class CognitiveNexus
class BayesianTrustGraph
class VectorStore
class KnowledgeGraph

# Key Functions
def process_query(query: str, mode: str = "balanced") -> QueryResponse
def add_document(doc: Document) -> str
def get_trusted_sources(query: str) -> List[Source]
```

#### P2P Networking
```python
# core/p2p/
class MeshProtocol
class PeerManager
class MessageRouter
class CryptoUtils

# Key Functions
def connect_to_network() -> bool
def send_message(peer_id: str, message: dict) -> bool
def discover_peers() -> List[PeerInfo]
```

### Infrastructure Modules (`infrastructure/`)

#### Gateway
```python
# infrastructure/gateway/
class UnifiedAPIGateway
class WebSocketManager
class AuthManager
class RateLimiter

# Key Functions
def start_gateway() -> None
def register_endpoint(path: str, handler: Callable) -> None
def authenticate_request(token: str) -> UserInfo
```

#### Fog Computing
```python
# infrastructure/fog/
class FogNode
class ResourceManager
class TaskScheduler
class ProofSystem

# Key Functions
def deploy_to_fog(task: Task) -> TaskResult
def verify_proof(proof: Proof) -> bool
def get_fog_resources() -> ResourceInfo
```

#### Digital Twin
```python
# infrastructure/twin/
class DigitalTwin
class PrivacyEngine
class LearningEngine
class ProfileManager

# Key Functions
def create_twin(user_id: str) -> DigitalTwin
def update_learning_data(data: LearningData) -> None
def get_personalized_response(query: str) -> Response
```

## Command-Line Interfaces

### Core CLI Tools

#### Main System CLI (`ui/cli/`)
```bash
# System Management
python -m ui.cli.system_manager --help
python -m ui.cli.agent_forge --help
python -m ui.cli.dashboard_launcher --help

# Available Commands
aivillage system status
aivillage agents list
aivillage fog deploy
aivillage p2p connect
```

#### Infrastructure P2P CLI (`infrastructure/p2p/cli.py`)
```bash
# P2P Network Management
python -m infrastructure.p2p.cli --help

# Commands
p2p-mesh start --topology mesh
p2p-mesh status
p2p-mesh peers
p2p-mesh send --peer-id <id> --message <msg>
```

#### Digital Twin CLI (`infrastructure/twin/cli.py`)
```bash
# Digital Twin Management
python -m infrastructure.twin.cli --help

# Commands
twin create --user-id <id>
twin update --profile-id <id> --data <data>
twin export --profile-id <id>
twin metrics --profile-id <id>
```

#### Compliance CLI (`infrastructure/shared/compliance/compliance_cli.py`)
```bash
# Compliance and Governance
python -m infrastructure.shared.compliance.compliance_cli --help

# Commands
compliance scan
compliance report
compliance remediate
```

#### Tokenomics CLI (`core/domain/tokenomics/governance/cli.py`)
```bash
# Tokenomics and DAO Governance
python -m core.domain.tokenomics.governance.cli --help

# Commands
governance vote --proposal-id <id>
governance status
tokenomics balance
tokenomics transfer --to <address> --amount <amt>
```

### Utility Scripts

#### Security Tools
```bash
# Security Monitoring
python scripts/security_audit_websocket_endpoints.py
python security/scripts/setup-monitoring.py
python security/scripts/generate-sbom.py
python security/scripts/aggregate-security-results.py
```

#### Development Tools
```bash
# Architecture Analysis
python scripts/architectural_analysis.py
python scripts/coupling_metrics.py
python scripts/run_quality_analysis.py
python scripts/architecture_dashboard.py
```

#### Deployment Tools
```bash
# Production Deployment
python scripts/deploy_production.py
python scripts/verify_production_integration.py
python scripts/start_service_mesh.py
```

## External Integration Points

### AI/ML Service Providers
- **OpenRouter** - Multi-model API gateway
- **OpenAI** - GPT models integration
- **Anthropic** - Claude models integration
- **Hugging Face** - Model hub and transformers

### Vector Databases
- **ChromaDB** - Default vector store
- **Qdrant** - Production vector database
- **Pinecone** - Cloud vector database
- **Weaviate** - Knowledge graph vector store

### Blockchain & Crypto
- **Betanet** - Blockchain network integration
- **SCION** - Secure network architecture
- **DHT** - Distributed hash table
- **Mixnet** - Privacy routing

### Cloud Services
- **AWS** - Cloud infrastructure
- **Docker** - Containerization
- **Kubernetes** - Orchestration
- **Prometheus** - Monitoring

### Messaging & Communication
- **WebRTC** - Peer-to-peer communication
- **libp2p** - P2P networking library
- **BitChat** - Encrypted messaging
- **WebSocket** - Real-time communication

## API Contracts & Schemas

### OpenAPI Specifications
- `docs/api/openapi.yaml` - Complete API specification
- `docs/api/UNIFIED_API_SPECIFICATION.yaml` - Unified gateway spec

### Core Data Models

#### Chat Request/Response
```yaml
ChatRequest:
  type: object
  required: [message]
  properties:
    message: { type: string }
    conversation_id: { type: string }
    agent_preference: { enum: [king, magi, sage, oracle, navigator, any] }
    mode: { enum: [fast, balanced, comprehensive, creative] }

ChatResponse:
  type: object
  properties:
    response: { type: string }
    conversation_id: { type: string }
    agent_used: { type: string }
    processing_time_ms: { type: integer }
    metadata: { type: object }
```

#### Query Processing
```yaml
QueryRequest:
  type: object
  required: [query]
  properties:
    query: { type: string }
    mode: { enum: [fast, balanced, comprehensive, creative, analytical] }
    include_sources: { type: boolean, default: true }
    max_results: { type: integer, minimum: 1, maximum: 50 }

QueryResponse:
  type: object
  properties:
    query_id: { type: string }
    response: { type: string }
    sources: { type: array, items: { $ref: '#/components/schemas/Source' } }
    metadata: { type: object }
```

#### Agent Management
```yaml
Agent:
  type: object
  properties:
    id: { type: string }
    name: { type: string }
    category: { enum: [governance, infrastructure, knowledge, culture, economy, language, health] }
    capabilities: { type: array, items: { type: string } }
    status: { enum: [available, busy, offline] }
    current_load: { type: number, minimum: 0, maximum: 1 }
```

#### P2P Network
```yaml
P2PStatusResponse:
  type: object
  properties:
    status: { enum: [connected, connecting, disconnected] }
    peer_count: { type: integer }
    transports: { type: object }
    network_health: { type: object }

Peer:
  type: object
  properties:
    id: { type: string }
    transport: { enum: [bitchat, betanet] }
    status: { enum: [connected, connecting, disconnected] }
    capabilities: { type: array }
    last_seen: { type: string, format: date-time }
```

#### Digital Twin
```yaml
DigitalTwinProfile:
  type: object
  properties:
    user_id: { type: string }
    model_size_mb: { type: number }
    learning_stats: { type: object }
    privacy_settings: { type: object }
    last_updated: { type: string, format: date-time }
```

### Error Handling
```yaml
ErrorResponse:
  type: object
  properties:
    detail: { type: string }
    error_code: { type: string }
    timestamp: { type: string, format: date-time }
    request_id: { type: string }

RateLimitResponse:
  allOf:
    - $ref: '#/components/schemas/ErrorResponse'
    - properties:
        retry_after: { type: integer }
        limit: { type: integer }
        window_seconds: { type: integer }
```

## Security & Authentication

### Authentication Methods
- **Bearer Token** - JWT-based authentication
- **API Key** - Header-based authentication (`x-api-key`)
- **OAuth 2.0** - Third-party authentication

### Security Features
- **Rate Limiting** - Per-endpoint request limits
- **CORS** - Cross-origin resource sharing
- **TLS/SSL** - Encrypted communications
- **WebSocket Security** - Authenticated WebSocket connections
- **P2P Encryption** - End-to-end encrypted P2P messaging

### Authorization Levels
- **Public** - No authentication required (health checks)
- **Authenticated** - Valid token required
- **Premium** - Enhanced rate limits
- **Admin** - Administrative privileges

### Security Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1693228800
Authorization: Bearer <jwt-token>
x-api-key: <api-key>
Idempotency-Key: <unique-key>
```

## Summary

The AIVillage platform exposes a comprehensive API surface across multiple domains:

- **75+ REST endpoints** across core services
- **10+ WebSocket channels** for real-time communication
- **50+ public classes** for programmatic access
- **15+ CLI tools** for system management
- **20+ external integrations** with AI/ML providers

All APIs follow OpenAPI 3.0 specifications with comprehensive error handling, rate limiting, and security controls. The platform supports both synchronous HTTP requests and asynchronous WebSocket communications, with full P2P mesh networking capabilities for distributed operations.