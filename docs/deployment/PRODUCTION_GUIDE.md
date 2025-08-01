# AIVillage Production Deployment Guide

⚠️ **Critical**: `server.py` is for DEVELOPMENT ONLY. This guide covers production deployment using the actual microservices architecture.

## Production vs Development Architecture

### ❌ **What NOT to Use in Production**

**`server.py`** - Development Server Only
```python
# This file is marked "DEVELOPMENT ONLY" for good reasons:
- Single-threaded FastAPI server
- No load balancing or scaling
- Development middleware and debugging enabled
- Not optimized for production workloads
- Missing production security features
```

### ✅ **What to Use in Production**

**Microservices Architecture**
- **Gateway Service**: API gateway and routing
- **Twin Service**: Digital twin management
- **MCP Servers**: Model Context Protocol services
- **Mesh Network**: P2P distributed communication

## Production-Ready Components

### 1. **Gateway Service** 
**Location**: `experimental/services/gateway/`
**Purpose**: Main API gateway and request routing
**Features**:
- RESTful API endpoints
- Authentication and authorization
- Request routing and load balancing
- Service discovery and health checks

### 2. **Twin Service**
**Location**: `experimental/services/twin/`
**Purpose**: Digital twin management and personalization
**Features**:
- User preference management
- Personalized AI interactions
- Privacy-focused edge processing
- Adaptive learning capabilities

### 3. **MCP Servers**
**Location**: `src/mcp_servers/hyperag/`
**Purpose**: Specialized AI service components
**Services**:
- **Guardian**: Security and policy enforcement
- **Planning**: Query classification and task planning
- **Repair**: Automated system repair (InnovatorAgent)
- **Memory**: Hypergraph knowledge management
- **Retrieval**: Personalized PageRank retrieval

### 4. **Mesh Network Layer**
**Location**: `mesh_network_manager.py`
**Purpose**: Distributed P2P communication
**Features**:
- Peer discovery and connection management
- Intelligent routing with health monitoring
- Fault tolerance and automatic recovery
- Message delivery guarantees

## Container Deployment

Use the provided `docker-compose.yml` for production deployment, NOT the development server.

## Environment Configuration

**`.env.production`**:
```bash
# Environment
ENV=production
DEBUG=false

# Services
GATEWAY_PORT=8080
TWIN_PORT=8081
MCP_PORT=8082
MESH_PORT=9000

# Security (CHANGE THESE IN PRODUCTION)
JWT_SECRET=your-production-jwt-secret-change-this
API_KEY=your-production-api-key-change-this

# Mesh Network
NODE_ID=production-node-1
MAX_CONNECTIONS=50
HEALTH_CHECK_INTERVAL=30
```

## Deployment Verification

### **Production Readiness Checklist**

- [ ] **All services start successfully**
- [ ] **Health checks pass for all components**
- [ ] **Mesh network peers discover each other**
- [ ] **API endpoints respond correctly**
- [ ] **SSL certificates configured**

### **Smoke Tests**

```bash
# Run production smoke tests
pytest tests/production/ --env=production

# Integration verification
pytest tests/integration/test_full_system_integration.py
```

## Critical Warning

**NEVER use `server.py` in production environments.** It is explicitly designed for development only and lacks the security, performance, and reliability features required for production deployment.

For production deployment questions, see the troubleshooting section or create an issue with the `production` label.