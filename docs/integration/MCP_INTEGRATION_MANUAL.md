# MCP (Model Control Protocol) Integration Manual

## Overview

The Model Control Protocol (MCP) is the core integration technology that enables seamless communication between AI models, services, and user interfaces across the AIVillage ecosystem. This manual provides comprehensive guidance for integrating MCP into your applications and services.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   MCP Architecture                      │
├─────────────────────────────────────────────────────────┤
│  Client Applications                                    │
│  ├── Web UI (Admin Dashboard)                         │
│  ├── Mobile Apps (Digital Twin)                       │
│  ├── Agent Forge Interface                            │
│  └── P2P/Fog Computing Nodes                          │
├─────────────────────────────────────────────────────────┤
│  MCP Protocol Layer                                    │
│  ├── Message Routing                                  │
│  ├── Authentication/Authorization                     │
│  ├── Request/Response Handling                        │
│  └── WebSocket Management                             │
├─────────────────────────────────────────────────────────┤
│  Backend Services                                      │
│  ├── Unified Agent Forge Backend                      │
│  ├── P2P Network Coordinator                          │
│  ├── Fog Computing Services                           │
│  └── Token/Governance Systems                         │
├─────────────────────────────────────────────────────────┤
│  Data Layer                                            │
│  ├── Model Storage                                    │
│  ├── Training Artifacts                               │
│  ├── P2P Mesh State                                   │
│  └── User Preferences                                 │
└─────────────────────────────────────────────────────────┘
```

## Core MCP Components

### 1. MCP Server Implementation

The MCP server handles protocol communication, authentication, and service orchestration:

```python
# infrastructure/gateway/unified_agent_forge_backend.py (excerpt)
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
from typing import Dict, Any, List

app = FastAPI(title="MCP Unified Backend")

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
websocket_connections = set()
service_registry = {}
authentication_cache = {}
```

### 2. WebSocket Connection Manager

Real-time bidirectional communication between clients and services:

```python
class WebSocketManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self.service_subscriptions: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_type: str = "default"):
        await websocket.accept()
        self.connections.append(websocket)

        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "client_type": client_type,
            "mcp_version": "2.0",
            "available_services": list(self.service_registry.keys())
        })

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

        # Remove from service subscriptions
        for service_list in self.service_subscriptions.values():
            if websocket in service_list:
                service_list.remove(websocket)

    async def broadcast(self, message: dict, service_filter: str = None):
        target_connections = self.connections

        if service_filter and service_filter in self.service_subscriptions:
            target_connections = self.service_subscriptions[service_filter]

        disconnected = []
        for connection in target_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"WebSocket send error: {e}")
                disconnected.append(connection)

        # Clean up disconnected connections
        for conn in disconnected:
            await self.disconnect(conn)

    async def subscribe_to_service(self, websocket: WebSocket, service_name: str):
        if service_name not in self.service_subscriptions:
            self.service_subscriptions[service_name] = []

        if websocket not in self.service_subscriptions[service_name]:
            self.service_subscriptions[service_name].append(websocket)
```

### 3. Authentication and JWT Integration

Secure access control with JSON Web Tokens:

```python
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

class MCPAuthenticator:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Could not validate credentials")

    def hash_password(self, password: str) -> str:
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

# Global authenticator instance
mcp_auth = MCPAuthenticator("your-secret-key-here")
```

## MCP API Endpoints

### Core Service Endpoints

```python
# Authentication endpoints
@app.post("/api/auth/login")
async def login(credentials: dict):
    username = credentials.get("username")
    password = credentials.get("password")

    # Validate credentials (implement your user validation)
    if not validate_user_credentials(username, password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create access token
    access_token = mcp_auth.create_access_token(
        data={"sub": username, "role": "user"}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 3600
    }

@app.get("/api/auth/verify")
async def verify_token(token: str = Depends(oauth2_scheme)):
    payload = mcp_auth.verify_token(token)
    return {"valid": True, "user": payload.get("sub")}

# Service discovery endpoints
@app.get("/api/services")
async def list_services():
    return {
        "services": list(service_registry.keys()),
        "total": len(service_registry),
        "mcp_version": "2.0"
    }

@app.get("/api/services/{service_name}/status")
async def get_service_status(service_name: str):
    if service_name not in service_registry:
        raise HTTPException(status_code=404, detail="Service not found")

    return service_registry[service_name]

# Model management endpoints
@app.get("/api/models")
async def list_models():
    return {
        "models": list(model_storage.values()),
        "total": len(model_storage),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/models/{model_id}/chat")
async def chat_with_model(model_id: str, request: dict):
    if model_id not in model_storage:
        raise HTTPException(status_code=404, detail="Model not found")

    model = model_storage[model_id]
    response = generate_model_response(model, request.get("message", ""))

    return {
        "model_id": model_id,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }
```

### Agent Forge Integration Endpoints

```python
# Agent Forge specific endpoints
@app.post("/api/agent-forge/training/start")
async def start_training(request: dict, background_tasks: BackgroundTasks):
    phase_name = request.get("phase_name", "Cognate")
    parameters = request.get("parameters", {})

    task_id = str(uuid.uuid4())

    # Start training in background
    if phase_name == "Cognate":
        background_tasks.add_task(execute_cognate_training, task_id, parameters)

    return {
        "task_id": task_id,
        "phase_name": phase_name,
        "status": "started",
        "message": f"{phase_name} training initiated"
    }

@app.get("/api/agent-forge/training/{task_id}/status")
async def get_training_status(task_id: str):
    if task_id not in training_instances:
        raise HTTPException(status_code=404, detail="Training task not found")

    return training_instances[task_id]

@app.get("/api/agent-forge/phases")
async def list_training_phases():
    phases = [
        "Cognate", "EvoMerge", "Quiet-STaR", "BitNet",
        "Forge Training", "Tool/Persona", "ADAS", "Final Compression"
    ]

    return {
        "phases": phases,
        "status": {phase: phase_status.get(phase, {"status": "ready"}) for phase in phases}
    }
```

### P2P/Fog Computing Endpoints

```python
# P2P network endpoints
@app.get("/api/p2p/status")
async def get_p2p_status():
    return {
        "bitchat": {
            "connected": mobile_bridge.connected if mobile_bridge else False,
            "active_peers": 8,
            "message_throughput": "150/hour"
        },
        "betanet": {
            "connected": mixnode_client.connected if mixnode_client else False,
            "active_circuits": 3,
            "anonymity_level": "high"
        },
        "mesh_topology": "adaptive",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/p2p/peers")
async def get_peer_list():
    return {
        "total_peers": 12,
        "connected_peers": 8,
        "peers": [
            {
                "peer_id": f"peer_{i}",
                "address": f"10.0.1.{100+i}",
                "latency_ms": 50 + (i * 10),
                "services": ["compute", "storage", "relay"]
            }
            for i in range(8)
        ]
    }

# Fog computing endpoints
@app.get("/api/fog/resources")
async def get_fog_resources():
    return {
        "active_nodes": fog_coordinator.active_nodes if fog_coordinator else 0,
        "total_compute_hours": 48.5,
        "available_memory_gb": 256.8,
        "utilization_percent": 65.3,
        "token_rewards_distributed": 2450,
        "harvesting_devices": 8
    }

@app.get("/api/fog/marketplace")
async def get_marketplace_services():
    return {
        "total_services": fog_marketplace.total_offerings if fog_marketplace else 0,
        "service_categories": ["compute", "storage", "bandwidth", "ml_inference"],
        "average_price_per_hour": 1.25,
        "active_contracts": 24
    }

@app.get("/api/fog/tokens")
async def get_token_balance():
    return {
        "balance": 1250.75,
        "staked_amount": 500.0,
        "rewards_earned": 890.25,
        "network_participation": True,
        "token_symbol": "FOG"
    }
```

## WebSocket Integration

### Client-Side WebSocket Implementation

```javascript
// JavaScript client for web applications
class MCPClient {
    constructor(baseUrl, authToken = null) {
        this.baseUrl = baseUrl;
        this.authToken = authToken;
        this.ws = null;
        this.messageHandlers = new Map();
        this.subscriptions = new Set();
    }

    async connect() {
        const wsUrl = this.baseUrl.replace('http', 'ws') + '/ws';
        this.ws = new WebSocket(wsUrl);

        return new Promise((resolve, reject) => {
            this.ws.onopen = () => {
                console.log('MCP WebSocket connected');

                // Send authentication if available
                if (this.authToken) {
                    this.send({
                        type: 'authenticate',
                        token: this.authToken
                    });
                }

                resolve();
            };

            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            };

            this.ws.onclose = (event) => {
                console.log('MCP WebSocket disconnected:', event.code);
                this.handleReconnection();
            };

            this.ws.onerror = (error) => {
                console.error('MCP WebSocket error:', error);
                reject(error);
            };
        });
    }

    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }

    subscribe(messageType, handler) {
        if (!this.messageHandlers.has(messageType)) {
            this.messageHandlers.set(messageType, []);
        }
        this.messageHandlers.get(messageType).push(handler);
        this.subscriptions.add(messageType);
    }

    handleMessage(message) {
        const handlers = this.messageHandlers.get(message.type) || [];
        handlers.forEach(handler => handler(message));
    }

    // Service-specific methods
    async startTraining(phaseName, parameters = {}) {
        this.send({
            type: 'start_training',
            phase_name: phaseName,
            parameters: parameters
        });
    }

    async subscribeToTrainingUpdates() {
        this.subscribe('training_update', (message) => {
            console.log('Training update:', message.data);
            // Handle training progress updates
        });

        this.send({
            type: 'subscribe_service',
            service: 'agent_forge_training'
        });
    }

    async subscribeToP2PUpdates() {
        this.subscribe('p2p_network_update', (message) => {
            console.log('P2P network update:', message.data);
            // Handle P2P network status updates
        });

        this.send({
            type: 'subscribe_service',
            service: 'p2p_network'
        });
    }

    handleReconnection() {
        setTimeout(() => {
            console.log('Attempting MCP reconnection...');
            this.connect().catch(console.error);
        }, 5000);
    }
}

// Usage example
const mcpClient = new MCPClient('ws://localhost:8083', authToken);

mcpClient.connect().then(() => {
    // Subscribe to training updates
    mcpClient.subscribeToTrainingUpdates();

    // Subscribe to P2P network updates
    mcpClient.subscribeToP2PUpdates();

    // Start Cognate training
    mcpClient.startTraining('Cognate', {
        real_training: true,
        max_steps: 2000
    });
});
```

### Python Client Implementation

```python
import asyncio
import websockets
import json
import logging

class MCPPythonClient:
    def __init__(self, base_url: str, auth_token: str = None):
        self.base_url = base_url
        self.auth_token = auth_token
        self.websocket = None
        self.message_handlers = {}
        self.running = False

    async def connect(self):
        ws_url = self.base_url.replace('http', 'ws') + '/ws'

        try:
            self.websocket = await websockets.connect(ws_url)
            self.running = True

            # Send authentication if available
            if self.auth_token:
                await self.send({
                    'type': 'authenticate',
                    'token': self.auth_token
                })

            # Start message handling loop
            asyncio.create_task(self.handle_messages())

            logging.info("MCP Python client connected")

        except Exception as e:
            logging.error(f"Failed to connect to MCP server: {e}")
            raise

    async def send(self, message: dict):
        if self.websocket:
            await self.websocket.send(json.dumps(message))

    async def handle_messages(self):
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get('type')

                if message_type in self.message_handlers:
                    handler = self.message_handlers[message_type]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
        except websockets.exceptions.ConnectionClosed:
            logging.warning("MCP WebSocket connection closed")
            self.running = False
        except Exception as e:
            logging.error(f"Error handling MCP messages: {e}")

    def subscribe(self, message_type: str, handler):
        self.message_handlers[message_type] = handler

    async def start_training(self, phase_name: str, parameters: dict = None):
        await self.send({
            'type': 'start_training',
            'phase_name': phase_name,
            'parameters': parameters or {}
        })

    async def get_training_status(self):
        await self.send({
            'type': 'get_training_status'
        })

    async def close(self):
        if self.websocket:
            await self.websocket.close()
        self.running = False

# Usage example
async def main():
    client = MCPPythonClient('ws://localhost:8083')

    # Set up message handlers
    client.subscribe('training_update', lambda msg: print(f"Training: {msg}"))
    client.subscribe('p2p_update', lambda msg: print(f"P2P: {msg}"))

    await client.connect()

    # Start training
    await client.start_training('Cognate', {'real_training': True})

    # Keep the client running
    while client.running:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Service Integration Patterns

### 1. Agent Forge Integration

```python
# Agent Forge MCP integration example
class AgentForgeMCPIntegration:
    def __init__(self, mcp_client: MCPPythonClient):
        self.mcp_client = mcp_client
        self.training_status = {}

    async def setup_training_pipeline(self):
        # Subscribe to training updates
        self.mcp_client.subscribe('training_update', self.handle_training_update)
        self.mcp_client.subscribe('training_complete', self.handle_training_complete)
        self.mcp_client.subscribe('training_error', self.handle_training_error)

    async def handle_training_update(self, message):
        phase = message.get('phase')
        data = message.get('data', {})

        self.training_status[phase] = data

        # Log progress
        progress = data.get('progress', 0)
        current_message = data.get('message', 'Processing...')

        logging.info(f"{phase} training: {progress*100:.1f}% - {current_message}")

    async def handle_training_complete(self, message):
        phase = message.get('phase')
        data = message.get('data', {})

        logging.info(f"{phase} training completed successfully")

        # Trigger next phase if applicable
        next_phases = {
            'Cognate': 'EvoMerge',
            'EvoMerge': 'Quiet-STaR',
            'Quiet-STaR': 'BitNet'
        }

        if phase in next_phases:
            next_phase = next_phases[phase]
            logging.info(f"Starting next phase: {next_phase}")
            await self.mcp_client.start_training(next_phase)

    async def handle_training_error(self, message):
        phase = message.get('phase')
        error = message.get('data', {}).get('error', 'Unknown error')

        logging.error(f"{phase} training failed: {error}")

        # Implement retry logic or error recovery
        await self.retry_training(phase)

    async def retry_training(self, phase: str, max_retries: int = 3):
        for attempt in range(max_retries):
            logging.info(f"Retrying {phase} training (attempt {attempt + 1})")

            try:
                await self.mcp_client.start_training(phase, {'retry': True})
                break
            except Exception as e:
                logging.error(f"Retry {attempt + 1} failed: {e}")

                if attempt == max_retries - 1:
                    logging.error(f"All retries exhausted for {phase} training")
```

### 2. P2P Network Integration

```python
# P2P network MCP integration
class P2PNetworkMCPIntegration:
    def __init__(self, mcp_client: MCPPythonClient):
        self.mcp_client = mcp_client
        self.peer_status = {}
        self.network_metrics = {}

    async def setup_p2p_monitoring(self):
        # Subscribe to P2P updates
        self.mcp_client.subscribe('p2p_status_update', self.handle_p2p_status)
        self.mcp_client.subscribe('peer_connected', self.handle_peer_connected)
        self.mcp_client.subscribe('peer_disconnected', self.handle_peer_disconnected)
        self.mcp_client.subscribe('network_metrics', self.handle_network_metrics)

    async def handle_p2p_status(self, message):
        data = message.get('data', {})

        # Update BitChat status
        bitchat_status = data.get('bitchat', {})
        if bitchat_status.get('connected'):
            logging.info(f"BitChat: {bitchat_status['active_peers']} peers connected")

        # Update BetaNet status
        betanet_status = data.get('betanet', {})
        if betanet_status.get('connected'):
            logging.info(f"BetaNet: {betanet_status['active_circuits']} circuits active")

    async def handle_peer_connected(self, message):
        peer_info = message.get('peer', {})
        peer_id = peer_info.get('peer_id')

        self.peer_status[peer_id] = peer_info
        logging.info(f"Peer connected: {peer_id}")

    async def handle_peer_disconnected(self, message):
        peer_id = message.get('peer_id')

        if peer_id in self.peer_status:
            del self.peer_status[peer_id]

        logging.info(f"Peer disconnected: {peer_id}")

    async def handle_network_metrics(self, message):
        metrics = message.get('metrics', {})
        self.network_metrics.update(metrics)

        # Log important metrics
        latency = metrics.get('avg_latency_ms', 0)
        throughput = metrics.get('throughput_mbps', 0)

        logging.info(f"Network: {latency}ms latency, {throughput}Mbps throughput")

    async def broadcast_to_network(self, message: str, priority: str = 'normal'):
        await self.mcp_client.send({
            'type': 'p2p_broadcast',
            'message': message,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        })
```

### 3. Fog Computing Integration

```python
# Fog computing MCP integration
class FogComputingMCPIntegration:
    def __init__(self, mcp_client: MCPPythonClient):
        self.mcp_client = mcp_client
        self.resource_status = {}
        self.marketplace_data = {}

    async def setup_fog_monitoring(self):
        # Subscribe to fog computing updates
        self.mcp_client.subscribe('fog_resources_update', self.handle_resource_update)
        self.mcp_client.subscribe('fog_marketplace_update', self.handle_marketplace_update)
        self.mcp_client.subscribe('token_balance_update', self.handle_token_update)
        self.mcp_client.subscribe('harvesting_session_update', self.handle_harvesting_update)

    async def handle_resource_update(self, message):
        resources = message.get('data', {})
        self.resource_status.update(resources)

        # Log resource utilization
        cpu_util = resources.get('utilization', {}).get('cpu_utilization', 0)
        memory_util = resources.get('utilization', {}).get('memory_utilization', 0)

        logging.info(f"Fog resources: {cpu_util}% CPU, {memory_util}% memory")

    async def handle_marketplace_update(self, message):
        marketplace = message.get('data', {})
        self.marketplace_data.update(marketplace)

        # Log marketplace activity
        total_services = marketplace.get('marketplace_stats', {}).get('total_offerings', 0)
        active_contracts = marketplace.get('marketplace_stats', {}).get('active_contracts', 0)

        logging.info(f"Marketplace: {total_services} services, {active_contracts} contracts")

    async def handle_token_update(self, message):
        token_data = message.get('data', {})

        balance = token_data.get('user_balance', {}).get('balance', 0)
        rewards = token_data.get('user_balance', {}).get('total_earned', 0)

        logging.info(f"FOG tokens: {balance} balance, {rewards} earned")

    async def handle_harvesting_update(self, message):
        harvesting_data = message.get('data', {})

        active_devices = harvesting_data.get('harvesting', {}).get('active_devices', 0)
        tokens_distributed = harvesting_data.get('rewards', {}).get('tokens_distributed_today', 0)

        logging.info(f"Harvesting: {active_devices} devices, {tokens_distributed} tokens distributed")

    async def request_compute_service(self, requirements: dict):
        await self.mcp_client.send({
            'type': 'fog_service_request',
            'service_type': 'compute_instance',
            'requirements': requirements,
            'max_price_per_hour': requirements.get('budget', 1.0)
        })
```

## Security and Authentication

### JWT Token Management

```python
# Advanced JWT management for MCP
class MCPTokenManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_tokens = {}
        self.refresh_tokens = {}

    def create_token_pair(self, user_id: str, permissions: list = None):
        # Create access token (short-lived)
        access_payload = {
            "sub": user_id,
            "permissions": permissions or [],
            "type": "access",
            "exp": datetime.utcnow() + timedelta(minutes=15)
        }
        access_token = jwt.encode(access_payload, self.secret_key, algorithm="HS256")

        # Create refresh token (long-lived)
        refresh_payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=7)
        }
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm="HS256")

        # Store tokens
        self.active_tokens[access_token] = access_payload
        self.refresh_tokens[refresh_token] = refresh_payload

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 900  # 15 minutes
        }

    def refresh_access_token(self, refresh_token: str):
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=["HS256"])

            if payload.get("type") != "refresh":
                raise ValueError("Invalid token type")

            user_id = payload.get("sub")

            # Create new access token
            return self.create_token_pair(user_id)

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Refresh token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

    def verify_permissions(self, token: str, required_permissions: list):
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            user_permissions = payload.get("permissions", [])

            # Check if user has all required permissions
            if not all(perm in user_permissions for perm in required_permissions):
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            return payload

        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

### Role-Based Access Control

```python
# RBAC implementation for MCP services
class MCPRoleManager:
    def __init__(self):
        self.roles = {
            "admin": {
                "permissions": [
                    "agent_forge:start_training",
                    "agent_forge:stop_training",
                    "p2p:manage_network",
                    "fog:admin_marketplace",
                    "system:full_access"
                ]
            },
            "developer": {
                "permissions": [
                    "agent_forge:start_training",
                    "agent_forge:view_models",
                    "p2p:view_network",
                    "fog:use_marketplace"
                ]
            },
            "user": {
                "permissions": [
                    "agent_forge:view_models",
                    "agent_forge:chat_models",
                    "p2p:basic_access",
                    "fog:basic_access"
                ]
            }
        }

    def get_permissions(self, role: str) -> list:
        return self.roles.get(role, {}).get("permissions", [])

    def has_permission(self, user_role: str, required_permission: str) -> bool:
        user_permissions = self.get_permissions(user_role)

        # Check for exact match
        if required_permission in user_permissions:
            return True

        # Check for wildcard permissions
        permission_parts = required_permission.split(":")
        if len(permission_parts) == 2:
            service, action = permission_parts
            wildcard_permission = f"{service}:*"
            if wildcard_permission in user_permissions:
                return True

        # Check for admin access
        if "system:full_access" in user_permissions:
            return True

        return False

# Middleware for permission checking
async def check_permissions(request: Request, required_permissions: list):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = auth_header.split(" ")[1]
    payload = mcp_auth.verify_token(token)

    user_role = payload.get("role", "user")
    role_manager = MCPRoleManager()

    for permission in required_permissions:
        if not role_manager.has_permission(user_role, permission):
            raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")

    return payload
```

## Configuration Management

### Environment Configuration

```yaml
# mcp_config.yaml
mcp:
  server:
    host: "0.0.0.0"
    port: 8083
    workers: 4
    debug: false

  authentication:
    secret_key: "${MCP_SECRET_KEY}"
    algorithm: "HS256"
    access_token_expire_minutes: 15
    refresh_token_expire_days: 7

  websocket:
    max_connections: 1000
    ping_interval: 30
    ping_timeout: 10

  services:
    agent_forge:
      enabled: true
      max_concurrent_trainings: 3
      model_storage_path: "./models"

    p2p_network:
      enabled: true
      bitchat_bridge: true
      betanet_mixnode: true

    fog_computing:
      enabled: true
      harvesting: true
      marketplace: true
      token_system: true

  logging:
    level: "INFO"
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: "./logs/mcp.log"
    max_size_mb: 100
    backup_count: 5

  security:
    cors_origins: ["*"]
    rate_limiting:
      enabled: true
      requests_per_minute: 100

  database:
    url: "sqlite:///./mcp_data.db"
    pool_size: 5
    echo: false
```

### Configuration Loading

```python
# Configuration management
import yaml
import os
from pathlib import Path

class MCPConfig:
    def __init__(self, config_file: str = "mcp_config.yaml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> dict:
        config_path = Path(self.config_file)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Environment variable substitution
        config = self.substitute_env_vars(config)

        return config

    def substitute_env_vars(self, obj):
        if isinstance(obj, dict):
            return {key: self.substitute_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

# Global configuration instance
mcp_config = MCPConfig()
```

## Monitoring and Metrics

### Health Check Endpoints

```python
# Health monitoring for MCP services
@app.get("/api/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "services": {}
    }

    # Check Agent Forge service
    if REAL_TRAINING_AVAILABLE:
        health_status["services"]["agent_forge"] = {
            "status": "healthy",
            "active_trainings": len(training_instances),
            "total_models": len(model_storage)
        }
    else:
        health_status["services"]["agent_forge"] = {
            "status": "limited",
            "message": "Real training not available, using simulation"
        }

    # Check P2P services
    if P2P_FOG_AVAILABLE:
        health_status["services"]["p2p_network"] = {
            "status": "healthy",
            "bitchat_connected": mobile_bridge.connected if mobile_bridge else False,
            "betanet_connected": mixnode_client.connected if mixnode_client else False
        }
    else:
        health_status["services"]["p2p_network"] = {
            "status": "unavailable",
            "message": "P2P services not initialized"
        }

    # Check Fog computing
    if fog_coordinator:
        health_status["services"]["fog_computing"] = {
            "status": "healthy",
            "active_nodes": len(fog_coordinator.active_nodes),
            "harvesting_active": fog_coordinator.harvest_manager.is_running
        }
    else:
        health_status["services"]["fog_computing"] = {
            "status": "unavailable",
            "message": "Fog coordinator not running"
        }

    # Overall status determination
    service_statuses = [s.get("status") for s in health_status["services"].values()]
    if all(status == "healthy" for status in service_statuses):
        health_status["status"] = "healthy"
    elif any(status == "healthy" for status in service_statuses):
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unhealthy"

    return health_status

@app.get("/api/metrics")
async def get_metrics():
    return {
        "websocket_connections": len(manager.connections),
        "active_training_instances": len(training_instances),
        "total_models": len(model_storage),
        "uptime_seconds": time.time() - startup_time,
        "memory_usage_mb": get_memory_usage(),
        "cpu_usage_percent": get_cpu_usage(),
        "request_count": get_request_count(),
        "error_count": get_error_count(),
        "timestamp": datetime.now().isoformat()
    }
```

### Performance Monitoring

```python
# Performance monitoring middleware
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = await func(*args, **kwargs)

            # Record successful request
            duration = time.time() - start_time
            record_request_metric(func.__name__, duration, "success")

            return result

        except Exception as e:
            # Record failed request
            duration = time.time() - start_time
            record_request_metric(func.__name__, duration, "error")

            # Log error
            logging.error(f"Error in {func.__name__}: {e}")
            raise

    return wrapper

# Metrics storage
request_metrics = []
error_counts = {}

def record_request_metric(endpoint: str, duration: float, status: str):
    request_metrics.append({
        "endpoint": endpoint,
        "duration": duration,
        "status": status,
        "timestamp": time.time()
    })

    # Keep only last 1000 metrics
    if len(request_metrics) > 1000:
        request_metrics.pop(0)

    # Update error counts
    if status == "error":
        error_counts[endpoint] = error_counts.get(endpoint, 0) + 1

# Apply monitoring to key endpoints
@app.post("/api/agent-forge/training/start")
@monitor_performance
async def monitored_start_training(request: dict, background_tasks: BackgroundTasks):
    # Implementation here
    pass
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. WebSocket Connection Issues

```python
# WebSocket debugging
async def debug_websocket_connection():
    try:
        # Test WebSocket endpoint
        async with websockets.connect('ws://localhost:8083/ws') as websocket:
            await websocket.send(json.dumps({"type": "ping"}))
            response = await websocket.recv()
            print(f"WebSocket test successful: {response}")

    except Exception as e:
        print(f"WebSocket connection failed: {e}")

        # Check common issues
        if "Connection refused" in str(e):
            print("- Check if MCP server is running on port 8083")
        elif "Handshake failed" in str(e):
            print("- Check CORS configuration")
        elif "SSL" in str(e):
            print("- Try using ws:// instead of wss:// for development")
```

#### 2. Authentication Problems

```bash
# Test authentication endpoint
curl -X POST http://localhost:8083/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test"}'

# Verify token
curl -X GET http://localhost:8083/api/auth/verify \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

#### 3. Service Integration Issues

```python
# Service availability check
async def check_service_availability():
    services = ["agent_forge", "p2p_network", "fog_computing"]

    for service in services:
        try:
            response = requests.get(f"http://localhost:8083/api/services/{service}/status")
            if response.status_code == 200:
                print(f"✅ {service}: Available")
            else:
                print(f"❌ {service}: Unavailable (HTTP {response.status_code})")
        except Exception as e:
            print(f"❌ {service}: Error - {e}")
```

#### 4. Performance Issues

```python
# Performance diagnostics
async def diagnose_performance():
    # Check memory usage
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    print(f"Memory usage: {memory_mb:.1f} MB")

    # Check WebSocket connections
    print(f"Active WebSocket connections: {len(manager.connections)}")

    # Check training instances
    print(f"Active training instances: {len(training_instances)}")

    # Check request metrics
    recent_requests = [m for m in request_metrics if time.time() - m["timestamp"] < 300]
    avg_duration = sum(r["duration"] for r in recent_requests) / max(len(recent_requests), 1)

    print(f"Average request duration (last 5 min): {avg_duration:.3f}s")
```

### Logging Configuration

```python
# Enhanced logging setup
import logging
from logging.handlers import RotatingFileHandler

def setup_mcp_logging():
    # Create logger
    logger = logging.getLogger("mcp")
    logger.setLevel(logging.INFO)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        'logs/mcp.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize logging
mcp_logger = setup_mcp_logging()
```

## Conclusion

The MCP (Model Control Protocol) provides a robust, scalable foundation for integrating AI services, P2P networks, and fog computing capabilities. Key benefits include:

- **Unified Communication**: Single protocol for all service interactions
- **Real-time Updates**: WebSocket-based bidirectional communication
- **Secure Authentication**: JWT-based authentication with role-based access
- **Service Discovery**: Automatic discovery and registration of available services
- **Performance Monitoring**: Built-in metrics and health checking
- **Cross-platform Support**: Works across web, mobile, and desktop applications

For successful MCP integration:

1. Start with basic WebSocket connection and authentication
2. Implement service-specific message handlers
3. Add error handling and reconnection logic
4. Configure appropriate security measures
5. Set up monitoring and alerting
6. Test thoroughly across different network conditions

The MCP system enables seamless coordination between Agent Forge training, P2P network operations, and fog computing services, providing a comprehensive platform for distributed AI applications.
