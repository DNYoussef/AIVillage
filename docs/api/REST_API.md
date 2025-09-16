# AIVillage REST API Documentation

## Overview

The AIVillage Unified API Gateway provides a comprehensive REST API for interacting with all system components including Agent Forge, P2P networking, fog computing, and agent communication.

**Base URL**: `https://api.aivillage.app` (Production) or `http://localhost:8080` (Development)

**API Version**: v1

**Authentication**: Bearer Token (JWT)

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [Agent Forge API](#agent-forge-api)
4. [Agent Communication API](#agent-communication-api)
5. [P2P Network API](#p2p-network-api)
6. [Fog Computing API](#fog-computing-api)
7. [RAG System API](#rag-system-api)
8. [File Management API](#file-management-api)
9. [WebSocket API](#websocket-api)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)
12. [SDK Examples](#sdk-examples)

## Authentication

### JWT Authentication

All API endpoints (except health checks) require JWT authentication.

**Request Header**:
```http
Authorization: Bearer <jwt_token>
```

**Token Structure**:
```json
{
  "sub": "user_id",
  "scopes": ["read", "write", "admin"],
  "tier": "premium",
  "exp": 1640995200,
  "iat": 1640908800
}
```

### Obtain Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password",
  "mfa_code": "123456"  // Optional, if MFA enabled
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "scopes": ["read", "write"]
}
```

### Refresh Token

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

## Core Endpoints

### Health Check

**GET** `/healthz`

Check system health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "services": {
    "agent_forge": {
      "status": "running",
      "available": true
    },
    "p2p_fog": {
      "status": "running",
      "available": true
    },
    "websocket": {
      "status": "running",
      "active_connections": 42,
      "total_connections": 1337
    }
  },
  "version": "1.0.0"
}
```

### System Information

**GET** `/`

Get API information and available endpoints.

**Response**:
```json
{
  "success": true,
  "data": {
    "service": "AIVillage Unified API Gateway",
    "version": "1.0.0",
    "status": "operational",
    "features": [
      "Agent Forge 7-phase training pipeline",
      "P2P/Fog computing integration",
      "JWT authentication with MFA",
      "Real-time WebSocket updates",
      "Production-grade security",
      "Comprehensive API documentation"
    ],
    "endpoints": {
      "health": "GET /healthz",
      "training": "POST /v1/models/train",
      "models": "GET /v1/models",
      "chat": "POST /v1/chat",
      "query": "POST /v1/query",
      "upload": "POST /v1/upload",
      "p2p_status": "GET /v1/p2p/status",
      "fog_nodes": "GET /v1/fog/nodes",
      "tokens": "GET /v1/tokens",
      "websocket": "ws://host/ws"
    }
  },
  "message": "",
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_123456789"
}
```

## Agent Forge API

### Start Training

**POST** `/v1/models/train`

Start a new Agent Forge training pipeline.

**Request**:
```json
{
  "phase_name": "cognate",
  "parameters": {
    "base_models": [
      "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
      "Qwen/Qwen2-1.5B-Instruct"
    ],
    "target_architecture": "auto",
    "merge_strategy": "evolutionary"
  },
  "real_training": true,
  "max_steps": 2000,
  "batch_size": 4
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "task_id": "task_abc123def456",
    "phase": "cognate",
    "status": "started",
    "real_training": true,
    "user_id": "user_123",
    "estimated_duration": "2-4 hours",
    "progress_url": "ws://localhost:8080/ws"
  },
  "message": "Training started for phase cognate",
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_987654321"
}
```

### List Models

**GET** `/v1/models`

Retrieve all trained models for the authenticated user.

**Query Parameters**:
- `limit`: Number of models to return (default: 10, max: 50)
- `offset`: Number of models to skip (default: 0)
- `status`: Filter by status (`training`, `completed`, `failed`, `all`)
- `phase`: Filter by training phase

**Response**:
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "model_id": "model_xyz789",
        "name": "Custom Agent Model v1",
        "status": "completed",
        "phases_completed": [
          "cognate",
          "evomerge",
          "quietstar",
          "bitnet",
          "training",
          "tool_baking",
          "adas",
          "final_compression"
        ],
        "created_at": "2025-01-15T08:00:00Z",
        "completed_at": "2025-01-15T10:15:00Z",
        "metrics": {
          "accuracy": 0.94,
          "compression_ratio": 0.65,
          "inference_speed": "12ms",
          "model_size": "1.2GB"
        },
        "download_url": "/v1/models/model_xyz789/download"
      }
    ],
    "total_count": 1,
    "user_id": "user_123"
  },
  "message": "Models retrieved successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Get Training Status

**GET** `/v1/models/train/{task_id}/status`

Get the current status of a training task.

**Response**:
```json
{
  "success": true,
  "data": {
    "task_id": "task_abc123def456",
    "status": "training",
    "current_phase": "forge_training",
    "progress": {
      "overall_progress": 0.65,
      "current_phase_progress": 0.75,
      "phases_completed": 4,
      "total_phases": 8,
      "estimated_time_remaining": "45 minutes"
    },
    "metrics": {
      "loss": 0.023,
      "accuracy": 0.91,
      "learning_rate": 0.0001,
      "batch_size": 4,
      "current_step": 1500,
      "total_steps": 2000
    },
    "logs_url": "/v1/models/train/task_abc123def456/logs"
  },
  "message": "Training status retrieved",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Chat with Model

**POST** `/v1/chat`

Chat with a trained model.

**Request**:
```json
{
  "model_id": "model_xyz789",
  "message": "What is the capital of France?",
  "conversation_id": "conv_123abc",
  "stream": false,
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 150,
    "top_p": 0.9
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "model_id": "model_xyz789",
    "response": "The capital of France is Paris. It is located in the north-central part of the country and serves as the political, cultural, and economic center of France.",
    "conversation_id": "conv_123abc",
    "user_id": "user_123",
    "usage": {
      "prompt_tokens": 12,
      "completion_tokens": 35,
      "total_tokens": 47
    },
    "metadata": {
      "inference_time_ms": 145,
      "model_version": "1.0.0",
      "confidence_score": 0.98
    }
  },
  "message": "Chat completed successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## Agent Communication API

### List Agents

**GET** `/v1/agents`

Get list of available agents.

**Response**:
```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "agent_id": "governance_king",
        "name": "The King",
        "description": "Ultimate constitutional authority and governance",
        "category": "governance",
        "capabilities": [
          "rag_access",
          "memory_access",
          "p2p_communication",
          "mcp_tools"
        ],
        "status": "active",
        "load": 0.25
      }
    ],
    "total_agents": 54,
    "active_agents": 42,
    "categories": [
      "governance",
      "infrastructure",
      "knowledge",
      "culture",
      "economy",
      "specialized"
    ]
  },
  "message": "Agents retrieved successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Send Message to Agent

**POST** `/v1/agents/{agent_id}/message`

Send a message to a specific agent.

**Request**:
```json
{
  "message": {
    "type": "query",
    "content": "What is the current system status?",
    "priority": "normal",
    "context": {
      "session_id": "sess_123",
      "user_context": "dashboard_view"
    }
  },
  "expect_response": true,
  "timeout": 30
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "message_id": "msg_abc123",
    "agent_id": "governance_king",
    "status": "delivered",
    "response": {
      "type": "status_report",
      "content": "System is operating normally. All services are healthy and 42 agents are active.",
      "metadata": {
        "processing_time_ms": 125,
        "confidence": 0.95,
        "sources": ["system_monitor", "agent_registry"]
      }
    },
    "delivery_time": "2025-01-15T10:30:00Z"
  },
  "message": "Message sent and response received",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Agent Conversation

**POST** `/v1/agents/conversation`

Start a multi-turn conversation with an agent.

**Request**:
```json
{
  "agent_id": "knowledge_oracle",
  "conversation_id": "conv_abc123",
  "messages": [
    {
      "role": "user",
      "content": "Explain quantum computing"
    }
  ],
  "context": {
    "user_level": "intermediate",
    "previous_topics": ["machine_learning", "algorithms"]
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "conversation_id": "conv_abc123",
    "agent_id": "knowledge_oracle",
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing",
        "timestamp": "2025-01-15T10:29:00Z"
      },
      {
        "role": "assistant",
        "content": "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers...",
        "timestamp": "2025-01-15T10:30:00Z",
        "metadata": {
          "knowledge_sources": ["quantum_physics_db", "computing_history"],
          "confidence": 0.92
        }
      }
    ],
    "status": "active"
  },
  "message": "Conversation updated",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## P2P Network API

### P2P Status

**GET** `/v1/p2p/status`

Get P2P network status and connectivity information.

**Response**:
```json
{
  "success": true,
  "data": {
    "status": "operational",
    "node_id": "node_abc123def456",
    "transports": {
      "bitchat": {
        "status": "connected",
        "peers": 12,
        "battery_level": 0.85,
        "signal_strength": "strong"
      },
      "betanet": {
        "status": "connected",
        "peers": 8,
        "throughput_mbps": 125.5,
        "latency_ms": 15
      },
      "quic": {
        "status": "connected",
        "peers": 5,
        "throughput_mbps": 850.2,
        "latency_ms": 8
      }
    },
    "routing_table": {
      "direct_routes": 25,
      "mesh_routes": 180,
      "server_routes": 5
    },
    "performance": {
      "messages_sent": 1523,
      "messages_received": 1489,
      "bytes_sent": 2456789,
      "bytes_received": 2398765,
      "uptime_seconds": 86400
    }
  },
  "message": "P2P status retrieved",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Discover Peers

**POST** `/v1/p2p/discover`

Initiate peer discovery process.

**Request**:
```json
{
  "discovery_types": ["local", "bluetooth", "dht"],
  "timeout_seconds": 30,
  "max_peers": 50
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "discovery_id": "disc_xyz789",
    "peers_found": [
      {
        "peer_id": "peer_123abc",
        "transport": "bitchat",
        "signal_strength": 85,
        "device_type": "mobile",
        "capabilities": ["agent_communication", "file_sharing"]
      },
      {
        "peer_id": "peer_456def",
        "transport": "betanet",
        "latency_ms": 12,
        "device_type": "desktop",
        "capabilities": ["fog_compute", "model_training"]
      }
    ],
    "discovery_duration_ms": 2500,
    "total_peers": 2
  },
  "message": "Peer discovery completed",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Send P2P Message

**POST** `/v1/p2p/message`

Send a message through the P2P network.

**Request**:
```json
{
  "recipient": "peer_123abc",
  "message": {
    "type": "agent_communication",
    "payload": {
      "from_agent": "governance_king",
      "to_agent": "infrastructure_coordinator",
      "content": "Request system status update"
    }
  },
  "priority": "normal",
  "encryption": "standard",
  "delivery_receipt": true
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "message_id": "msg_p2p_789xyz",
    "recipient": "peer_123abc",
    "status": "sent",
    "transport_used": "bitchat",
    "route": ["node_current", "node_relay1", "peer_123abc"],
    "delivery_time_ms": 245,
    "receipt_expected": true
  },
  "message": "P2P message sent successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## Fog Computing API

### List Fog Nodes

**GET** `/v1/fog/nodes`

Get information about available fog computing nodes.

**Query Parameters**:
- `status`: Filter by node status (`active`, `busy`, `offline`, `all`)
- `tier`: Filter by privacy tier (`bronze`, `silver`, `gold`, `platinum`)
- `location`: Filter by geographic location
- `capabilities`: Filter by node capabilities

**Response**:
```json
{
  "success": true,
  "data": {
    "nodes": [
      {
        "node_id": "fog_node_123",
        "status": "active",
        "location": {
          "region": "us-east-1",
          "city": "New York",
          "coordinates": [40.7128, -74.0060]
        },
        "resources": {
          "cpu_cores": 16,
          "memory_gb": 64,
          "storage_gb": 1000,
          "gpu_count": 2,
          "gpu_type": "NVIDIA A100"
        },
        "pricing": {
          "bronze_tier": 0.50,
          "silver_tier": 0.75,
          "gold_tier": 1.00,
          "platinum_tier": 1.50
        },
        "capabilities": [
          "model_training",
          "inference",
          "data_processing",
          "secure_computation"
        ],
        "current_load": 0.25,
        "trust_score": 0.98
      }
    ],
    "total_nodes": 1,
    "active_nodes": 1,
    "summary": {
      "total_cpu_cores": 256,
      "total_memory_gb": 1024,
      "total_gpu_count": 32,
      "average_load": 0.35
    }
  },
  "message": "Fog nodes retrieved",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Submit Compute Job

**POST** `/v1/fog/jobs`

Submit a computation job to the fog network.

**Request**:
```json
{
  "job_name": "model_training_job",
  "job_type": "training",
  "requirements": {
    "cpu_cores": 8,
    "memory_gb": 32,
    "gpu_count": 1,
    "estimated_duration_hours": 2
  },
  "privacy_tier": "gold",
  "code": {
    "repository": "https://github.com/user/ml-training",
    "branch": "main",
    "entry_point": "train.py"
  },
  "data": {
    "input_data_url": "https://storage.aivillage.app/datasets/training_data.zip",
    "output_location": "user_storage/results/"
  },
  "environment": {
    "python_version": "3.11",
    "packages": ["torch", "transformers", "datasets"]
  },
  "max_cost": 5.00
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "job_id": "job_fog_abc123",
    "status": "queued",
    "assigned_node": "fog_node_123",
    "estimated_start_time": "2025-01-15T10:35:00Z",
    "estimated_completion_time": "2025-01-15T12:35:00Z",
    "cost_estimate": {
      "compute_cost": 2.00,
      "storage_cost": 0.50,
      "network_cost": 0.25,
      "total_cost": 2.75
    },
    "progress_url": "/v1/fog/jobs/job_fog_abc123/status"
  },
  "message": "Compute job submitted successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Job Status

**GET** `/v1/fog/jobs/{job_id}/status`

Get the status of a fog computing job.

**Response**:
```json
{
  "success": true,
  "data": {
    "job_id": "job_fog_abc123",
    "status": "running",
    "progress": {
      "percentage": 65,
      "current_stage": "model_training",
      "stages_completed": ["data_loading", "preprocessing"],
      "estimated_time_remaining": "35 minutes"
    },
    "resource_usage": {
      "cpu_utilization": 0.85,
      "memory_usage_gb": 28.5,
      "gpu_utilization": 0.92,
      "network_io_mbps": 45.2
    },
    "costs": {
      "current_cost": 1.45,
      "estimated_final_cost": 2.23
    },
    "logs_url": "/v1/fog/jobs/job_fog_abc123/logs"
  },
  "message": "Job status retrieved",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## RAG System API

### Query Knowledge Base

**POST** `/v1/query`

Query the RAG system for information.

**Request**:
```json
{
  "query": "What is the P2P network architecture?",
  "max_results": 10,
  "include_sources": true,
  "mode": "comprehensive",
  "context": {
    "user_level": "technical",
    "domain": "networking"
  },
  "filters": {
    "document_types": ["architecture", "technical_specs"],
    "confidence_threshold": 0.7
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "query": "What is the P2P network architecture?",
    "answer": "The P2P network architecture in AIVillage implements a sophisticated multi-transport system with BitChat (BLE mesh), BetaNet (HTX), and QUIC protocols. It features intelligent routing, mobile-first design, and automatic protocol selection based on device type and network conditions.",
    "sources": [
      {
        "document_id": "doc_p2p_arch",
        "title": "P2P Networking Architecture",
        "relevance_score": 0.95,
        "excerpt": "The Transport Manager is the central coordination system that handles protocol selection and routing...",
        "metadata": {
          "document_type": "architecture",
          "last_updated": "2025-01-15T09:00:00Z",
          "author": "Architecture Team"
        }
      }
    ],
    "confidence": 0.92,
    "processing_time_ms": 185,
    "knowledge_graph_paths": [
      {
        "path": ["P2P Network", "Transport Manager", "Protocol Selection"],
        "strength": 0.88
      }
    ]
  },
  "message": "Query processed successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Add Knowledge

**POST** `/v1/knowledge`

Add new knowledge to the RAG system.

**Request**:
```json
{
  "content": "The new quantum encryption module provides enhanced security for P2P communications...",
  "metadata": {
    "title": "Quantum Encryption Module",
    "document_type": "technical_specification",
    "category": "security",
    "author": "Security Team",
    "version": "1.0",
    "tags": ["encryption", "quantum", "security", "p2p"]
  },
  "chunk_strategy": "semantic",
  "generate_embeddings": true,
  "update_knowledge_graph": true
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "document_id": "doc_quantum_enc_123",
    "chunks_created": 5,
    "embeddings_generated": 5,
    "knowledge_graph_updates": {
      "nodes_added": 3,
      "relationships_added": 7
    },
    "processing_time_ms": 1250
  },
  "message": "Knowledge added successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## File Management API

### Upload File

**POST** `/v1/upload`

Upload and process a file for knowledge extraction.

**Request** (multipart/form-data):
```http
POST /v1/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>

--boundary
Content-Disposition: form-data; name="file"; filename="document.pdf"
Content-Type: application/pdf

[binary file content]
--boundary
Content-Disposition: form-data; name="metadata"
Content-Type: application/json

{
  "category": "research",
  "tags": ["ai", "machine learning"],
  "auto_extract": true,
  "privacy_level": "internal"
}
--boundary--
```

**Response**:
```json
{
  "success": true,
  "data": {
    "file_id": "file_abc123def456",
    "filename": "document.pdf",
    "size": 2456789,
    "content_type": "application/pdf",
    "status": "processed",
    "user_id": "user_123",
    "extraction_results": {
      "pages_processed": 25,
      "text_extracted": true,
      "images_extracted": 5,
      "tables_extracted": 3,
      "chunks_created": 45
    },
    "knowledge_integration": {
      "documents_created": 45,
      "embeddings_generated": 45,
      "knowledge_graph_updates": 12
    },
    "download_url": "/v1/files/file_abc123def456/download"
  },
  "message": "File uploaded and processed successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### List Files

**GET** `/v1/files`

List uploaded files for the authenticated user.

**Query Parameters**:
- `limit`: Number of files to return (default: 20, max: 100)
- `offset`: Number of files to skip (default: 0)
- `category`: Filter by category
- `content_type`: Filter by MIME type
- `status`: Filter by processing status

**Response**:
```json
{
  "success": true,
  "data": {
    "files": [
      {
        "file_id": "file_abc123def456",
        "filename": "document.pdf",
        "size": 2456789,
        "content_type": "application/pdf",
        "status": "processed",
        "category": "research",
        "uploaded_at": "2025-01-15T09:00:00Z",
        "processed_at": "2025-01-15T09:05:00Z",
        "metadata": {
          "pages": 25,
          "language": "en",
          "tags": ["ai", "machine learning"]
        }
      }
    ],
    "total_count": 1,
    "user_id": "user_123"
  },
  "message": "Files retrieved successfully",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## WebSocket API

### Connect to WebSocket

**WS** `/ws`

Establish a WebSocket connection for real-time updates.

**Connection Headers**:
```http
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: <key>
Sec-WebSocket-Version: 13
Authorization: Bearer <token>  // Optional but recommended
```

**Welcome Message**:
```json
{
  "type": "connection_established",
  "message": "Connected to AIVillage Unified API Gateway",
  "services": {
    "agent_forge": true,
    "p2p_fog": true
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "connection_id": "conn_abc123"
}
```

### WebSocket Message Types

#### Client to Server Messages

**Ping**:
```json
{
  "type": "ping",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Subscribe to Updates**:
```json
{
  "type": "subscribe",
  "topics": ["training_progress", "agent_messages", "p2p_events"],
  "filters": {
    "user_id": "user_123",
    "priority": "high"
  }
}
```

**Get Status**:
```json
{
  "type": "get_status",
  "components": ["agents", "p2p", "fog"]
}
```

#### Server to Client Messages

**Pong**:
```json
{
  "type": "pong",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Training Progress**:
```json
{
  "type": "training_progress",
  "task_id": "task_abc123def456",
  "progress": 0.75,
  "phase": "forge_training",
  "metrics": {
    "loss": 0.023,
    "accuracy": 0.91
  },
  "user_id": "user_123",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Agent Message**:
```json
{
  "type": "agent_message",
  "from_agent": "governance_king",
  "to_agent": "user_session",
  "message": {
    "content": "System status update: All services operational",
    "priority": "normal"
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**P2P Event**:
```json
{
  "type": "p2p_event",
  "event": "peer_connected",
  "peer_id": "peer_123abc",
  "transport": "bitchat",
  "metadata": {
    "device_type": "mobile",
    "capabilities": ["agent_communication"]
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "success": false,
  "error_code": "VALIDATION_ERROR",
  "message": "Invalid request parameters",
  "details": {
    "field": "model_id",
    "reason": "Model not found",
    "allowed_values": ["model_xyz789", "model_abc123"]
  },
  "timestamp": "2025-01-15T10:30:00Z",
  "request_id": "req_987654321",
  "documentation": "https://docs.aivillage.app/errors/VALIDATION_ERROR"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid authentication token |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource does not exist |
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `SERVICE_UNAVAILABLE` | 503 | Required service is not available |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `INSUFFICIENT_RESOURCES` | 503 | Not enough computational resources |
| `MODEL_TRAINING_FAILED` | 422 | Model training process failed |
| `P2P_NETWORK_ERROR` | 503 | P2P network connectivity issue |
| `AGENT_COMMUNICATION_FAILED` | 503 | Agent communication error |
| `FOG_COMPUTE_ERROR` | 503 | Fog computing service error |

## Rate Limiting

### Rate Limits by Tier

| Tier | Requests/Minute | Requests/Hour | Notes |
|------|-----------------|---------------|-------|
| **Standard** | 60 | 1,000 | Default tier |
| **Premium** | 200 | 5,000 | Paid subscription |
| **Enterprise** | 500 | 10,000 | Enterprise plan |

### Rate Limit Headers

All responses include rate limit information:

```http
X-RateLimit-Tier: premium
X-RateLimit-Limit: 200
X-RateLimit-Remaining: 195
X-RateLimit-Reset: 1640995200
X-RateLimit-Reset-After: 45
```

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded for tier premium",
  "details": {
    "limit": 200,
    "window": "1 minute",
    "reset_at": "2025-01-15T10:31:00Z",
    "suggested_retry_after": 45
  },
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## SDK Examples

### Python SDK

```python
import asyncio
from aivillage_sdk import AIVillageClient

# Initialize client
client = AIVillageClient(
    base_url="https://api.aivillage.app",
    api_key="your_api_key"
)

async def main():
    # Authenticate
    await client.authenticate()
    
    # Start model training
    training_job = await client.models.train(
        phase_name="cognate",
        parameters={
            "base_models": ["Qwen/Qwen2-1.5B-Instruct"],
            "target_architecture": "auto"
        },
        max_steps=1000
    )
    
    print(f"Training started: {training_job.task_id}")
    
    # Monitor progress via WebSocket
    async with client.websocket() as ws:
        await ws.subscribe(["training_progress"])
        
        async for message in ws:
            if message.type == "training_progress":
                print(f"Progress: {message.progress:.1%}")
                if message.progress >= 1.0:
                    break
    
    # Chat with trained model
    response = await client.chat(
        model_id=training_job.model_id,
        message="Hello, how are you?"
    )
    print(f"Model response: {response.message}")
    
    # Query knowledge base
    knowledge = await client.query(
        "What is the Agent Forge pipeline?",
        max_results=5
    )
    print(f"Knowledge: {knowledge.answer}")
    
    # Send message to agent
    agent_response = await client.agents.send_message(
        agent_id="governance_king",
        message={"content": "What is the system status?"}
    )
    print(f"Agent response: {agent_response.content}")

asyncio.run(main())
```

### JavaScript SDK

```javascript
const { AIVillageClient } = require('@aivillage/sdk');

// Initialize client
const client = new AIVillageClient({
  baseURL: 'https://api.aivillage.app',
  apiKey: 'your_api_key'
});

async function main() {
  // Authenticate
  await client.authenticate();
  
  // Start model training
  const trainingJob = await client.models.train({
    phaseName: 'cognate',
    parameters: {
      baseModels: ['Qwen/Qwen2-1.5B-Instruct'],
      targetArchitecture: 'auto'
    },
    maxSteps: 1000
  });
  
  console.log(`Training started: ${trainingJob.taskId}`);
  
  // Monitor progress
  const ws = await client.websocket();
  await ws.subscribe(['training_progress']);
  
  ws.on('training_progress', (message) => {
    console.log(`Progress: ${(message.progress * 100).toFixed(1)}%`);
    if (message.progress >= 1.0) {
      ws.close();
    }
  });
  
  // Query knowledge base
  const knowledge = await client.query(
    'What is the Agent Forge pipeline?',
    { maxResults: 5 }
  );
  console.log(`Knowledge: ${knowledge.answer}`);
}

main().catch(console.error);
```

### cURL Examples

**Authentication**:
```bash
curl -X POST https://api.aivillage.app/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "password123"
  }'
```

**Start Training**:
```bash
curl -X POST https://api.aivillage.app/v1/models/train \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "phase_name": "cognate",
    "parameters": {
      "base_models": ["Qwen/Qwen2-1.5B-Instruct"]
    },
    "max_steps": 1000
  }'
```

**Query Knowledge**:
```bash
curl -X POST https://api.aivillage.app/v1/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the P2P network architecture?",
    "max_results": 5
  }'
```

**Upload File**:
```bash
curl -X POST https://api.aivillage.app/v1/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.pdf" \
  -F 'metadata={"category":"research","tags":["ai"]}'
```

---

*Last Updated: January 2025*
*Version: 3.0.0*
*Status: Production Ready*

For more information, visit the [AIVillage Documentation Portal](https://docs.aivillage.app).