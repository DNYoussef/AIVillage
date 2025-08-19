# MCP Integration - Protocol Implementation

## Overview

This document details the complete implementation of the Model Control Protocol (MCP) 2024-11-05 compliance within AIVillage, covering the protocol handlers, request/response formats, error handling, and security mechanisms that enable unified agent-system interaction.

## 🏗️ Protocol Architecture

### Core Protocol Components

The MCP implementation consists of three primary layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    PROTOCOL LAYER                           │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            MCPProtocolHandler                       │   │
│  │  • Request routing and validation                   │   │
│  │  • Response formatting and error handling           │   │
│  │  • Security enforcement and audit logging           │   │
│  │  • Performance monitoring and metrics               │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            MCPRequest / MCPResponse                 │   │
│  │  • JSON-RPC 2.0 message wrapping                   │   │
│  │  • Timestamp and UUID tracking                     │   │
│  │  • Parameter validation and type checking          │   │
│  │  • Metadata attachment and processing time         │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Error Handling System                    │   │
│  │  • MCPError hierarchy with specific error types    │   │
│  │  • Authentication and authorization errors         │   │
│  │  • Validation and resource not found errors        │   │
│  │  • Internal server error handling and logging      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### MCP Protocol Handler Implementation

**Location**: `packages/rag/mcp_servers/hyperag/protocol.py:115`

The `MCPProtocolHandler` class serves as the central coordinator for all MCP operations:

```python
class MCPProtocolHandler:
    """Handles MCP protocol requests for HypeRAG."""

    def __init__(
        self,
        permission_manager: PermissionManager,
        model_registry: ModelRegistry,
        storage_backend: Any | None = None,
    ) -> None:
        self.permission_manager = permission_manager
        self.model_registry = model_registry
        self.storage_backend = storage_backend
        self.model_registry_path = Path("data/model_registry.json")

        # Complete handler registry for all MCP operations
        self.handlers = {
            # Core operations
            "hyperag/query": self.handle_query,
            "hyperag/creative": self.handle_creative_query,
            "hyperag/repair": self.handle_repair,

            # Knowledge management
            "hyperag/knowledge/add": self.handle_add_knowledge,
            "hyperag/knowledge/search": self.handle_search_knowledge,
            "hyperag/knowledge/update": self.handle_update_knowledge,
            "hyperag/knowledge/delete": self.handle_delete_knowledge,

            # Adapter management
            "hyperag/adapter/upload": self.handle_upload_adapter,
            "hyperag/adapter/list": self.handle_list_adapters,
            "hyperag/adapter/activate": self.handle_activate_adapter,
            "hyperag/adapter/deactivate": self.handle_deactivate_adapter,

            # Guardian operations
            "hyperag/guardian/validate": self.handle_guardian_validate,
            "hyperag/guardian/override": self.handle_guardian_override,

            # System operations
            "hyperag/health": self.handle_health_check,
            "hyperag/metrics": self.handle_metrics,
            "hyperag/audit": self.handle_audit_log,

            # Model management
            "hyperag/model/register": self.handle_register_model,
            "hyperag/model/stats": self.handle_model_stats,
        }
```

**Key Features**:
- **Complete Handler Registry**: 16 different MCP operations supported
- **Security Integration**: Permission manager and authentication context
- **Model Registry**: Agent-specific model management and statistics
- **Storage Backend**: Pluggable storage for knowledge and memory
- **Performance Tracking**: Request timing and metadata collection

## 🔄 Request/Response Flow

### MCP Request Processing Pipeline

The complete request processing flow implements comprehensive validation, authentication, and error handling:

```python
async def handle_request(self, request: MCPRequest, context: AuthContext | None = None) -> MCPResponse:
    """Handle an MCP request."""
    try:
        # 1. Request Validation
        if not request.method:
            raise InvalidRequest("Missing method")

        if request.method not in self.handlers:
            raise NotFound(f"Unknown method: {request.method}")

        # 2. Handler Resolution
        handler = self.handlers[request.method]

        # 3. Authentication Check
        start_time = time.time()
        if context:
            result = await handler(context, **request.params)
        elif request.method in ["hyperag/health"]:  # Public endpoints
            result = await handler(**request.params)
        else:
            raise AuthenticationRequired

        # 4. Performance Metadata
        processing_time = time.time() - start_time
        if isinstance(result, dict):
            result["metadata"] = result.get("metadata", {})
            result["metadata"]["processing_time_ms"] = round(processing_time * 1000, 2)

        return MCPResponse(result=result, request_id=request.request_id)

    except MCPError as e:
        logger.warning(f"MCP error for {request.method}: {e.message}")
        return MCPResponse(error=e, request_id=request.request_id)

    except Exception as e:
        logger.exception(f"Unexpected error for {request.method}: {e!s}")
        return MCPResponse(
            error=InternalError(f"Unexpected error: {e!s}"),
            request_id=request.request_id,
        )
```

### Request/Response Data Structures

**MCPRequest Wrapper**:
```python
class MCPRequest:
    """MCP request wrapper."""

    def __init__(self, method: str, params: dict[str, Any], request_id: str | None = None) -> None:
        self.method = method
        self.params = params
        self.request_id = request_id or str(uuid.uuid4())
        self.timestamp = datetime.now()
```

**MCPResponse Wrapper**:
```python
class MCPResponse:
    """MCP response wrapper."""

    def __init__(
        self,
        result: Any = None,
        error: MCPError | None = None,
        request_id: str | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.request_id = request_id
        self.timestamp = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        response = {"jsonrpc": "2.0", "id": self.request_id}

        if self.error:
            response["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "data": self.error.data,
            }
        else:
            response["result"] = self.result

        return response
```

## 🚨 Error Handling System

### Comprehensive Error Hierarchy

The MCP implementation includes a complete error hierarchy for different failure scenarios:

```python
class MCPError(Exception):
    """Base class for MCP protocol errors."""

    def __init__(self, code: str, message: str, data: dict[str, Any] | None = None) -> None:
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(f"{code}: {message}")

class AuthenticationRequired(MCPError):
    """Authentication required error."""
    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__("AUTH_REQUIRED", message)

class PermissionDenied(MCPError):
    """Permission denied error."""
    def __init__(self, message: str = "Permission denied") -> None:
        super().__init__("PERMISSION_DENIED", message)

class InvalidRequest(MCPError):
    """Invalid request error."""
    def __init__(self, message: str = "Invalid request") -> None:
        super().__init__("INVALID_REQUEST", message)

class NotFound(MCPError):
    """Resource not found error."""
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__("NOT_FOUND", message)

class InternalError(MCPError):
    """Internal server error."""
    def __init__(self, message: str = "Internal server error") -> None:
        super().__init__("INTERNAL_ERROR", message)
```

### Error Response Format

All errors follow the JSON-RPC 2.0 error response format with AIVillage-specific extensions:

```json
{
  "jsonrpc": "2.0",
  "id": "request-uuid-here",
  "error": {
    "code": "PERMISSION_DENIED",
    "message": "Insufficient permissions for governance operations",
    "data": {
      "required_permission": "hyperag:governance:write",
      "user_permissions": ["hyperag:read", "hyperag:write"],
      "suggested_action": "Contact administrator for governance permissions"
    }
  }
}
```

## 🔧 Tool Implementation Examples

### Core Query Operation

**Location**: `packages/rag/mcp_servers/hyperag/protocol.py:204`

The `handle_query` method demonstrates the complete MCP tool implementation pattern:

```python
@require_permission(HypeRAGPermissions.READ)
@audit_operation("query")
async def handle_query(
    self,
    context: AuthContext,
    query: str,
    mode: str = "NORMAL",
    user_id: str | None = None,
    plan_hints: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Handle standard query request."""

    # 1. Context Setup
    if not user_id:
        user_id = context.user_id

    agent_type = context.role

    # 2. Query Planning
    plan_context = {"user_id": user_id, "agent_role": context.role}
    if plan_hints:
        plan_context.update(plan_hints)

    plan = await self.model_registry.process_with_model(
        context.agent_id, agent_type, "plan", query, plan_context
    )

    # 3. Knowledge Retrieval
    if not self.storage_backend:
        raise InternalError("Storage backend not configured")

    retrieval_limit = getattr(plan, "max_depth", 10)
    raw_results = await self.storage_backend.search_knowledge(query, limit=retrieval_limit)

    # 4. Node Construction
    retrieved_nodes = [
        Node(
            id=item["id"],
            content=item["content"],
            node_type=item.get("content_type", "text"),
            confidence=item.get("relevance", 1.0),
            metadata=item.get("metadata", {}),
        )
        for item in raw_results
    ]

    # 5. Knowledge Graph Construction
    knowledge_graph = await self.model_registry.process_with_model(
        context.agent_id, agent_type, "construct", retrieved_nodes, plan
    )

    # 6. Reasoning Process
    reasoning_result = await self.model_registry.process_with_model(
        context.agent_id, agent_type, "reason", knowledge_graph, query, plan
    )

    # 7. Response Format
    return {
        "request_id": plan.query_id,
        "status": "success",
        "mode_used": plan.mode.value,
        "result": {
            "answer": reasoning_result.answer,
            "confidence": reasoning_result.confidence,
            "reasoning_path": [asdict(step) for step in reasoning_result.reasoning_steps],
            "sources": [asdict(node) for node in reasoning_result.sources],
        },
        "guardian_decision": {
            "action": "APPLY",
            "semantic_score": 0.9,
            "utility_score": 0.85,
            "safety_score": 0.95,
        },
        "plan": asdict(plan),
    }
```

**Key Implementation Features**:
- **Permission Decorators**: `@require_permission` ensures proper authorization
- **Audit Logging**: `@audit_operation` creates compliance audit trail
- **Agent-Specific Processing**: Uses agent's registered model for reasoning
- **Comprehensive Response**: Includes reasoning path, sources, and metadata
- **Guardian Integration**: Safety and utility scoring for all responses

### Knowledge Management Operations

**Add Knowledge Tool**:
```python
@require_permission(HypeRAGPermissions.WRITE)
@audit_operation("add_knowledge")
async def handle_add_knowledge(
    self,
    context: AuthContext,
    content: str,
    content_type: str = "text",
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Handle add knowledge request."""
    node_id = str(uuid.uuid4())

    if not self.storage_backend:
        raise InternalError("Storage backend not configured")

    await self.storage_backend.add_knowledge(node_id, content, content_type, metadata)

    return {
        "node_id": node_id,
        "status": "success",
        "message": "Knowledge added successfully",
    }
```

**Search Knowledge Tool**:
```python
@require_permission(HypeRAGPermissions.READ)
@audit_operation("search_knowledge")
async def handle_search_knowledge(
    self,
    context: AuthContext,
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Handle search knowledge request."""
    if not self.storage_backend:
        raise InternalError("Storage backend not configured")

    results = await self.storage_backend.search_knowledge(query, limit, filters)

    return {
        "results": results,
        "total_count": len(results),
        "query": query,
    }
```

### Model Management Operations

**Model Registration**:
```python
@require_permission(HypeRAGPermissions.ADMIN)
@audit_operation("register_model")
async def handle_register_model(
    self,
    context: AuthContext,
    agent_id: str,
    model_config: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Handle model registration request."""

    # 1. Validation
    if not isinstance(model_config, dict):
        raise InvalidRequest("Model configuration must be a dictionary")

    required_fields = ["model_name", "model_type"]
    for field in required_fields:
        if field not in model_config:
            raise InvalidRequest(f"Missing required field: {field}")

    # 2. Load Existing Registry
    def _load_registry() -> dict[str, Any]:
        if self.model_registry_path.exists():
            with self.model_registry_path.open("r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    registry = await asyncio.to_thread(_load_registry)

    # 3. Prevent Duplicates
    if agent_id in registry:
        raise InvalidRequest(f"Model already registered for agent {agent_id}")

    # 4. Persist Configuration
    timestamp = datetime.now().isoformat()
    metadata = {"config": model_config, "registered_at": timestamp}
    registry[agent_id] = metadata

    def _save_registry(data: dict[str, Any]) -> None:
        self.model_registry_path.parent.mkdir(parents=True, exist_ok=True)
        with self.model_registry_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    await asyncio.to_thread(_save_registry, registry)

    return {
        "agent_id": agent_id,
        "status": "registered",
        "message": "Model registered successfully",
        "model_metadata": metadata,
    }
```

## 🔐 Security Implementation

### Permission-Based Authorization

Every MCP operation includes permission checking through decorators:

```python
@require_permission(HypeRAGPermissions.READ)
@audit_operation("operation_name")
async def handler_method(self, context: AuthContext, ...):
    """Handler method with security enforcement."""
```

### Authentication Context

All operations receive an `AuthContext` containing:

```python
@dataclass
class AuthContext:
    user_id: str
    agent_id: str
    session_id: str
    role: str
    permissions: set[str]
    expires_at: datetime
    ip_address: str
```

### Audit Logging

The `@audit_operation` decorator ensures all operations are logged for compliance:

```python
def audit_operation(operation_name: str):
    """Decorator to audit MCP operations."""
    def decorator(func):
        async def wrapper(self, context: AuthContext, *args, **kwargs):
            # Log operation start
            start_time = time.time()

            try:
                result = await func(self, context, *args, **kwargs)
                # Log successful operation
                await self.permission_manager.log_audit_event(
                    user_id=context.user_id,
                    operation=operation_name,
                    status="success",
                    duration_ms=round((time.time() - start_time) * 1000, 2)
                )
                return result
            except Exception as e:
                # Log failed operation
                await self.permission_manager.log_audit_event(
                    user_id=context.user_id,
                    operation=operation_name,
                    status="error",
                    error_message=str(e),
                    duration_ms=round((time.time() - start_time) * 1000, 2)
                )
                raise
        return wrapper
    return decorator
```

## 📊 Performance and Monitoring

### Request Timing

Every MCP request includes performance metadata:

```python
# Performance metadata added to all responses
processing_time = time.time() - start_time
if isinstance(result, dict):
    result["metadata"] = result.get("metadata", {})
    result["metadata"]["processing_time_ms"] = round(processing_time * 1000, 2)
```

### Health Check Implementation

**Location**: `packages/rag/mcp_servers/hyperag/protocol.py:556`

```python
async def handle_health_check(self, **kwargs) -> dict[str, Any]:
    """Handle health check request (no auth required)."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "permission_manager": "healthy",
            "model_registry": "healthy",
            "storage_backend": ("healthy" if self.storage_backend else "not_configured"),
        },
    }
```

### Metrics Collection

```python
@require_permission(HypeRAGPermissions.MONITOR)
@audit_operation("metrics")
async def handle_metrics(self, context: AuthContext, **kwargs) -> dict[str, Any]:
    """Handle metrics request."""
    model_stats = self.model_registry.get_model_stats()

    return {
        "timestamp": datetime.now().isoformat(),
        "models": model_stats,
        "active_sessions": len(await self.permission_manager.get_active_sessions()),
        "system_info": {"uptime_seconds": time.time() - getattr(self, "start_time", time.time())},
    }
```

## 🌐 MCP Server Implementation

### Standard MCP Server

**Location**: `packages/rag/mcp_servers/hyperag/mcp_server.py:29`

The `HypeRAGMCPServer` implements the standard MCP 2024-11-05 protocol:

```python
class HypeRAGMCPServer:
    """Standard MCP server for HypeRAG."""

    def __init__(self) -> None:
        self.permission_manager = None
        self.model_registry = None
        self.protocol_handler = None
        self.initialized = False
        # Local HyperRAG pipeline used for all queries and memory operations
        self.pipeline = HyperRAGPipeline()

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        # Handle standard MCP methods
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": True,
                        "resources": True,
                        "logging": True,
                    },
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "hyperag_query",
                            "description": "Query the HypeRAG knowledge graph",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Natural language query",
                                    },
                                    "context": {
                                        "type": "string",
                                        "description": "Additional context",
                                    },
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "hyperag_memory",
                            "description": "Store or retrieve memories",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "enum": ["store", "retrieve", "search"],
                                        "description": "Memory operation",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content to store or search for",
                                    },
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Optional tags",
                                    },
                                },
                                "required": ["action"],
                            },
                        },
                    ]
                },
            }
```

### Tool Execution

```python
if method == "tools/call":
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "hyperag_query":
        result = await self._handle_query(arguments, auth_context)
    elif tool_name == "hyperag_memory":
        result = await self._handle_memory(arguments, auth_context)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
    }
```

## 🔗 Client Integration

### MCP Client Implementation

**Location**: `packages/p2p/communications/mcp_client.py:13`

```python
class MCPClient:
    """JSON-RPC 2.0 over HTTPS with mTLS & JWT."""

    def __init__(self, endpoint: str, cert: str, key: str, ca: str) -> None:
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.verify = ca
        self.session.cert = (cert, key)

    def _make_token(self) -> str:
        return jwt.encode({"aud": "mcp"}, JWT_SECRET, algorithm="HS256")

    def call(self, method: str, params: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": str(uuid.uuid4()),
        }
        headers = {"Authorization": f"Bearer {self._make_token()}"}
        resp = self.session.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
```

**Key Features**:
- **JSON-RPC 2.0 Protocol**: Standard message format for MCP communication
- **HTTPS with mTLS**: Mutual TLS authentication for secure transport
- **JWT Authentication**: Bearer token authentication with HS256 signing
- **Request/Response**: Complete request/response cycle with proper error handling

## 🚀 Integration Examples

### Agent MCP Tool Usage

```python
# Agent makes MCP query
async def agent_query_example():
    mcp_client = MCPClient(
        endpoint="https://hyperag-mcp.aivillage.local",
        cert="agent.crt",
        key="agent.key",
        ca="ca.crt"
    )

    # Query knowledge
    response = mcp_client.call("hyperag_query", {
        "query": "What are the key principles of federated learning?",
        "context": "Research for agent training optimization",
        "mode": "comprehensive"
    })

    return response["result"]

# Store memory
async def agent_memory_example():
    response = mcp_client.call("hyperag_memory", {
        "action": "store",
        "content": "Learned that gradient compression improves federated learning efficiency",
        "tags": ["learning", "optimization", "federated"]
    })

    return response["result"]["item_id"]
```

### System Integration Workflow

```python
async def complete_mcp_workflow():
    """Complete MCP integration workflow."""

    # 1. Agent authentication
    auth_context = await authenticate_agent("sage_agent")

    # 2. Knowledge query through MCP
    query_result = await mcp_client.call("hyperag_query", {
        "query": "Current system performance metrics",
        "user_id": auth_context.user_id
    })

    # 3. Add knowledge based on findings
    if query_result["confidence"] < 0.7:
        await mcp_client.call("hyperag_knowledge_add", {
            "content": "System requires performance optimization",
            "content_type": "system_observation",
            "metadata": {"priority": "high", "source": "sage_agent"}
        })

    # 4. Register model updates
    await mcp_client.call("hyperag_model_register", {
        "agent_id": "sage_agent",
        "model_config": {
            "model_name": "sage_v2.1",
            "model_type": "reasoning",
            "capabilities": ["analysis", "research", "governance"]
        }
    })

    return {"status": "workflow_complete", "operations": 3}
```

---

This protocol implementation provides the foundation for secure, standardized, and efficient communication between all AIVillage agents and systems through the Model Control Protocol standard.
