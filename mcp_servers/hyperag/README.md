# HypeRAG MCP Server

A Model Context Protocol (MCP) server implementation for HypeRAG, providing brain-inspired dual-memory knowledge retrieval with creativity, repair, and safety validation capabilities.

## Features

- **Multi-Agent Support**: Role-based access control for King, Sage, Magi, and other agent types
- **Model Injection**: Agent-specific reasoning models with hot-swapping
- **Dual-Memory Architecture**: Hippo-Index (episodic) + Hypergraph-KG (semantic)
- **Permission System**: Granular RBAC with time-based and resource-based controls
- **Audit Trail**: Comprehensive logging of all operations
- **Guardian Gate**: Safety validation and policy enforcement
- **Real-time Communication**: WebSocket-based JSON-RPC protocol

## Quick Start

### Installation

```bash
cd mcp_servers/hyperag
pip install -r requirements.txt
```

### Configuration

Copy and customize the configuration file:

```bash
cp config/hyperag_mcp.yaml config/hyperag_mcp_local.yaml
# Edit config/hyperag_mcp_local.yaml with your settings
```

### Running the Server

```python
from mcp_servers.hyperag import HypeRAGMCPServer
import asyncio

async def main():
    server = HypeRAGMCPServer("config/hyperag_mcp_local.yaml")
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
```

Or run directly:

```bash
python -m mcp_servers.hyperag.server
```

## API Usage

### Authentication

The server supports JWT tokens and API keys:

```json
{
  "jsonrpc": "2.0",
  "method": "hyperag/query",
  "params": {
    "auth": "Bearer <jwt-token>",
    "query": "What are the connections between quantum computing and neural networks?"
  },
  "id": "request-1"
}
```

### Basic Query

```json
{
  "jsonrpc": "2.0",
  "method": "hyperag/query",
  "params": {
    "query": "Explain machine learning",
    "mode": "NORMAL",
    "plan_hints": {
      "max_depth": 3,
      "time_budget_ms": 2000
    }
  },
  "id": "query-123"
}
```

### Creative Query

```json
{
  "jsonrpc": "2.0",
  "method": "hyperag/creative",
  "params": {
    "source_concept": "blockchain",
    "target_concept": "biological_evolution",
    "creativity_parameters": {
      "mode": "analogical",
      "max_hops": 5,
      "min_surprise": 0.7
    }
  },
  "id": "creative-456"
}
```

### Knowledge Management

```json
{
  "jsonrpc": "2.0",
  "method": "hyperag/knowledge/add",
  "params": {
    "content": "Quantum computers use quantum bits (qubits) for computation.",
    "content_type": "text",
    "metadata": {
      "domain": "quantum_computing",
      "source": "textbook"
    }
  },
  "id": "add-789"
}
```

## Agent Model Integration

### Custom Agent Models

Implement the `AgentReasoningModel` interface for custom agent behavior:

```python
from mcp_servers.hyperag.models import AgentReasoningModel, QueryPlan, KnowledgeGraph, ReasoningResult

class CustomAgentModel(AgentReasoningModel):
    async def plan_query(self, query: str, context=None) -> QueryPlan:
        # Custom query planning logic
        return QueryPlan(
            query_id=str(uuid.uuid4()),
            mode=QueryMode.NORMAL,
            max_depth=4,
            confidence_threshold=0.8
        )

    async def construct_knowledge(self, retrieved, plan, context=None) -> KnowledgeGraph:
        # Custom knowledge graph construction
        kg = KnowledgeGraph(nodes={}, edges={})
        # ... implement construction logic
        return kg

    async def reason(self, knowledge, query, plan, context=None) -> ReasoningResult:
        # Custom reasoning logic
        return ReasoningResult(
            answer="Custom reasoning result",
            confidence=0.9,
            reasoning_steps=[],
            sources=[]
        )

# Register the model
model_registry.register_model_class("custom", CustomAgentModel)
```

## Permission System

### Roles and Permissions

| Role | Permissions | Description |
|------|-------------|-------------|
| `king` | Full access | Complete system control |
| `sage` | Read/Write/Graph | Strategic knowledge management |
| `magi` | Read/Code docs | Development-focused access |
| `watcher` | Read/Monitor | Observation only |
| `external` | Limited read | Public knowledge access |
| `guardian` | Gate control | Safety validation |
| `innovator` | Repair propose | Graph repair suggestions |

### Custom Permissions

Add custom permission rules in the configuration:

```yaml
permissions:
  resource_rules:
    custom_namespace:
      - rule: "custom:${user_id}/*"
        permission: "hyperag:write"
        roles: ["custom_role"]
```

## Storage Backends

### Supported Backends

- **Hippo-Index**: DuckDB (columnar), Redis (cache)
- **Hypergraph-KG**: Neo4j (graph), Qdrant (vectors)
- **Cache Layer**: Redis, In-memory
- **Audit Storage**: File, Database

### Configuration

```yaml
storage:
  hippo_index:
    backend: "duckdb"
    connection_string: "data/hippo_index.db"

  hypergraph_kg:
    backend: "neo4j"
    uri: "bolt://localhost:7687"

  vector_store:
    backend: "qdrant"
    host: "localhost"
    port: 6333
```

## Monitoring

### Health Check

```bash
curl -X POST http://localhost:8765 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"hyperag/health","id":"health"}'
```

### Metrics

```json
{
  "jsonrpc": "2.0",
  "method": "hyperag/metrics",
  "params": {
    "auth": "Bearer <token>"
  },
  "id": "metrics"
}
```

### Audit Log

```json
{
  "jsonrpc": "2.0",
  "method": "hyperag/audit",
  "params": {
    "auth": "Bearer <token>",
    "user_id": "optional",
    "limit": 100
  },
  "id": "audit"
}
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black mcp_servers/hyperag/
isort mcp_servers/hyperag/
mypy mcp_servers/hyperag/
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8765

CMD ["python", "-m", "mcp_servers.hyperag.server"]
```

## Security Considerations

1. **Change Default Secrets**: Update JWT secret and API keys
2. **Enable TLS**: Use HTTPS in production
3. **Network Security**: Restrict access with firewall rules
4. **Regular Updates**: Keep dependencies updated
5. **Audit Monitoring**: Review audit logs regularly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

See the main project LICENSE file for details.
