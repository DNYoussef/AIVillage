# AIVillage API Documentation

## Overview

This directory contains comprehensive API documentation for all AIVillage components.

## Core APIs

### 🤖 Agent System
- [`agent_forge/`](agent_forge/) - Agent creation and management
- [`communications/`](communications/) - Inter-agent messaging protocol
- [`distributed_agents/`](distributed_agents/) - Distributed agent deployment

### 🔗 Infrastructure
- [`core/p2p/`](core/p2p/) - Peer-to-peer networking
- [`core/resources/`](core/resources/) - Resource management and profiling
- [`infrastructure/`](infrastructure/) - Mesh networking and device coordination

### 🧠 AI/ML Systems  
- [`production/compression/`](production/compression/) - Model compression pipeline
- [`production/evolution/`](production/evolution/) - Agent evolution system
- [`production/rag/`](production/rag/) - Retrieval-augmented generation
- [`production/federated_learning/`](production/federated_learning/) - Federated learning coordination

### 🔧 Utilities
- [`mcp_servers/`](mcp_servers/) - Model Context Protocol servers
- [`twin_runtime/`](twin_runtime/) - Digital twin execution
- [`monitoring/`](monitoring/) - System monitoring and analytics

## Quick Navigation

| Component | Description | Status |
|-----------|-------------|--------|
| [Agent Forge](agent_forge/README.md) | Agent template system and orchestration | ✅ Complete |
| [P2P Network](core/p2p/README.md) | Distributed communication layer | ✅ Complete |
| [Compression](production/compression/README.md) | Model compression (4x ratio) | ✅ Complete |
| [Evolution](production/evolution/README.md) | Agent self-improvement | ✅ Complete |
| [RAG System](production/rag/README.md) | Retrieval-augmented generation pipeline | 🟡 Partial |
| [Federated Learning](production/federated_learning/README.md) | Privacy-preserving training | ✅ Complete |
| [Resource Management](core/resources/README.md) | Device profiling and constraints | ✅ Complete |

## Usage Examples

### Creating an Agent
```python
from agent_forge.orchestrator import ForgeOrchestrator
from agent_forge.templates import AgentTemplate

# Load agent template
template = AgentTemplate.load("king")

# Create orchestrator
orchestrator = ForgeOrchestrator()

# Deploy agent
agent = await orchestrator.deploy_agent(template, device_constraints={
    "memory_mb": 2048,
    "cpu_cores": 4
})
```

### Starting P2P Network
```python
from core.p2p.p2p_node import P2PNode
from core.resources.device_profiler import DeviceProfiler

# Initialize device profiler
profiler = DeviceProfiler()
await profiler.start_monitoring()

# Create P2P node
node = P2PNode(device_profiler=profiler)
await node.start_server(port=9000)

# Connect to peer
await node.connect_to_peer("192.168.1.100", 9000)
```

### Using Compression Pipeline
```python
from production.compression import CompressionPipeline
from production.compression.config import CompressionConfig

# Configure compression
config = CompressionConfig(
    input_model_path="./models/base_model",
    output_model_path="./models/compressed_model",
    bitnet_zero_threshold=0.02,
    target_compression_ratio=4.0
)

# Run compression
pipeline = CompressionPipeline(config)
result = await pipeline.compress_model()

print(f"Compression ratio: {result.compression_ratio}x")
```

### Querying the RAG System
```bash
python -m src.production.rag.rag_system.main query --question "What is RAG?"
```

## Development

### Adding New APIs

1. Create module documentation in appropriate directory
2. Include code examples and usage patterns
3. Document all public classes and methods
4. Add integration examples

### Documentation Standards

- Use Google-style docstrings
- Include type hints
- Provide working code examples
- Document error cases and exceptions
- Include performance characteristics

## Status Legend

- ✅ **Complete**: Production-ready with full documentation
- 🟡 **Partial**: Working but needs documentation improvements  
- 🔴 **Missing**: Planned but not yet implemented
- 📝 **Draft**: Documentation in progress
