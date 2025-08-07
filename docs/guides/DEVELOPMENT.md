# AIVillage Development Guide

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- 4GB+ RAM (8GB recommended)
- CUDA-compatible GPU (optional but recommended)

### Development Environment Setup

1. **Clone and setup**:
   ```bash
   git clone https://github.com/DNYoussef/AIVillage.git
   cd AIVillage

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

   # Install development dependencies
   make dev-setup
   ```

2. **Verify installation**:
   ```bash
   # Run validation
   make validate-all

   # Check health
   make health-check
   ```

## Development Workflow

### 1. Code Quality Standards

We maintain high code quality through automated tools:

```bash
# Auto-fix formatting and style
make fix

# Run all quality checks
make lint

# Run security scans
make security
```

### 2. Testing Strategy

```bash
# Run all tests
make test

# Run specific test suites
make test-unit          # Fast unit tests
make test-integration   # Slower integration tests
make test-performance   # Benchmark tests
```

### 3. Pre-commit Hooks

Pre-commit hooks run automatically on each commit:

```bash
# Install hooks (done by make dev-setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

## Architecture Overview

### Core Components

```
AIVillage Architecture
├── Agent Layer (18 specialized agents)
├── Communication Layer (P2P mesh networking)
├── Resource Layer (device profiling & constraints)
├── AI/ML Layer (compression, evolution, RAG, federated learning)
└── Infrastructure Layer (monitoring, security, deployment)
```

### Key Design Principles

1. **Mobile-First**: Optimized for 2-4GB RAM devices
2. **Distributed**: No single point of failure
3. **Self-Evolving**: Agents improve over time
4. **Privacy-Preserving**: Federated learning with differential privacy
5. **Production-Ready**: Comprehensive testing and monitoring

## Adding New Features

### 1. Agent Types

To add a new agent type:

```python
# 1. Create template
# src/production/agent_forge/templates/new_agent_template.json
{
  "agent_id": "new_agent",
  "specification": {
    "name": "NewAgent",
    "description": "Description of capabilities",
    "primary_capabilities": ["capability1", "capability2"],
    "resource_requirements": {
      "cpu": "medium",
      "memory": "low",
      "network": "high"
    }
  }
}

# 2. Update master config
# Add to src/production/agent_forge/templates/master_config.json

# 3. Create agent implementation
# src/production/agent_forge/agents/new_agent.py

# 4. Add tests
# tests/unit/agent_forge/test_new_agent.py
```

### 2. P2P Protocols

```python
# 1. Define message type
class NewProtocolMessage:
    def __init__(self, data: dict):
        self.type = "NEW_PROTOCOL"
        self.data = data

# 2. Register handler
node.register_handler("NEW_PROTOCOL", handle_new_protocol)

# 3. Implement handler
async def handle_new_protocol(message: dict, writer):
    # Process message
    result = process_protocol_data(message)

    # Send response
    await node.send_response(writer, result)
```

### 3. Compression Algorithms

```python
# 1. Create algorithm implementation
class NewCompressionAlgorithm(CompressionAlgorithm):
    def compress(self, model: torch.nn.Module) -> torch.nn.Module:
        # Implement compression logic
        return compressed_model

    def get_compression_ratio(self) -> float:
        return self.compression_ratio

# 2. Register with pipeline
from production.compression import register_algorithm
register_algorithm("new_algorithm", NewCompressionAlgorithm)

# 3. Add configuration support
# Update CompressionConfig in compression/config.py
```

## Performance Guidelines

### Memory Management

- Target 2-4GB RAM devices
- Use lazy loading for large models
- Implement memory monitoring and cleanup
- Profile memory usage regularly

### Network Efficiency

- Compress messages before transmission
- Implement batching for small messages
- Use connection pooling
- Handle network failures gracefully

### CPU Optimization

- Use async/await for I/O operations
- Implement CPU-intensive tasks as background jobs
- Consider multiprocessing for parallel work
- Profile CPU usage and optimize hotspots

## Debugging

### Logging

```python
import logging
logger = logging.getLogger(__name__)

# Use structured logging
logger.info("Agent deployed", extra={
    "agent_id": agent_id,
    "device_id": device_id,
    "memory_usage": memory_mb
})
```

### Monitoring

```python
# Add metrics
from monitoring.metrics import Metrics
metrics = Metrics()

with metrics.timer("operation_duration"):
    result = perform_operation()

metrics.increment("operation_count")
metrics.gauge("memory_usage", get_memory_usage())
```

### Common Issues

1. **Import Errors**: Check PYTHONPATH and __init__.py files
2. **Memory Leaks**: Use memory profilers and implement cleanup
3. **Network Timeouts**: Implement proper timeout handling
4. **Resource Constraints**: Monitor device resources and adapt

## Testing

### Unit Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestAgentDeployment:
    @pytest.fixture
    def orchestrator(self):
        return ForgeOrchestrator()

    @patch('agent_forge.orchestrator.DeviceProfiler')
    async def test_deploy_agent_success(self, mock_profiler, orchestrator):
        # Setup mocks
        mock_profiler.return_value.get_available_memory.return_value = 4096

        # Test deployment
        agent = await orchestrator.deploy_agent("king")

        # Assertions
        assert agent.agent_type == "king"
        assert agent.status == "running"
```

### Integration Tests

```python
@pytest.mark.integration
class TestP2PNetworking:
    async def test_multi_node_communication(self):
        # Setup multiple nodes
        nodes = [P2PNode() for _ in range(3)]

        # Start nodes
        for i, node in enumerate(nodes):
            await node.start_server(port=9000 + i)

        # Connect nodes
        await nodes[0].connect_to_peer("localhost", 9001)
        await nodes[1].connect_to_peer("localhost", 9002)

        # Test message passing
        message = {"type": "TEST", "data": "hello"}
        await nodes[0].broadcast_message(message)

        # Verify delivery
        received = await nodes[2].receive_message(timeout=5.0)
        assert received["data"] == "hello"
```

### Performance Tests

```python
@pytest.mark.benchmark
def test_compression_performance(benchmark):
    """Test compression speed meets requirements."""
    model = create_test_model(size="1GB")

    result = benchmark(compress_model, model)

    # Verify performance requirements
    assert result.compression_ratio >= 3.8
    assert result.compression_time < 300  # 5 minutes max
```

## Documentation

### Code Documentation

```python
class AgentDeployer:
    """Deploys agents across distributed devices.

    This class handles the deployment of AI agents to appropriate devices
    based on resource constraints and performance requirements.

    Attributes:
        device_manager: Manages device discovery and capabilities
        resource_monitor: Monitors resource usage across devices

    Example:
        ```python
        deployer = AgentDeployer()
        agent = await deployer.deploy_agent("king", constraints={
            "memory_mb": 2048,
            "cpu_cores": 4
        })
        ```
    """

    async def deploy_agent(
        self,
        agent_type: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> DeployedAgent:
        """Deploy an agent with resource constraints.

        Args:
            agent_type: Type of agent to deploy (e.g., "king", "sage")
            constraints: Resource constraints for deployment

        Returns:
            DeployedAgent instance with deployment information

        Raises:
            InsufficientResourcesError: If no device meets constraints
            AgentDeploymentError: If deployment fails

        Note:
            This method automatically selects the best available device
            based on current resource availability and agent requirements.
        """
```

### API Documentation

Use the docs/api/ structure to document public APIs with examples.

## Contributing

### Pull Request Process

1. Create feature branch from `develop`
2. Implement changes with tests
3. Run full validation: `make validate-all`
4. Submit PR with clear description
5. Address review feedback
6. Merge after approval

### Code Review Checklist

- [ ] Tests cover new functionality
- [ ] Documentation updated
- [ ] Performance impact considered
- [ ] Security implications reviewed
- [ ] Mobile compatibility maintained
- [ ] Error handling implemented
- [ ] Logging added appropriately

## Resources

- [Architecture Overview](../architecture/README.md)
- [Deployment Guide](../deployment/README.md)
- [API Reference](../api/README.md)
- [Troubleshooting](TROUBLESHOOTING.md)

## Help

- Create GitHub issues for bugs/features
- Use Discussions for questions
- Check existing documentation first
- Include minimal reproduction examples
