# AI Village Multi-Agent Platform
[![API Docs](https://img.shields.io/badge/docs-latest-blue)](https://atlantisai.github.io/atlantis) [![Coverage](docs/assets/coverage.svg)](#)

AI Village is an experimental multi-agent platform exploring autonomous agent architectures and advanced compression techniques.

> ⚠️ **Development Status**: This is an experimental prototype. Many documented features are planned but not yet implemented. See [Implementation Status](#implementation-status) for details.

> **Working Features** (Validated through testing):
> - **Compression Pipeline**: experimental prototype SeedLM, BitNet, and VPTQ compression (4/5 tests pass)
> - **Core Communication**: Robust message handling and protocol management (23/23 tests pass)
> - **RAG Pipeline**: Retrieval-Augmented Generation system
> - **FastAPI Server**: Development server with web UI (development only)
> - **Microservices**: Gateway and Twin services for production deployment
>
> **Planned Features**: planned planned self-evolving (not yet implemented) (not yet implemented) system, HippoRAG, full agent specialization, and Quiet-STaR integration remain future work. The `SelfEvolvingSystem` class is a stub implementation.

Refer to [docs/feature_matrix.md](docs/feature_matrix.md) for a status overview of all major components.

<!--feature-matrix-start-->
| Sub-system | Status | Test Results |
|------------|--------|--------------|
| **Compression Pipeline** | ✅ | **80% (4/5 tests pass)** |
| **Core Communication** | ✅ | **100% (23/23 tests pass)** |
| Twin Runtime | ✅ | Microservice functional |
| Gateway Service | ✅ | Microservice functional |
| King / Sage / Magi | 🟡 | Basic prototypes, limited functionality |
| Self‑Evolving System | 🔴 | **Import failures (stub confirmed)** |
| HippoRAG | 🔴 | Not implemented |
| Mesh Credits | 🟡 | Prototype only |
| ADAS Optimisation | 🟡 | Basic implementation |
| ConfidenceEstimator | 🟡 | Prototype only |
<!--feature-matrix-end-->

The [messaging protocol decision](docs/adr/0002-messaging-protocol.md) is documented in **ADR-0002**. gRPC/WebSocket support described there is not yet implemented.

The [server.py restriction to dev/test only](docs/adr/ADR-0010-monolith-test-harness-only.md) is documented in **ADR-0010**. Production services should use the gateway and twin microservices.

See [docs/roadmap.md](docs/roadmap.md) for upcoming milestones.

## Implementation Status

### ✅ experimental prototype Components
- **Compression Pipeline**: Comprehensive SeedLM, BitNet, and VPTQ implementations with CLI tools
- **Core Communication**: Message handling, protocol management, broadcasting, and history tracking
- **Microservices**: Gateway and Twin services for production deployment
- **Test Infrastructure**: 126 test files with professional pytest configuration (50% coverage threshold)

### 🟡 Prototype Components
- **King/Sage/Magi Agents**: Basic coordination and RAG functionality, limited specialization
- **RAG Pipeline**: Basic retrieval and reasoning modules
- **Model Training**: Agent Forge with ADAS optimization (prototype stage)

### 🔴 Planned Components
- **planned planned self-evolving (not yet implemented) (not yet implemented) System**: Currently a stub implementation for demos only
- **HippoRAG**: Not implemented, mentioned in documentation only
- **Quiet-STaR Integration**: Thought generation not integrated with agents
- **Full Agent Specialization**: Agents have minimal specialized capabilities

### 📊 Validation Methodology
This status is based on comprehensive testing of 126 test files, revealing a 42% trust score between documentation and implementation. Working components consistently pass tests, while problematic components show import failures and dependency issues.

## Quick Start

```bash
# Clone and enter the repository
git clone https://github.com/yourusername/ai-village.git
cd ai-village

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the development server (dev/test only)
python server.py
```

> ⚠️ **Important**: `server.py` is for development and testing only. For production deployments, use the Gateway and Twin microservices in `services/`. See [ADR-0010](docs/adr/ADR-0010-monolith-test-harness-only.md) for details.

Open `http://localhost:8000/` to access the basic dashboard.

For advanced setup instructions and detailed usage examples see:
- [docs/advanced_setup.md](docs/advanced_setup.md)
- [docs/usage_examples.md](docs/usage_examples.md)
