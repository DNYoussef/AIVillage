# AI Village Self-Improving System
[![API Docs](https://img.shields.io/badge/docs-latest-blue)](https://atlantisai.github.io/atlantis) [![Coverage](docs/assets/coverage.svg)](#)

AI Village is an experimental multi-agent platform that explores self-evolving architectures. Only a subset of the vision is implemented today.

> **Status**
> The following pieces are usable:
> - Retrieval-Augmented Generation (RAG) pipeline
> - FastAPI server with a simple web UI
> - Experimental model merging utilities
>
> Features such as Quiet-STaR thought generation and the full SAGE framework remain future work. The `SelfEvolvingSystem` class is a lightweight stub used for demos.

Refer to [docs/feature_matrix.md](docs/feature_matrix.md) for a status overview of all major components.

<!--feature-matrix-start-->
| Sub-system | Status |
|------------|--------|
| Twin Runtime | ✅ |
| King / Sage / Magi | ✅ |
| Self‑Evolving System | 🔴 |
| HippoRAG | 🔴 |
| Mesh Credits | ✅ |
| ADAS Optimisation | ✅ |
| ConfidenceEstimator | ✅ |
<!--feature-matrix-end-->

The [messaging protocol decision](docs/adr/0002-messaging-protocol.md) is documented in **ADR-0002**. gRPC/WebSocket support described there is not yet implemented.

The [server.py restriction to dev/test only](docs/adr/ADR-0010-monolith-test-harness-only.md) is documented in **ADR-0010**. Production services should use the gateway and twin microservices.

See [docs/roadmap.md](docs/roadmap.md) for upcoming milestones.

## Quick Start

```bash
# Clone and enter the repository
git clone https://github.com/yourusername/ai-village.git
cd ai-village

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the API server
python server.py
```

Open `http://localhost:8000/` to access the basic dashboard.

For advanced setup instructions and detailed usage examples see:
- [docs/advanced_setup.md](docs/advanced_setup.md)
- [docs/usage_examples.md](docs/usage_examples.md)
