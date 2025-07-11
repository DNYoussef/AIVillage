# AI Village Self-Improving System
[![API Docs](https://img.shields.io/badge/docs-latest-blue)](https://atlantisai.github.io/atlantis) [![Coverage](docs/assets/coverage.svg)](#)

AI Village implements a multi-agent system with a self-evolving architecture. The running code includes a Retrieval-Augmented Generation (RAG) pipeline, a FastAPI server and experimental model merging utilities.

> **Status**
> The following pieces of the project are usable today:
> - Retrieval-Augmented Generation (RAG) pipeline
> - FastAPI server with a simple web UI
> - Experimental model merging utilities
>
> All other phases &mdash; including Quiet-STaR thought generation, expert vectors and ADAS optimization &mdash; remain future work and are outlined for reference only in the documentation.  The `SelfEvolvingSystem` class present in the code is a lightweight stub used for demos and tests. The SAGE Framework remains a placeholder and is not yet implemented.

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
The [messaging protocol decision](docs/adr/0002-messaging-protocol.md) is documented in **ADR-0002**.

See [docs/roadmap.md](docs/roadmap.md) for a short overview of completed and planned milestones.

## Quick Start

```bash
# clone the repository
git clone https://github.com/yourusername/ai-village.git
cd ai-village

# setup Python 3.10+ environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# launch the server
python server.py
```

Detailed installation and environment configuration steps are provided in [docs/advanced_setup.md](docs/advanced_setup.md). Usage examples and pipeline walkthroughs live in [docs/usage_examples.md](docs/usage_examples.md). For an overview of the repository layout see [docs/system_overview.md](docs/system_overview.md).
