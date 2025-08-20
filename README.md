# AIVillage — Distributed AI Platform

[![CI Pipeline](https://github.com/DNYoussef/AIVillage/workflows/Main%20CI%2FCD%20Pipeline/badge.svg)](https://github.com/DNYoussef/AIVillage/actions)
[![Tests](https://img.shields.io/badge/tests-196%20passing%20%2F%20295%20total-yellow)](#-testing--quality)
[![Security Scan](https://img.shields.io/badge/security-hardened-green)](#-security)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A sophisticated multi-agent AI system with self‑evolution, distributed fog computing, advanced compression, and autonomous agent orchestration. AIVillage brings the AI to your data, coordinating specialized agents across phones, laptops, edge devices, and cloud nodes while preserving privacy.

---



## Overview

AIVillage shifts from centralized AI to a distributed, privacy‑preserving approach. Personal data never leaves your device by default; models run locally and collaborate across a resilient P2P + fog network when needed.

### Core Use Cases

* **Personal AI Assistant** (Digital Twin): On‑device (1–10 MB) models learn preferences via surprise‑based learning; local processing with differential privacy for any optional sharing.
* **Distributed Computing**: Idle devices contribute compute through a market‑driven fog network with SLAs.
* **Knowledge Management**: HyperRAG forms Bayesian trust networks with democratic governance (Sage/Curator/King quorum).
* **Edge AI Development**: Agent Forge 7‑phase pipeline distributes training across fog + federated participants.

---

## Key Features

### 🤖 Multi‑Agent Architecture (23 Specialized Agents)

**Leadership/Governance**: King (orchestrator), Auditor, Legal, Shield, Sword

**Infrastructure**: Coordinator, Gardener, Magi, Navigator, Sustainer

**Knowledge**: Curator, Oracle, Sage, Shaman, Strategist

**Culture/Economy**: Ensemble, Horticulturist, Maker, Banker‑Economist, Merchant

**Specialized Services**: Medic, Polyglot, Tutor

Unified capabilities across agents: HyperRAG read‑only memory via MCP servers, P2P messaging (BitChat/BetaNet), Quiet‑STaR reflection tokens, ADAS self‑modification, proprioception‑style resource awareness.

### 🛠️ Agent Forge — 7‑Phase Pipeline (Production‑Ready)

1. **EvoMerge** (linear/slerp, TIES/DARE, frankenmerge/DFS → 8 combos + NSGA‑II) **+ HRRM Bootstrap**
2. **Quiet‑STaR** (reasoning token baking; Grokfast acceleration)
3. **BitNet 1.58** (ternary pre‑compression)
4. **Forge Training** (edge‑of‑chaos, self‑modeling, dream cycles; Grokfast)
5. **Tool & Persona Baking** (identity + tools fused into weights)
6. **ADAS** (Transformers² vector composition architecture search)
7. **Final Compression** (SeedLM + VPTQ + hypercompression)

**HRRM Bootstrap System**: Three specialized ~50M models (Planner, Reasoner, Memory) provide 30× faster EvoMerge iteration with pre-optimized architectures and specialized capabilities.

**Distributed Training**: Federated (FedAvg) + fog offloading with battery/thermal‑aware scheduling and failure‑tolerant aggregation.

### 🌫️ Fog Computing (Production)

* **Gateway**: OpenAPI 3.1; Admin/Jobs/Sandboxes/Usage APIs; RBAC & quotas
* **Scheduler**: NSGA‑II Pareto optimization (cost/latency/reliability)
* **Marketplace**: Spot/on‑demand, trust‑based matching
* **Edge Capability Beacon**: mDNS discovery; live battery/thermal/network telemetry; dynamic pricing
* **WASI Runtime**: Secure sandboxing with CPU/mem/I/O quotas; cross‑platform
* **SLA Classes**: S (attested replication), A (replicated), B (best‑effort)

**Performance Benchmarks**

| Metric             | Achievement |    Typical |
| ------------------ | ----------: | ---------: |
| Scheduling Latency |    < 100 ms |      1–5 s |
| Price Discovery    |     < 50 ms | 200–500 ms |
| Edge Discovery     |      5–30 s |    1–5 min |
| Trade Success      |       ≥ 95% |     70–80% |
| Utilization        |      70–85% |     40–60% |

### 📚 HyperRAG & Digital Twins

* **Bayesian Trust Graphs** with calibrated confidences & trust propagation
* **Democratic Governance** (2/3 quorum among Sage/Curator/King)
* **Distributed Storage** with local caching & privacy zones
* **Digital Twin Concierge**: On‑device models, surprise‑based learning, automatic data deletion, DP noise on optional egress

### 🌐 P2P Communication

* **BitChat**: Offline BLE mesh (≤7 hops), 204‑byte chunking, battery‑aware routing
* **BetaNet**: Encrypted internet transport via HTX; QUIC bridge via unified Transport Manager
* **Transport Manager**: Intelligent routing/failover, data‑budget & thermal awareness

---

## 🚀 Quick Start

### Prerequisites

* Python 3.9+ (3.11 recommended)
* Git LFS
* Docker (optional)

### Install & Verify

```bash
git clone https://github.com/DNYoussef/AIVillage.git
cd AIVillage
make dev-install
make ci-pre-flight
```

### First Run

```bash
make serve
make test-fast
make format lint
```

---

## 🏗️ Architecture Overview

```
📱 Apps Layer          → Mobile apps, web interfaces, CLI tools
🌫️ Fog Layer          → Distributed computing, edge orchestration
🧠 Core Layer         → Agents, RAG, Agent Forge, tokenomics
🌐 Infrastructure      → P2P networking, edge mgmt, APIs, security
🛠️ DevOps Layer       → CI/CD, monitoring, deployment
```

### Deep Dive

* **Applications**: iOS/Android, Web, CLI
* **Fog Computing**: Resource abstraction, NSGA‑II scheduling, SLA orchestration
* **Core AI**: Agents, HyperRAG, Agent Forge, token systems
* **Infrastructure**: P2P transports, edge manager, API gateways, WASI runtime
* **DevOps**: CI/CD, artifact collection, monitoring, quality gates

### Integration Patterns

Event‑driven communication, resource abstraction (local/edge/fog/cloud), progressive enhancement (graceful offline), strict privacy boundaries.

### Technology Stack

Python 3.9+; PyTorch/Transformers/ONNX; WebRTC & BLE; WASI; Docker/Kubernetes; Helm; Prometheus.

---

## 🔒 Security Architecture

* **Crypto**: AES‑GCM (data), Ed25519 (signatures), X25519 (KEX)
* **Zero‑Trust**: AuthN/AuthZ on every boundary; RBAC; namespace isolation
* **Differential Privacy**: Calibrated noise on shared aggregates
* **Secure Enclaves**: Optional TEEs where available
* **Audit Trails**: Tamper‑evident logs; distributed retention

---

## ⚙️ Installation & Setup Guide

### Dev Environment

```bash
make dev-install
# 1) venv & deps  2) pre-commit  3) DB init  4) keygen  5) defaults  6) dev certs
```

### Config Management

```bash
cp .env.example .env
# Edit values for API keys, P2P bootstrap nodes, fog endpoints, budgets, GDPR mode, etc.
```

### Container & Kubernetes

```bash
make docker-build && docker-compose up -d
curl http://localhost:8000/health
# or Helm
helm install aivillage deploy/helm/aivillage \
  --set image.tag=latest --set environment=production --set replicaCount=3
```

---

## 📦 Usage & API Guide

### Fog Jobs (Python SDK)

```python
from fog_client import FogClient
client = FogClient(api_key="your-api-key")
result = await client.run_job_with_budget(job_spec, max_cost_usd=10.0, sla_class="A", timeout_seconds=300)
```

### Agent Collaboration

```python
village = AIVillage()
sage = village.get_agent("sage"); king = village.get_agent("king")
analysis = await sage.analyze({"context": "roadmap", "data": dataset})
final = await king.coordinate([analysis], {"decision_criteria": ["roi","risk"]})
```

### P2P Setup

```python
from aivillage.p2p import P2PNetwork
net = P2PNetwork(device_id="...", capabilities={"compute_power":"high"})
await net.initialize({"bitchat": {"enabled": True, "max_hops": 7}, "betanet": {"enabled": True}})
await net.join_network()
```

---

## 🔄 Automation & Development Workflow

### CI/CD (7 Stages)

1. Pre‑flight (<30s)  2) Code quality  3) Tests (3.9/3.11; multi‑OS)  4) Security (Bandit/Safety/Semgrep/secrets)  5) Performance  6) Build/Package  7) Deploy (staging→prod gate)

### Pre‑commit Hooks

Installed via `make dev-install`; run `pre-commit run --all-files`.

### Make Targets (selection)

```bash
make help
make format lint type-check security
make test test-unit test-integration test-fast
make ci-pre-flight ci-local ci
make docs
make docker-build deploy-staging deploy-production
```

---

## 🧪 Testing & Quality

**Structure**

```
tests/
├─ unit/  ├─ integration/  ├─ e2e/  ├─ validation/ (system, p2p, mobile, security)
├─ benchmarks/  ├─ security/  ├─ conftest.py  └─ pytest.ini
```

**Run**

```bash
pytest tests/ -v
pytest -m "unit|integration|validation|security|benchmark"
pytest tests/ --cov=packages --cov=src --cov-report=html
```

**Consolidation Outcomes**: \~350→\~270 test files; 23,662 duplicate lines removed; unified fixtures; markers; parallel‑ready.

---

## 🚀 Deployment & Operations

**Env Vars (example)**

```bash
export AIVILLAGE_ENVIRONMENT=production
export FOG_NETWORK_URL=https://fog.aivillage.com
export GDPR_COMPLIANCE_MODE=strict
export ENABLE_AUDIT_LOGGING=true
```

**Docker Compose (prod)**: Health checks, secrets, persistent volumes.

**Kubernetes (Helm)**: HPA, anti‑affinity, TLS ingress, resource requests/limits.

### Monitoring & Observability

* Health endpoints (`/health`, `/metrics` Prometheus)
* Perf metrics (P95/P99 latencies, utilization, SLA compliance)
* Distributed tracing; alerting rules

---

## 🤝 Contributing

1. Fork & clone → `make dev-install` → branch from `develop`
2. Follow code style (Black 120, Ruff, type hints, Google docstrings)
3. Ensure `make ci-local` passes; include tests & docs
4. Conventional commits; open PR for review

**Quality gates**: pre‑commit ✓, CI ✓, security ✓, coverage ✓, review ✓

---

## 🔒 Security

* Static & SAST (Bandit, Semgrep); dependency scanning (Safety)
* Secret detection; least‑privilege defaults; input validation; HTTPS everywhere

---

## 📚 Documentation

See **docs/** for Architecture, API, Development, Deployment, Guides. Generate API docs via `make docs`.

---

## 📜 License & Acknowledgments

MIT License — see [LICENSE](LICENSE).

Thanks to contributors, researchers, and open‑source communities whose work underpins AIVillage.

---

*Last Updated: August 19, 2025*
