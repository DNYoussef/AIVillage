# System Architecture

The AIVillage system is composed of modular services and libraries that communicate over a mesh-enabled messaging layer.

## Core Components
- **Core Services (`src/core/`)** – provides chat engine, message protocols, and P2P node implementation that underpins all communication.
- **Communications (`src/communications/`)** – manages credits ledger and community interactions; exposed via the mesh network.
- **Agent Forge (`src/agent_forge/`)** – orchestrates model training, evaluation, and evolution. Integrates with the RAG system for knowledge retrieval.
- **RAG System (`src/rag_system/` and `src/agent_forge/rag_integration.py`)** – supplies retrieval-augmented context to agents; currently a stub pending full implementation.
- **Digital Twin (`src/digital_twin/`)** – simulates personalized tutoring environments and connects to monitoring for performance metrics.
- **Monitoring (`src/monitoring/`)** – Prometheus/Grafana metrics, alert manager, and dashboards.
- **MCP Servers (`src/mcp_servers/`)** – expose model context protocol endpoints such as the HyperRAG server.
- **Deployment (`deploy/`, `docker-compose.yml`)** – container definitions and orchestration for running the system.

## Integration Points
- The mesh network manager and `core.p2p` layer enable distributed communication between services.
- Agent Forge consumes RAG results via the RAG integration module.
- Credits and community hub use the communications module to track resource usage across the mesh.
- Digital Twin workloads report metrics to the monitoring stack for analysis.

This document reflects the current codebase and will evolve as components mature.
