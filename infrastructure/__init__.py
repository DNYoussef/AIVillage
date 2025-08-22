"""
AIVillage Infrastructure Layer - Technical Implementation

This layer contains all technical infrastructure components:
- gateway: FastAPI entry point with auth, routing, rate limiting
- twin: Digital Twin Engine for personal AI models
- mcp: Model Control Protocol for agent communication
- p2p: P2P communication (BitChat, BetaNet, mesh)
- shared: Shared utilities and common code
- fog: Fog computing infrastructure
- data: Data persistence (PostgreSQL, Neo4j, Redis, Vector DB)
- messaging: Event-driven architecture components

Infrastructure components should never depend on the core business logic.
"""
