# Feature Matrix

| Feature | Status | Notes |
| ------- | ------ | ----- |
| Retrieval-Augmented Generation pipeline | Implemented | Basic retrieval and reasoning modules in `rag_system/` |
| FastAPI server | Implemented | `server.py` exposes simple query endpoint |
| Self-Evolving System | Placeholder | `SelfEvolvingSystem` stub; not integrated across agents |
| Twin Runtime | ✅ v0.2.0 | Extracted microservice in `services/twin` |
| Twin Extraction | Prototype | Endpoints at `/v1/chat` & `/v1/user/{id}`; LRU eviction documented in ADR-0001 |
| Gateway service | ✅ v0.2.0 | Prometheus metrics and rate limiting |
| Mesh Networking / Federated Learning | Placeholder | skeleton modules in `communications/`, not fully functional |
| Expert Vectors training | Planned | documented in `docs/geometry_aware_training.md` but unimplemented |
| ADAS optimization | Prototype | basic optimizer in `agent_forge/adas` |
| Quiet-STaR module | Planned | not implemented; tests marked xfail |
| ADR-0002: Messaging Protocol | Proposed | gRPC with WebSocket fallback |
