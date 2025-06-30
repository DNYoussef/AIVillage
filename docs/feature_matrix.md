# Feature Matrix

| Feature | Status | Notes |
| ------- | ------ | ----- |
| Retrieval-Augmented Generation pipeline | Implemented | Basic retrieval and reasoning modules in `rag_system/` |
| FastAPI server | Implemented | `server.py` exposes simple query endpoint |
| Self-Evolving System | Placeholder | `SelfEvolvingSystem` stub; not integrated across agents |
| Twin Extraction | Planned | ADR-0001 describes design; no code yet |
| Mesh Networking / Federated Learning | Placeholder | skeleton modules in `communications/`, not fully functional |
| Expert Vectors training | Planned | documented in `docs/geometry_aware_training.md` but unimplemented |
| ADAS optimization | Prototype | basic optimizer in `agent_forge/adas` |
| Quiet-STaR module | Planned | not implemented; tests marked xfail |
| ADR-0002: Messaging Protocol | Proposed | gRPC with WebSocket fallback |
