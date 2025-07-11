# Architecture

The repository started as a monolithic Python project. It now includes two
microservices – a Gateway and the Twin runtime – which communicate over HTTP.
gRPC/WebSocket support proposed in ADR-0002 is not yet implemented.
Most other modules remain in-process libraries. Quiet-STaR and expert vectors
remain stubs, while a basic ADAS prototype is available.

<!--feature-matrix-start-->
| Sub-system | Status |
|------------|--------|
| Twin Runtime | ✅ v0.2.0 |
| King / Sage / Magi | ✅ |
| Self‑Evolving System | 🔴 |
| HippoRAG | 🔴 |
| Mesh Credits | ✅ |
| ADAS Optimisation | ✅ |
| ConfidenceEstimator | ✅ |
<!--feature-matrix-end-->
