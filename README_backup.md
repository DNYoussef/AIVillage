# AIVillage: Experimental AI Infrastructure

[![API Docs](https://img.shields.io/badge/docs-latest-blue)](docs/) [![Coverage](docs/assets/coverage.svg)](#) [![Quality Gates](https://img.shields.io/badge/quality-gates-passing-green)](#quality-gates)

> **Development Status**: Active development with production-ready compression, evolution, and RAG components alongside experimental agent and mesh networking systems.

AIVillage is an experimental AI platform providing components for model compression, evolution, RAG, and agent specialization. The codebase separates production-ready components from experimental development work.

## ⚠️ Development Server Notice

**`server.py` is DEVELOPMENT ONLY** - Use Gateway/Twin services for production:
- `server.py`: Development and testing only (not production-ready)
- Production: Use `experimental/services/gateway.py` and `experimental/services/twin.py`
- See `docs/architecture.md` for production deployment guidance