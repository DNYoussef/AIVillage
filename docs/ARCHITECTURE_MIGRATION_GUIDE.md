# Architecture Migration Guide

## Overview

This guide documents the reorganization of AIVillage codebase according to the architectural blueprint defined in `docs/architecture/ARCHITECTURE.md`.

## New Folder Structure

The codebase has been reorganized to match the architectural layers:

```
AIVillage/
├── gateway/                    # 🚪 Gateway Layer (FastAPI Entry Point)
│   ├── auth/                   # Authentication, JWT, OAuth2
│   ├── routing/                # Request routing
│   ├── rate_limiting/          # Rate limiting and quotas
│   └── ui/                     # User interface (moved from packages/ui)
├── fog/                        # 🌫️ Fog Computing Infrastructure
│   ├── gateway/                # Fog gateway APIs
│   ├── scheduler/              # NSGA-II Scheduler
│   ├── marketplace/            # Marketplace engine
│   └── edge/                   # Edge capability beacon
├── twin/                       # 🔄 Twin Server (Digital Twin Engine)
│   ├── engine/                 # Personal AI models
│   ├── privacy/                # Privacy-first components
│   └── learning/               # Learning engine
├── mcp/                        # 🛠️ MCP Layer (Model Control Protocol)
│   ├── tools/                  # Agent tools
│   ├── memory/                 # Memory servers
│   └── servers/                # RAG servers
├── p2p/                        # 🌐 P2P Communication Layer
│   ├── bitchat/                # BitChat (BLE)
│   ├── betanet/                # BetaNet (HTTP)
│   └── mesh/                   # Mesh routing
├── rag/                        # 📚 RAG System
├── agents/                     # 🤖 Agent Layer
├── data/                       # 💾 Data Stores
├── shared/                     # Common utilities and shared code
│   ├── auth/                   # Shared authentication
│   ├── config/                 # Configuration management
│   ├── logging/                # Logging utilities
│   └── utils/                  # Utility functions
└── tools/                      # Development and operational tools
```

## Migration Summary

### Moved Components

| Old Location | New Location | Purpose |
|-------------|-------------|---------|
| `packages/core/bin/` | `gateway/` | FastAPI entry points |
| `packages/ui/` | `gateway/ui/` | User interface components |
| `packages/core/legacy/` | `twin/` | Digital twin engine |
| `packages/core/common/` | `shared/` | Shared utilities |
| `packages/core/tools/` | `tools/` | Development tools |

### Maintained Locations

- `fog/` - Already correctly positioned
- `rag/` - Already correctly positioned
- `agents/` - Already correctly positioned
- `data/` - Already correctly positioned

## Compatibility Layer

A compatibility layer has been added in `packages/__init__.py` to maintain backward compatibility with existing imports. This allows existing code to continue working while the migration is completed.

## Import Updates

### Old Imports
```python
from packages.core.bin import server
from packages.ui.App import App
from packages.core.legacy.chat_engine import ChatEngine
from packages.core.common.flags import FeatureFlag
```

### New Imports
```python
from gateway import server
from gateway.ui.App import App
from twin.chat_engine import ChatEngine
from shared.flags import FeatureFlag
```

## Migration Benefits

1. **Clear Architectural Boundaries**: Each layer has distinct responsibilities
2. **Improved Maintainability**: Code organization matches system architecture
3. **Better Scalability**: Architectural layers can scale independently
4. **Enhanced Security**: Clear separation of concerns improves security boundaries
5. **Simplified Navigation**: Developers can easily find code by architectural layer

## Next Steps

1. Update import statements in existing code gradually
2. Remove compatibility layer once migration is complete
3. Add architectural fitness functions to prevent regression
4. Update documentation to reflect new structure

## Architecture Alignment

This reorganization aligns with the AIVillage architecture:

- **Gateway Layer**: Entry point for all requests with authentication and routing
- **Fog Computing**: Distributed computing infrastructure
- **Twin Server**: Personal AI models and privacy-first components
- **MCP Layer**: Model Control Protocol for agent communication
- **P2P Layer**: Peer-to-peer communication and mesh networking
- **RAG System**: Retrieval-augmented generation
- **Agent Layer**: Specialized AI agents with democratic governance
- **Data Stores**: Persistent storage layers

For detailed architectural information, see `docs/architecture/ARCHITECTURE.md`.
