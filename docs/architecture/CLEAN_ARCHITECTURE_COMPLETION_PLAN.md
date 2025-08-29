# AIVillage Clean Architecture Completion Plan

## Current State Analysis

✅ **Infrastructure Layer (60% Complete)**
- `gateway/` - API gateway with admin, routing, scheduling
- `twin/` - Digital twin system with security, compression
- `mcp/` - MCP protocol implementation (basic structure)
- `p2p/` - P2P networking (basic structure)
- `shared/` - Common utilities, config, auth

🔄 **Missing Components**
- Apps layer for UI consolidation
- Core business logic separation
- Complete infrastructure services
- DevOps automation structure
- Shared libraries organization
- External integrations layer

## Clean Architecture Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIVillage Clean Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│  📱 apps/                    │  🧠 core/                        │
│  ├── mobile/                 │  ├── agents/                     │
│  │   ├── ios/               │  │   ├── governance/              │
│  │   ├── android/           │  │   ├── knowledge/               │
│  │   └── shared/            │  │   ├── infrastructure/          │
│  ├── web/                   │  │   └── specialized/             │
│  ├── cli/                   │  ├── agent-forge/                │
│  └── desktop/               │  │   ├── training/                │
│                             │  │   ├── evolution/               │
│  🔗 infrastructure/          │  │   └── compression/             │
│  ├── gateway/ ✅            │  ├── rag/                         │
│  ├── twin/ ✅               │  │   ├── ingestion/               │
│  ├── mcp/ ✅                │  │   ├── retrieval/               │
│  ├── p2p/ ✅                │  │   └── generation/              │
│  ├── shared/ ✅             │  └── domain/                      │
│  ├── fog/ (new)             │      ├── tokenomics/             │
│  ├── data/ (new)            │      └── governance/             │
│  └── messaging/ (new)       │                                  │
├─────────────────────────────────────────────────────────────────┤
│  ⚙️ devops/                  │  📚 libs/                        │
│  ├── ci-cd/                 │  ├── crypto/                     │
│  ├── deployment/            │  ├── networking/                 │
│  ├── monitoring/            │  ├── ml-utils/                   │
│  └── automation/            │  └── common/                     │
│                             │                                  │
│  🔌 integrations/           │  📋 config/                      │
│  ├── external-apis/         │  ├── environments/               │
│  ├── blockchain/            │  └── services/                   │
│  └── cloud-services/        │                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Definitions and Responsibilities

### 1. Apps Layer (`apps/`)
**Purpose**: User interfaces and client applications
**Dependencies**: Can use `core/`, `infrastructure/`, `libs/`
**Cannot depend on**: Other apps

```
apps/
├── mobile/                     # Mobile applications
│   ├── ios/                   # iOS native app
│   │   ├── AIVillage/         # Main iOS project
│   │   ├── Shared/            # Shared iOS code
│   │   └── Tests/             # iOS tests
│   ├── android/               # Android native app
│   │   ├── app/               # Main Android module
│   │   ├── shared/            # Shared Android code
│   │   └── tests/             # Android tests
│   └── shared/                # Cross-platform mobile code
│       ├── components/        # Shared UI components
│       ├── services/          # Mobile services
│       ├── utils/             # Mobile utilities
│       └── types/             # TypeScript types
├── web/                       # Web applications
│   ├── admin-dashboard/       # Admin interface
│   ├── user-portal/           # User interface
│   ├── developer-console/     # Developer tools
│   └── shared/                # Shared web components
├── cli/                       # Command-line interfaces
│   ├── admin-cli/             # Administrative CLI
│   ├── developer-cli/         # Developer tools CLI
│   └── user-cli/              # User interaction CLI
└── desktop/                   # Desktop applications
    ├── electron/              # Electron app
    └── native/                # Native desktop apps
```

### 2. Core Layer (`core/`)
**Purpose**: Business logic and domain models
**Dependencies**: Only `libs/` and external libraries
**Cannot depend on**: `apps/`, `infrastructure/`, `devops/`

```
core/
├── agents/                    # Agent system business logic
│   ├── governance/            # King, legal, auditor agents
│   │   ├── king/             # King agent logic
│   │   ├── legal/            # Legal compliance
│   │   └── auditor/          # Audit functions
│   ├── knowledge/             # Knowledge management agents
│   │   ├── sage/             # Sage agent logic
│   │   ├── oracle/           # Oracle predictions
│   │   └── curator/          # Content curation
│   ├── infrastructure/        # Infrastructure agents
│   │   ├── magi/             # Magi agent logic
│   │   ├── navigator/        # Network navigation
│   │   └── sustainer/        # System maintenance
│   ├── specialized/           # Domain-specific agents
│   │   ├── creative/         # Creative tasks
│   │   ├── financial/        # Financial operations
│   │   └── social/           # Social interactions
│   └── contracts/             # Agent interface contracts
├── agent-forge/               # Agent training and evolution
│   ├── training/              # Training pipelines
│   │   ├── curriculum/       # Curriculum learning
│   │   ├── pipelines/        # Training workflows
│   │   └── validation/       # Model validation
│   ├── evolution/             # Evolutionary algorithms
│   │   ├── mutation/         # Model mutations
│   │   ├── selection/        # Model selection
│   │   └── fitness/          # Fitness evaluation
│   ├── compression/           # Model compression
│   │   ├── quantization/     # Quantization methods
│   │   ├── pruning/          # Model pruning
│   │   └── distillation/     # Knowledge distillation
│   └── models/                # Model definitions
├── rag/                       # RAG system business logic
│   ├── ingestion/             # Data ingestion logic
│   │   ├── parsers/          # Document parsers
│   │   ├── chunking/         # Text chunking
│   │   └── embedding/        # Embedding generation
│   ├── retrieval/             # Information retrieval
│   │   ├── indexing/         # Index management
│   │   ├── search/           # Search algorithms
│   │   └── ranking/          # Result ranking
│   ├── generation/            # Response generation
│   │   ├── synthesis/        # Information synthesis
│   │   ├── reasoning/        # Reasoning chains
│   │   └── validation/       # Response validation
│   └── memory/                # Memory management
└── domain/                    # Domain models and business rules
    ├── tokenomics/            # Economic models
    │   ├── rewards/          # Reward calculations
    │   ├── governance/       # DAO governance
    │   └── treasury/         # Treasury management
    ├── identity/              # Identity management
    ├── security/              # Security policies
    └── compliance/            # Compliance rules
```

### 3. Infrastructure Layer (`infrastructure/`)
**Purpose**: Technical implementation details
**Dependencies**: Only `libs/` and external libraries
**Cannot depend on**: `apps/`, `core/`

```
infrastructure/
├── gateway/ ✅                # API gateway (existing)
│   ├── api/                  # REST/GraphQL APIs
│   ├── auth/                 # Authentication
│   ├── routing/              # Request routing
│   ├── rate_limiting/        # Rate limiting
│   └── monitoring/           # Gateway monitoring
├── twin/ ✅                  # Digital twin system (existing)
│   ├── engine/               # Twin engine
│   ├── security/             # Twin security
│   ├── compression/          # Data compression
│   └── database/             # Twin storage
├── mcp/ ✅                   # MCP protocol (existing - expand)
│   ├── servers/              # MCP servers
│   ├── clients/              # MCP clients
│   ├── protocol/             # Protocol implementation
│   └── tools/                # MCP tools
├── p2p/ ✅                   # P2P networking (existing - expand)
│   ├── betanet/              # BetaNet protocol
│   ├── bitchat/              # BitChat messaging
│   ├── mesh/                 # Mesh networking
│   └── federation/           # Network federation
├── shared/ ✅                # Shared infrastructure (existing)
│   ├── auth/                 # Authentication utilities
│   ├── config/               # Configuration management
│   ├── logging/              # Logging infrastructure
│   └── utils/                # Common utilities
├── fog/                      # Fog computing (NEW)
│   ├── coordination/         # Fog coordination
│   ├── resource-allocation/  # Resource management
│   ├── edge-computing/       # Edge computation
│   └── mobile-optimization/  # Mobile edge optimization
├── data/                     # Data layer (NEW)
│   ├── storage/              # Storage backends
│   ├── caching/              # Caching systems
│   ├── streaming/            # Data streaming
│   └── persistence/          # Data persistence
└── messaging/                # Messaging systems (NEW)
    ├── queues/               # Message queues
    ├── pubsub/               # Pub/Sub systems
    ├── events/               # Event systems
    └── notifications/        # Notification delivery
```

### 4. DevOps Layer (`devops/`)
**Purpose**: Development operations and automation
**Dependencies**: Can reference all layers for deployment

```
devops/
├── ci-cd/                    # Continuous integration/deployment
│   ├── github-actions/       # GitHub workflows
│   ├── quality-gates/        # Code quality checks
│   ├── security-scans/       # Security scanning
│   └── deployment-pipelines/ # Deployment automation
├── deployment/               # Deployment configurations
│   ├── docker/               # Container definitions
│   ├── kubernetes/           # K8s manifests
│   ├── terraform/            # Infrastructure as code
│   └── environments/         # Environment configs
├── monitoring/               # Monitoring and observability
│   ├── metrics/              # Metrics collection
│   ├── logging/              # Log aggregation
│   ├── alerting/             # Alert management
│   └── dashboards/           # Monitoring dashboards
└── automation/               # Automation scripts
    ├── backup/               # Backup automation
    ├── maintenance/          # Maintenance scripts
    ├── migration/            # Data migration
    └── provisioning/         # Resource provisioning
```

### 5. Libs Layer (`libs/`)
**Purpose**: Shared libraries and utilities
**Dependencies**: Only external libraries
**Cannot depend on**: Any other project layers

```
libs/
├── crypto/                   # Cryptographic utilities
│   ├── encryption/           # Encryption algorithms
│   ├── signing/              # Digital signatures
│   ├── hashing/              # Hash functions
│   └── key-management/       # Key management
├── networking/               # Networking utilities
│   ├── protocols/            # Network protocols
│   ├── discovery/            # Service discovery
│   ├── routing/              # Network routing
│   └── security/             # Network security
├── ml-utils/                 # Machine learning utilities
│   ├── training/             # Training utilities
│   ├── inference/            # Inference utilities
│   ├── evaluation/           # Model evaluation
│   └── optimization/         # Model optimization
└── common/                   # Common utilities
    ├── data-structures/      # Data structures
    ├── algorithms/           # Algorithms
    ├── validation/           # Input validation
    └── serialization/        # Data serialization
```

### 6. Integrations Layer (`integrations/`)
**Purpose**: External system integrations
**Dependencies**: Can use `libs/`, external APIs only

```
integrations/
├── external-apis/            # External API integrations
│   ├── openai/               # OpenAI integration
│   ├── anthropic/            # Anthropic integration
│   ├── openrouter/           # OpenRouter integration
│   └── cloud-providers/      # Cloud provider APIs
├── blockchain/               # Blockchain integrations
│   ├── ethereum/             # Ethereum integration
│   ├── polygon/              # Polygon integration
│   └── smart-contracts/      # Smart contract interfaces
└── cloud-services/           # Cloud service integrations
    ├── aws/                  # AWS services
    ├── gcp/                  # Google Cloud Platform
    └── azure/                # Microsoft Azure
```

## Module Boundary Contracts

### Interface Definitions

```python
# Core Agent Contract
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AgentInterface(ABC):
    """Core agent interface - all agents must implement"""
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request"""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Health check for agent"""
        pass

# Infrastructure Service Contract
class InfrastructureService(ABC):
    """Infrastructure service interface"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize service with configuration"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Graceful shutdown"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Service health status"""
        pass
```

### Dependency Rules

1. **Strict Layer Isolation**
   - Core cannot import from infrastructure
   - Infrastructure cannot import from apps
   - Apps can import from core and infrastructure
   - Libs are dependency-free utilities only

2. **Interface-Based Communication**
   - All cross-layer communication through interfaces
   - No direct class dependencies across layers
   - Use dependency injection for implementations

3. **Configuration-Driven Integration**
   - All integrations configured externally
   - No hardcoded dependencies
   - Environment-specific configurations

## Migration Strategy

### Phase 1: Structure Creation (Week 1)
1. Create all directory structures
2. Move clear candidates to new locations
3. Update import paths incrementally
4. Maintain backward compatibility bridges

### Phase 2: Core Separation (Week 2)
1. Extract business logic from infrastructure
2. Create interface contracts
3. Implement dependency injection
4. Update tests for new structure

### Phase 3: Infrastructure Completion (Week 3)
1. Add missing infrastructure components
2. Consolidate duplicate implementations
3. Implement cross-layer communication
4. Complete integration testing

### Phase 4: Apps Consolidation (Week 4)
1. Consolidate UI components
2. Implement shared UI libraries
3. Create platform-specific builds
4. Complete end-to-end testing

### Phase 5: DevOps Integration (Week 5)
1. Update CI/CD pipelines
2. Implement monitoring for new structure
3. Create deployment automation
4. Performance optimization

## Migration Commands

```bash
# Phase 1: Create structure
mkdir -p apps/{mobile/{ios,android,shared},web,cli,desktop}
mkdir -p core/{agents/{governance,knowledge,infrastructure,specialized},agent-forge,rag,domain}
mkdir -p infrastructure/{fog,data,messaging}
mkdir -p devops/{ci-cd,deployment,monitoring,automation}
mkdir -p libs/{crypto,networking,ml-utils,common}
mkdir -p integrations/{external-apis,blockchain,cloud-services}

# Phase 2: Move packages content
# (Detailed file-by-file migration)

# Phase 3: Update imports
# (Automated import updating)

# Phase 4: Clean up
# (Remove old structures)
```

## Validation Criteria

### Architectural Compliance
- [ ] No circular dependencies between layers
- [ ] All cross-layer communication through interfaces
- [ ] Configuration-driven integrations
- [ ] Clear separation of concerns

### Code Quality
- [ ] All files under 500 lines
- [ ] Functions under 50 lines
- [ ] Maximum 3 parameters per function
- [ ] No magic numbers or strings

### Testing Coverage
- [ ] Unit tests for all business logic
- [ ] Integration tests for all APIs
- [ ] End-to-end tests for critical paths
- [ ] Performance tests for bottlenecks

### Documentation
- [ ] README in every major directory
- [ ] API documentation for all interfaces
- [ ] Architecture decision records
- [ ] Deployment guides

## Success Metrics

### Before Migration
- 5,000+ files across packages/
- Multiple implementations of same features
- Unclear boundaries and responsibilities
- Difficult to locate functionality

### After Migration
- Clear layer separation with defined responsibilities
- Single source of truth for each feature
- Well-defined interfaces and contracts
- Easy navigation and maintenance

## Next Steps

1. **Review and approve** this architectural plan
2. **Create directory structures** following the design
3. **Implement interface contracts** for cross-layer communication
4. **Begin systematic migration** starting with clear candidates
5. **Update CI/CD pipelines** to support new structure
6. **Create comprehensive documentation** for new architecture

This clean architecture completion will transform AIVillage into a maintainable, scalable, and understandable system that follows industry best practices while preserving all existing functionality.