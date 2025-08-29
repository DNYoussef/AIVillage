# Architecture - Consolidated Documentation

**Last Updated:** 2025-08-27
**Status:** Unified architectural truth consolidating 32+ architectural documents
**Evidence-Based Assessment:** Integration with actual codebase verification

## 🎯 System Architecture Overview

AIVillage is a distributed multi-agent AI platform that has evolved from a monolithic Python project into a sophisticated layered architecture with autonomous evolution capabilities, fog computing infrastructure, and advanced P2P networking. The system follows clean architecture principles with clear separation of concerns across distinct layers.

### Architectural Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIVillage Clean Architecture                 │
├─────────────────────────────────────────────────────────────────┤
│  📱 Apps Layer              │  🧠 Core Layer                     │
│  ├── mobile/ (iOS/Android)  │  ├── agents/ (23+ specialized)    │
│  ├── web/ (Admin/Portal)    │  ├── agent-forge/ (Training)      │
│  ├── cli/ (Development)     │  ├── hyperrag/ (Knowledge)        │
│  └── desktop/ (Native)      │  └── domain/ (Business Logic)     │
├─────────────────────────────────────────────────────────────────┤
│  🔗 Infrastructure Layer    │  ⚙️ DevOps Layer                  │
│  ├── gateway/ (API/Auth)    │  ├── ci-cd/ (Quality Gates)       │
│  ├── twin/ (Digital Twin)   │  ├── deployment/ (K8s/Docker)     │
│  ├── fog/ (Edge Computing)  │  ├── monitoring/ (Observability)  │
│  ├── p2p/ (Mesh Network)    │  └── automation/ (Operations)     │
│  └── mcp/ (Protocol)        │                                  │
├─────────────────────────────────────────────────────────────────┤
│  📦 Packages Layer          │  🔌 Integrations Layer            │
│  ├── crypto/ (Security)     │  ├── blockchain/ (Smart Contract)  │
│  ├── ml-utils/ (AI Tools)   │  ├── external-apis/ (OpenRouter)  │
│  ├── networking/ (Utils)    │  └── cloud-services/ (AWS/GCP)    │
│  └── common/ (Shared)       │                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🏗️ Component Architecture

### Core Components

#### 1. Agent System (Multi-Agent Orchestration)
**Status:** Partially Implemented
**Location:** `core/agents/`, `experiments/agents/`

- **King Agent**: Strategic planning and high-level governance
- **Sage Agent**: Knowledge management and information synthesis
- **Magi Agent**: Infrastructure management and technical operations
- **Oracle Agent**: Prediction and forecasting capabilities
- **Navigator Agent**: Network routing and mesh coordination
- **23 Specialized Agents**: Domain-specific expertise (governance, knowledge, infrastructure)

**Democratic Governance:** 2/3 quorum voting system for collective decision-making

#### 2. Agent Forge (Training & Evolution Pipeline)
**Status:** Production Ready
**Location:** `core/agent-forge/`, `src/agent_forge/`

**7-Phase Training Pipeline:**
1. **EvoMerge**: Evolutionary model optimization with tournament selection
2. **Quiet-STaR**: Internal reasoning with thought tokens (`<|startofthought|>`)
3. **BitNet**: 1-bit quantization for mobile deployment
4. **Training**: Curriculum learning and specialization
5. **Tool/Persona Baking**: Role-specific fine-tuning
6. **ADAS**: Architecture Discovery and Self-modification
7. **Final Compression**: VPTQ + HyperFn optimization

**Performance Characteristics:**
- **4-8x model compression** through BitNet + VPTQ
- **50-generation evolution** with resource constraints
- **W&B integration** for experiment tracking
- **HRRM Bootstrap**: Three 50M parameter models for acceleration

#### 3. HyperRAG System (Retrieval-Augmented Generation)
**Status:** Multiple Implementations (Consolidation Required)
**Locations:** `core/hyperrag/`, `core/rag/`, 8+ duplicate implementations

**Cognitive Nexus Architecture:**
- **Ingestion Pipeline**: Multi-modal document parsing and chunking
- **Bayesian Trust Networks**: Probabilistic knowledge validation
- **Vector Store**: High-performance semantic search with embeddings
- **Knowledge Graph**: Structured relationship mapping (Neo4j)
- **Confidence Estimation**: Response reliability scoring
- **Sub-100ms latency** target for production queries

#### 4. Fog Computing Infrastructure
**Status:** Production Ready
**Location:** `infrastructure/fog/`

**Distributed Edge Computing:**
- **NSGA-II Scheduler**: Multi-objective optimization for resource allocation
- **Marketplace Engine**: Spot/on-demand bidding for compute resources
- **Edge Capability Beacon**: Mobile device integration with WASI runner
- **SLA Classes**: S-class (replicated+attested), A-class (replicated), B-class (best-effort)
- **BetaNet Integration**: Bridge adapters for advanced transport protocols

**Performance Metrics:**
- Job Scheduling: <100ms (NSGA-II optimization)
- Market Price Discovery: <50ms (real-time bidding)
- Edge Device Discovery: 5-30 seconds (mDNS + capability beacon)
- Resource Utilization: 70-85% across fog network

### Infrastructure Components

#### 1. Gateway Layer (API & Authentication)
**Status:** Operational
**Location:** `infrastructure/gateway/`

- **FastAPI-based**: HTTP/WebSocket entry point
- **Authentication**: JWT tokens, API keys, OAuth2 integration
- **Rate Limiting**: Request throttling and quota management
- **Request Routing**: Intelligent routing to appropriate services
- **Admin APIs**: System administration and monitoring

#### 2. Digital Twin Engine
**Status:** Partially Operational
**Location:** `infrastructure/twin/`

- **Personal AI Models**: 1-10MB models running locally on devices
- **Privacy-First**: All personal data stays on device (never cloud)
- **Surprise-Based Learning**: Models improve via prediction accuracy
- **Resource Management**: Battery/thermal-aware processing
- **Encrypted Storage**: AES encryption with compliance features

#### 3. P2P Communication Layer
**Status:** Operational (Multi-Protocol)
**Location:** `infrastructure/p2p/`

**Multi-Transport Architecture:**
- **BitChat**: Bluetooth Low Energy mesh networking for offline scenarios
- **BetaNet**: Encrypted HTTP/WebSocket transport for internet connectivity
- **LibP2P**: Peer-to-peer protocol with routing and discovery
- **Mesh Routing**: Intelligent message routing with failover capabilities
- **Federation**: Network bridging and inter-protocol communication

**Network Characteristics:**
- **Mobile-First**: Battery and bandwidth-aware protocols
- **Offline Support**: Message queuing for disconnected peers
- **Security**: End-to-end encryption with identity verification

#### 4. MCP (Model Control Protocol) Layer
**Status:** Operational
**Location:** `infrastructure/mcp/`

- **Agent Tools**: Standardized interfaces for agent capabilities
- **Memory Servers**: Persistent and working memory management
- **RAG Servers**: Knowledge retrieval and augmentation services
- **Inter-Service Communication**: Unified protocol for component interaction

### Integration Patterns

#### 1. Event-Driven Architecture
- **System Events**: Standardized event format across all components
- **Event Bus**: Cross-layer communication without tight coupling
- **Pub/Sub Pattern**: Asynchronous message handling
- **Event Sourcing**: Complete audit trail of system changes

#### 2. Microservices Communication
- **REST APIs**: Standard HTTP interfaces for synchronous communication
- **GraphQL**: Flexible query interface for complex data needs
- **WebSocket**: Real-time bidirectional communication
- **gRPC**: High-performance inter-service communication

## 📋 Architectural Decision Records (ADRs)

### Active ADRs

#### ADR-S4-02: Response Confidence Calibration
**Status:** Accepted (2025-07-15)
**Decision:** Integrate ConformalCalibrator with beta-binomial behind `CALIBRATION_ENABLED` flag
**Impact:** +1.1ms latency, +4MB RAM, enables risk-aware meta-agents
**Rollout:** Shadow traffic → 10% → 100% gradual deployment

#### ADR-FG-01: Fog Computing Integration
**Status:** Accepted (2025-08-01)
**Decision:** NSGA-II scheduler with marketplace bidding for edge resources
**Impact:** Distributed computing with 70-85% resource utilization
**Architecture:** Bridge adapters maintain BetaNet bounty separation

#### ADR-QS-01: Quiet-STaR Implementation
**Status:** Accepted (2025-07-20)
**Decision:** Thought token injection with A/B testing harness
**Impact:** 5-10% accuracy improvement on reasoning tasks
**Integration:** Complete with Agent Forge pipeline

#### ADR-EVO-01: EvoMerge Integration
**Status:** Accepted (2025-07-10)
**Decision:** 50-generation evolution with HRRM bootstrap acceleration
**Impact:** 30x faster iteration with resource-constrained optimization
**Validation:** Tournament selection with Pareto frontier analysis

### Superseded ADRs

#### ADR-MON-01: Monolithic Architecture (Superseded)
**Original Decision:** Single Python application with embedded components
**Superseded By:** Clean architecture with layer separation
**Migration:** Completed systematic refactoring to distributed components

#### ADR-RAG-01: Single RAG Implementation (Superseded)
**Original Decision:** Unified RAG system in single location
**Current Reality:** 8+ duplicate implementations requiring consolidation
**Resolution Required:** Critical priority consolidation to single source of truth

## 🔧 Design Patterns & Standards

### Connascence-Based Coupling Management

**Coupling Strength Hierarchy** (Weakest → Strongest):
1. **Static Forms**: Name (CoN) → Type (CoT) → Meaning (CoM) → Position (CoP) → Algorithm (CoA)
2. **Dynamic Forms**: Execution (CoE) → Timing (CoTg) → Value (CoV) → Identity (CoI)

**Management Rules:**
- **Strong connascence acceptable** within same class/function only
- **Weak connascence preferred** for cross-module communication
- **Degree reduction** through facades when >2 places co-vary
- **Locality enforcement** to keep coupling contained

### Clean Architecture Principles

1. **Dependency Inversion**: High-level modules don't depend on low-level modules
2. **Interface Segregation**: Clients shouldn't depend on unused interfaces
3. **Single Responsibility**: Each component has one clear purpose
4. **Open/Closed**: Open for extension, closed for modification

### Module Boundary Contracts

```python
# Agent Interface Contract
class IAgent(ABC):
    @abstractmethod
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process incoming request with standardized format"""
        pass

    @abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        pass

# Infrastructure Service Contract
class IInfrastructureService(ABC):
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize service with configuration"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Service health status with metrics"""
        pass
```

## 🎯 Architecture Quality Gates

### Fitness Functions (Automated Validation)

1. **Layer Boundary Enforcement**
   - No imports from higher layers to lower layers
   - All cross-layer communication through interfaces
   - Configuration-driven dependencies only

2. **Connascence Metrics**
   - Positional parameter ratio: <20% functions with >3 positional params
   - Magic literal density: <5 magic numbers per 100 lines
   - Algorithm duplication count: 0 duplicate implementations
   - Global reference tracking: minimize singleton usage

3. **Code Quality Gates**
   - Maximum file size: 500 lines
   - Maximum function size: 50 lines
   - Cyclomatic complexity: <10 per function
   - Test coverage: >80% for core business logic

4. **Performance Requirements**
   - RAG query latency: <100ms
   - API response time: <200ms
   - Agent response time: <500ms
   - System startup time: <30 seconds

### Quality Monitoring

```yaml
# Pre-commit Hooks
hooks:
  - id: connascence-checker
    name: Connascence Analysis
  - id: coupling-metrics
    name: Coupling Metrics Validation
  - id: anti-pattern-detector
    name: Anti-pattern Detection
  - id: layer-boundary-validator
    name: Architecture Layer Validation
```

---

## ❌ ARCHITECTURE REALITY GAP

### Component Architecture Gaps

#### 1. **RAG System Fragmentation vs. Unified Design**
- **Designed**: Single, coherent HyperRAG system with cognitive nexus
- **Reality**: **8+ duplicate implementations** across different locations
  - `core/hyperrag/`, `core/rag/`, `src/production/rag/`
  - `experiments/rag/`, `packages/rag/`, multiple others
- **Impact**: CRITICAL - No single source of truth, conflicting implementations
- **Evidence**: 70-80% code redundancy identified in consolidation analysis

#### 2. **Agent System Integration vs. Scattered Implementations**
- **Designed**: Unified 23-agent system with democratic governance
- **Reality**: **7+ separate agent implementations** with no coordination
  - `core/agents/`, `experiments/agents/`, `packages/agents/`
  - Import errors prevent agent communication
- **Impact**: HIGH - No actual multi-agent coordination possible
- **Evidence**: ModuleNotFoundError prevents any agent-to-agent communication

#### 3. **Clean Layer Separation vs. Circular Dependencies**
- **Designed**: Clean architecture with strict layer boundaries
- **Reality**: **Circular imports** and boundary violations throughout codebase
- **Impact**: MEDIUM - Architecture principles not enforced
- **Evidence**: Infrastructure importing from core, apps importing from infrastructure bidirectionally

### Pattern Compliance Gaps

#### 1. **Interface-Based Communication vs. Direct Coupling**
- **Standard**: All cross-layer communication through well-defined interfaces
- **Reality**: Direct class dependencies and tight coupling across modules
- **Violation**: Strong connascence across module boundaries (forbidden)
- **Examples**: Direct imports instead of dependency injection

#### 2. **Single Source of Truth vs. Duplicate Implementations**
- **Standard**: One implementation per feature/component
- **Reality**: Same features implemented 3-10 times in different locations
- **Major Duplications**:
  - RAG systems (8+ implementations)
  - Agent systems (7+ implementations)
  - P2P communication (6+ implementations)
  - Compression systems (3+ implementations)

#### 3. **Configuration-Driven Integration vs. Hardcoded Dependencies**
- **Standard**: All integrations configured externally, no hardcoded values
- **Reality**: Magic numbers, hardcoded paths, embedded configuration
- **Impact**: Difficult to test, deploy, and maintain across environments

### Critical Architecture Debt

#### Missing Implementations
1. **Unified Agent Coordination System**: Democratic governance not operational
2. **Complete RAG Pipeline**: Embeddings and vector search not functional
3. **Integrated Testing Framework**: Tests scattered across 20+ locations
4. **Production Monitoring**: Limited observability and health checking
5. **Security Layer**: Authentication and authorization incomplete

#### Architecture Violations
1. **Layer Boundary Violations**: 50+ cases of improper dependency direction
2. **Interface Contract Violations**: Direct coupling instead of interface usage
3. **Connascence Violations**: Strong coupling across module boundaries
4. **Single Responsibility Violations**: Components mixing multiple concerns

#### Technical Debt Areas
1. **Code Redundancy**: 40-50% of codebase estimated redundant
2. **Circular Dependencies**: Multiple import cycles preventing clean separation
3. **Configuration Chaos**: Settings scattered across 50+ files
4. **Test Fragmentation**: No centralized testing strategy

### Architecture Resolution Priorities

#### CRITICAL (Immediate - Next 2 Weeks)
1. **RAG System Consolidation**: Merge 8+ implementations into single source of truth
   - Choose best implementation as base (likely `core/hyperrag/`)
   - Preserve unique features from other implementations
   - Delete redundant code after consolidation
   - **Impact**: Eliminates 15,000+ redundant lines of code

2. **Agent System Unification**: Fix import errors and create working agent coordination
   - Resolve ModuleNotFoundError issues preventing agent communication
   - Implement actual democratic governance system
   - Create working agent-to-agent coordination protocols
   - **Impact**: Enables core multi-agent functionality

3. **Testing Consolidation**: Centralize all tests into organized structure
   - Move scattered tests to unified `tests/` directory
   - Organize by type: unit, integration, e2e, benchmarks
   - **Impact**: 200+ test files properly organized

#### HIGH (Next 1 Month)
1. **Layer Boundary Enforcement**: Implement architectural fitness functions
   - Add pre-commit hooks for dependency validation
   - Fix circular import cycles
   - Enforce interface-based communication
   - **Impact**: Prevents further architecture erosion

2. **Infrastructure Completion**: Fill gaps in infrastructure layer
   - Complete MCP protocol implementation
   - Enhance P2P networking capabilities
   - Implement comprehensive monitoring
   - **Impact**: Operational production readiness

3. **Configuration Unification**: Centralize all configuration management
   - Move scattered configs to unified system
   - Implement environment-specific configurations
   - Remove hardcoded values throughout codebase
   - **Impact**: Simplifies deployment and maintenance

#### MEDIUM (Next 3 Months)
1. **Performance Optimization**: Meet stated performance targets
   - RAG queries: <100ms (currently unknown due to fragmentation)
   - API responses: <200ms (currently inconsistent)
   - System startup: <30 seconds (currently varies by component)

2. **Security Implementation**: Complete security architecture
   - End-to-end authentication and authorization
   - Privacy compliance (GDPR, FERPA) as designed
   - Audit trails and compliance reporting

3. **DevOps Automation**: Complete CI/CD pipeline with quality gates
   - Automated architecture validation
   - Performance regression testing
   - Security scanning integration

### Measurement Criteria

#### Success Metrics
- **Code Redundancy**: 40-50% → 0%
- **Architecture Violations**: 50+ → 0
- **Integration Test Pass Rate**: 12.5% → 90%
- **Service Operational Rate**: 28.6% → 95%
- **Documentation Coverage**: 60% → 95%

#### Monthly Assessment Schedule
- **Week 1**: RAG system consolidation progress
- **Week 2**: Agent system integration testing
- **Week 3**: Infrastructure gap analysis
- **Week 4**: Overall architecture health assessment

---

This consolidated architecture document represents the unified truth across all 32+ architectural documents found in the codebase, providing both the aspirational design and the current reality gaps that must be addressed for the system to achieve its architectural vision.

**Next Actions:**
1. Address critical architecture gaps through systematic consolidation
2. Implement architecture fitness functions to prevent regression
3. Create monthly architecture health assessments
4. Establish architecture review board for major decisions
