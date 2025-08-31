# Architectural Coupling Analysis - GraphFixer

## Executive Summary

This analysis examines the coupling patterns, dependencies, and architectural relationships within the GraphFixer system. The analysis reveals high coupling with the TrustGraph system, moderate coupling with vector engines, and opportunities for significant decoupling through interface-based design patterns.

## Coupling Classification Framework

We use the following coupling severity classification:
- **VERY HIGH**: Tight structural dependencies, difficult to change
- **HIGH**: Strong functional dependencies, moderate change impact  
- **MEDIUM**: Interface-based dependencies, manageable change impact
- **LOW**: Loosely coupled through protocols, minimal change impact
- **VERY LOW**: Independent components, no change impact

## 1. External Dependency Analysis

### 1.1 TrustGraph Dependency (VERY HIGH Coupling)

**Dependency Type**: Structural and Data Coupling
**Severity**: VERY HIGH ⚠️⚠️⚠️

**Analysis**:
```python
# Direct structural access patterns found:
- Line 430: self.trust_graph.nodes.items()  # Direct dict access
- Line 444: node.incoming_edges, node.outgoing_edges  # Direct attribute access  
- Line 480: node.embedding  # Direct attribute access
- Line 494: np.dot(emb1, emb2)  # Assumes specific embedding format
- Line 597: edge.source_id, edge.target_id  # Direct edge structure access
```

**Coupling Indicators**:
1. **Knowledge of Internal Structure**: GraphFixer knows about `nodes` dict and `edges` dict
2. **Direct Attribute Access**: Accesses `node.incoming_edges`, `node.outgoing_edges`, `node.trust_score`
3. **Format Assumptions**: Assumes embeddings are NumPy arrays with specific dimensions
4. **Method Assumptions**: Expects specific methods like `update_trust_from_evidence()`

**Mitigation Strategies**:

**Strategy 1: Repository Pattern**
```python
class GraphRepository(Protocol):
    """Abstract repository for graph operations - reduces coupling"""
    
    async def get_all_nodes(self) -> AsyncIterator[GraphNodeView]:
        """Get all nodes without exposing internal structure"""
        ...
    
    async def get_node_connections(self, node_id: str) -> List[ConnectionView]:
        """Get node connections without exposing edge structure"""
        ...
    
    async def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get embedding without assuming storage format"""
        ...
    
    async def get_trust_metrics(self, node_id: str) -> TrustMetrics:
        """Get trust information without direct attribute access"""
        ...

@dataclass
class GraphNodeView:
    """View object that hides internal node structure"""
    id: str
    concept: str
    trust_score: float
    connection_count: int
    embedding: Optional[np.ndarray] = None

@dataclass  
class ConnectionView:
    """View object for node connections"""
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
```

**Strategy 2: Event-Driven Decoupling**
```python
class GraphEventHandler:
    """Handles graph events without direct coupling"""
    
    async def on_node_added(self, event: NodeAddedEvent):
        """React to node additions"""
        gaps = await self.gap_detector.analyze_new_node(event.node_view)
        await self.publish_gaps_detected(gaps)
    
    async def on_edge_added(self, event: EdgeAddedEvent):
        """React to edge additions"""  
        gaps = await self.gap_detector.reanalyze_affected_nodes(event.affected_nodes)
        await self.publish_gaps_updated(gaps)
```

### 1.2 VectorEngine Dependency (MEDIUM Coupling)

**Dependency Type**: Functional Coupling
**Severity**: MEDIUM ⚠️

**Analysis**:
```python
# VectorEngine usage patterns:
- Line 471: if not self.vector_engine  # Optional dependency
- Line 480: node.embedding is not None  # Assumes embedding availability
- Line 493: similarity = np.dot(emb1, emb2)  # Assumes NumPy format
```

**Coupling Indicators**:
1. **Optional Dependency**: Gracefully handles missing vector engine
2. **Format Assumptions**: Assumes embeddings are NumPy arrays
3. **Direct Computation**: Performs similarity computation directly

**Current Mitigation**: Already uses protocol-based interface
```python
class VectorEngineProtocol(Protocol):
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        ...
```

**Enhanced Mitigation Strategy**:
```python
class SimilarityService:
    """Decoupled similarity computation service"""
    
    def __init__(self, vector_engine: Optional[VectorEngineProtocol] = None):
        self.vector_engine = vector_engine
        self.similarity_cache = {}
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity with caching and fallback"""
        if not self.vector_engine:
            return self._lexical_similarity(text1, text2)  # Fallback
        
        cache_key = (hash(text1), hash(text2))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        emb1 = await self.vector_engine.get_embedding(text1)
        emb2 = await self.vector_engine.get_embedding(text2)
        
        similarity = self._compute_cosine_similarity(emb1, emb2)
        self.similarity_cache[cache_key] = similarity
        return similarity
```

### 1.3 NumPy Dependency (LOW Coupling)

**Dependency Type**: Library Coupling
**Severity**: LOW ✅

**Analysis**: Well-encapsulated scientific computing dependency. NumPy is stable and standard for mathematical operations.

## 2. Internal Coupling Analysis

### 2.1 Method Interdependence (HIGH Coupling)

**Coupling Pattern**: Sequential Method Dependencies

**Analysis**:
```python
# Method call chain with tight coupling:
detect_knowledge_gaps()  
    ├── _run_detection_method()      # Tight coupling to detection methods
    │   ├── _detect_structural_gaps()
    │   ├── _detect_semantic_gaps()
    │   └── _detect_trust_inconsistencies()
    ├── _deduplicate_gaps()          # Depends on gap structure
    └── _rank_gaps_by_priority()     # Depends on gap priority calculation

propose_solutions()
    ├── _propose_missing_nodes()     # Tight coupling to gap types  
    ├── _propose_missing_relationships()
    └── _rank_node_proposals()       # Depends on proposal structure
```

**Decoupling Strategy: Strategy Pattern**
```python
class GapDetectionStrategy(ABC):
    """Abstract strategy for gap detection - decouples methods"""
    
    @abstractmethod
    async def detect(self, context: DetectionContext) -> List[DetectedGap]:
        """Detect gaps using specific strategy"""
        pass

class StructuralGapStrategy(GapDetectionStrategy):
    """Structural gap detection strategy"""
    
    async def detect(self, context: DetectionContext) -> List[DetectedGap]:
        return await self._analyze_structure(context.graph_snapshot)

class SemanticGapStrategy(GapDetectionStrategy):
    """Semantic gap detection strategy"""
    
    def __init__(self, similarity_service: SimilarityService):
        self.similarity_service = similarity_service
    
    async def detect(self, context: DetectionContext) -> List[DetectedGap]:
        return await self._analyze_semantic_gaps(context)

class GapDetectionOrchestrator:
    """Orchestrates gap detection strategies - loose coupling"""
    
    def __init__(self):
        self.strategies: Dict[str, GapDetectionStrategy] = {}
    
    def register_strategy(self, name: str, strategy: GapDetectionStrategy):
        """Register detection strategy"""
        self.strategies[name] = strategy
    
    async def detect_all_gaps(self, context: DetectionContext, 
                             methods: List[str] = None) -> List[DetectedGap]:
        """Detect gaps using specified strategies"""
        active_methods = methods or list(self.strategies.keys())
        
        # Execute strategies in parallel - no coupling between them
        tasks = [
            self.strategies[method].detect(context)
            for method in active_methods 
            if method in self.strategies
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect successful results
        all_gaps = []
        for result in results:
            if not isinstance(result, Exception):
                all_gaps.extend(result)
        
        return all_gaps
```

### 2.2 Data Structure Coupling (MEDIUM Coupling)

**Coupling Pattern**: Shared Data Structures

**Analysis**:
```python
# Shared data structures create coupling:
DetectedGap       # Used across detection methods
ProposedNode      # Used across proposal methods  
ProposedRelationship  # Used across proposal methods
GapAnalysisResult     # Used for result aggregation
```

**Current Mitigation**: Centralized types in `graph_types.py` ✅

**Enhanced Strategy: Domain Events**
```python
@dataclass
class GapDetectedEvent:
    """Domain event for gap detection - loose coupling"""
    gap_id: str
    gap_type: str
    confidence: float
    detection_method: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass  
class ProposalGeneratedEvent:
    """Domain event for proposal generation"""
    proposal_id: str
    proposal_type: str  # 'node' or 'relationship'
    gap_id: str
    confidence: float
    timestamp: datetime

class EventBus:
    """Event bus for loose coupling between components"""
    
    def __init__(self):
        self.handlers: Dict[Type, List[Callable]] = {}
    
    def subscribe(self, event_type: Type, handler: Callable):
        """Subscribe to event type"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    async def publish(self, event: Any):
        """Publish event to subscribers"""
        event_type = type(event)
        if event_type in self.handlers:
            tasks = [handler(event) for handler in self.handlers[event_type]]
            await asyncio.gather(*tasks, return_exceptions=True)
```

## 3. Configuration Coupling Analysis

### 3.1 Hard-coded Constants (HIGH Coupling)

**Problem**: Magic numbers scattered throughout code

**Analysis**:
```python
# Hard-coded values create coupling:
- Line 447: if 1 <= total_connections <= 2  # Magic numbers
- Line 496: if similarity > 0.7  # Hard-coded threshold
- Line 606: if trust_variance > 0.2  # Hard-coded threshold  
- Line 638: expected_edges = total_nodes * 1.5  # Magic multiplier
```

**Solution: Configuration Service**
```python
@dataclass
class GraphFixerConfig:
    """Centralized configuration - reduces coupling"""
    
    # Structural analysis thresholds
    weak_connection_min: int = 1
    weak_connection_max: int = 2
    high_trust_threshold: float = 0.6
    
    # Semantic analysis thresholds  
    similarity_threshold: float = 0.7
    min_embedding_dimensions: int = 50
    
    # Trust analysis thresholds
    trust_variance_threshold: float = 0.2
    min_trust_sources: int = 2
    
    # Connectivity analysis
    expected_edges_multiplier: float = 1.5
    connectivity_threshold_ratio: float = 0.5
    
    # Performance settings
    max_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    batch_size: int = 1000
    
    @classmethod
    def from_environment(cls) -> 'GraphFixerConfig':
        """Create config from environment variables"""
        return cls(
            similarity_threshold=float(os.getenv('GF_SIMILARITY_THRESHOLD', 0.7)),
            trust_variance_threshold=float(os.getenv('GF_TRUST_VARIANCE_THRESHOLD', 0.2)),
            # ... other env vars
        )

class ConfigurableGraphFixer:
    """GraphFixer with externalized configuration"""
    
    def __init__(self, config: GraphFixerConfig, ...):
        self.config = config
        # ... rest of initialization
    
    async def _detect_structural_gaps(self) -> List[DetectedGap]:
        """Uses configuration instead of magic numbers"""
        gaps = []
        
        for node_id, node in self.trust_graph.nodes.items():
            total_connections = len(node.incoming_edges) + len(node.outgoing_edges)
            
            # Use configuration instead of hard-coded values
            if (self.config.weak_connection_min <= total_connections <= self.config.weak_connection_max 
                and node.trust_score > self.config.high_trust_threshold):
                
                gap = DetectedGap(
                    gap_type=GapType.WEAK_CONNECTION,
                    source_nodes=[node_id],
                    description=f"High-trust node '{node.concept}' is under-connected",
                    confidence=0.7,
                    priority=0.5,
                    detection_method="structural_analysis",
                )
                gaps.append(gap)
        
        return gaps
```

## 4. Temporal Coupling Analysis

### 4.1 Method Execution Order Dependencies (MEDIUM Coupling)

**Problem**: Methods must be called in specific order

**Analysis**:
```python
# Required execution sequence:
1. initialize()           # Must be called first
2. detect_knowledge_gaps()  # Must complete before propose_solutions()
3. propose_solutions()    # Depends on gaps from step 2
4. validate_proposal()    # Depends on proposals from step 3
```

**Solution: Pipeline Pattern**
```python
class GraphFixerPipeline:
    """Pipeline pattern to manage execution sequence"""
    
    def __init__(self, graph_fixer: GraphFixer):
        self.graph_fixer = graph_fixer
        self.pipeline_state = {}
    
    async def execute(self, query: str, context: Dict[str, Any]) -> GapAnalysisResult:
        """Execute complete pipeline with proper sequencing"""
        
        # Stage 1: Initialization
        result = GapAnalysisResult()
        
        if not self.graph_fixer.initialized:
            await self.graph_fixer.initialize()
        
        # Stage 2: Gap Detection
        start_time = time.time()
        gaps = await self.graph_fixer.detect_knowledge_gaps(
            query=query,
            retrieved_info=context.get('retrieved_info'),
            focus_area=context.get('focus_area')
        )
        result.gaps_detected = gaps
        result.total_gaps_found = len(gaps)
        
        # Stage 3: Solution Proposals (only if gaps found)
        if gaps:
            proposed_nodes, proposed_relationships = await self.graph_fixer.propose_solutions(
                gaps=gaps,
                max_proposals=context.get('max_proposals')
            )
            result.proposed_nodes = proposed_nodes
            result.proposed_relationships = proposed_relationships
            result.total_proposals = len(proposed_nodes) + len(proposed_relationships)
        
        # Stage 4: Quality Metrics
        result.analysis_time_ms = (time.time() - start_time) * 1000
        result.avg_gap_confidence = statistics.mean([g.confidence for g in gaps]) if gaps else 0.0
        
        return result
```

## 5. Interface Segregation Analysis

### 5.1 Fat Interface Problem (MEDIUM Coupling)

**Problem**: GraphFixer has too many responsibilities

**Current Interface**:
```python
class GraphFixer:
    # Gap detection responsibilities
    async def detect_knowledge_gaps(...)
    async def _detect_structural_gaps(...)
    async def _detect_semantic_gaps(...)
    # ... more detection methods
    
    # Proposal responsibilities  
    async def propose_solutions(...)
    async def _propose_missing_nodes(...)
    # ... more proposal methods
    
    # Validation responsibilities
    async def validate_proposal(...)
    
    # Analysis responsibilities
    async def analyze_graph_completeness(...)
    # ... more analysis methods
```

**Solution: Interface Segregation**
```python
class IGapDetector(Protocol):
    """Focused interface for gap detection"""
    async def detect_gaps(self, context: DetectionContext) -> List[DetectedGap]:
        ...

class IProposalGenerator(Protocol):
    """Focused interface for proposal generation"""
    async def generate_proposals(self, gaps: List[DetectedGap]) -> Tuple[List[ProposedNode], List[ProposedRelationship]]:
        ...

class IValidationManager(Protocol):
    """Focused interface for validation"""
    async def validate_proposal(self, proposal: Union[ProposedNode, ProposedRelationship], 
                               feedback: str, is_accepted: bool) -> bool:
        ...

class IGraphAnalyzer(Protocol):
    """Focused interface for graph analysis"""
    async def analyze_completeness(self) -> Dict[str, Any]:
        ...

class GraphFixerFacade:
    """Facade that composes focused services - loose coupling"""
    
    def __init__(self, 
                 gap_detector: IGapDetector,
                 proposal_generator: IProposalGenerator,
                 validation_manager: IValidationManager,
                 graph_analyzer: IGraphAnalyzer):
        self.gap_detector = gap_detector
        self.proposal_generator = proposal_generator  
        self.validation_manager = validation_manager
        self.graph_analyzer = graph_analyzer
    
    async def analyze_and_propose(self, query: str) -> GapAnalysisResult:
        """Orchestrate the complete workflow"""
        context = DetectionContext(query=query)
        
        # Each component is loosely coupled
        gaps = await self.gap_detector.detect_gaps(context)
        nodes, relationships = await self.proposal_generator.generate_proposals(gaps)
        completeness = await self.graph_analyzer.analyze_completeness()
        
        return GapAnalysisResult(
            gaps_detected=gaps,
            proposed_nodes=nodes,
            proposed_relationships=relationships,
            metadata={'completeness': completeness}
        )
```

## 6. Dependency Injection Analysis

### 6.1 Current Dependency Injection (PARTIAL)

**Analysis**: GraphFixer uses constructor injection but not consistently

```python
class GraphFixer:
    def __init__(self,
                 trust_graph=None,      # Optional, should be required
                 vector_engine=None,    # Optional, good
                 min_confidence_threshold: float = 0.3,  # Good
                 max_proposals_per_gap: int = 3):        # Good
```

**Enhanced Dependency Injection**:
```python
class GraphFixerDependencies:
    """Dependency container for GraphFixer"""
    
    def __init__(self):
        self.graph_repository: Optional[IGraphRepository] = None
        self.similarity_service: Optional[ISimilarityService] = None
        self.config_service: Optional[IConfigService] = None
        self.event_bus: Optional[IEventBus] = None
        self.metrics_collector: Optional[IMetricsCollector] = None
    
    def validate(self):
        """Validate required dependencies"""
        if not self.graph_repository:
            raise ValueError("Graph repository is required")

class GraphFixer:
    """Fully dependency-injected GraphFixer"""
    
    def __init__(self, dependencies: GraphFixerDependencies):
        dependencies.validate()
        
        self.graph_repo = dependencies.graph_repository
        self.similarity_service = dependencies.similarity_service or NullSimilarityService()
        self.config = dependencies.config_service or DefaultConfigService()
        self.event_bus = dependencies.event_bus or NullEventBus()
        self.metrics = dependencies.metrics_collector or NullMetricsCollector()
    
    @classmethod
    def create_default(cls, trust_graph) -> 'GraphFixer':
        """Factory method for default configuration"""
        deps = GraphFixerDependencies()
        deps.graph_repository = TrustGraphRepository(trust_graph)
        deps.similarity_service = VectorSimilarityService()
        deps.config_service = EnvironmentConfigService()
        return cls(deps)
```

## 7. Testing Coupling Analysis

### 7.1 Testability Issues (HIGH Coupling)

**Problems Identified**:
1. **Hard Dependencies**: Difficult to mock TrustGraph
2. **Global State**: Class-level caches make testing difficult  
3. **Side Effects**: Methods modify shared state
4. **Complex Setup**: Requires complex test fixtures

**Solution: Test-Friendly Architecture**
```python
class TestableGraphFixer:
    """Testable version with dependency injection"""
    
    def __init__(self, 
                 graph_repo: IGraphRepository,
                 similarity_service: ISimilarityService = None,
                 config: GraphFixerConfig = None):
        self.graph_repo = graph_repo
        self.similarity_service = similarity_service or NullSimilarityService()
        self.config = config or GraphFixerConfig()
        
        # Inject cache for testability
        self.cache = InMemoryCache()  # Can be mocked
    
    async def detect_structural_gaps(self, 
                                   graph_snapshot: Optional[GraphSnapshot] = None) -> List[DetectedGap]:
        """Testable method with optional snapshot injection"""
        snapshot = graph_snapshot or await self.graph_repo.create_snapshot()
        return await self._analyze_structure(snapshot)

# Test example
async def test_structural_gap_detection():
    # Arrange - easy mocking
    mock_repo = Mock(spec=IGraphRepository)
    mock_repo.create_snapshot.return_value = create_test_snapshot()
    
    graph_fixer = TestableGraphFixer(
        graph_repo=mock_repo,
        config=GraphFixerConfig(weak_connection_max=1)  # Configurable
    )
    
    # Act
    gaps = await graph_fixer.detect_structural_gaps()
    
    # Assert  
    assert len(gaps) > 0
    assert all(gap.gap_type == GapType.WEAK_CONNECTION for gap in gaps)
```

## 8. Coupling Metrics and Recommendations

### 8.1 Coupling Metrics

| Component | Current Coupling Level | Target Coupling Level | Priority |
|-----------|----------------------|---------------------|----------|
| TrustGraph Dependency | VERY HIGH ⚠️⚠️⚠️ | MEDIUM ⚠️ | HIGH |
| VectorEngine Dependency | MEDIUM ⚠️ | LOW ✅ | MEDIUM |
| Method Interdependence | HIGH ⚠️⚠️ | LOW ✅ | HIGH |
| Configuration Coupling | HIGH ⚠️⚠️ | VERY LOW ✅ | HIGH |
| Temporal Coupling | MEDIUM ⚠️ | LOW ✅ | MEDIUM |
| Interface Segregation | HIGH ⚠️⚠️ | LOW ✅ | HIGH |

### 8.2 Decoupling Roadmap

**Phase 1: Foundation (2-3 weeks)**
1. Extract configuration to external config service
2. Implement repository pattern for graph access
3. Add dependency injection container
4. Create focused interfaces (ISeparation)

**Phase 2: Service Extraction (4-6 weeks)**
1. Extract similarity service with caching
2. Implement event-driven architecture
3. Create pipeline orchestrator
4. Add comprehensive testing infrastructure

**Phase 3: Architecture Evolution (2-3 months)**
1. Implement microservice boundaries
2. Add distributed event bus
3. Create configuration service
4. Implement monitoring and metrics

### 8.3 Success Metrics

**Coupling Reduction Targets**:
- **Method Dependencies**: Reduce from 15 direct calls to 5 interface calls
- **Configuration Coupling**: Eliminate all magic numbers (currently 12 identified)
- **Testing Setup**: Reduce test fixture complexity by 80%
- **Service Boundaries**: Enable independent deployment of 5 services

**Quality Improvements Expected**:
- **Testability**: 90% test coverage achievable with mocked dependencies
- **Maintainability**: Single responsibility per service component  
- **Scalability**: Independent scaling of computation-intensive services
- **Flexibility**: Configuration changes without code deployment

This coupling analysis provides the architectural foundation for transforming GraphFixer from a monolithic, tightly-coupled system into a loosely-coupled, service-oriented architecture suitable for modern deployment patterns.