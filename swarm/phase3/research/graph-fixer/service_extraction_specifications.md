# Service Extraction Specifications - GraphFixer

## 1. ML Inference Service (Priority 1 - Ready for Extraction)

### Service Definition
**Purpose**: Handles all machine learning inference operations for semantic analysis, confidence scoring, and pattern recognition.

### Current Code Location
- **File**: `core/hyperrag/cognitive/graph_fixer.py`
- **Lines**: 488-520 (semantic similarity), 765-775 (learning), 828-853 (statistics)
- **Dependencies**: NumPy, embeddings

### API Specification
```python
class MLInferenceService:
    async def compute_semantic_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float
    
    async def batch_similarity_computation(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray]
    ) -> List[float]
    
    async def calculate_confidence_scores(
        self,
        base_score: float,
        evidence: List[float],
        weights: List[float] = None
    ) -> float
    
    async def analyze_trust_distribution(
        self,
        trust_scores: List[float]
    ) -> Dict[str, float]
    
    async def update_learning_metrics(
        self,
        validation_feedback: ValidationFeedback
    ) -> bool
```

### Extraction Benefits
- **Performance**: Can be optimized with specialized hardware (GPU/TPU)
- **Scalability**: Horizontal scaling for computation-heavy operations
- **Caching**: Dedicated caching layer for similarity computations
- **Optimization**: Vector operations can be heavily optimized

### Migration Strategy
1. **Phase 1**: Extract to separate module within same codebase
2. **Phase 2**: Containerize as independent service
3. **Phase 3**: Add specialized hardware support and optimization

---

## 2. Graph Optimization Service (Priority 2)

### Service Definition
**Purpose**: Analyzes graph structure, completeness metrics, connectivity patterns, and provides optimization recommendations.

### Current Code Location
- **File**: `core/hyperrag/cognitive/graph_fixer.py`
- **Lines**: 335-361 (completeness analysis), 777-853 (metrics calculation)
- **Dependencies**: TrustGraph interface, NumPy

### API Specification
```python
class GraphOptimizationService:
    async def analyze_structural_completeness(
        self,
        graph_snapshot: GraphSnapshot
    ) -> StructuralMetrics
    
    async def analyze_trust_distribution(
        self,
        graph_snapshot: GraphSnapshot
    ) -> TrustMetrics
    
    async def analyze_connectivity_patterns(
        self,
        graph_snapshot: GraphSnapshot
    ) -> ConnectivityMetrics
    
    async def calculate_overall_completeness(
        self,
        structural: StructuralMetrics,
        trust: TrustMetrics,
        connectivity: ConnectivityMetrics
    ) -> float
    
    async def recommend_optimizations(
        self,
        metrics: CombinedMetrics
    ) -> List[OptimizationRecommendation]
```

### Data Structures
```python
@dataclass
class GraphSnapshot:
    nodes: Dict[str, NodeData]
    edges: Dict[str, EdgeData]
    timestamp: datetime
    
@dataclass
class StructuralMetrics:
    completeness_score: float
    total_nodes: int
    total_edges: int
    average_degree: float
    isolated_nodes: int
    
@dataclass
class OptimizationRecommendation:
    type: OptimizationType
    priority: float
    description: str
    expected_impact: float
    implementation_cost: float
```

### Extraction Benefits
- **Monitoring**: Dedicated service for graph health monitoring
- **Performance**: Can run as background service for continuous analysis
- **Alerting**: Can trigger alerts for graph quality degradation
- **Reporting**: Specialized reporting and visualization capabilities

---

## 3. Node Proposal Service (Priority 3)

### Service Definition
**Purpose**: Generates proposals for missing nodes based on graph analysis and neighboring concepts.

### Current Code Location
- **File**: `core/hyperrag/cognitive/graph_fixer.py`
- **Lines**: 659-704 (node proposal generation)
- **Dependencies**: TrustGraph interface

### API Specification
```python
class NodeProposalService:
    async def propose_missing_nodes(
        self,
        gap: DetectedGap,
        graph_context: GraphContext
    ) -> List[ProposedNode]
    
    async def analyze_neighboring_concepts(
        self,
        node_ids: List[str],
        graph_context: GraphContext
    ) -> Set[str]
    
    async def generate_bridging_concepts(
        self,
        source_concepts: List[str],
        target_concepts: List[str]
    ) -> List[BridgingConcept]
    
    async def score_node_proposals(
        self,
        proposals: List[ProposedNode],
        graph_context: GraphContext
    ) -> List[ProposedNode]
```

### Data Structures
```python
@dataclass
class GraphContext:
    local_subgraph: SubgraphData
    concept_embeddings: Dict[str, np.ndarray]
    trust_scores: Dict[str, float]
    
@dataclass
class BridgingConcept:
    concept_name: str
    bridged_concepts: List[str]
    confidence: float
    utility_score: float
```

### Extraction Benefits
- **Specialization**: Can focus on advanced concept generation algorithms
- **A/B Testing**: Can test different proposal strategies
- **Learning**: Dedicated learning from proposal acceptance/rejection
- **Optimization**: Can be optimized for specific domain knowledge

---

## 4. Relationship Analysis Service (Priority 4)

### Service Definition
**Purpose**: Analyzes missing relationships, proposes connections between nodes, and strengthens weak connections.

### Current Code Location
- **File**: `core/hyperrag/cognitive/graph_fixer.py`
- **Lines**: 706-733 (relationship proposals), 284-294 (relationship strengthening)
- **Dependencies**: TrustGraph interface

### API Specification
```python
class RelationshipAnalysisService:
    async def propose_missing_relationships(
        self,
        gap: DetectedGap,
        graph_context: GraphContext
    ) -> List[ProposedRelationship]
    
    async def analyze_relationship_strength(
        self,
        node_pairs: List[Tuple[str, str]],
        graph_context: GraphContext
    ) -> Dict[Tuple[str, str], float]
    
    async def suggest_bridge_relationships(
        self,
        isolated_clusters: List[List[str]],
        graph_context: GraphContext
    ) -> List[ProposedRelationship]
    
    async def optimize_relationship_types(
        self,
        existing_relationships: List[GraphEdge]
    ) -> List[RelationshipOptimization]
```

### Extraction Benefits
- **Graph Theory**: Can implement advanced graph algorithms
- **Pattern Recognition**: Specialized relationship pattern detection
- **Domain Adaptation**: Can adapt to specific relationship types
- **Performance**: Optimized graph traversal algorithms

---

## 5. Gap Detection Service (Priority 5 - Complex Extraction)

### Service Definition
**Purpose**: Detects various types of knowledge gaps using multiple analysis methods.

### Current Code Location
- **File**: `core/hyperrag/cognitive/graph_fixer.py`
- **Lines**: 232-261 (main detection), 398-657 (individual methods)
- **Dependencies**: TrustGraph, VectorEngine, NumPy

### API Specification
```python
class GapDetectionService:
    async def detect_all_gaps(
        self,
        query: Optional[str] = None,
        retrieved_info: Optional[List[Any]] = None,
        focus_area: Optional[str] = None,
        methods: Optional[List[str]] = None
    ) -> List[DetectedGap]
    
    async def detect_structural_gaps(
        self,
        graph_snapshot: GraphSnapshot
    ) -> List[DetectedGap]
    
    async def detect_semantic_gaps(
        self,
        query: Optional[str],
        focus_area: Optional[str],
        embeddings_context: EmbeddingsContext
    ) -> List[DetectedGap]
    
    async def detect_trust_inconsistencies(
        self,
        graph_snapshot: GraphSnapshot
    ) -> List[DetectedGap]
    
    async def detect_connectivity_gaps(
        self,
        graph_snapshot: GraphSnapshot
    ) -> List[DetectedGap]
```

### Complex Dependencies
```python
@dataclass
class EmbeddingsContext:
    node_embeddings: Dict[str, np.ndarray]
    similarity_threshold: float
    vector_engine_interface: VectorEngineProtocol
    
@dataclass  
class DetectionConfiguration:
    similarity_threshold: float = 0.7
    trust_variance_threshold: float = 0.2
    connectivity_threshold_ratio: float = 0.5
    min_nodes_for_analysis: int = 3
```

### Extraction Challenges
- **High Coupling**: Deep integration with graph structures
- **Multi-Method Coordination**: Complex orchestration of detection methods
- **Performance**: Computationally intensive operations
- **State Management**: Needs access to full graph state

### Migration Strategy
1. **Phase 1**: Extract detection methods as separate functions
2. **Phase 2**: Create service interface with graph snapshot pattern  
3. **Phase 3**: Implement event-driven gap detection triggers
4. **Phase 4**: Add streaming/incremental detection capabilities

---

## Implementation Priority Matrix

| Service | Complexity | Coupling | Benefits | Priority |
|---------|------------|----------|----------|----------|
| ML Inference | Low | Low | High | 1 |
| Graph Optimization | Medium | Medium | High | 2 |  
| Node Proposal | Medium | High | Medium | 3 |
| Relationship Analysis | Medium | High | Medium | 4 |
| Gap Detection | High | Very High | High | 5 |

## Cross-Service Communication Patterns

```
Gap Detection Service
        ↓ (DetectedGap)
    ┌─→ Node Proposal Service 
    │   ↓ (ProposedNode)
    └─→ Relationship Analysis Service
        ↓ (ProposedRelationship)
        
ML Inference Service ←→ All Services (similarity, confidence)
Graph Optimization Service ←→ Gap Detection (metrics, analysis)
```

## Shared Infrastructure Requirements

1. **Graph Snapshot Service**: Provides consistent graph state snapshots
2. **Caching Layer**: Redis/Memcached for similarity computations and metrics
3. **Message Queue**: For async processing and service communication
4. **Metrics Collection**: Prometheus/monitoring for service performance
5. **Configuration Service**: Centralized configuration management

This specification provides the technical foundation for extracting services from the GraphFixer system while maintaining functionality and improving scalability.