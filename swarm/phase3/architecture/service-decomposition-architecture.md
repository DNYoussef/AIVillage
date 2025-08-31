# Phase 3 Service Decomposition Architecture
## Detailed Service Design Specifications

### Service Architecture Overview

This document provides detailed architectural specifications for the decomposed services from the three God classes. Each service is designed with clean interfaces, minimal dependencies, and clear responsibilities.

## FogCoordinator Service Decomposition

### 1. FogOrchestrationService
```python
# Primary Responsibilities:
# - System lifecycle management
# - Component coordination
# - Health monitoring

class FogOrchestrationService:
    """Core orchestration service for fog computing system."""
    
    # Key Methods (estimated 8-10 methods):
    async def start_system() -> bool
    async def stop_system() -> bool  
    async def get_system_status() -> SystemStatus
    async def coordinate_components() -> bool
    async def handle_component_failure(component: str) -> bool
    async def perform_health_check() -> HealthStatus
    
    # Dependencies:
    # - Service Registry (for component discovery)
    # - Event Bus (for inter-service communication)
    # - Configuration Manager (for system settings)
```

### 2. FogHarvestingService
```python
# Primary Responsibilities:
# - Mobile device registration
# - Compute harvesting coordination
# - Resource allocation

class FogHarvestingService:
    """Manages mobile compute harvesting operations."""
    
    # Key Methods (estimated 7-9 methods):
    async def register_device(device_info: DeviceInfo) -> bool
    async def start_harvesting(device_id: str) -> bool
    async def stop_harvesting(device_id: str) -> bool
    async def get_harvest_status(device_id: str) -> HarvestStatus
    async def update_harvest_policy(policy: HarvestPolicy) -> bool
    
    # Dependencies:
    # - MobileResourceManager (device management)
    # - FogHarvestManager (harvesting operations)
    # - Event Bus (status notifications)
```

### 3. FogMarketplaceService
```python
# Primary Responsibilities:
# - Service marketplace integration
# - SLA tier management
# - Resource pricing

class FogMarketplaceService:
    """Manages fog computing marketplace operations."""
    
    # Key Methods (estimated 8-10 methods):
    async def register_service(service: ServiceInfo) -> bool
    async def discover_services(criteria: SearchCriteria) -> List[ServiceInfo]
    async def manage_sla_tiers(tier: SLATier) -> bool
    async def process_marketplace_request(request: MarketRequest) -> Response
    async def update_pricing(pricing: PricingInfo) -> bool
    
    # Dependencies:
    # - FogMarketplace (marketplace operations)
    # - EnhancedSLATierManager (SLA management)
    # - Pricing Manager (cost calculations)
```

### 4. FogPrivacyService
```python
# Primary Responsibilities:
# - Privacy policy management
# - Onion routing coordination
# - Hidden service hosting

class FogPrivacyService:
    """Manages privacy and anonymity features."""
    
    # Key Methods (estimated 6-8 methods):
    async def create_hidden_service(config: HiddenServiceConfig) -> str
    async def manage_onion_routing(request: RoutingRequest) -> bool
    async def enforce_privacy_policy(policy: PrivacyPolicy) -> bool
    async def get_privacy_status() -> PrivacyStatus
    
    # Dependencies:
    # - OnionRouter (routing operations)
    # - Privacy Policy Engine (policy enforcement)
    # - Event Bus (privacy events)
```

### 5. FogTokenomicsService
```python
# Primary Responsibilities:
# - Token distribution
# - Reward calculations
# - Economic incentives

class FogTokenomicsService:
    """Manages token economics and rewards."""
    
    # Key Methods (estimated 6-8 methods):
    async def distribute_rewards(participants: List[str]) -> bool
    async def calculate_token_rewards(activity: ActivityInfo) -> TokenAmount
    async def manage_token_system() -> bool
    async def get_token_balance(user_id: str) -> TokenBalance
    async def process_token_transaction(tx: Transaction) -> bool
    
    # Dependencies:
    # - FogTokenSystem (token operations)
    # - Activity Monitor (reward calculations)
    # - Event Bus (transaction events)
```

### 6. FogSystemStatsService
```python
# Primary Responsibilities:
# - Performance metrics
# - System statistics
# - Health monitoring

class FogSystemStatsService:
    """Collects and manages system statistics."""
    
    # Key Methods (estimated 5-7 methods):
    async def collect_metrics() -> MetricsData
    async def generate_stats_report() -> StatsReport
    async def track_performance(component: str) -> PerformanceData
    async def get_system_health() -> HealthMetrics
    
    # Dependencies:
    # - Metrics Collector (data gathering)
    # - Statistics Engine (data processing)
    # - Storage Service (metrics persistence)
```

## FogOnionCoordinator Service Decomposition

### 1. PrivacyTaskService
```python
# Primary Responsibilities:
# - Privacy-aware task submission
# - Task routing with privacy requirements
# - Privacy validation

class PrivacyTaskService:
    """Manages privacy-aware task processing."""
    
    # Key Methods (estimated 9-11 methods):
    async def submit_task(task: PrivacyAwareTask) -> bool
    async def validate_privacy_requirements(task: PrivacyAwareTask) -> bool
    async def route_task_privately(task: PrivacyAwareTask) -> bool
    async def get_task_status(task_id: str) -> TaskStatus
    async def cancel_task(task_id: str) -> bool
    
    # Dependencies:
    # - OnionCircuitService (circuit assignment)
    # - Privacy Validator (requirement validation)
    # - Task Router (routing logic)
```

### 2. OnionCircuitService
```python
# Primary Responsibilities:
# - Circuit creation and management
# - Circuit pool maintenance
# - Circuit rotation

class OnionCircuitService:
    """Manages onion routing circuits."""
    
    # Key Methods (estimated 10-12 methods):
    async def create_circuit(privacy_level: PrivacyLevel) -> OnionCircuit
    async def maintain_circuit_pool() -> bool
    async def rotate_circuits() -> bool
    async def assign_circuit_to_task(task_id: str) -> OnionCircuit
    async def cleanup_expired_circuits() -> bool
    
    # Dependencies:
    # - OnionRouter (circuit operations)
    # - Circuit Pool Manager (pool operations)
    # - Privacy Policy Engine (privacy requirements)
```

### 3. HiddenServiceManagementService
```python
# Primary Responsibilities:
# - Hidden service hosting
# - Service registration
# - Access control

class HiddenServiceManagementService:
    """Manages hidden service operations."""
    
    # Key Methods (estimated 8-10 methods):
    async def create_hidden_service(config: ServiceConfig) -> HiddenService
    async def register_service(service: PrivacyAwareService) -> bool
    async def manage_access_control(service_id: str, rules: AccessRules) -> bool
    async def get_service_by_address(address: str) -> PrivacyAwareService
    async def update_service_config(service_id: str, config: ServiceConfig) -> bool
    
    # Dependencies:
    # - Hidden Service Manager (service operations)
    # - Access Control Engine (permission management)
    # - Service Registry (service discovery)
```

### 4. PrivacyGossipService
```python
# Primary Responsibilities:
# - Secure gossip protocols
# - Privacy-preserving communication
# - Mixnet integration

class PrivacyGossipService:
    """Manages privacy-preserving inter-node communication."""
    
    # Key Methods (estimated 6-8 methods):
    async def send_private_gossip(message: GossipMessage) -> bool
    async def receive_gossip_message(message: GossipMessage) -> bool
    async def initialize_mixnet_client() -> bool
    async def manage_gossip_routing() -> bool
    
    # Dependencies:
    # - Mixnet Client (anonymous communication)
    # - Gossip Protocol Engine (message routing)
    # - Privacy Policy Engine (communication policies)
```

## GraphFixer Service Decomposition

### 1. GapDetectionService
```python
# Primary Responsibilities:
# - Knowledge gap identification
# - Structural analysis
# - Semantic gap detection

class GapDetectionService:
    """Detects gaps in knowledge graphs."""
    
    # Key Methods (estimated 10-12 methods):
    async def detect_structural_gaps() -> List[DetectedGap]
    async def detect_semantic_gaps(query: str) -> List[DetectedGap]
    async def detect_path_gaps(info: List[Any]) -> List[DetectedGap]
    async def detect_trust_inconsistencies() -> List[DetectedGap]
    async def detect_connectivity_gaps() -> List[DetectedGap]
    async def rank_gaps_by_priority(gaps: List[DetectedGap]) -> List[DetectedGap]
    
    # Dependencies:
    # - Graph Analysis Engine (structural analysis)
    # - Semantic Analyzer (semantic processing)
    # - Trust Scorer (trust analysis)
```

### 2. KnowledgeProposalService
```python
# Primary Responsibilities:
# - Solution proposal generation
# - Node and relationship proposals
# - Proposal ranking

class KnowledgeProposalService:
    """Generates proposals to fill knowledge gaps."""
    
    # Key Methods (estimated 8-10 methods):
    async def propose_missing_nodes(gap: DetectedGap) -> List[ProposedNode]
    async def propose_missing_relationships(gap: DetectedGap) -> List[ProposedRelationship]
    async def rank_node_proposals(proposals: List[ProposedNode]) -> List[ProposedNode]
    async def rank_relationship_proposals(proposals: List[ProposedRelationship]) -> List[ProposedRelationship]
    async def generate_solution_alternatives(gap: DetectedGap) -> List[Solution]
    
    # Dependencies:
    # - Proposal Generator (solution generation)
    # - Ranking Engine (proposal scoring)
    # - Knowledge Base (existing knowledge)
```

### 3. GraphAnalysisService
```python
# Primary Responsibilities:
# - Graph completeness analysis
# - Connectivity analysis
# - Trust distribution analysis

class GraphAnalysisService:
    """Analyzes graph structure and properties."""
    
    # Key Methods (estimated 9-11 methods):
    async def analyze_structural_completeness() -> Dict[str, Any]
    async def analyze_semantic_completeness() -> Dict[str, Any]
    async def analyze_trust_distribution() -> Dict[str, Any]
    async def analyze_connectivity_patterns() -> Dict[str, Any]
    async def calculate_graph_metrics() -> GraphMetrics
    
    # Dependencies:
    # - Graph Theory Engine (structural analysis)
    # - Statistical Analyzer (distribution analysis)
    # - Connectivity Analyzer (path analysis)
```

### 4. ValidationService
```python
# Primary Responsibilities:
# - Proposal validation
# - Quality assessment
# - Learning from feedback

class ValidationService:
    """Validates knowledge proposals and learns from feedback."""
    
    # Key Methods (estimated 7-9 methods):
    async def validate_proposal(proposal: Union[ProposedNode, ProposedRelationship]) -> ValidationResult
    async def assess_proposal_quality(proposal: Any) -> QualityScore
    async def learn_from_validation(proposal: Any, accepted: bool) -> bool
    async def update_validation_models() -> bool
    async def get_validation_confidence(proposal: Any) -> float
    
    # Dependencies:
    # - Validation Engine (proposal assessment)
    # - Machine Learning Models (quality prediction)
    # - Feedback Processor (learning system)
```

### 5. GraphMetricsService
```python
# Primary Responsibilities:
# - Performance metrics collection
# - Statistical reporting
# - System monitoring

class GraphMetricsService:
    """Collects and manages graph processing metrics."""
    
    # Key Methods (estimated 6-8 methods):
    async def collect_processing_metrics() -> MetricsData
    async def generate_performance_report() -> PerformanceReport
    async def track_gap_detection_performance() -> PerformanceData
    async def monitor_proposal_success_rates() -> SuccessMetrics
    async def calculate_system_efficiency() -> EfficiencyMetrics
    
    # Dependencies:
    # - Metrics Collector (data gathering)
    # - Performance Analyzer (metric processing)
    # - Reporting Engine (report generation)
```

## Inter-Service Communication Architecture

### Communication Patterns

#### 1. Event-Driven Architecture
```
Services communicate through async events:
- Service A publishes events
- Service B subscribes to relevant events
- Loose coupling through event bus
```

#### 2. Request-Response Pattern
```
Direct service-to-service calls for:
- Synchronous operations
- Critical path operations
- Data retrieval
```

#### 3. Circuit Breaker Pattern
```
Resilient communication with:
- Automatic failure detection
- Fallback mechanisms
- Recovery monitoring
```

### Service Interface Standards

#### 1. Common Interface Pattern
```python
@dataclass
class ServiceConfig:
    """Standard service configuration."""
    service_name: str
    version: str
    dependencies: List[str]
    health_check_interval: int = 30

class BaseService:
    """Base class for all services."""
    
    async def start(self) -> bool
    async def stop(self) -> bool
    async def health_check() -> HealthStatus
    async def get_metrics() -> ServiceMetrics
```

#### 2. Dependency Injection
```python
class ServiceRegistry:
    """Central service registry for dependency injection."""
    
    def register_service(self, service: BaseService) -> None
    def get_service(self, service_type: Type) -> BaseService
    def inject_dependencies(self, service: BaseService) -> None
```

### Data Flow Architecture

#### FogCoordinator Data Flow:
```
External Request → FogOrchestrationService → [Service Selection] → 
Specific Service → Response Aggregation → External Response
```

#### FogOnionCoordinator Data Flow:
```
Privacy Task → PrivacyTaskService → OnionCircuitService → 
HiddenServiceManagementService → PrivacyGossipService → Task Completion
```

#### GraphFixer Data Flow:
```
Graph Analysis Request → GapDetectionService → KnowledgeProposalService → 
ValidationService → GraphAnalysisService → GraphMetricsService → Results
```

## Performance Optimization Strategy

### 1. Async Operation Patterns
```python
# Concurrent service operations
async def process_concurrent_requests():
    tasks = [
        service_a.process_request(req1),
        service_b.process_request(req2),
        service_c.process_request(req3)
    ]
    results = await asyncio.gather(*tasks)
    return aggregate_results(results)
```

### 2. Caching Strategy
```python
# Service-level caching
class CacheableService(BaseService):
    def __init__(self):
        self._cache = TTLCache(maxsize=1000, ttl=300)
    
    async def get_cached_result(self, key: str) -> Any:
        if key in self._cache:
            return self._cache[key]
        
        result = await self._compute_result(key)
        self._cache[key] = result
        return result
```

### 3. Connection Pooling
```python
# Inter-service connection management
class ServiceConnectionPool:
    def __init__(self, max_connections: int = 10):
        self._pool = asyncio.Queue(maxsize=max_connections)
        self._active_connections = set()
    
    async def get_connection(self, service: str) -> ServiceConnection:
        # Pool management logic
        pass
```

## Testing Strategy

### Unit Testing Approach
```python
# Service-specific unit tests
class TestFogOrchestrationService:
    @pytest.fixture
    def service(self):
        return FogOrchestrationService(mock_dependencies=True)
    
    async def test_start_system(self, service):
        result = await service.start_system()
        assert result is True
    
    async def test_system_failure_handling(self, service):
        # Test failure scenarios
        pass
```

### Integration Testing Strategy
```python
# Cross-service integration tests
class TestServiceIntegration:
    async def test_fog_coordinator_workflow(self):
        # Test complete workflow across services
        orchestration = FogOrchestrationService()
        harvesting = FogHarvestingService()
        marketplace = FogMarketplaceService()
        
        # Test service interactions
        pass
```

### Performance Testing Framework
```python
# Performance benchmarking
class ServicePerformanceBenchmark:
    async def benchmark_service_performance(self, service: BaseService):
        start_time = time.time()
        
        # Execute performance test
        await service.process_test_workload()
        
        execution_time = time.time() - start_time
        return PerformanceMetrics(execution_time=execution_time)
```

## Deployment Strategy

### Service Containerization
```dockerfile
# Standard service container template
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service/ .
CMD ["python", "-m", "service.main"]

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

### Service Discovery Configuration
```yaml
# Service mesh configuration
services:
  fog_orchestration:
    port: 8001
    health_check: "/health"
    dependencies: ["service_registry", "event_bus"]
  
  fog_harvesting:
    port: 8002
    health_check: "/health"
    dependencies: ["fog_orchestration", "resource_manager"]
```

This architectural specification provides the detailed blueprint for implementing the service decomposition strategy, ensuring clean separation of concerns while maintaining system performance and reliability.