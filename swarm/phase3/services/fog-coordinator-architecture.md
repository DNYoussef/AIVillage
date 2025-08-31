# Fog Coordinator 6-Service Architecture Design

## Executive Summary

This document specifies the refactoring of the 754-line `fog_coordinator.py` into 6 independent services to address:

- **High Coupling**: 39.81 coupling score with 8+ subsystem dependencies
- **Circular Dependencies**: Complex initialization with 15+ methods
- **Mixed Concerns**: Harvesting, marketplace, privacy, and tokenomics in single class
- **Maintainability**: Monolithic structure inhibiting independent deployment

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Fog Coordinator Services                    │
├─────────────────────────────────────────────────────────────┤
│  FogOrchestrationService (110 lines) - System Lifecycle    │
│  │ ├── Service Discovery & Registration                    │
│  │ ├── Health Monitoring & Circuit Breaking               │
│  │ └── Service-to-Service Communication                   │
├─────────────────────────────────────────────────────────────┤
│  FogHarvestingService (90 lines) - Mobile Compute          │
│  │ ├── Device Discovery & Capability Assessment           │
│  │ ├── Resource Allocation & Burst Coordination          │
│  │ └── Participant Selection & Management                 │
├─────────────────────────────────────────────────────────────┤
│  FogMarketplaceService (110 lines) - Market Integration    │
│  │ ├── Pricing & Resource Valuation                      │
│  │ ├── Auction & Bidding Logic                           │
│  │ └── Revenue Distribution & Settlement                  │
├─────────────────────────────────────────────────────────────┤
│  FogPrivacyService (100 lines) - Privacy & Security        │
│  │ ├── Differential Privacy & Budget Management          │
│  │ ├── Onion Routing & Hidden Services                   │
│  │ └── Secure Aggregation & Byzantine Detection          │
├─────────────────────────────────────────────────────────────┤
│  FogTokenomicsService (90 lines) - Token Economics         │
│  │ ├── Credit System Integration                          │
│  │ ├── Compute Mining & Reward Distribution              │
│  │ └── Wallet Management & Transaction Processing        │
├─────────────────────────────────────────────────────────────┤
│  FogSystemStatsService (70 lines) - Metrics & Monitoring   │
│  │ ├── Performance Metrics Collection                     │
│  │ ├── Health Status & Alerting                          │
│  │ └── Historical Analytics & Reporting                  │
└─────────────────────────────────────────────────────────────┘
```

## Service Specifications

### 1. FogOrchestrationService (~110 lines)

**Primary Responsibility**: System lifecycle management and service coordination

**Core Responsibilities**:
- Service discovery and registration
- Inter-service communication and message routing
- Health monitoring and circuit breaking
- Configuration management and dependency injection

**Interface Contract**:
```python
class FogOrchestrationService:
    async def initialize_services(self) -> ServiceRegistry
    async def register_service(self, service_id: str, service_instance: Any) -> bool
    async def discover_service(self, service_type: str) -> Optional[ServiceInstance]
    async def health_check_all_services(self) -> Dict[str, HealthStatus]
    async def shutdown_services(self, graceful: bool = True) -> bool
    
    # Event routing between services
    async def route_event(self, event: ServiceEvent) -> None
    async def broadcast_event(self, event: ServiceEvent) -> None
```

**Dependencies**:
- P2P Node (external)
- Configuration System (external)
- Logging System (external)

**Configuration Requirements**:
```yaml
orchestration:
  service_discovery:
    timeout_seconds: 30
    retry_attempts: 3
  health_checks:
    interval_seconds: 60
    circuit_breaker_threshold: 5
  messaging:
    max_queue_size: 1000
    retry_backoff_ms: 500
```

**Performance Targets**:
- Service startup: < 5 seconds
- Health check cycle: < 1 second
- Event routing latency: < 10ms
- Memory footprint: < 50MB

### 2. FogHarvestingService (~90 lines)

**Primary Responsibility**: Mobile compute harvesting and resource management

**Core Responsibilities**:
- Device discovery and capability assessment
- Participant selection and management
- Resource allocation and burst coordination
- Device suitability evaluation

**Interface Contract**:
```python
class FogHarvestingService:
    async def discover_devices(self) -> List[DeviceCapability]
    async def assess_device_suitability(self, device_id: str) -> SuitabilityScore
    async def select_participants(self, requirements: ResourceRequirements) -> List[str]
    async def allocate_resources(self, participants: List[str], workload: ComputeWorkload) -> AllocationPlan
    async def coordinate_burst(self, burst_request: BurstRequest) -> str  # returns burst_id
    async def monitor_device_health(self, device_id: str) -> DeviceHealth
```

**Dependencies**:
- FogOrchestrationService (service discovery)
- P2P Network (device communication)

**Configuration Requirements**:
```yaml
harvesting:
  device_discovery:
    scan_interval_seconds: 120
    capability_cache_ttl: 300
  participant_selection:
    min_trust_score: 0.6
    min_battery_percent: 50
    min_ram_mb: 2048
  resource_allocation:
    burst_timeout_minutes: 30
    max_concurrent_bursts: 10
```

**Performance Targets**:
- Device discovery cycle: < 30 seconds
- Participant selection: < 5 seconds
- Resource allocation: < 2 seconds
- Concurrent device monitoring: 1000+ devices

### 3. FogMarketplaceService (~110 lines)

**Primary Responsibility**: Marketplace integration and economic coordination

**Core Responsibilities**:
- Resource pricing and valuation
- Auction mechanism and bidding logic
- Revenue distribution and settlement
- Market analytics and reporting

**Interface Contract**:
```python
class FogMarketplaceService:
    async def price_resources(self, resource_spec: ResourceSpec) -> PricingQuote
    async def create_auction(self, resource_request: ResourceRequest) -> AuctionId
    async def submit_bid(self, auction_id: str, bid: ResourceBid) -> bool
    async def settle_auction(self, auction_id: str) -> AuctionResult
    async def distribute_revenue(self, completed_job: CompletedJob) -> RevenueDistribution
    async def get_market_metrics(self) -> MarketMetrics
```

**Dependencies**:
- FogTokenomicsService (payment processing)
- FogHarvestingService (resource availability)
- FogOrchestrationService (service coordination)

**Configuration Requirements**:
```yaml
marketplace:
  pricing:
    base_compute_rate_per_gflop: 0.001
    network_multiplier: 1.2
    storage_rate_per_gb: 0.0001
  auctions:
    auction_duration_minutes: 10
    min_bid_increment: 0.01
    reserve_price_ratio: 0.7
  revenue_sharing:
    platform_fee_percent: 5
    provider_share_percent: 85
    validator_share_percent: 10
```

**Performance Targets**:
- Pricing calculation: < 100ms
- Auction processing: < 5 seconds
- Revenue settlement: < 1 second
- Market data updates: Real-time

### 4. FogPrivacyService (~100 lines)

**Primary Responsibility**: Privacy preservation and security enforcement

**Core Responsibilities**:
- Differential privacy and budget management
- Onion routing and hidden services
- Secure aggregation protocols
- Byzantine fault detection and mitigation

**Interface Contract**:
```python
class FogPrivacyService:
    async def initialize_privacy_budgets(self, participants: List[str]) -> Dict[str, float]
    async def apply_differential_privacy(self, data: torch.Tensor, epsilon: float) -> torch.Tensor
    async def create_hidden_service(self, service_spec: HiddenServiceSpec) -> str
    async def setup_onion_routing(self, route_spec: RouteSpec) -> OnionRoute
    async def secure_aggregate(self, gradients: Dict[str, torch.Tensor]) -> torch.Tensor
    async def detect_byzantine_behavior(self, participant_data: List[ParticipantData]) -> List[str]
```

**Dependencies**:
- FogOrchestrationService (service coordination)
- P2P Network (secure communications)
- Cryptography Libraries (external)

**Configuration Requirements**:
```yaml
privacy:
  differential_privacy:
    default_epsilon: 1.0
    default_delta: 1e-5
    budget_per_participant: 10.0
  secure_aggregation:
    threshold_participants: 3
    byzantine_tolerance: 0.33
  hidden_services:
    max_services_per_node: 5
    replication_factor: 3
    access_fee_tokens: 0.001
```

**Performance Targets**:
- DP noise application: < 50ms
- Secure aggregation: < 10 seconds
- Byzantine detection: < 5 seconds
- Hidden service deployment: < 30 seconds

### 5. FogTokenomicsService (~90 lines)

**Primary Responsibility**: Token economics and payment processing

**Core Responsibilities**:
- Credit system integration and management
- Compute mining and reward distribution
- Wallet operations and transaction processing
- Economic incentive alignment

**Interface Contract**:
```python
class FogTokenomicsService:
    async def initialize_wallet(self, wallet_id: str) -> WalletInfo
    async def process_payment(self, transaction: Transaction) -> TransactionResult
    async def calculate_mining_rewards(self, work_proof: WorkProof) -> float
    async def distribute_rewards(self, rewards: Dict[str, float]) -> bool
    async def verify_balance(self, wallet_id: str, amount: float) -> bool
    async def get_transaction_history(self, wallet_id: str) -> List[Transaction]
```

**Dependencies**:
- Credit System (external)
- FogOrchestrationService (service coordination)
- Blockchain/Ledger (external)

**Configuration Requirements**:
```yaml
tokenomics:
  compute_mining:
    base_reward_per_gflop_hour: 0.1
    quality_bonus_multiplier: 1.5
    reliability_threshold: 0.9
  transaction_processing:
    confirmation_blocks: 3
    gas_price_gwei: 20
    max_retry_attempts: 5
  wallet_management:
    default_balance: 10.0
    minimum_balance: 0.01
    auto_topup_threshold: 1.0
```

**Performance Targets**:
- Payment processing: < 2 seconds
- Reward calculation: < 1 second
- Balance verification: < 100ms
- Transaction throughput: 1000+ TPS

### 6. FogSystemStatsService (~70 lines)

**Primary Responsibility**: Metrics collection and system monitoring

**Core Responsibilities**:
- Performance metrics collection and aggregation
- Health status monitoring and alerting
- Historical analytics and trend analysis
- System performance reporting

**Interface Contract**:
```python
class FogSystemStatsService:
    async def collect_metrics(self) -> SystemMetrics
    async def record_event(self, event: MetricEvent) -> None
    async def get_performance_stats(self, time_window: TimeWindow) -> PerformanceStats
    async def generate_health_report(self) -> HealthReport
    async def setup_alerts(self, alert_config: AlertConfig) -> None
    async def export_analytics(self, format: ExportFormat) -> bytes
```

**Dependencies**:
- All other services (metrics collection)
- FogOrchestrationService (service coordination)
- Time Series Database (external)

**Configuration Requirements**:
```yaml
system_stats:
  metrics_collection:
    interval_seconds: 30
    batch_size: 100
    retention_days: 30
  performance_monitoring:
    cpu_threshold_percent: 80
    memory_threshold_percent: 85
    latency_threshold_ms: 1000
  alerting:
    email_notifications: true
    slack_webhook_url: "https://..."
    alert_cooldown_minutes: 15
```

**Performance Targets**:
- Metrics collection: < 5 seconds
- Real-time dashboard updates: < 1 second
- Historical query response: < 10 seconds
- Storage efficiency: 90%+ compression

## Service Integration Patterns

### Dependency Breaking Strategy

**Original Circular Dependencies**:
```
FogCoordinator → P2P → Evolution → FogCoordinator
FogCoordinator → Tokenomics → Credit → FogCoordinator
FogCoordinator → Privacy → Secure → FogCoordinator
```

**New Dependency Flow**:
```
FogOrchestrationService (coordinator)
├── Discovers → FogHarvestingService
├── Discovers → FogMarketplaceService  
├── Discovers → FogPrivacyService
├── Discovers → FogTokenomicsService
└── Discovers → FogSystemStatsService

Inter-service communication via:
- Event-driven messaging (async)
- Service registry lookup (sync)
- Circuit breaker pattern (resilience)
```

### Communication Patterns

**1. Event-Driven Architecture**:
```python
# Example: Device discovered event
device_discovered = DeviceDiscoveredEvent(
    device_id="node_123",
    capabilities=capabilities,
    timestamp=time.time()
)

# Harvesting service publishes
await orchestration.broadcast_event(device_discovered)

# Marketplace service subscribes and updates pricing
# Privacy service subscribes and initializes budgets
```

**2. Service Registry Pattern**:
```python
# Service registration
await orchestration.register_service("harvesting", harvesting_service)
await orchestration.register_service("marketplace", marketplace_service)

# Service discovery
marketplace = await orchestration.discover_service("marketplace")
pricing = await marketplace.price_resources(resource_spec)
```

**3. Circuit Breaker Pattern**:
```python
# Resilient service calls
@circuit_breaker(failure_threshold=5, timeout=30)
async def call_marketplace_service(self, request):
    marketplace = await self.orchestration.discover_service("marketplace")
    return await marketplace.process_request(request)
```

## Migration Strategy

### Phase 1: Service Extraction (Week 1)
1. Create service interface contracts
2. Extract core business logic from fog_coordinator.py
3. Implement service stubs with basic functionality
4. Set up service registry and discovery

### Phase 2: Dependency Resolution (Week 2)
1. Replace direct dependencies with service calls
2. Implement event-driven communication
3. Add circuit breaker patterns
4. Update tests for individual services

### Phase 3: Integration Testing (Week 3)
1. End-to-end integration tests
2. Performance benchmarking
3. Load testing with realistic scenarios
4. Security and privacy validation

### Phase 4: Production Deployment (Week 4)
1. Blue-green deployment strategy
2. Service monitoring and alerting
3. Rollback procedures
4. Documentation and training

## Benefits & Success Metrics

### Technical Benefits
- **Maintainability**: Individual services < 110 lines each
- **Testability**: Isolated unit tests for each service
- **Scalability**: Independent horizontal scaling
- **Reliability**: Circuit breaker and retry patterns

### Performance Benefits
- **Coupling Reduction**: From 39.81 to < 10.0 target
- **Startup Time**: From 15+ methods to < 5 seconds
- **Memory Usage**: 30% reduction through service isolation
- **Throughput**: 2x improvement via parallel processing

### Operational Benefits
- **Independent Deployment**: Services deploy separately
- **Fault Isolation**: Service failures don't cascade
- **Team Ownership**: Clear service boundaries
- **Monitoring**: Granular service-level metrics

## Risk Mitigation

### Technical Risks
1. **Service Communication Overhead**: Mitigated by async messaging and local caching
2. **Consistency Issues**: Mitigated by event sourcing and eventual consistency patterns
3. **Service Discovery Failures**: Mitigated by service registry redundancy

### Operational Risks
1. **Deployment Complexity**: Mitigated by containerization and orchestration
2. **Monitoring Complexity**: Mitigated by centralized logging and tracing
3. **Team Coordination**: Mitigated by clear API contracts and integration tests

This architecture provides a solid foundation for breaking the fog_coordinator.py monolith into maintainable, scalable services while preserving all existing functionality and improving overall system reliability.