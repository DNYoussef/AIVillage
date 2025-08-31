# Comprehensive Refactoring Recommendations

## Executive Summary

The current `DistributedFederatedLearning` coordinator is a 754-line monolithic "God Object" that violates fundamental software architecture principles. This analysis recommends breaking it into 8 independent microservices with clear boundaries, reducing complexity by 80%+ while enabling independent scaling, development, and deployment.

**Current State**: Single monolithic class managing 7 unrelated domains  
**Target State**: 8 loosely-coupled microservices with event-driven architecture  
**Migration Timeline**: 4-phase approach over 3-6 months  
**Expected Benefits**: 80% complexity reduction, 3x development velocity, 90% fault isolation  

## Critical Issues Requiring Immediate Action

### 🔴 **CRITICAL: Circular Dependencies** 
```python
# Current problematic pattern
class BurstCoordinator:
    def __init__(self, federated_coordinator: DistributedFederatedLearning):
        self.fl_coordinator = federated_coordinator  # CIRCULAR DEPENDENCY

class DistributedFederatedLearning:
    def __init__(self, ...):
        self.burst_coordinator = BurstCoordinator(self)  # CREATES CYCLE
```

**Impact**: Prevents independent testing, creates memory leaks, makes service extraction impossible  
**Solution**: Implement dependency injection with interfaces

### 🔴 **CRITICAL: Mixed Domain Concerns**
- Federated Learning mixed with marketplace operations
- Privacy services embedded in ML training logic  
- Tokenomics scattered across multiple unrelated functions
- P2P transport mixed with business logic

**Impact**: Changes in one domain break others, impossible to scale independently  
**Solution**: Extract services with clear domain boundaries

### 🔴 **CRITICAL: Shared Mutable State**
```python
# Multiple services modify shared coordinator state
self.available_participants  # Accessed by FL, Burst, Hidden Services
self.fl_stats              # Updated by multiple unrelated components  
self.privacy_budgets       # Mixed with participant management
```

**Impact**: Race conditions, data corruption, inconsistent state  
**Solution**: Implement event-driven architecture with service-owned state

## Detailed Refactoring Plan

### Phase 1: Foundation Services (Weeks 1-4)
**Objective**: Extract core infrastructure services to enable other extractions

#### 1.1 Extract Device Registry Service
**Priority**: HIGHEST 🔴  
**Rationale**: All other services depend on device information

```python
# NEW: Device Registry Service
class DeviceRegistryService:
    """Central device and capability management"""
    
    async def register_device(self, device: DeviceInfo) -> DeviceId:
        """Register device with comprehensive capability assessment"""
        
    async def query_devices(self, criteria: DeviceCriteria) -> List[DeviceInfo]:
        """Query devices for any service without exposing FL internals"""
        
    async def update_reputation(self, device_id: DeviceId, service: str, score: float) -> None:
        """Update device reputation per service domain"""

# MIGRATION: Update existing code
class DistributedFederatedLearning:
    def __init__(self, device_registry: DeviceRegistryService, ...):
        self._device_registry = device_registry  # Dependency injection
        # Remove: self.available_participants
        
    async def _discover_participants(self) -> None:
        # OLD: Direct participant management
        # NEW: Use device registry
        criteria = DeviceCriteria(
            min_ram_mb=2048,
            min_trust_score=0.6,
            supports_ml=True
        )
        devices = await self._device_registry.query_devices(criteria)
```

**Benefits**:
- ✅ Eliminates shared participant data between unrelated services
- ✅ Enables independent device management and reputation tracking
- ✅ Provides foundation for other service extractions

#### 1.2 Extract Communication Service  
**Priority**: HIGH 🟡  
**Rationale**: Abstracts P2P transport from business logic

```python
# NEW: Communication Service
class CommunicationService:
    """Abstract network transport and messaging"""
    
    async def send_typed_message(self, recipient: NodeId, message: TypedMessage) -> bool:
        """Type-safe message sending with serialization"""
        
    async def subscribe_to_message_type(self, msg_type: Type[Message], handler: MessageHandler) -> None:
        """Type-safe message subscription"""

# MIGRATION: Replace P2P handlers
class FederatedLearningService:
    def __init__(self, communication: CommunicationService, ...):
        self._communication = communication
        
    async def initialize(self):
        # OLD: String-based message discrimination
        # NEW: Type-safe message handling  
        await self._communication.subscribe_to_message_type(
            GradientSubmissionMessage, 
            self._handle_gradient_submission
        )
```

#### 1.3 Extract Monitoring Service
**Priority**: MEDIUM 🟢  
**Rationale**: Enables observability for service extraction process

```python
# NEW: Monitoring Service  
class MonitoringService:
    """System observability and performance tracking"""
    
    async def track_operation(self, service: str, operation: str, duration: float) -> None:
        """Track operation performance across services"""
        
    async def record_business_metric(self, metric: BusinessMetric) -> None:
        """Record business KPIs (tokens earned, services deployed, etc.)"""

# MIGRATION: Replace scattered statistics
class FederatedLearningService:
    async def complete_training_round(self, round_id: str):
        start_time = time.time()
        try:
            # ... training logic ...
            await self._monitoring.track_operation(
                "federated_learning", "training_round", time.time() - start_time
            )
        except Exception as e:
            await self._monitoring.record_error("federated_learning", "training_round", str(e))
```

### Phase 2: Business Service Extraction (Weeks 5-8)
**Objective**: Extract business domain services with clear boundaries

#### 2.1 Extract Tokenomics Service
**Priority**: HIGH 🟡  
**Rationale**: Payment verification needed before marketplace extraction

```python
# NEW: Tokenomics Service
class TokenomicsService:
    """Token economics and payment processing"""
    
    async def verify_payment_capability(self, wallet: WalletId, amount: TokenAmount, service: str) -> PaymentCapability:
        """Verify payment with detailed capability response"""
        
    async def process_service_payment(self, payment: ServicePayment) -> PaymentResult:
        """Process payment with proper accounting and audit trail"""
        
    async def calculate_contribution_rewards(self, contributions: List[Contribution]) -> List[Reward]:
        """Calculate rewards based on contribution type and quality"""

# MIGRATION: Remove tokenomics from coordinator
class DistributedFederatedLearning:
    def __init__(self, tokenomics: TokenomicsService, ...):
        self._tokenomics = tokenomics
        # Remove: self.credit_system, self.compute_mining
        
    async def complete_training_round(self, round: TrainingRound):
        # OLD: Direct token calculation
        # NEW: Event-driven reward processing
        await self._event_bus.publish(
            TrainingRoundCompletedEvent(
                participants=round.participants,
                performance_metrics=round.metrics
            )
        )
```

#### 2.2 Extract Harvest Management Service
**Priority**: HIGH 🟡  
**Rationale**: Core business functionality currently missing/scattered

```python
# NEW: Harvest Management Service (currently missing most functionality)
class HarvestManagementService:
    """Mobile compute resource harvesting with safety policies"""
    
    async def start_harvest_session(self, device_id: DeviceId, policy: HarvestPolicy) -> HarvestSession:
        """Start battery-aware, thermal-safe harvesting session"""
        
    async def monitor_harvest_conditions(self, session_id: str) -> HarvestStatus:
        """Continuously monitor battery, thermal, and user activity"""
        
    async def calculate_session_rewards(self, session: CompletedHarvestSession) -> TokenAmount:
        """Calculate rewards based on actual contribution and quality"""

# IMPLEMENTATION: Add missing capabilities from harvest_manager.py
@dataclass
class HarvestPolicy:
    min_battery_percent: int = 20
    max_cpu_temp: float = 45.0
    require_charging: bool = True
    max_cpu_percent: float = 50.0
    thermal_throttle_temp: float = 55.0
    
@dataclass  
class HarvestSession:
    session_id: str
    device_id: DeviceId
    start_time: datetime
    policy: HarvestPolicy
    resources_contributed: ResourceMetrics
    safety_events: List[SafetyEvent]  # Thermal throttling, battery low, etc.
```

**Current Gap**: The existing coordinator lacks most harvest management functionality. This service needs to be built from scratch using the patterns in `harvest_manager.py`.

### Phase 3: Complex Business Services (Weeks 9-12)  
**Objective**: Extract marketplace and privacy services

#### 3.1 Extract Marketplace Service
**Priority**: MEDIUM 🟢  
**Rationale**: Complex service requiring multiple dependencies

```python
# NEW: Marketplace Service (expand BurstCoordinator capabilities)
class MarketplaceService:
    """Comprehensive fog computing marketplace"""
    
    def __init__(
        self, 
        device_registry: DeviceRegistryService,
        tokenomics: TokenomicsService,
        communication: CommunicationService
    ):
        self._device_registry = device_registry
        self._tokenomics = tokenomics
        self._communication = communication
        
    async def create_service_offering(self, provider: ProviderId, offering: ServiceOffering) -> OfferingId:
        """Create marketplace offering with SLA guarantees"""
        
    async def match_service_request(self, request: ServiceRequest) -> List[ServiceMatch]:
        """Intelligent matching with pricing and availability"""
        
    async def allocate_compute_burst(self, request: BurstRequest) -> BurstAllocation:
        """Enhanced burst allocation with better node selection"""

# MIGRATION: Extract BurstCoordinator functionality
# OLD: BurstCoordinator embedded in FL coordinator
# NEW: Independent marketplace service with proper interfaces
```

#### 3.2 Extract Privacy Service  
**Priority**: MEDIUM 🟢  
**Rationale**: Complex privacy protocols requiring specialized implementation

```python
# NEW: Privacy Service (expand HiddenServiceManager capabilities)
class PrivacyService:
    """Comprehensive privacy layer with onion routing"""
    
    async def create_onion_circuit(self, hops: int, circuit_type: CircuitType) -> OnionCircuit:
        """Create multi-hop circuit with proper key management"""
        
    async def deploy_hidden_service(self, deployment: HiddenServiceDeployment) -> HiddenServiceId:
        """Deploy hidden service with proper data sharding and encryption"""
        
    async def route_anonymous_traffic(self, circuit: OnionCircuit, traffic: bytes) -> bytes:
        """Route traffic through onion circuit with proper mixing"""

# MIGRATION: Extract HiddenServiceManager + add missing onion routing
# Current HiddenServiceManager has basic functionality
# Need to integrate with onion_routing.py for complete privacy layer
```

### Phase 4: Federated Learning Service Refinement (Weeks 13-16)
**Objective**: Refactor FL coordinator to focus on core ML responsibilities

#### 4.1 Remove Extracted Functionality
```python
# REFINED: Federated Learning Service (focused on ML only)
class FederatedLearningService:
    """Focused federated learning coordination"""
    
    def __init__(
        self,
        device_registry: DeviceRegistryService,
        communication: CommunicationService,  
        monitoring: MonitoringService,
        event_bus: EventBus
    ):
        # Clean dependency injection - no circular dependencies
        self._device_registry = device_registry
        self._communication = communication
        self._monitoring = monitoring
        self._event_bus = event_bus
        
        # REMOVED: tokenomics, marketplace, privacy services
        # REMOVED: P2P message handlers (now in communication service)
        # REMOVED: participant management (now in device registry)
        
    async def coordinate_training_round(self, config: TrainingConfig) -> TrainingResult:
        """Focus solely on ML training coordination"""
        
        # 1. Select participants via device registry
        participants = await self._select_ml_participants(config.participant_criteria)
        
        # 2. Coordinate training phases  
        round_result = await self._execute_training_phases(participants, config)
        
        # 3. Publish completion event for other services
        await self._event_bus.publish(
            TrainingRoundCompletedEvent(
                round_id=round_result.round_id,
                participants=[p.device_id for p in participants],
                model_metrics=round_result.metrics,
                privacy_budget_consumed=round_result.privacy_cost
            )
        )
        
        return round_result
```

#### 4.2 Implement Event-Driven Integration
```python  
# Event-driven integration with other services
class FederatedLearningService:
    async def _select_ml_participants(self, criteria: ParticipantCriteria) -> List[MLParticipant]:
        """Select participants focused on ML requirements"""
        
        # Query device registry for ML-capable devices
        device_criteria = DeviceCriteria(
            min_ram_mb=criteria.min_memory_mb,
            min_compute_score=criteria.min_compute_capability,
            supports_frameworks=criteria.required_frameworks,
            min_reputation_score=criteria.min_trust_level
        )
        
        suitable_devices = await self._device_registry.query_devices(device_criteria)
        
        # Convert to ML-specific participant objects
        participants = []
        for device in suitable_devices:
            participant = MLParticipant(
                device_id=device.device_id,
                ml_capabilities=device.ml_capabilities,
                privacy_budget=await self._get_privacy_budget(device.device_id)
            )
            participants.append(participant)
            
        return participants
```

## Service Interface Specifications

### 1. Device Registry Service Interface
```python
@protocol
class DeviceRegistryService:
    """Device and capability management service"""
    
    async def register_device(self, device: DeviceRegistration) -> DeviceId: ...
    async def update_capabilities(self, device_id: DeviceId, capabilities: DeviceCapabilities) -> bool: ...
    async def query_devices(self, criteria: DeviceCriteria) -> List[DeviceInfo]: ...
    async def update_reputation(self, device_id: DeviceId, service: str, score: ReputationScore) -> bool: ...
    async def get_device_health(self, device_id: DeviceId) -> DeviceHealthStatus: ...
    
@dataclass
class DeviceCriteria:
    min_ram_mb: Optional[int] = None
    min_cpu_cores: Optional[int] = None
    min_compute_score: Optional[float] = None
    required_capabilities: List[str] = field(default_factory=list)
    min_reputation_score: Optional[float] = None
    max_latency_ms: Optional[int] = None
    geographic_region: Optional[str] = None
    availability_required: bool = True
```

### 2. Communication Service Interface
```python
@protocol  
class CommunicationService:
    """Network communication and messaging service"""
    
    async def send_message(self, recipient: NodeId, message: TypedMessage) -> MessageResult: ...
    async def broadcast_message(self, recipients: List[NodeId], message: TypedMessage) -> BroadcastResult: ...
    async def subscribe_to_messages(self, message_type: Type[Message], handler: MessageHandler) -> SubscriptionId: ...
    async def create_secure_channel(self, peer: NodeId, encryption_level: EncryptionLevel) -> ChannelId: ...
    async def estimate_latency(self, peer: NodeId) -> LatencyEstimate: ...

# Type-safe message definitions
@dataclass
class GradientSubmissionMessage(TypedMessage):
    message_type: str = "gradient_submission"
    round_id: str
    device_id: str  
    gradients: Dict[str, bytes]  # Serialized tensors
    submission_timestamp: datetime
    
@dataclass  
class ModelDistributionMessage(TypedMessage):
    message_type: str = "model_distribution"
    round_id: str
    model_state: Dict[str, bytes]  # Serialized model parameters
    training_config: TrainingConfig
```

### 3. Tokenomics Service Interface
```python
@protocol
class TokenomicsService:
    """Token economics and payment processing service"""
    
    async def verify_payment_capability(self, wallet: WalletId, cost: ServiceCost) -> PaymentVerification: ...
    async def process_payment(self, payment: PaymentRequest) -> PaymentResult: ...
    async def calculate_rewards(self, activity: RewardableActivity) -> RewardCalculation: ...
    async def distribute_rewards(self, distribution: RewardDistribution) -> DistributionResult: ...
    async def get_wallet_balance(self, wallet: WalletId) -> TokenBalance: ...
    
@dataclass
class ServiceCost:
    base_amount: TokenAmount
    service_type: str  # "compute_burst", "hidden_service", "fl_participation"
    usage_metrics: Dict[str, float]  # Hours, GB, etc.
    sla_tier: str = "standard"
    
@dataclass
class RewardableActivity:
    activity_type: str  # "harvest_contribution", "fl_participation", "service_hosting"
    participant_id: str
    metrics: ActivityMetrics
    quality_score: float
    duration: timedelta
```

## Event-Driven Architecture Implementation

### Event Bus Configuration
```python
# Event bus implementation using NATS or similar
class EventBus:
    """Distributed event bus for service coordination"""
    
    def __init__(self, nats_url: str = "nats://localhost:4222"):
        self._nc = await nats.connect(nats_url)
        self._js = self._nc.jetstream()
        
    async def publish_event(self, event: DomainEvent) -> None:
        """Publish domain event to interested services"""
        subject = f"events.{event.__class__.__name__}"
        data = event.to_json().encode()
        
        await self._js.publish(subject, data, headers={
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "source_service": event.source_service
        })
        
    async def subscribe_to_events(self, event_type: Type[DomainEvent], handler: EventHandler) -> None:
        """Subscribe to specific domain event types"""
        subject = f"events.{event_type.__name__}"
        
        async def message_handler(msg):
            try:
                event_data = json.loads(msg.data.decode())
                event = event_type.from_dict(event_data)
                await handler(event)
                await msg.ack()
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                await msg.nak()
                
        await self._js.subscribe(subject, cb=message_handler)
```

### Domain Event Definitions  
```python
# Base domain event
@dataclass
class DomainEvent:
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source_service: str = ""
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

# Specific domain events
@dataclass
class DeviceRegisteredEvent(DomainEvent):
    device_id: str
    device_type: str
    capabilities: DeviceCapabilities
    source_service: str = "device_registry"

@dataclass  
class TrainingRoundCompletedEvent(DomainEvent):
    round_id: str
    session_id: str
    participants: List[str]
    model_metrics: Dict[str, float]
    privacy_budget_consumed: float
    duration_seconds: float
    source_service: str = "federated_learning"

@dataclass
class HarvestSessionCompletedEvent(DomainEvent):
    session_id: str
    device_id: str
    resources_contributed: ResourceMetrics
    quality_score: float
    tokens_earned: int
    source_service: str = "harvest_management"
    
@dataclass
class ServiceContractCreatedEvent(DomainEvent):
    contract_id: str
    service_type: str
    provider_id: str
    customer_id: str
    agreed_price: TokenAmount
    sla_terms: Dict[str, Any]
    source_service: str = "marketplace"
```

## Configuration Management Strategy

### Service Configuration
```python
# Centralized configuration with environment variable override
@dataclass
class ServiceConfig:
    """Base configuration for all services"""
    
    # Service discovery
    consul_url: str = "http://localhost:8500"
    service_name: str = ""
    service_port: int = 8080
    
    # Event bus  
    nats_url: str = "nats://localhost:4222"
    event_stream_name: str = "ai_village_events"
    
    # Database
    database_url: str = "postgresql://localhost:5432/ai_village"
    redis_url: str = "redis://localhost:6379"
    
    # Monitoring
    prometheus_port: int = 9090
    jaeger_url: str = "http://localhost:14268/api/traces"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment variables"""
        return cls(
            consul_url=os.getenv("CONSUL_URL", cls.consul_url),
            nats_url=os.getenv("NATS_URL", cls.nats_url),
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            # ... other env vars
        )

# Service-specific configuration
@dataclass        
class FederatedLearningConfig(ServiceConfig):
    """Configuration for federated learning service"""
    
    service_name: str = "federated_learning"
    
    # FL-specific settings
    max_participants_per_round: int = 20
    min_participants_per_round: int = 3
    round_timeout_minutes: int = 10
    differential_privacy_epsilon: float = 1.0
    secure_aggregation_enabled: bool = True
    
    # Model settings
    supported_frameworks: List[str] = field(default_factory=lambda: ["pytorch", "tensorflow"])
    max_model_size_mb: int = 100
    
@dataclass
class TokenomicsConfig(ServiceConfig):
    """Configuration for tokenomics service"""
    
    service_name: str = "tokenomics"
    
    # Economic settings
    initial_token_supply: int = 1_000_000_000
    harvest_reward_rate: float = 10.0  # tokens per hour
    fl_participation_bonus: float = 1.5  # multiplier
    marketplace_fee_percent: float = 2.0
    
    # Payment settings  
    payment_timeout_seconds: int = 30
    retry_failed_payments: bool = True
    max_payment_retries: int = 3
```

### Docker Deployment Configuration
```yaml
# docker-compose.yml for development environment
version: '3.8'

services:
  # Infrastructure services
  consul:
    image: consul:1.15
    ports:
      - "8500:8500"
    command: agent -dev -client=0.0.0.0
    
  nats:
    image: nats:2.9
    ports:
      - "4222:4222"
    command: -js
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ai_village
      POSTGRES_USER: aivillage  
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:7
    ports:
      - "6379:6379"
      
  # Core services  
  device-registry:
    build: ./services/device-registry
    environment:
      - CONSUL_URL=http://consul:8500
      - NATS_URL=nats://nats:4222
      - DATABASE_URL=postgresql://aivillage:password@postgres:5432/ai_village
    depends_on:
      - consul
      - nats
      - postgres
      
  communication:
    build: ./services/communication
    environment:
      - CONSUL_URL=http://consul:8500
      - NATS_URL=nats://nats:4222  
      - REDIS_URL=redis://redis:6379
    depends_on:
      - consul
      - nats
      - redis
      
  # Business services
  federated-learning:
    build: ./services/federated-learning
    environment:
      - CONSUL_URL=http://consul:8500
      - NATS_URL=nats://nats:4222
      - DATABASE_URL=postgresql://aivillage:password@postgres:5432/ai_village
      - MAX_PARTICIPANTS_PER_ROUND=20
      - DIFFERENTIAL_PRIVACY_EPSILON=1.0
    depends_on:
      - device-registry
      - communication
      - monitoring
      
  harvest-management:
    build: ./services/harvest-management  
    environment:
      - CONSUL_URL=http://consul:8500
      - NATS_URL=nats://nats:4222
      - MIN_BATTERY_PERCENT=20
      - MAX_CPU_TEMP=45.0
      - HARVEST_REWARD_RATE=10.0
    depends_on:
      - device-registry
      - tokenomics
      
  tokenomics:
    build: ./services/tokenomics
    environment:
      - CONSUL_URL=http://consul:8500
      - NATS_URL=nats://nats:4222
      - DATABASE_URL=postgresql://aivillage:password@postgres:5432/ai_village
      - INITIAL_TOKEN_SUPPLY=1000000000
    depends_on:
      - consul
      - nats
      - postgres
      
volumes:
  postgres_data:
```

## Testing Strategy

### Unit Testing Per Service
```python
# Example: Device Registry Service unit tests
class TestDeviceRegistryService:
    
    @pytest.fixture
    async def service(self):
        # Mock dependencies for isolated testing
        mock_db = Mock()
        mock_event_bus = Mock()
        return DeviceRegistryService(mock_db, mock_event_bus)
        
    async def test_register_device_success(self, service):
        device_info = DeviceInfo(
            device_id="test_device",
            device_type="smartphone", 
            capabilities=DeviceCapabilities(ram_mb=4096, cpu_cores=4)
        )
        
        device_id = await service.register_device(device_info)
        
        assert device_id == "test_device"
        # Verify database calls
        service._db.insert_device.assert_called_once()
        # Verify event publication
        service._event_bus.publish_event.assert_called_once_with(
            DeviceRegisteredEvent(device_id="test_device", ...)
        )
        
    async def test_query_devices_by_criteria(self, service):
        criteria = DeviceCriteria(min_ram_mb=2048, supports_ml=True)
        
        devices = await service.query_devices(criteria)
        
        # Verify correct database query
        service._db.query_devices.assert_called_with(
            ram_mb__gte=2048,
            capabilities__supports_ml=True
        )
```

### Integration Testing
```python
# Integration test for service interactions
class TestServiceIntegration:
    
    @pytest.fixture
    async def test_environment(self):
        """Set up test environment with all services"""
        # Start test database, event bus, etc.
        async with TestEnvironment() as env:
            yield env
            
    async def test_federated_learning_end_to_end(self, test_environment):
        """Test complete FL workflow across services"""
        
        # 1. Register devices
        device_registry = test_environment.get_service("device_registry")
        device_id = await device_registry.register_device(test_device_info)
        
        # 2. Start FL training
        fl_service = test_environment.get_service("federated_learning")
        training_session = await fl_service.start_training(test_model, test_config)
        
        # 3. Verify participant selection
        participants = await fl_service.get_selected_participants(training_session.session_id)
        assert device_id in [p.device_id for p in participants]
        
        # 4. Complete training round  
        round_result = await fl_service.complete_training_round(training_session.session_id)
        
        # 5. Verify reward distribution
        tokenomics = test_environment.get_service("tokenomics")
        rewards = await tokenomics.get_pending_rewards(device_id)
        assert len(rewards) > 0
        assert rewards[0].reward_type == "fl_participation"
```

### Performance Testing
```python
# Performance benchmarks for refactored services
class TestPerformance:
    
    async def test_device_query_performance(self):
        """Ensure device queries scale to 10,000+ devices"""
        
        # Register 10,000 test devices
        device_registry = await get_test_service("device_registry")
        for i in range(10000):
            await device_registry.register_device(create_test_device(i))
            
        # Benchmark query performance
        start_time = time.time()
        devices = await device_registry.query_devices(
            DeviceCriteria(min_ram_mb=2048, supports_ml=True)
        )
        query_time = time.time() - start_time
        
        assert query_time < 0.1  # Should complete in <100ms
        assert len(devices) > 0
        
    async def test_concurrent_training_rounds(self):
        """Test multiple concurrent FL training sessions"""
        
        fl_service = await get_test_service("federated_learning") 
        
        # Start 10 concurrent training sessions
        sessions = []
        for i in range(10):
            session = await fl_service.start_training(
                create_test_model(i), create_test_config(i)
            )
            sessions.append(session)
            
        # Complete all sessions concurrently
        results = await asyncio.gather(*[
            fl_service.complete_training(session.session_id) 
            for session in sessions
        ])
        
        assert len(results) == 10
        assert all(r.success for r in results)
```

## Monitoring and Observability

### Service Health Monitoring
```python
# Health check endpoints for each service
class HealthCheckEndpoint:
    def __init__(self, service: Any):
        self._service = service
        
    async def health_check(self) -> HealthStatus:
        """Comprehensive health check for service"""
        checks = [
            self._check_database_connection(),
            self._check_event_bus_connection(),
            self._check_dependent_services(),
            self._check_resource_usage()
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        overall_status = HealthStatus.HEALTHY
        failed_checks = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception) or not result:
                overall_status = HealthStatus.UNHEALTHY  
                failed_checks.append(checks[i].__name__)
                
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.now(UTC),
            failed_checks=failed_checks,
            uptime=self._get_uptime(),
            version=self._get_service_version()
        )
```

### Metrics Collection
```python
# Prometheus metrics for each service
class ServiceMetrics:
    def __init__(self, service_name: str):
        self.service_name = service_name
        
        # Standard metrics for all services
        self.request_duration = Histogram(
            f'{service_name}_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'status']
        )
        
        self.request_count = Counter(
            f'{service_name}_requests_total', 
            'Total requests',
            ['method', 'endpoint', 'status']
        )
        
        self.active_connections = Gauge(
            f'{service_name}_active_connections',
            'Active service connections'
        )
        
    def record_request(self, method: str, endpoint: str, duration: float, status: str):
        self.request_duration.labels(method, endpoint, status).observe(duration)
        self.request_count.labels(method, endpoint, status).inc()

# Service-specific metrics
class FederatedLearningMetrics(ServiceMetrics):
    def __init__(self):
        super().__init__("federated_learning")
        
        self.active_training_sessions = Gauge(
            'fl_active_training_sessions',
            'Number of active training sessions'
        )
        
        self.training_round_duration = Histogram(
            'fl_training_round_duration_seconds',
            'Training round duration',
            ['participants_count', 'model_size']
        )
        
        self.participant_selection_time = Histogram(
            'fl_participant_selection_seconds', 
            'Time to select participants'
        )
```

## Migration Timeline and Milestones

### Phase 1: Foundation (Weeks 1-4)
**Week 1-2: Infrastructure Setup**
- [ ] Set up service discovery (Consul)
- [ ] Set up event bus (NATS)  
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Create base service templates and interfaces

**Week 3-4: Core Services**
- [ ] Extract Device Registry Service
- [ ] Extract Communication Service
- [ ] Extract Monitoring Service
- [ ] Update FL coordinator to use new services

**Milestone 1**: Core infrastructure services operational, FL service using device registry

### Phase 2: Business Services (Weeks 5-8)
**Week 5-6: Economic Layer**
- [ ] Extract Tokenomics Service
- [ ] Implement payment verification interfaces
- [ ] Update all services to use tokenomics interface

**Week 7-8: Resource Management**  
- [ ] Extract Harvest Management Service
- [ ] Implement harvest session management
- [ ] Integrate harvest rewards with tokenomics

**Milestone 2**: Payment and harvest services operational, event-driven reward distribution

### Phase 3: Advanced Services (Weeks 9-12)
**Week 9-10: Marketplace**
- [ ] Extract Marketplace Service
- [ ] Expand burst coordination capabilities
- [ ] Implement service contracts and SLA monitoring

**Week 11-12: Privacy Layer**
- [ ] Extract Privacy Service  
- [ ] Implement onion routing protocols
- [ ] Integrate hidden service management

**Milestone 3**: All business services extracted, marketplace and privacy operational

### Phase 4: Optimization (Weeks 13-16)
**Week 13-14: FL Service Refinement**
- [ ] Remove all extracted functionality from FL coordinator
- [ ] Focus FL service on core ML coordination
- [ ] Optimize ML-specific participant selection

**Week 15-16: System Optimization**
- [ ] Performance optimization across services
- [ ] End-to-end integration testing
- [ ] Production deployment preparation

**Milestone 4**: Complete microservices architecture, production-ready

## Success Metrics

### Quantitative Metrics
- **Complexity Reduction**: 80%+ reduction in lines per service (from 754 to <150)
- **Coupling Reduction**: 90%+ reduction in cross-service dependencies
- **Test Coverage**: 95%+ unit test coverage per service
- **Performance**: <100ms service response times, <1s end-to-end operations
- **Reliability**: 99.9%+ uptime per service, fault isolation working

### Qualitative Metrics
- **Developer Velocity**: Teams can develop services independently
- **Deployment Flexibility**: Services can be deployed/updated independently  
- **Operational Excellence**: Clear monitoring, alerting, and debugging capabilities
- **Business Agility**: New features can be added without affecting other services
- **Technical Debt**: Reduced coupling and complexity enable future evolution

## Risk Assessment and Mitigation

### High Risks 🔴
1. **Service Communication Latency**
   - **Risk**: Inter-service calls add latency vs. monolithic in-process calls
   - **Mitigation**: Async messaging, response caching, circuit breakers
   
2. **Data Consistency Across Services**  
   - **Risk**: Eventual consistency may cause temporary inconsistencies
   - **Mitigation**: Event sourcing, saga pattern, compensation logic
   
3. **Operational Complexity**
   - **Risk**: Managing 8 services vs. 1 monolith increases operational overhead
   - **Mitigation**: Container orchestration, automated deployment, comprehensive monitoring

### Medium Risks 🟡  
1. **Service Discovery Failures**
   - **Risk**: Services unable to find each other
   - **Mitigation**: Multiple service discovery methods, health checks, failover

2. **Event Bus Availability**
   - **Risk**: Event bus failure affects all service communication
   - **Mitigation**: Event bus clustering, persistent queues, circuit breakers
   
### Low Risks 🟢
1. **Migration Timeline Overrun**
   - **Risk**: 16-week timeline may be optimistic
   - **Mitigation**: Incremental migration, parallel development, buffer time

## Conclusion

The comprehensive refactoring of the federated coordinator from a 754-line monolith to 8 independent microservices will:

### Transform Architecture
- **From**: Single tightly-coupled coordinator managing 7 unrelated domains
- **To**: 8 loosely-coupled services with clear boundaries and responsibilities

### Enable Business Growth
- **Scalability**: Independent scaling based on demand per service type
- **Development Velocity**: Parallel development by specialized teams
- **Feature Flexibility**: Add new capabilities without affecting existing services
- **Operational Excellence**: Clear monitoring, fault isolation, and debugging

### Improve Technical Quality
- **Maintainability**: 80%+ reduction in complexity per service
- **Testability**: Independent unit and integration testing
- **Reliability**: Fault isolation prevents cascading failures
- **Performance**: Optimized resource usage per service domain

### Support AI Village Vision
The microservices architecture directly supports the AI Village goals:
- **Mobile Compute Harvesting**: Dedicated service with proper battery/thermal safety
- **Censorship-Resistant Services**: Specialized privacy service with onion routing
- **Economic Incentives**: Comprehensive tokenomics service with proper payment processing
- **Decentralized Marketplace**: Full-featured marketplace service with SLA enforcement

**Recommendation**: Proceed with the 4-phase migration plan, starting immediately with Phase 1 infrastructure services. The benefits significantly outweigh the risks, and the current monolithic architecture is unsustainable for the planned AI Village expansion.

**Investment Required**: 4 developers × 16 weeks = 64 person-weeks  
**Expected ROI**: 3x development velocity improvement, 90% fault reduction, infinite scalability potential