# Service Extraction and Natural Boundaries Strategy

## Overview
Based on the architectural analysis, this document proposes a comprehensive service extraction strategy that breaks the monolithic federated coordinator into 8 independent, loosely-coupled services with clear boundaries and responsibilities.

## Proposed Service Architecture

```mermaid
graph TB
    subgraph "Core Services"
        FLS[Federated Learning Service]
        DRS[Device Registry Service]
        CS[Communication Service]
    end
    
    subgraph "Business Services"
        HMS[Harvest Management Service]
        MPS[Marketplace Service]
        PS[Privacy Service]
        TS[Tokenomics Service]
    end
    
    subgraph "Infrastructure Services"
        MS[Monitoring Service]
        SDS[Service Discovery]
        EB[Event Bus]
    end
    
    FLS --> DRS
    FLS --> CS
    FLS --> EB
    
    HMS --> DRS
    HMS --> TS
    HMS --> EB
    
    MPS --> DRS
    MPS --> TS
    MPS --> EB
    
    PS --> DRS
    PS --> CS
    PS --> EB
    
    TS --> EB
    
    MS --> EB
    MS --> SDS
    
    SDS --> EB
```

## Service Boundaries and Responsibilities

### 1. **Federated Learning Service** (Core Domain)
**Purpose**: Coordinate distributed ML training  
**Natural Boundary**: Machine learning training lifecycle  
**Current Lines**: ~400/754 (53% - appropriate)

#### Responsibilities:
- **Training Orchestration**: Manage FL training rounds and phases
- **Model Management**: Distribute global model and aggregate gradients
- **Privacy Preservation**: Implement differential privacy and secure aggregation
- **Quality Assurance**: Byzantine robustness and hierarchical aggregation

#### Interface Definition:
```python
class FederatedLearningService:
    async def initialize_training(self, model: MLModel, config: TrainingConfig) -> TrainingSession:
        """Initialize a new federated learning session"""
    
    async def start_training_round(self, session_id: str) -> TrainingRound:
        """Start a new training round with participant selection"""
    
    async def get_training_status(self, session_id: str) -> TrainingStatus:
        """Get current training session status and metrics"""
    
    async def stop_training(self, session_id: str) -> TrainingResults:
        """Stop training and return final results"""

# Events published
class TrainingRoundStartedEvent(DomainEvent):
    session_id: str
    participants: List[str]
    round_config: TrainingConfig

class TrainingRoundCompletedEvent(DomainEvent):
    session_id: str
    round_id: str
    metrics: TrainingMetrics
    model_performance: float
```

#### Dependencies:
- ✅ Device Registry Service (participant selection)
- ✅ Communication Service (message routing)
- ✅ Event Bus (domain events)

#### What to Extract:
- Remove BurstCoordinator and HiddenServiceManager
- Remove tokenomics integration
- Remove P2P message handling
- Keep core FL logic and privacy features

---

### 2. **Harvest Management Service** (Business Domain)
**Purpose**: Manage mobile compute resource harvesting  
**Natural Boundary**: Device resource collection and contribution tracking  
**Current Lines**: Scattered/Missing (~50 lines, should be ~300)

#### Responsibilities:
- **Battery-Aware Harvesting**: Detect optimal harvesting conditions
- **Resource Policy Enforcement**: Manage CPU/memory/bandwidth limits
- **Harvest Session Management**: Track contribution sessions and rewards
- **Device Capability Assessment**: Profile hardware capabilities
- **Thermal Safety Management**: Prevent device overheating

#### Interface Definition:
```python
class HarvestManagementService:
    async def register_harvester(self, device_id: str, capabilities: DeviceCapabilities) -> HarvesterProfile:
        """Register a device for compute harvesting"""
    
    async def start_harvest_session(self, device_id: str, policy: HarvestPolicy) -> HarvestSession:
        """Start harvesting session with safety policies"""
    
    async def stop_harvest_session(self, session_id: str) -> ContributionLedger:
        """Stop session and return contribution metrics"""
    
    async def get_available_harvesters(self, criteria: HarvestCriteria) -> List[HarvesterInfo]:
        """Get available devices for harvesting"""
    
    async def update_harvest_policy(self, device_id: str, policy: HarvestPolicy) -> bool:
        """Update harvesting policies for device"""

# Events published  
class HarvestSessionStartedEvent(DomainEvent):
    device_id: str
    session_id: str
    expected_duration: timedelta

class HarvestContributionEvent(DomainEvent):
    device_id: str
    session_id: str
    resources_contributed: ResourceMetrics
    tokens_earned: int
```

#### Key Components to Extract:
```python
# From harvest_manager.py - these are missing from current coordinator
@dataclass
class HarvestPolicy:
    min_battery_percent: int = 20
    max_cpu_temp: float = 45.0
    require_charging: bool = True
    max_cpu_percent: float = 50.0

@dataclass  
class HarvestSession:
    session_id: str
    device_id: str
    cpu_cycles: int = 0
    memory_mb_hours: float = 0.0
    tokens_earned: int = 0

@dataclass
class ContributionLedger:
    device_id: str
    total_sessions: int = 0
    total_tokens_earned: int = 0
    reliability_score: float = 1.0
```

#### Dependencies:
- ✅ Device Registry Service (device information)
- ✅ Tokenomics Service (reward calculation)
- ✅ Event Bus (contribution events)

---

### 3. **Marketplace Service** (Business Domain)
**Purpose**: Fog computing marketplace for services and resources  
**Natural Boundary**: Service offerings, contracts, and marketplace operations  
**Current Lines**: ~115 (BurstCoordinator) + Missing marketplace logic

#### Responsibilities:
- **Service Offering Management**: Register and manage service offerings
- **Demand Matching**: Match service requests with providers
- **Contract Management**: Service contracts and SLA enforcement
- **Dynamic Pricing**: Supply/demand-based pricing algorithms  
- **Multi-Region Coordination**: Manage fog zones and availability

#### Interface Definition:
```python
class MarketplaceService:
    async def register_service_offering(self, offering: ServiceOffering) -> OfferingId:
        """Register a new service offering in marketplace"""
    
    async def request_service(self, request: ServiceRequest) -> ServiceContract:
        """Request a service and create contract with provider"""
    
    async def allocate_compute_burst(self, requirements: ComputeRequirements, budget: TokenAmount) -> BurstAllocation:
        """Allocate compute burst for high-performance tasks"""
    
    async def monitor_sla_compliance(self, contract_id: str) -> SLAStatus:
        """Monitor service level agreement compliance"""
    
    async def get_marketplace_stats(self) -> MarketplaceMetrics:
        """Get marketplace statistics and pricing information"""

# Events published
class ServiceContractCreatedEvent(DomainEvent):
    contract_id: str
    service_type: ServiceType
    provider_id: str
    customer_id: str
    agreed_price: TokenAmount

class SLABreachEvent(DomainEvent):
    contract_id: str
    breach_type: str
    penalty_amount: TokenAmount
```

#### Key Components to Extract:
```python
# From fog_marketplace.py - these are missing from current coordinator
@dataclass
class ServiceOffering:
    offering_id: str
    provider_id: str
    service_type: ServiceType
    cpu_cores: int
    memory_gb: float
    base_price: Decimal
    uptime_guarantee: float = 99.0

@dataclass
class ServiceContract:
    contract_id: str
    customer_id: str
    provider_id: str
    agreed_price: Decimal
    sla_terms: Dict[str, Any]
    usage_hours: float = 0.0
    uptime_percent: float = 100.0
```

#### Dependencies:
- ✅ Device Registry Service (provider capabilities)
- ✅ Tokenomics Service (payment processing)
- ✅ Event Bus (marketplace events)

---

### 4. **Privacy Service** (Business Domain)  
**Purpose**: Censorship-resistant privacy layer and hidden services  
**Natural Boundary**: Onion routing, hidden services, and privacy protocols  
**Current Lines**: ~173 (HiddenServiceManager) + Missing onion routing

#### Responsibilities:
- **Onion Circuit Management**: Multi-hop circuit construction and maintenance
- **Hidden Service Hosting**: Deploy and manage .fog hidden services
- **Privacy Protocol Coordination**: Traffic mixing and padding
- **Introduction Point Management**: Manage service introduction points
- **Data Encryption and Sharding**: Secure data distribution across nodes

#### Interface Definition:
```python
class PrivacyService:
    async def create_onion_circuit(self, hops: int, exit_criteria: ExitCriteria) -> Circuit:
        """Create multi-hop onion circuit for anonymous communication"""
    
    async def deploy_hidden_service(self, data: bytes, privacy_level: PrivacyLevel, budget: TokenAmount) -> HiddenServiceId:
        """Deploy censorship-resistant hidden service"""
    
    async def access_hidden_service(self, service_id: HiddenServiceId, accessor_wallet: str) -> bytes:
        """Access hidden service and charge access fees"""
    
    async def route_traffic(self, circuit_id: str, traffic: bytes) -> bytes:
        """Route traffic through onion circuit"""
    
    async def get_service_stats(self, service_id: HiddenServiceId) -> ServiceStats:
        """Get statistics for hidden service"""

# Events published
class HiddenServiceDeployedEvent(DomainEvent):
    service_id: HiddenServiceId
    owner_wallet: str
    hosting_nodes: List[str]
    privacy_level: PrivacyLevel

class CircuitEstablishedEvent(DomainEvent):
    circuit_id: str
    hops: List[str]
    estimated_latency: float
```

#### Key Components to Extract:
```python
# From onion_routing.py and hidden service manager
@dataclass
class Circuit:
    circuit_id: str
    hops: List[NodeId]
    encryption_keys: List[bytes]
    established_at: datetime

@dataclass
class HiddenService:
    service_id: HiddenServiceId
    owner_wallet: str
    hosting_nodes: List[str]
    privacy_level: PrivacyLevel
    access_count: int = 0
    revenue_earned: TokenAmount = 0
```

#### Dependencies:
- ✅ Device Registry Service (node selection for circuits/hosting)
- ✅ Communication Service (circuit communication)
- ✅ Tokenomics Service (hidden service payments)
- ✅ Event Bus (privacy events)

---

### 5. **Tokenomics Service** (Business Domain)
**Purpose**: Token economics, payments, and incentive alignment  
**Natural Boundary**: Token management, payments, and economic incentives  
**Current Lines**: Scattered (~30 lines) + Missing tokenomics logic

#### Responsibilities:
- **Payment Processing**: Verify and process service payments
- **Reward Distribution**: Calculate and distribute contribution rewards  
- **Token Supply Management**: Manage token economics and inflation
- **Staking and Governance**: Handle token staking and voting mechanisms
- **Economic Analytics**: Track economic metrics and optimize incentives

#### Interface Definition:
```python
class TokenomicsService:
    async def verify_payment_capability(self, wallet: WalletId, amount: TokenAmount, service_type: str) -> bool:
        """Verify wallet can pay for service"""
    
    async def process_payment(self, payment_request: PaymentRequest) -> PaymentResult:
        """Process payment between parties"""
    
    async def calculate_rewards(self, contributions: List[Contribution]) -> List[RewardAllocation]:
        """Calculate rewards for resource contributions"""
    
    async def distribute_rewards(self, allocations: List[RewardAllocation]) -> DistributionResult:
        """Distribute calculated rewards to contributors"""
    
    async def get_wallet_balance(self, wallet: WalletId) -> TokenAmount:
        """Get current wallet balance"""
    
    async def get_economic_metrics(self) -> EconomicMetrics:
        """Get system-wide economic metrics"""

# Events published
class PaymentProcessedEvent(DomainEvent):
    transaction_id: str
    from_wallet: WalletId
    to_wallet: WalletId
    amount: TokenAmount
    service_type: str

class RewardsDistributedEvent(DomainEvent):
    distribution_id: str
    total_amount: TokenAmount
    recipient_count: int
    reward_type: str  # harvest, fl_participation, hosting, etc.
```

#### Key Components to Extract:
```python
# From tokenomics files
@dataclass  
class PaymentRequest:
    from_wallet: WalletId
    to_wallet: WalletId
    amount: TokenAmount
    service_type: str
    metadata: Dict[str, Any]

@dataclass
class RewardAllocation:
    recipient_wallet: WalletId
    amount: TokenAmount
    reward_type: str
    contribution_metrics: Dict[str, float]
```

#### Dependencies:
- ✅ Event Bus (payment and reward events)
- ✅ Monitoring Service (economic metrics)

---

### 6. **Device Registry Service** (Core Infrastructure)
**Purpose**: Central device and capability management  
**Natural Boundary**: Device information, capabilities, and reputation  
**Current Lines**: Scattered in participant management (~100 lines)

#### Responsibilities:
- **Device Registration**: Register devices and their capabilities
- **Capability Assessment**: Evaluate device suitability for different services
- **Reputation Management**: Track device reputation across all services
- **Geographic Coordination**: Manage device geographic distribution
- **Health Monitoring**: Track device availability and performance

#### Interface Definition:
```python
class DeviceRegistryService:
    async def register_device(self, device_info: DeviceInfo) -> DeviceId:
        """Register device with capabilities in registry"""
    
    async def update_device_capabilities(self, device_id: DeviceId, capabilities: DeviceCapabilities) -> bool:
        """Update device capability information"""
    
    async def get_devices_by_criteria(self, criteria: DeviceCriteria) -> List[DeviceInfo]:
        """Query devices matching specific criteria"""
    
    async def update_device_reputation(self, device_id: DeviceId, service: str, score: float) -> bool:
        """Update device reputation for specific service"""
    
    async def get_device_reputation(self, device_id: DeviceId, service: str) -> ReputationScore:
        """Get device reputation for service"""
    
    async def mark_device_offline(self, device_id: DeviceId) -> bool:
        """Mark device as offline/unavailable"""

# Events published
class DeviceRegisteredEvent(DomainEvent):
    device_id: DeviceId
    device_type: str
    capabilities: DeviceCapabilities
    region: str

class DeviceReputationUpdatedEvent(DomainEvent):
    device_id: DeviceId
    service: str
    old_score: float
    new_score: float
```

#### Dependencies:
- ✅ Event Bus (device events)
- ✅ Monitoring Service (device health metrics)

---

### 7. **Communication Service** (Core Infrastructure)
**Purpose**: Abstract network transport and messaging  
**Natural Boundary**: Network communication protocols and message routing  
**Current Lines**: Embedded in P2P handlers (~80 lines)

#### Responsibilities:
- **Protocol Abstraction**: Support multiple transport protocols (BitChat/Betanet/libp2p)
- **Message Routing**: Route messages between services and nodes
- **Circuit Management**: Manage network circuits and connections
- **NAT Traversal**: Handle network connectivity issues
- **Message Serialization**: Handle message encoding/decoding

#### Interface Definition:
```python
class CommunicationService:
    async def send_message(self, recipient: NodeId, message: Message) -> bool:
        """Send message to specific node"""
    
    async def broadcast_message(self, message: Message, recipients: List[NodeId]) -> List[NodeId]:
        """Broadcast message to multiple recipients"""
    
    async def create_circuit(self, nodes: List[NodeId], circuit_type: CircuitType) -> CircuitId:
        """Create communication circuit through specified nodes"""
    
    async def subscribe_to_messages(self, message_type: Type[Message], handler: MessageHandler) -> SubscriptionId:
        """Subscribe to specific message types"""
    
    async def unsubscribe(self, subscription_id: SubscriptionId) -> bool:
        """Unsubscribe from message type"""

# Events published
class MessageDeliveredEvent(DomainEvent):
    message_id: str
    recipient: NodeId
    delivery_time: datetime
    success: bool

class CircuitEstablishedEvent(DomainEvent):
    circuit_id: str
    nodes: List[NodeId]
    latency_ms: float
```

#### Dependencies:
- ✅ Event Bus (communication events)
- ✅ Device Registry Service (node lookup)

---

### 8. **Monitoring Service** (Infrastructure)
**Purpose**: System observability and performance tracking  
**Natural Boundary**: Metrics collection, analysis, and insights  
**Current Lines**: Basic stats tracking (~20 lines, should be ~200)

#### Responsibilities:
- **Metrics Collection**: Collect performance metrics from all services
- **Health Monitoring**: Monitor service and system health
- **Analytics and Insights**: Generate performance insights and recommendations
- **Alerting**: Alert on system issues and anomalies
- **Dashboard Data**: Provide data for monitoring dashboards

#### Interface Definition:
```python
class MonitoringService:
    async def record_metric(self, service: str, metric_name: str, value: float, tags: Dict[str, str] = None) -> bool:
        """Record performance metric"""
    
    async def record_event(self, service: str, event_type: str, metadata: Dict[str, Any]) -> bool:
        """Record system event"""
    
    async def get_service_health(self, service: str) -> HealthStatus:
        """Get health status for service"""
    
    async def get_system_metrics(self, time_range: TimeRange) -> SystemMetrics:
        """Get system-wide metrics for time range"""
    
    async def detect_anomalies(self, service: str, metric: str) -> List[Anomaly]:
        """Detect performance anomalies"""

# Events published  
class HealthStatusChangedEvent(DomainEvent):
    service: str
    old_status: HealthStatus
    new_status: HealthStatus
    timestamp: datetime

class AnomalyDetectedEvent(DomainEvent):
    service: str
    metric: str
    anomaly_score: float
    description: str
```

#### Dependencies:
- ✅ Event Bus (monitoring events)
- ✅ All services (metrics collection)

## Service Communication Patterns

### 1. **Event-Driven Communication** (Primary)
Services communicate primarily through domain events via the event bus:

```python
# Example: FL service publishes training completion
await event_bus.publish_event(
    TrainingRoundCompletedEvent(
        session_id="fl_123",
        round_id="round_5",
        participants=["device1", "device2"],
        performance_metrics={"accuracy": 0.85, "loss": 0.15}
    )
)

# Tokenomics service subscribes to calculate rewards
class TokenomicsService:
    async def handle_training_completed(self, event: TrainingRoundCompletedEvent):
        rewards = await self.calculate_fl_participation_rewards(
            event.participants, event.performance_metrics
        )
        await self.distribute_rewards(rewards)
```

### 2. **Direct API Calls** (For Immediate Response)
Services make direct calls when immediate response is needed:

```python
# Marketplace verifies payment before allocating resources
payment_verified = await tokenomics_service.verify_payment_capability(
    wallet=request.customer_wallet,
    amount=service_cost,
    service_type="compute_burst"
)
if not payment_verified:
    raise InsufficientFundsError()
```

### 3. **Circuit Breaker Pattern**
Services implement circuit breakers for fault tolerance:

```python
class ServiceClient:
    def __init__(self, service_endpoint: str):
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=ServiceUnavailableError
        )
    
    async def call_service(self, request: ServiceRequest):
        return await self._circuit_breaker.call(
            self._make_service_call, request
        )
```

## Migration Strategy

### Phase 1: Extract Core Infrastructure Services
1. **Device Registry Service** - Central device management
2. **Communication Service** - Abstract network transport  
3. **Monitoring Service** - Metrics and observability
4. **Service Discovery** - Service registration and discovery

### Phase 2: Extract Business Services  
1. **Harvest Management Service** - Mobile compute harvesting
2. **Tokenomics Service** - Payment and reward processing

### Phase 3: Extract Complex Business Services
1. **Marketplace Service** - Service marketplace and contracts
2. **Privacy Service** - Onion routing and hidden services

### Phase 4: Refactor Federated Learning Service
1. Remove extracted functionality from FL coordinator
2. Update to use service interfaces instead of direct access
3. Focus on core FL training orchestration

### Phase 5: Integration Testing and Optimization
1. End-to-end testing of service interactions
2. Performance optimization and tuning
3. Monitoring and alerting setup

## Service Deployment Model

### Independent Deployment
Each service can be deployed independently:

```yaml
# docker-compose.yml
services:
  device-registry:
    image: aivillage/device-registry:latest
    environment:
      - DB_HOST=postgres
    depends_on:
      - postgres
      - event-bus

  harvest-manager:
    image: aivillage/harvest-manager:latest
    depends_on:
      - device-registry
      - tokenomics
      - event-bus

  marketplace:
    image: aivillage/marketplace:latest
    depends_on:
      - device-registry
      - tokenomics
      - event-bus
```

### Service Configuration
Services configured through environment variables:

```python
class ServiceConfig:
    # Service discovery
    SERVICE_DISCOVERY_URL: str = "http://consul:8500"
    
    # Event bus
    EVENT_BUS_URL: str = "nats://nats:4222"
    
    # Database
    DATABASE_URL: str = "postgresql://user:pass@postgres:5432/db"
    
    # Service-specific config
    HARVEST_MIN_BATTERY_PERCENT: int = 20
    MARKETPLACE_DEFAULT_SLA_TIER: str = "standard"
```

## Benefits of Service Extraction

### Scalability Benefits
- ✅ Independent scaling of each service based on demand
- ✅ Resource optimization per service requirements
- ✅ Horizontal scaling support for high-demand services

### Development Benefits  
- ✅ Team ownership of individual services
- ✅ Independent development and deployment cycles
- ✅ Technology stack flexibility per service
- ✅ Improved testability through service isolation

### Operational Benefits
- ✅ Fault isolation - failures don't cascade
- ✅ Independent monitoring and alerting per service
- ✅ Easier debugging and troubleshooting
- ✅ Rolling updates without system-wide downtime

### Business Benefits
- ✅ Feature development velocity improvement
- ✅ Reduced coordination overhead between teams
- ✅ Ability to retire or replace individual services
- ✅ Support for different business models per service

## Conclusion

The service extraction strategy transforms the monolithic federated coordinator into a distributed microservices architecture with clear boundaries, responsibilities, and communication patterns. This enables the AI Village fog computing system to:

1. **Scale independently** based on demand for each service type
2. **Develop faster** with dedicated teams per service domain  
3. **Deploy safely** with fault isolation and rolling updates
4. **Maintain easily** with clear separation of concerns
5. **Extend flexibly** with new services and capabilities

The proposed 8-service architecture maintains all existing functionality while enabling future growth and evolution of the AI Village ecosystem.