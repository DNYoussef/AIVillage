# Dependency Breaking Implementation Guide

## Current Circular Dependency Analysis

### Identified Circular Dependencies in `fog_coordinator.py`

Based on the code analysis, the following circular dependency patterns were identified:

```python
# Pattern 1: P2P Integration Cycle
DistributedFederatedLearning → P2PNode → MeshNetwork → FogMetricsCollector → DistributedFederatedLearning

# Pattern 2: Tokenomics Integration Cycle  
DistributedFederatedLearning → VILLAGECreditSystem → ComputeMiningSystem → DistributedFederatedLearning

# Pattern 3: Evolution System Cycle
DistributedFederatedLearning → InfrastructureAwareEvolution → P2PNode → DistributedFederatedLearning

# Pattern 4: Nested Component Cycle
DistributedFederatedLearning → BurstCoordinator → HiddenServiceManager → DistributedFederatedLearning
```

### Coupling Score Breakdown

| Component | Current Lines | Coupling Score | Dependencies |
|-----------|---------------|----------------|--------------|
| Main FL Class | 1100+ | 25.3 | 8 major systems |
| Initialization | 150+ | 12.7 | 5 external systems |
| Message Handlers | 200+ | 8.4 | 3 P2P systems |
| Training Logic | 400+ | 15.2 | 6 subsystems |
| Burst Coordinator | 180+ | 7.8 | 3 nested dependencies |
| Hidden Services | 160+ | 6.1 | 4 crypto/privacy systems |

**Total Coupling Score: 39.81** (Target: < 10.0)

## Dependency Breaking Strategy

### 1. Event-Driven Decoupling Pattern

**Before: Direct Dependencies**
```python
class DistributedFederatedLearning:
    def __init__(self, p2p_node, credit_system, mesh_network, fog_metrics):
        self.p2p_node = p2p_node  # Direct coupling
        self.credit_system = credit_system  # Direct coupling
        self.mesh_network = mesh_network  # Direct coupling
        self.fog_metrics = fog_metrics  # Direct coupling
        
    async def run_training_round(self):
        # Direct method calls create tight coupling
        participants = self.p2p_node.get_suitable_peers()
        cost = self.credit_system.calculate_cost(participants)
        metrics = self.fog_metrics.collect_training_metrics()
```

**After: Event-Driven Decoupling**
```python
class FogOrchestrationService:
    async def coordinate_training_round(self):
        # Publish event instead of direct calls
        event = TrainingRoundRequestedEvent(
            requirements=ResourceRequirements(min_participants=5),
            budget=100.0
        )
        await self.event_bus.publish(event)

class FogHarvestingService:
    async def handle_training_round_requested(self, event: TrainingRoundRequestedEvent):
        # React to event instead of being called directly
        participants = await self.select_participants(event.requirements)
        response_event = ParticipantsSelectedEvent(participants=participants)
        await self.event_bus.publish(response_event)
```

### 2. Dependency Injection Container

**Implementation Pattern:**
```python
class ServiceContainer:
    def __init__(self):
        self._services = {}
        self._factories = {}
        
    def register_factory(self, service_type: str, factory_func):
        """Register a factory function for lazy service creation."""
        self._factories[service_type] = factory_func
        
    async def get_service(self, service_type: str):
        """Get or create service instance."""
        if service_type not in self._services:
            if service_type in self._factories:
                self._services[service_type] = await self._factories[service_type]()
            else:
                raise ValueError(f"Service {service_type} not registered")
        return self._services[service_type]

# Usage in services
class FogMarketplaceService:
    def __init__(self, container: ServiceContainer):
        self._container = container
        
    async def price_resources(self, spec: ResourceSpec):
        # Lazy dependency resolution - no circular imports
        tokenomics = await self._container.get_service('tokenomics')
        base_rate = await tokenomics.get_base_compute_rate()
        return self._calculate_pricing(spec, base_rate)
```

### 3. Interface Segregation

**Before: Fat Interface**
```python
class DistributedFederatedLearning:
    # 40+ methods mixing concerns
    async def initialize_federated_learning(self): pass
    async def discover_participants(self): pass  
    async def price_resources(self): pass
    async def apply_differential_privacy(self): pass
    async def process_payments(self): pass
    async def collect_metrics(self): pass
    # ... 35 more methods
```

**After: Segregated Interfaces**
```python
# Each service has focused, single-responsibility interface
class IParticipantManager:
    async def discover_participants(self): pass
    async def assess_device_suitability(self): pass
    async def select_participants(self): pass

class IPricingManager:
    async def calculate_resource_price(self): pass
    async def create_auction(self): pass
    async def settle_payment(self): pass

class IPrivacyManager:
    async def apply_differential_privacy(self): pass
    async def manage_privacy_budgets(self): pass
    async def detect_byzantine_behavior(self): pass
```

### 4. Service Registry Pattern

**Implementation:**
```python
class ServiceRegistry:
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._health_checks: Dict[str, Callable] = {}
        
    async def register(self, name: str, service: Any, health_check: Optional[Callable] = None):
        """Register a service with optional health check."""
        self._services[name] = service
        if health_check:
            self._health_checks[name] = health_check
            
    async def get(self, name: str) -> Any:
        """Get service by name with circuit breaker."""
        if name not in self._services:
            raise ServiceNotFoundError(f"Service {name} not registered")
            
        service = self._services[name]
        
        # Health check before returning service
        if name in self._health_checks:
            is_healthy = await self._health_checks[name](service)
            if not is_healthy:
                raise ServiceUnavailableError(f"Service {name} is unhealthy")
                
        return service

# Usage eliminates direct dependencies
class FogHarvestingService:
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        
    async def coordinate_burst(self, request: BurstRequest):
        # No direct dependency on marketplace or tokenomics
        marketplace = await self.registry.get('marketplace')
        tokenomics = await self.registry.get('tokenomics')
        
        # Proceed with business logic
        quote = await marketplace.price_resources(request.resource_spec)
        payment_ok = await tokenomics.verify_balance(request.wallet, quote.total_price)
```

## Step-by-Step Migration Plan

### Phase 1: Extract Service Interfaces (Days 1-2)

1. **Create Interface Contracts**
   ```bash
   # Create service interface files
   touch services/interfaces/orchestration.py
   touch services/interfaces/harvesting.py  
   touch services/interfaces/marketplace.py
   touch services/interfaces/privacy.py
   touch services/interfaces/tokenomics.py
   touch services/interfaces/stats.py
   ```

2. **Define Data Models**
   ```python
   # services/models/common.py
   from dataclasses import dataclass
   from typing import List, Dict, Any
   
   @dataclass
   class DeviceCapability:
       device_id: str
       compute_gflops: float
       # ... other fields
   
   @dataclass 
   class ResourceRequest:
       requirements: ResourceRequirements
       max_budget: float
       # ... other fields
   ```

### Phase 2: Implement Service Stubs (Days 3-4)

1. **Create Minimal Service Implementations**
   ```python
   # services/implementations/harvesting_service.py
   class FogHarvestingService(IFogHarvestingService):
       def __init__(self, registry: ServiceRegistry):
           self.registry = registry
           self.devices: Dict[str, DeviceCapability] = {}
           
       async def discover_devices(self) -> List[DeviceCapability]:
           # Stub implementation
           return list(self.devices.values())
           
       async def select_participants(self, requirements: ResourceRequirements) -> List[str]:
           # Simple implementation to start
           suitable_devices = []
           for device in self.devices.values():
               if self._meets_requirements(device, requirements):
                   suitable_devices.append(device.device_id)
           return suitable_devices[:requirements.max_nodes]
   ```

2. **Set Up Service Registry**
   ```python
   # services/container.py
   class ServiceContainer:
       async def initialize(self, config: Dict[str, Any]):
           registry = ServiceRegistry()
           
           # Register services in dependency order
           await registry.register('harvesting', FogHarvestingService(registry))
           await registry.register('marketplace', FogMarketplaceService(registry))
           await registry.register('privacy', FogPrivacyService(registry))
           await registry.register('tokenomics', FogTokenomicsService(registry))
           await registry.register('stats', FogSystemStatsService(registry))
           
           # Orchestration service coordinates others
           orchestration = FogOrchestrationService(registry)
           await registry.register('orchestration', orchestration)
           
           return registry
   ```

### Phase 3: Migrate Core Logic (Days 5-8)

1. **Extract Participant Management Logic**
   ```python
   # Migration from fog_coordinator.py lines 326-400
   async def _discover_participants(self) -> None:
       # OLD: Direct P2P calls
       suitable_peers = self.p2p_node.get_suitable_evolution_peers(min_count=1)
       
   # NEW: Event-driven approach  
   class FogHarvestingService:
       async def discover_devices(self) -> List[DeviceCapability]:
           # Publish discovery request event
           event = DeviceDiscoveryRequestedEvent()
           await self.event_bus.publish(event)
           
           # Collect responses from P2P layer
           responses = await self.event_bus.wait_for_responses(
               event_type='device_discovered',
               timeout_seconds=30
           )
           
           return [DeviceCapability(**response.payload) for response in responses]
   ```

2. **Extract Pricing Logic**
   ```python
   # Migration from BurstCoordinator logic (lines 1246-1350)
   class FogMarketplaceService:
       async def price_resources(self, spec: ResourceSpec) -> PricingQuote:
           # Calculate base pricing without direct dependencies
           base_compute_rate = await self._get_base_compute_rate()
           
           price_components = {
               'compute': spec.compute_gflops * base_compute_rate * spec.duration_hours,
               'memory': spec.memory_gb * 0.001 * spec.duration_hours,  
               'network': spec.network_mbps * 0.0001 * spec.duration_hours,
               'storage': spec.storage_gb * 0.00001 * spec.duration_hours
           }
           
           base_price = sum(price_components.values())
           
           # Apply market multipliers
           demand_multiplier = await self._calculate_demand_multiplier()
           quality_multiplier = self._get_quality_multiplier(spec.quality_level)
           
           total_price = base_price * demand_multiplier * quality_multiplier
           
           return PricingQuote(
               resource_spec=spec,
               base_price=base_price,
               quality_multiplier=quality_multiplier,
               demand_multiplier=demand_multiplier, 
               total_price=total_price,
               quote_id=f"quote_{uuid.uuid4().hex[:8]}"
           )
   ```

### Phase 4: Event System Integration (Days 9-10)

1. **Implement Event Bus**
   ```python
   # services/events/event_bus.py
   class EventBus:
       def __init__(self):
           self._subscribers: Dict[str, List[Callable]] = {}
           self._event_queue = asyncio.Queue()
           
       async def subscribe(self, event_type: str, handler: Callable):
           if event_type not in self._subscribers:
               self._subscribers[event_type] = []
           self._subscribers[event_type].append(handler)
           
       async def publish(self, event: ServiceEvent):
           await self._event_queue.put(event)
           
       async def start_processing(self):
           while True:
               event = await self._event_queue.get()
               await self._dispatch_event(event)
               
       async def _dispatch_event(self, event: ServiceEvent):
           handlers = self._subscribers.get(event.event_type, [])
           tasks = [handler(event) for handler in handlers]
           await asyncio.gather(*tasks, return_exceptions=True)
   ```

2. **Wire Up Event Handlers**
   ```python
   # Integration events between services
   class ServiceIntegration:
       async def setup_event_handlers(self, event_bus: EventBus, registry: ServiceRegistry):
           harvesting = await registry.get('harvesting')
           marketplace = await registry.get('marketplace')
           tokenomics = await registry.get('tokenomics')
           
           # Device discovery flow
           await event_bus.subscribe('device_discovered', harvesting.handle_device_discovered)
           await event_bus.subscribe('device_discovered', marketplace.update_supply_metrics)
           
           # Resource pricing flow  
           await event_bus.subscribe('resource_priced', tokenomics.validate_pricing)
           await event_bus.subscribe('auction_completed', tokenomics.process_payments)
           
           # Privacy budget flow
           privacy = await registry.get('privacy')
           await event_bus.subscribe('participant_selected', privacy.allocate_privacy_budget)
   ```

### Phase 5: Testing & Validation (Days 11-14)

1. **Unit Tests for Each Service**
   ```python
   # tests/unit/test_harvesting_service.py
   import pytest
   from unittest.mock import AsyncMock, MagicMock
   
   @pytest.mark.asyncio
   async def test_device_discovery():
       registry = MagicMock()
       service = FogHarvestingService(registry)
       
       # Mock device capabilities
       mock_devices = [
           DeviceCapability(device_id="device1", compute_gflops=10.0, ...),
           DeviceCapability(device_id="device2", compute_gflops=15.0, ...)
       ]
       
       service._discover_from_p2p = AsyncMock(return_value=mock_devices)
       
       discovered = await service.discover_devices()
       
       assert len(discovered) == 2
       assert discovered[0].device_id == "device1"
   ```

2. **Integration Tests**
   ```python
   # tests/integration/test_service_integration.py
   @pytest.mark.asyncio
   async def test_complete_training_flow():
       container = ServiceContainer()
       registry = await container.initialize(test_config)
       
       orchestration = await registry.get('orchestration')
       
       # Test complete federated learning flow
       result = await orchestration.coordinate_training_round(
           requirements=ResourceRequirements(min_participants=3),
           budget=50.0
       )
       
       assert result.status == "completed"
       assert len(result.participants) >= 3
   ```

## Performance Impact Analysis

### Expected Improvements

| Metric | Current | Target | Improvement |
|--------|---------|---------|------------|
| Coupling Score | 39.81 | < 10.0 | 75% reduction |
| Startup Time | ~15 seconds | < 5 seconds | 67% improvement |
| Memory Usage | ~200MB | ~140MB | 30% reduction |
| Test Coverage | 45% | 85% | 89% improvement |
| Deployment Units | 1 monolith | 6 services | Independent scaling |

### Service Communication Overhead

**Estimated Latency Overhead:**
- Event bus dispatch: ~2-5ms per event
- Service registry lookup: ~1ms per lookup  
- Cross-service calls: ~5-15ms additional latency

**Mitigation Strategies:**
1. Local caching of frequently accessed services
2. Async message queuing for non-critical events
3. Circuit breaker patterns for resilience
4. Connection pooling for service calls

## Monitoring & Observability

### Service Health Metrics

```python
@dataclass
class ServiceHealthMetrics:
    service_name: str
    request_count: int
    error_count: int
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    dependency_status: Dict[str, str]
```

### Distributed Tracing

```python
# services/monitoring/tracing.py
import opentelemetry.trace as trace

tracer = trace.get_tracer(__name__)

class FogHarvestingService:
    async def discover_devices(self) -> List[DeviceCapability]:
        with tracer.start_as_current_span("discover_devices") as span:
            span.set_attribute("operation", "device_discovery")
            
            devices = await self._internal_discover()
            
            span.set_attribute("devices_found", len(devices))
            return devices
```

This dependency breaking strategy provides a clear path from the current monolithic structure to a well-architected microservices system, eliminating circular dependencies and improving maintainability, testability, and scalability.