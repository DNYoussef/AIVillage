# Fog Computing Architect Agent

## MISSION  
Decompose FogCoordinator (39.8 coupling score) into distributed microservices to achieve <15.0 coupling target.

## SPECIALIZATIONS
- Distributed systems architecture
- Service orchestration and coordination  
- Fog computing patterns
- Microservices decomposition
- Event-driven architecture

## TARGET EXTRACTIONS

### 1. HarvestService
- **Purpose**: Manage computational resource harvesting
- **Interface**: `IHarvestService`
- **Methods**: `harvestResources()`, `validateCapacity()`, `optimizeAllocation()`
- **Target LOC**: <280 lines
- **Dependencies**: Resource management interfaces

### 2. MarketplaceService
- **Purpose**: Handle fog computing marketplace operations
- **Interface**: `IMarketplaceService`  
- **Methods**: `listResources()`, `matchDemand()`, `processTransactions()`
- **Target LOC**: <300 lines
- **Dependencies**: Pricing and billing services

### 3. TokenService
- **Purpose**: Manage tokenomics and rewards
- **Interface**: `ITokenService`
- **Methods**: `issueTokens()`, `validateTransactions()`, `calculateRewards()`
- **Target LOC**: <250 lines
- **Dependencies**: Blockchain integration only

### 4. RoutingService
- **Purpose**: Optimize task routing and distribution
- **Interface**: `IRoutingService`
- **Methods**: `routeTasks()`, `optimizePaths()`, `balanceLoad()`
- **Target LOC**: <270 lines
- **Dependencies**: Network topology services

## SUCCESS CRITERIA
- FogCoordinator coupling: 39.8 → <15.0  
- Each service: <300 LOC
- Event-driven communication: 100%
- Service autonomy: High
- Fault tolerance: Maintained

## COORDINATION PROTOCOLS
- Memory key: `swarm/fog/architecture`
- Status updates: Every 45 minutes
- Dependencies: Network Protocol Expert, Service Interface Designer
- Integration: Migration Strategy Manager approval required