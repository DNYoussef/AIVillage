# Network Protocol Expert Agent

## MISSION
Separate PathPolicy algorithm (1,438 LOC) into 6 specialized network services with <300 LOC each.

## SPECIALIZATIONS
- Network routing algorithms
- Protocol switching and optimization
- Path selection strategies  
- Network topology analysis
- Performance monitoring

## TARGET SERVICES

### 1. RouteSelectionService
- **Purpose**: Core route selection algorithms
- **Interface**: `IRouteSelectionService`
- **Methods**: `selectOptimalRoute()`, `evaluateRoutes()`, `rankPaths()`
- **Target LOC**: <280 lines
- **Focus**: Route optimization algorithms

### 2. ProtocolManager
- **Purpose**: Network protocol management
- **Interface**: `IProtocolManager`
- **Methods**: `switchProtocols()`, `validateProtocol()`, `optimizeProtocol()`
- **Target LOC**: <250 lines
- **Focus**: Protocol abstraction layer

### 3. NetworkMonitor  
- **Purpose**: Real-time network monitoring
- **Interface**: `INetworkMonitor`
- **Methods**: `monitorLatency()`, `trackThroughput()`, `detectCongestion()`
- **Target LOC**: <240 lines
- **Focus**: Performance metrics collection

### 4. PathOptimizer
- **Purpose**: Path optimization strategies
- **Interface**: `IPathOptimizer`
- **Methods**: `optimizePath()`, `calculateMetrics()`, `adjustRouting()`
- **Target LOC**: <220 lines
- **Focus**: Advanced optimization algorithms

### 5. TopologyAnalyzer
- **Purpose**: Network topology analysis
- **Interface**: `ITopologyAnalyzer`
- **Methods**: `analyzeTopology()`, `identifyBottlenecks()`, `suggestImprovements()`
- **Target LOC**: <200 lines
- **Focus**: Network structure analysis

### 6. PolicyEngine
- **Purpose**: Routing policy enforcement
- **Interface**: `IPolicyEngine`
- **Methods**: `enforcePolicy()`, `validateRules()`, `updatePolicies()`
- **Target LOC**: <180 lines
- **Focus**: Policy management and enforcement

## SUCCESS CRITERIA
- PathPolicy: 1,438 LOC → 6 services <300 LOC each
- Algorithm separation: Complete
- Performance: Maintained or improved
- Interface compliance: 100%
- Policy flexibility: Enhanced

## COORDINATION PROTOCOLS
- Memory key: `swarm/network/protocols`  
- Status updates: Every 30 minutes
- Dependencies: Performance Optimization Specialist
- Validation: Testing Coordinator integration tests required