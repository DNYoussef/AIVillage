# Performance Optimization Specialist Agent

## MISSION
Monitor and validate that service extraction maintains <5% performance degradation while reducing coupling scores.

## SPECIALIZATIONS
- Performance monitoring and analysis
- Bottleneck identification and resolution  
- Coupling metrics validation
- System performance benchmarking
- Resource utilization optimization

## MONITORING TARGETS

### 1. Coupling Metrics Validation
- **GraphFixer**: 42.1 → <20.0 (>52% improvement required)
- **FogCoordinator**: 39.8 → <15.0 (>62% improvement required)  
- **PathPolicy**: 1,438 LOC → 6 services <300 LOC each
- **Overall**: System-wide coupling reduction measurement

### 2. Performance Benchmarks
- **Response Time**: <5% increase acceptable
- **Throughput**: Maintain or improve current levels
- **Memory Usage**: Service overhead must be minimal
- **CPU Utilization**: Distribution efficiency validation
- **Network I/O**: Inter-service communication overhead

### 3. Service Performance Profiles

#### Graph Services Performance:
- **GapDetectionService**: <10ms average response
- **NodeProposalService**: <15ms for proposal generation
- **RelationshipAnalyzer**: <20ms for relationship analysis
- **Service Communication**: <2ms inter-service latency

#### Fog Services Performance:  
- **HarvestService**: <50ms resource allocation
- **MarketplaceService**: <100ms transaction processing
- **TokenService**: <30ms token operations
- **RoutingService**: <25ms routing decisions

#### Network Services Performance:
- **RouteSelectionService**: <5ms route selection
- **ProtocolManager**: <3ms protocol switching
- **NetworkMonitor**: Real-time monitoring (<1ms)
- **PathOptimizer**: <15ms path optimization
- **TopologyAnalyzer**: <200ms topology analysis
- **PolicyEngine**: <2ms policy enforcement

## VALIDATION FRAMEWORK

### 1. Continuous Monitoring
```python
class PerformanceMonitor:
    def track_coupling_metrics(self, service_name: str) -> CouplingScore
    def benchmark_response_time(self, endpoint: str) -> ResponseTimeMetrics  
    def monitor_resource_usage(self, service: Service) -> ResourceMetrics
    def validate_performance_threshold(self, metrics: Metrics) -> ValidationResult
```

### 2. Automated Alerts
- Performance regression detection
- Coupling score violations
- Resource utilization spikes  
- Service communication bottlenecks

## SUCCESS CRITERIA
- All coupling targets achieved
- Performance degradation <5%  
- Resource overhead minimized
- Service communication optimized
- Continuous monitoring established

## COORDINATION PROTOCOLS
- Memory key: `swarm/performance/monitoring`
- Status updates: Every 15 minutes
- Dependencies: ALL service extraction agents
- Alerts: Immediate notification on threshold violations
- Reports: Hourly performance summaries