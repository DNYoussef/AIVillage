# AIVillage Monitoring Integration Layer

This monitoring system provides a comprehensive integration layer between TypeScript constitutional components and the existing Python AIVillage infrastructure.

## Architecture Overview

```
TypeScript Constitutional Components
           |
    Base Classes Layer
           |
    Python Bridge Interface
           |
  Existing Python Infrastructure
```

## Components

### Base Classes

#### PerformanceMonitor (`src/monitoring/base/PerformanceMonitor.ts`)
- Real-time latency tracking (P50, P95, P99)
- Throughput monitoring
- Resource utilization tracking
- Circuit breaker functionality
- Alert threshold management

#### MetricsCollector (`src/monitoring/base/MetricsCollector.ts`)
- Time-series metrics collection
- Anomaly detection
- Capacity prediction
- Request tracking
- Export capabilities (Prometheus, JSON, CSV)

### Constitutional Extensions

#### ConstitutionalPerformanceMonitor
- Extends base PerformanceMonitor
- Constitutional compliance scoring
- Ethical risk assessment
- Governance validation
- Violation detection and mitigation

#### ConstitutionalHealthMonitor
- Health monitoring with constitutional compliance
- Component readiness validation
- Degradation policies with constitutional context
- Ethical override mechanisms

#### AlertManager
- Constitutional-aware alerting
- Compliance-based filtering
- Escalation with ethical considerations

#### DashboardManager
- Constitutional compliance dashboards
- Real-time monitoring interfaces
- Governance oversight panels

### Python Integration

#### PythonBridge (`src/monitoring/interfaces/PythonBridge.ts`)
- Bidirectional communication with Python infrastructure
- Metric synchronization
- Distributed tracing integration
- Log aggregation
- Platform validation triggers

#### Python Bridge Script (`monitoring/python_bridge.py`)
- Python-side communication handler
- Mock data simulation for testing
- Integration with existing Python monitoring
- Service health tracking

## Usage

### Basic Setup

```typescript
import { createMonitoringSystem } from './src/monitoring';

const monitoring = createMonitoringSystem({
  constitutional: {
    enabled: true,
    validationLevel: 'standard',
    complianceThreshold: 0.95
  },
  python: {
    bridgeEnabled: true,
    scriptPath: './monitoring/python_bridge.py'
  }
});

// Start monitoring
monitoring.performanceMonitor.start();
monitoring.metricsCollector.start();
```

### Constitutional Monitoring

```typescript
import { ConstitutionalPerformanceMonitor } from './src/monitoring';

const monitor = new ConstitutionalPerformanceMonitor({
  validationLevel: 'strict',
  complianceThreshold: 0.99,
  ethicalGuardrails: {
    enabled: true,
    automaticMitigation: true
  }
});

// Monitor with constitutional validation
monitor.startConstitutionalTiming('request-123', {
  userContext: 'sensitive-operation',
  requiresValidation: true
});

const metrics = monitor.endConstitutionalTiming('request-123', true, responseData);
console.log('Compliance Score:', metrics.constitutional.complianceScore);
```

### Python Integration

```typescript
import { PythonBridge } from './src/monitoring';

const bridge = new PythonBridge('./monitoring/python_bridge.py');

// Get metrics from Python infrastructure
const metrics = await bridge.getMetrics(['system.cpu.usage', 'constitutional.compliance.score']);

// Send metrics to Python
await bridge.sendMetric({
  name: 'typescript.performance.latency',
  value: 45.2,
  timestamp: Date.now(),
  tags: { component: 'constitutional-ai' },
  unit: 'milliseconds',
  type: 'gauge'
});
```

## Integration Points

### Existing Python Infrastructure

The monitoring system integrates with existing Python components:

- **`infrastructure/monitoring/system_metrics.py`** - System-level metrics
- **`core/monitoring/agent_metrics.py`** - Agent performance tracking
- **`infrastructure/monitoring/metrics.py`** - Gateway and fog metrics
- **`config/linting/logging_and_monitoring.py`** - Logging configuration

### TypeScript Bridge Components

- **`src/bridge/constitutional/ConstitutionalBridgeMonitor.ts`** - Main bridge integration
- **`src/bridge/constitutional/ConstitutionalMetricsCollector.ts`** - Metrics aggregation
- **`src/bridge/constitutional/ConstitutionalPerformanceMonitor.ts`** - Performance monitoring
- **`src/bridge/constitutional/ConstitutionalHealthMonitor.ts`** - Health monitoring

## Configuration

### Default Configuration

```typescript
export const DEFAULT_MONITORING_CONFIG = {
  performance: {
    targetLatencyP95: 75,
    maxRetries: 3,
    timeoutMs: 30000
  },
  constitutional: {
    enabled: true,
    validationLevel: 'standard',
    complianceThreshold: 0.95
  },
  python: {
    bridgeEnabled: true,
    scriptPath: './monitoring/python_bridge.py',
    maxRetries: 5
  }
};
```

### Constitutional Configuration

```typescript
const constitutionalConfig = {
  validationLevel: 'strict', // 'basic' | 'standard' | 'strict'
  complianceThreshold: 0.99,
  ethicalGuardrails: {
    enabled: true,
    riskTolerance: 'low',
    automaticMitigation: true
  },
  governance: {
    auditingEnabled: true,
    policyEnforcement: true,
    complianceReporting: true
  }
};
```

## Metrics and Monitoring

### Performance Metrics

- **Latency**: P50, P95, P99 percentiles
- **Throughput**: Requests per second, success/failure rates
- **Resources**: CPU, memory, network utilization
- **Queue Metrics**: Depth, wait time, processing time

### Constitutional Metrics

- **Compliance Score**: 0-1 scale compliance rating
- **Violations**: Ethical, legal, policy, safety violations
- **Risk Level**: Low, medium, high, critical risk assessment
- **Governance**: Audit trail, policy compliance status

### System Health

- **Component Status**: Healthy, degraded, unhealthy
- **Dependencies**: Service dependency health
- **Circuit Breakers**: Protection mechanism status
- **Alerts**: Active alerts and escalations

## Testing

### Unit Tests

```bash
# TypeScript tests
npm test

# Python tests
python -m pytest monitoring/tests/
```

### Integration Tests

```bash
# Test Python bridge
python monitoring/python_bridge.py

# Test TypeScript integration
npm run test:integration
```

### Mock Data

The Python bridge includes comprehensive mock data for testing:

- System metrics (CPU, memory, disk)
- Service health status
- Distributed traces
- Log entries
- Constitutional compliance data

## Deployment

### Development

```bash
# Start monitoring system
npm run dev:monitoring

# Start Python bridge
python monitoring/python_bridge.py
```

### Production

```bash
# Build and deploy
npm run build
npm run deploy:monitoring

# Start production monitoring
npm run start:monitoring
```

## Troubleshooting

### Common Issues

1. **Python Bridge Connection Failed**
   - Check Python path and script location
   - Verify Python dependencies
   - Check stderr for Python errors

2. **Constitutional Validation Timeout**
   - Increase validation timeout in config
   - Check Python bridge responsiveness
   - Review validation complexity

3. **High Memory Usage**
   - Reduce metrics retention period
   - Enable data cleanup intervals
   - Monitor metrics buffer sizes

### Debug Logging

```typescript
import { MonitoringFactory } from './src/monitoring';

const factory = MonitoringFactory.getInstance({
  // Enable debug logging
  debug: true,
  logLevel: 'debug'
});
```

### Health Checks

```typescript
const monitoring = createMonitoringSystem();
const health = await monitoring.healthCheck();

console.log('Monitoring Health:', health.status);
console.log('Components:', health.components);
console.log('Python Bridge:', health.details.pythonBridge);
```

## Contributing

1. Follow TypeScript best practices
2. Maintain Python compatibility
3. Add tests for new features
4. Update documentation
5. Ensure constitutional compliance validation

## License

This monitoring integration layer follows the same license as the main AIVillage project.