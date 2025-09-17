# Constitutional BetaNet Protocol Adapter

A high-performance, 7-layer protocol translation bridge for bidirectional communication between AIVillage HTTP/REST and BetaNet protocol systems.

## Overview

The Constitutional BetaNet Adapter provides enterprise-grade protocol translation with built-in performance optimization, circuit breaker patterns, connection pooling, and comprehensive monitoring. Target performance: **<75ms p95 latency** with fault tolerance and constitutional compliance validation.

## Architecture

### 7-Layer Protocol Stack

The adapter implements the full OSI model for robust protocol translation:

1. **Physical Layer** - Connection management and data transmission
2. **Data Link Layer** - Framing, error detection, and checksums
3. **Network Layer** - Routing, fragmentation, and reassembly
4. **Transport Layer** - Reliability, flow control, and connection management
5. **Session Layer** - Session establishment, state management, and termination
6. **Presentation Layer** - Encryption, compression, and data format conversion
7. **Application Layer** - Message processing, validation, and routing

### Key Components

- **ConstitutionalBetaNetAdapter** - Main adapter class with 7-layer translation
- **AdapterFactory** - Factory pattern for creating configured adapters
- **ConnectionPool** - Efficient connection reuse and management
- **CircuitBreaker** - Fault tolerance with automatic recovery
- **MetricsCollector** - Performance monitoring and analytics

## Features

### Performance Optimization
- **Connection Pooling** - Reuse connections for efficiency
- **Circuit Breaker** - Automatic failure detection and recovery
- **Async/Await** - Non-blocking operations throughout
- **Target Latency** - <75ms p95 latency requirement
- **Load Balancing** - Multiple connection endpoints

### Protocol Translation
- **Bidirectional** - AIVillageRequest ↔ BetaNetMessage translation
- **Version Negotiation** - Support for multiple protocol versions
- **Message Types** - Data, Control, Discovery, Handshake, Error, Heartbeat
- **Priority Handling** - Message priority and QoS support
- **Fragmentation** - Large message handling with reassembly

### Security & Compliance
- **Constitutional Validation** - Built-in compliance checking
- **Security Levels** - Public, Internal, Confidential, Secret
- **Encryption** - End-to-end encryption support
- **Authentication** - Session-based authentication
- **Audit Trails** - Comprehensive logging and monitoring

### Monitoring & Observability
- **Real-time Metrics** - Latency, throughput, error rates
- **Health Checks** - Component-level health monitoring
- **Event Emission** - Comprehensive event system
- **Performance Tracking** - P50, P95, P99 latency tracking
- **Circuit Breaker State** - Fault tolerance monitoring

## Quick Start

### Basic Usage

```typescript
import { ConstitutionalBetaNetAdapter, adapterFactory } from './bridge';

// Create adapter with default configuration
const adapter = adapterFactory.createAdapter('my-adapter');

// Translate HTTP request to BetaNet message
const aivillageRequest = {
  method: 'GET',
  path: '/api/users/123',
  headers: { 'Content-Type': 'application/json' },
  timestamp: Date.now(),
  sessionId: 'session_123'
};

const betaNetMessage = await adapter.translateRequestToBetaNet(aivillageRequest);

// Translate BetaNet message back to HTTP response
const aivillageResponse = await adapter.translateResponseFromBetaNet(betaNetMessage);

// Get performance metrics
const metrics = adapter.getPerformanceMetrics();
console.log('P95 Latency:', metrics.p95Latency);
```

### High Availability Setup

```typescript
import { adapterFactory } from './bridge';

// Create HA adapter with optimized settings
const haAdapter = adapterFactory.createHighAvailabilityAdapter('ha-adapter');

// Monitor health
const healthCheck = await adapterFactory.performHealthCheck('ha-adapter');
console.log('Health Status:', healthCheck.status);
```

### Development Setup

```typescript
import { createDevAdapter } from './bridge';

// Create development adapter with relaxed settings
const devAdapter = createDevAdapter('dev-adapter');
```

## Configuration

### Default Configuration

```typescript
const config = {
  connectionPool: {
    maxConnections: 10,
    maxIdleTime: 300000, // 5 minutes
    cleanupInterval: 60000 // 1 minute
  },
  circuitBreaker: {
    failureThreshold: 5,
    resetTimeout: 60000,
    monitoringPeriod: 60000
  },
  performance: {
    targetLatencyP95: 75, // 75ms target
    maxRetries: 3,
    timeoutMs: 30000
  },
  security: {
    encryptionEnabled: true,
    compressionEnabled: true,
    defaultSecurityLevel: 'internal'
  }
};
```

### Custom Configuration

```typescript
import { adapterFactory, validateAdapterConfig } from './bridge';

const customConfig = {
  performance: {
    targetLatencyP95: 50, // Aggressive latency target
    maxRetries: 5
  },
  connectionPool: {
    maxConnections: 20
  }
};

// Validate configuration
const validation = validateAdapterConfig(customConfig);
if (validation.isValid) {
  const adapter = adapterFactory.createAdapter('custom-adapter', customConfig);
}
```

## Protocol Translation

### Message Types

#### AIVillage Request → BetaNet Message
```typescript
// HTTP GET → BetaNet DATA_TRANSFER
// HTTP POST → BetaNet DATA_TRANSFER
// HTTP DELETE → BetaNet CONTROL
// HTTP OPTIONS → BetaNet DISCOVERY
```

#### BetaNet Message Types
- **DISCOVERY** - Service discovery and capability negotiation
- **HANDSHAKE** - Protocol version and feature negotiation
- **DATA_TRANSFER** - Application data exchange
- **CONTROL** - System control commands
- **ERROR** - Error reporting and handling
- **HEARTBEAT** - Connection liveness checks
- **TERMINATION** - Clean connection shutdown

### Security Levels
- **PUBLIC** - No special protection required
- **INTERNAL** - Internal system use only
- **CONFIDENTIAL** - Sensitive business data
- **SECRET** - Highly classified information

### Constitutional Flags
Headers like `x-constitutional-flags` are automatically parsed:
```typescript
// privacy-protected, audit-required, data-classification
```

## Session Management

```typescript
// Create session
const sessionId = await adapter.createSession({
  destination: 'user_service',
  securityLevel: 'confidential',
  timeout: 300000 // 5 minutes
});

// Use session
const request = {
  // ... request data
  sessionId: sessionId
};

// Terminate session
await adapter.terminateSession(sessionId);
```

## Protocol Negotiation

```typescript
// Negotiate protocol version and capabilities
const negotiation = await adapter.negotiateProtocol('1.1', [
  'encryption',
  'compression',
  'fragmentation',
  'priority_queuing'
]);

console.log('Negotiated version:', negotiation.version);
console.log('Supported features:', negotiation.supportedFeatures);
```

## Performance Monitoring

### Real-time Metrics
```typescript
const metrics = adapter.getPerformanceMetrics();

console.log({
  requestCount: metrics.requestCount,
  averageLatency: metrics.averageLatency,
  p95Latency: metrics.p95Latency,
  p99Latency: metrics.p99Latency,
  errorRate: metrics.errorRate,
  throughput: metrics.throughput,
  activeConnections: metrics.activeConnections,
  circuitBreakerState: metrics.circuitBreakerState
});
```

### Health Checks
```typescript
const health = await adapterFactory.performHealthCheck('adapter-id');

console.log({
  status: health.status, // healthy, warning, critical
  checks: {
    connectionPool: health.checks.connectionPool,
    circuitBreaker: health.checks.circuitBreaker,
    protocolStack: health.checks.protocolStack,
    performance: health.checks.performance
  },
  errors: health.errors
});
```

## Event Handling

```typescript
// Listen for events
adapter.on('requestTranslated', (event) => {
  console.log('Request translated:', event.latency, 'ms');
});

adapter.on('responseTranslated', (event) => {
  console.log('Response translated:', event.latency, 'ms');
});

adapter.on('translationError', (event) => {
  console.error('Translation error:', event.error);
});

adapter.on('circuitBreakerStateChange', (state) => {
  console.log('Circuit breaker state:', state);
});
```

## Error Handling

### Circuit Breaker States
- **CLOSED** - Normal operation
- **OPEN** - Failing, requests blocked
- **HALF_OPEN** - Testing recovery

### Error Types
- **PROTOCOL_ERROR** - Protocol violation
- **AUTHENTICATION_FAILED** - Auth failure
- **AUTHORIZATION_DENIED** - Access denied
- **TIMEOUT** - Operation timeout
- **NETWORK_ERROR** - Network failure
- **INVALID_MESSAGE** - Malformed message
- **RESOURCE_UNAVAILABLE** - Resource unavailable
- **CIRCUIT_BREAKER_OPEN** - Circuit breaker open

## Testing

### Unit Tests
```bash
npm test
```

### Performance Tests
```typescript
// Load testing with 100 requests
// Concurrent testing with 50 parallel requests
// Latency verification (<75ms P95)
```

### Integration Tests
```typescript
// End-to-end protocol translation
// Session management
// Error handling
// Circuit breaker behavior
```

## Deployment

### Production
```typescript
const prodAdapter = adapterFactory.createHighAvailabilityAdapter('prod-adapter');
```

### Development
```typescript
const devAdapter = adapterFactory.createDevelopmentAdapter('dev-adapter');
```

### Monitoring Setup
```typescript
// Enable comprehensive monitoring
const config = {
  monitoring: {
    metricsEnabled: true,
    metricsInterval: 10000,
    healthCheckInterval: 30000,
    eventLoggingEnabled: true
  }
};
```

## API Reference

### Core Classes
- `ConstitutionalBetaNetAdapter` - Main adapter implementation
- `AdapterFactory` - Factory for creating adapters
- `ConnectionPool` - Connection pooling and management
- `CircuitBreaker` - Fault tolerance implementation
- `MetricsCollector` - Performance metrics collection

### Utility Functions
- `createDefaultAdapter()` - Create with default config
- `createHAAdapter()` - Create high-availability adapter
- `createDevAdapter()` - Create development adapter
- `validateAdapterConfig()` - Validate configuration
- `getAdapterSummary()` - Get adapter status summary
- `shutdownAllAdapters()` - Graceful shutdown

## Performance Targets

- **P95 Latency**: <75ms
- **Throughput**: 1000+ requests/second
- **Error Rate**: <5%
- **Availability**: 99.9%
- **Connection Efficiency**: 10+ reused connections
- **Circuit Breaker**: <60s recovery time

## Constitutional Compliance

The adapter includes built-in constitutional validation:
- Privacy protection flags
- Audit requirements
- Data classification
- Security level enforcement
- Access control validation

## Security Considerations

- End-to-end encryption support
- Session-based authentication
- Constitutional flag validation
- Security level enforcement
- Audit trail generation
- Circuit breaker protection

## License

See project LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Ensure <75ms P95 latency
5. Submit pull request