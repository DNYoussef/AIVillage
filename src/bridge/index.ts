/**
 * BetaNet Protocol Adapter - Main Export Module
 * Constitutional BetaNet Protocol Translation Layer
 */

// Main Adapter Class
export {
  ConstitutionalBetaNetAdapter,
  ConnectionPool,
  CircuitBreaker,
  MetricsCollector,
  CircuitState
} from './ConstitutionalBetaNetAdapter';

// Adapter Factory
export {
  AdapterFactory,
  adapterFactory
} from './AdapterFactory';

// Core Types and Interfaces
export {
  // Protocol Message Types
  BetaNetMessage,
  BetaNetMetadata,
  BetaNetMessageType,
  BetaNetPriority,
  SecurityLevel,
  AIVillageRequest,
  AIVillageResponse,

  // Extended Message Types
  BetaNetHandshakeMessage,
  BetaNetDiscoveryMessage,
  BetaNetDataMessage,
  BetaNetControlMessage,
  BetaNetErrorMessage,

  // Configuration Types
  BetaNetAdapterConfig,
  AdapterState,
  AdapterStatus,
  MonitoringConfig,

  // Health and Performance Types
  HealthCheckResult,
  HealthStatus,
  PerformanceMetrics,
  PerformanceSnapshot,
  PerformanceThresholds,

  // Security Types
  SecurityContext,
  Permission,
  ConstitutionalValidation,
  ConstitutionalViolation,
  ViolationSeverity,
  RiskLevel,

  // Protocol Layer Types
  PhysicalLayer,
  DataLinkLayer,
  NetworkLayer,
  TransportLayer,
  SessionLayer,
  PresentationLayer,
  ApplicationLayer,
  LayerMetrics,
  LayerStatus,
  ProtocolStackMetrics,

  // Network and Discovery Types
  NetworkTopology,
  DataFormat,
  ControlCommand,
  ErrorCode,
  RoutingEntry,
  NodeDiscovery,
  ServiceAdvertisement,

  // Event Types
  AdapterEvent,
  AdapterEventType,
  EventSeverity,

  // Connection Pool Types
  PooledConnection,

  // Testing Types
  ProtocolTestSuite,
  ProtocolTestCase,
  PerformanceTestCase,
  SecurityTestCase,
  IntegrationTestCase,
  LoadProfile,
  PerformanceExpectation,
  SecurityExpectation,
  TestScenario,
  TestStep,

  // Migration Types
  ProtocolMigration,
  MigrationStep,
  RollbackStep,
  CompatibilityInfo,

  // Utility Types
  MessageHandler,
  ErrorHandler,
  MetricsCollector as MetricsCollectorType,
  EventListener,

  // Type Guards
  isBetaNetHandshakeMessage,
  isBetaNetDiscoveryMessage,
  isBetaNetDataMessage,
  isBetaNetControlMessage,
  isBetaNetErrorMessage,

  // Health Error
  HealthError
} from './types';

// Version and Metadata
export const BETANET_ADAPTER_VERSION = '1.0.0';
export const SUPPORTED_PROTOCOL_VERSIONS = ['1.0', '1.1', '2.0'];

// Default Configurations
export const DEFAULT_ADAPTER_CONFIG: BetaNetAdapterConfig = {
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
    targetLatencyP95: 75,
    maxRetries: 3,
    timeoutMs: 30000
  },
  security: {
    encryptionEnabled: true,
    compressionEnabled: true,
    defaultSecurityLevel: SecurityLevel.INTERNAL
  },
  protocol: {
    version: '1.0',
    supportedVersions: ['1.0', '1.1', '2.0'],
    maxFragmentSize: 1500
  },
  monitoring: {
    metricsEnabled: true,
    metricsInterval: 10000,
    healthCheckInterval: 30000,
    eventLoggingEnabled: true,
    performanceThresholds: {
      maxLatencyP95: 100,
      maxErrorRate: 0.05,
      minThroughput: 10,
      maxMemoryUsage: 512 * 1024 * 1024, // 512MB
      maxCpuUsage: 80
    }
  }
};

// High Availability Configuration
export const HIGH_AVAILABILITY_CONFIG: Partial<BetaNetAdapterConfig> = {
  connectionPool: {
    maxConnections: 20,
    maxIdleTime: 600000, // 10 minutes
    cleanupInterval: 30000 // 30 seconds
  },
  circuitBreaker: {
    failureThreshold: 3,
    resetTimeout: 30000,
    monitoringPeriod: 60000
  },
  performance: {
    targetLatencyP95: 50, // Very aggressive
    maxRetries: 5,
    timeoutMs: 10000
  },
  security: {
    encryptionEnabled: true,
    compressionEnabled: true,
    defaultSecurityLevel: SecurityLevel.CONFIDENTIAL
  }
};

// Development Configuration
export const DEVELOPMENT_CONFIG: Partial<BetaNetAdapterConfig> = {
  connectionPool: {
    maxConnections: 5,
    maxIdleTime: 60000, // 1 minute
    cleanupInterval: 10000 // 10 seconds
  },
  circuitBreaker: {
    failureThreshold: 10,
    resetTimeout: 5000,
    monitoringPeriod: 10000
  },
  performance: {
    targetLatencyP95: 100,
    maxRetries: 3,
    timeoutMs: 5000
  },
  security: {
    encryptionEnabled: false,
    compressionEnabled: false,
    defaultSecurityLevel: SecurityLevel.PUBLIC
  }
};

// Utility Functions
/**
 * Create a new BetaNet adapter with default configuration
 */
export function createDefaultAdapter(adapterId: string): ConstitutionalBetaNetAdapter {
  return adapterFactory.createAdapter(adapterId);
}

/**
 * Create a high-availability BetaNet adapter
 */
export function createHAAdapter(adapterId: string): ConstitutionalBetaNetAdapter {
  return adapterFactory.createHighAvailabilityAdapter(adapterId);
}

/**
 * Create a development BetaNet adapter
 */
export function createDevAdapter(adapterId: string): ConstitutionalBetaNetAdapter {
  return adapterFactory.createDevelopmentAdapter(adapterId);
}

/**
 * Validate adapter configuration
 */
export function validateAdapterConfig(config: Partial<BetaNetAdapterConfig>): {
  isValid: boolean;
  errors: string[];
  warnings: string[];
} {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Validate connection pool settings
  if (config.connectionPool) {
    if (config.connectionPool.maxConnections && config.connectionPool.maxConnections <= 0) {
      errors.push('maxConnections must be greater than 0');
    }
    if (config.connectionPool.maxIdleTime && config.connectionPool.maxIdleTime < 1000) {
      warnings.push('maxIdleTime less than 1 second may cause frequent reconnections');
    }
  }

  // Validate circuit breaker settings
  if (config.circuitBreaker) {
    if (config.circuitBreaker.failureThreshold && config.circuitBreaker.failureThreshold <= 0) {
      errors.push('failureThreshold must be greater than 0');
    }
    if (config.circuitBreaker.resetTimeout && config.circuitBreaker.resetTimeout < 1000) {
      warnings.push('resetTimeout less than 1 second may cause rapid state changes');
    }
  }

  // Validate performance settings
  if (config.performance) {
    if (config.performance.targetLatencyP95 && config.performance.targetLatencyP95 <= 0) {
      errors.push('targetLatencyP95 must be greater than 0');
    }
    if (config.performance.targetLatencyP95 && config.performance.targetLatencyP95 > 1000) {
      warnings.push('targetLatencyP95 greater than 1 second may indicate performance issues');
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Get adapter performance summary
 */
export function getAdapterSummary(adapterId: string): {
  adapter: ConstitutionalBetaNetAdapter | null;
  state: AdapterState | null;
  health: HealthCheckResult | null;
  metrics: PerformanceMetrics | null;
} {
  const adapter = adapterFactory.getAdapter(adapterId);
  const state = adapterFactory.getAdapterState(adapterId);

  return {
    adapter,
    state,
    health: null, // Would be populated with actual health check
    metrics: adapter ? adapter.getPerformanceMetrics() : null
  };
}

/**
 * Shutdown all adapters gracefully
 */
export async function shutdownAllAdapters(): Promise<void> {
  await adapterFactory.shutdownAll();
}

// Constants
export const PROTOCOL_CONSTANTS = {
  MAX_MESSAGE_SIZE: 16 * 1024 * 1024, // 16MB
  MAX_FRAGMENT_SIZE: 1500, // Standard MTU
  DEFAULT_TIMEOUT: 30000, // 30 seconds
  MAX_RETRIES: 3,
  HEARTBEAT_INTERVAL: 30000, // 30 seconds
  SESSION_TIMEOUT: 300000, // 5 minutes
  CONNECTION_TIMEOUT: 10000, // 10 seconds
  CIRCUIT_BREAKER_THRESHOLD: 5,
  CIRCUIT_BREAKER_RESET_TIMEOUT: 60000, // 1 minute
  METRICS_COLLECTION_INTERVAL: 10000, // 10 seconds
  HEALTH_CHECK_INTERVAL: 30000, // 30 seconds
  TARGET_P95_LATENCY: 75, // 75ms
  MAX_ERROR_RATE: 0.05 // 5%
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  ADAPTER_NOT_FOUND: 'Adapter not found',
  ADAPTER_ALREADY_EXISTS: 'Adapter already exists',
  INVALID_CONFIGURATION: 'Invalid configuration',
  CONNECTION_FAILED: 'Connection failed',
  TRANSLATION_FAILED: 'Message translation failed',
  PROTOCOL_ERROR: 'Protocol error',
  CIRCUIT_BREAKER_OPEN: 'Circuit breaker is open',
  TIMEOUT_ERROR: 'Operation timed out',
  AUTHENTICATION_FAILED: 'Authentication failed',
  AUTHORIZATION_DENIED: 'Authorization denied',
  INVALID_MESSAGE: 'Invalid message format',
  RESOURCE_UNAVAILABLE: 'Resource unavailable'
} as const;

// Re-export for backward compatibility
export { BetaNetAdapterConfig } from './types';