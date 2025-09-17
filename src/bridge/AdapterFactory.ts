/**
 * BetaNet Adapter Factory
 * Factory pattern for creating and configuring protocol adapters
 */

import {
  ConstitutionalBetaNetAdapter,
  CircuitBreaker,
  ConnectionPool,
  MetricsCollector,
  CircuitState
} from './ConstitutionalBetaNetAdapter';
import {
  BetaNetAdapterConfig,
  AdapterState,
  AdapterStatus,
  MonitoringConfig,
  HealthCheckResult,
  HealthStatus,
  SecurityLevel
} from './types';

export class AdapterFactory {
  private static instance: AdapterFactory;
  private adapters: Map<string, ConstitutionalBetaNetAdapter> = new Map();
  private configurations: Map<string, BetaNetAdapterConfig> = new Map();

  private constructor() {}

  static getInstance(): AdapterFactory {
    if (!AdapterFactory.instance) {
      AdapterFactory.instance = new AdapterFactory();
    }
    return AdapterFactory.instance;
  }

  /**
   * Create a new BetaNet adapter with custom configuration
   */
  createAdapter(
    adapterId: string,
    config?: Partial<BetaNetAdapterConfig>
  ): ConstitutionalBetaNetAdapter {
    if (this.adapters.has(adapterId)) {
      throw new Error(`Adapter with ID '${adapterId}' already exists`);
    }

    const fullConfig = this.mergeWithDefaults(config);
    this.configurations.set(adapterId, fullConfig);

    const adapter = new ConstitutionalBetaNetAdapter();
    this.setupAdapterConfiguration(adapter, fullConfig);
    this.setupAdapterMonitoring(adapter, adapterId, fullConfig.monitoring);

    this.adapters.set(adapterId, adapter);
    return adapter;
  }

  /**
   * Get existing adapter by ID
   */
  getAdapter(adapterId: string): ConstitutionalBetaNetAdapter | null {
    return this.adapters.get(adapterId) || null;
  }

  /**
   * Remove and cleanup adapter
   */
  async removeAdapter(adapterId: string): Promise<void> {
    const adapter = this.adapters.get(adapterId);
    if (adapter) {
      await adapter.shutdown();
      this.adapters.delete(adapterId);
      this.configurations.delete(adapterId);
    }
  }

  /**
   * List all active adapters
   */
  listAdapters(): string[] {
    return Array.from(this.adapters.keys());
  }

  /**
   * Get adapter state
   */
  getAdapterState(adapterId: string): AdapterState | null {
    const adapter = this.adapters.get(adapterId);
    if (!adapter) return null;

    const metrics = adapter.getPerformanceMetrics();

    return {
      status: this.determineAdapterStatus(metrics),
      activeConnections: metrics.activeConnections,
      totalRequests: metrics.requestCount,
      totalErrors: Math.floor(metrics.requestCount * metrics.errorRate),
      averageLatency: metrics.averageLatency,
      circuitBreakerState: metrics.circuitBreakerState,
      lastHealthCheck: Date.now(),
      version: '1.0.0'
    };
  }

  /**
   * Perform health check on adapter
   */
  async performHealthCheck(adapterId: string): Promise<HealthCheckResult | null> {
    const adapter = this.adapters.get(adapterId);
    const config = this.configurations.get(adapterId);

    if (!adapter || !config) return null;

    const metrics = adapter.getPerformanceMetrics();
    const timestamp = Date.now();

    return {
      status: this.determineOverallHealth(metrics, config),
      timestamp,
      checks: {
        connectionPool: this.checkConnectionPool(metrics),
        circuitBreaker: this.checkCircuitBreaker(metrics),
        protocolStack: this.checkProtocolStack(metrics),
        performance: this.checkPerformance(metrics, config)
      },
      metrics: {
        latencyP50: metrics.averageLatency,
        latencyP95: metrics.p95Latency,
        latencyP99: metrics.p99Latency,
        throughput: metrics.throughput,
        errorRate: metrics.errorRate,
        activeConnections: metrics.activeConnections,
        queueDepth: 0 // Would be implemented with actual queue
      },
      errors: this.collectHealthErrors(metrics, config)
    };
  }

  /**
   * Update adapter configuration
   */
  async updateAdapterConfiguration(
    adapterId: string,
    configUpdate: Partial<BetaNetAdapterConfig>
  ): Promise<boolean> {
    const adapter = this.adapters.get(adapterId);
    const currentConfig = this.configurations.get(adapterId);

    if (!adapter || !currentConfig) return false;

    const newConfig = { ...currentConfig, ...configUpdate };
    this.configurations.set(adapterId, newConfig);

    // Apply configuration changes
    await this.setupAdapterConfiguration(adapter, newConfig);

    return true;
  }

  /**
   * Create adapter with high-availability configuration
   */
  createHighAvailabilityAdapter(adapterId: string): ConstitutionalBetaNetAdapter {
    const haConfig: Partial<BetaNetAdapterConfig> = {
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

    return this.createAdapter(adapterId, haConfig);
  }

  /**
   * Create adapter optimized for development/testing
   */
  createDevelopmentAdapter(adapterId: string): ConstitutionalBetaNetAdapter {
    const devConfig: Partial<BetaNetAdapterConfig> = {
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

    return this.createAdapter(adapterId, devConfig);
  }

  /**
   * Shutdown all adapters
   */
  async shutdownAll(): Promise<void> {
    const shutdownPromises = Array.from(this.adapters.values()).map(adapter =>
      adapter.shutdown()
    );

    await Promise.all(shutdownPromises);
    this.adapters.clear();
    this.configurations.clear();
  }

  // Private helper methods
  private mergeWithDefaults(config?: Partial<BetaNetAdapterConfig>): BetaNetAdapterConfig {
    const defaults: BetaNetAdapterConfig = {
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

    return this.deepMerge(defaults, config || {});
  }

  private deepMerge(target: any, source: any): any {
    const result = { ...target };

    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        result[key] = this.deepMerge(target[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }

    return result;
  }

  private setupAdapterConfiguration(
    adapter: ConstitutionalBetaNetAdapter,
    config: BetaNetAdapterConfig
  ): void {
    // Configuration would be applied to adapter instance
    // This is a mock implementation as the adapter constructor doesn't accept config
    console.log(`Applied configuration to adapter:`, config);
  }

  private setupAdapterMonitoring(
    adapter: ConstitutionalBetaNetAdapter,
    adapterId: string,
    monitoringConfig: MonitoringConfig
  ): void {
    if (!monitoringConfig.metricsEnabled) return;

    // Set up metrics collection
    const metricsInterval = setInterval(() => {
      const metrics = adapter.getPerformanceMetrics();
      this.publishMetrics(adapterId, metrics);
    }, monitoringConfig.metricsInterval);

    // Set up health checks
    const healthCheckInterval = setInterval(async () => {
      const healthResult = await this.performHealthCheck(adapterId);
      if (healthResult) {
        this.publishHealthCheck(adapterId, healthResult);
      }
    }, monitoringConfig.healthCheckInterval);

    // Set up event logging
    if (monitoringConfig.eventLoggingEnabled) {
      adapter.on('requestTranslated', (event) => {
        this.logEvent(adapterId, 'REQUEST_TRANSLATED', event);
      });

      adapter.on('responseTranslated', (event) => {
        this.logEvent(adapterId, 'RESPONSE_TRANSLATED', event);
      });

      adapter.on('translationError', (event) => {
        this.logEvent(adapterId, 'TRANSLATION_ERROR', event);
      });

      adapter.on('circuitBreakerStateChange', (state) => {
        this.logEvent(adapterId, 'CIRCUIT_BREAKER_STATE_CHANGE', { state });
      });
    }

    // Cleanup on adapter shutdown
    adapter.once('shutdown', () => {
      clearInterval(metricsInterval);
      clearInterval(healthCheckInterval);
    });
  }

  private determineAdapterStatus(metrics: any): AdapterStatus {
    if (metrics.circuitBreakerState === CircuitState.OPEN) {
      return AdapterStatus.FAILING;
    }

    if (metrics.errorRate > 0.1 || metrics.p95Latency > 200) {
      return AdapterStatus.DEGRADED;
    }

    return AdapterStatus.ACTIVE;
  }

  private determineOverallHealth(metrics: any, config: BetaNetAdapterConfig): HealthStatus {
    if (metrics.circuitBreakerState === CircuitState.OPEN) {
      return HealthStatus.CRITICAL;
    }

    if (metrics.p95Latency > config.monitoring.performanceThresholds.maxLatencyP95 ||
        metrics.errorRate > config.monitoring.performanceThresholds.maxErrorRate) {
      return HealthStatus.WARNING;
    }

    return HealthStatus.HEALTHY;
  }

  private checkConnectionPool(metrics: any): HealthStatus {
    return metrics.activeConnections > 0 ? HealthStatus.HEALTHY : HealthStatus.WARNING;
  }

  private checkCircuitBreaker(metrics: any): HealthStatus {
    switch (metrics.circuitBreakerState) {
      case CircuitState.CLOSED:
        return HealthStatus.HEALTHY;
      case CircuitState.HALF_OPEN:
        return HealthStatus.WARNING;
      case CircuitState.OPEN:
        return HealthStatus.CRITICAL;
      default:
        return HealthStatus.UNKNOWN;
    }
  }

  private checkProtocolStack(metrics: any): HealthStatus {
    // Mock implementation - would check each layer
    return HealthStatus.HEALTHY;
  }

  private checkPerformance(metrics: any, config: BetaNetAdapterConfig): HealthStatus {
    const thresholds = config.monitoring.performanceThresholds;

    if (metrics.p95Latency > thresholds.maxLatencyP95 * 1.5 ||
        metrics.errorRate > thresholds.maxErrorRate * 2) {
      return HealthStatus.CRITICAL;
    }

    if (metrics.p95Latency > thresholds.maxLatencyP95 ||
        metrics.errorRate > thresholds.maxErrorRate) {
      return HealthStatus.WARNING;
    }

    return HealthStatus.HEALTHY;
  }

  private collectHealthErrors(metrics: any, config: BetaNetAdapterConfig): any[] {
    const errors: any[] = [];
    const thresholds = config.monitoring.performanceThresholds;

    if (metrics.p95Latency > thresholds.maxLatencyP95) {
      errors.push({
        code: 'HIGH_LATENCY',
        message: `P95 latency ${metrics.p95Latency}ms exceeds threshold ${thresholds.maxLatencyP95}ms`,
        severity: 'medium',
        timestamp: Date.now(),
        component: 'performance'
      });
    }

    if (metrics.errorRate > thresholds.maxErrorRate) {
      errors.push({
        code: 'HIGH_ERROR_RATE',
        message: `Error rate ${(metrics.errorRate * 100).toFixed(2)}% exceeds threshold ${(thresholds.maxErrorRate * 100).toFixed(2)}%`,
        severity: 'high',
        timestamp: Date.now(),
        component: 'reliability'
      });
    }

    return errors;
  }

  private publishMetrics(adapterId: string, metrics: any): void {
    // Would publish to monitoring system
    console.log(`Metrics for ${adapterId}:`, metrics);
  }

  private publishHealthCheck(adapterId: string, healthResult: any): void {
    // Would publish to monitoring system
    console.log(`Health check for ${adapterId}:`, healthResult);
  }

  private logEvent(adapterId: string, eventType: string, data: any): void {
    // Would log to centralized logging system
    console.log(`Event ${eventType} for ${adapterId}:`, data);
  }
}

// Singleton export
export const adapterFactory = AdapterFactory.getInstance();