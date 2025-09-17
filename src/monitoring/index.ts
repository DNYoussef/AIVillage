/**
 * AIVillage Monitoring Module - Unified Exports
 * Comprehensive monitoring, metrics collection, and constitutional compliance
 */

// Base Classes
export {
  PerformanceMonitor,
  PerformanceMetrics,
  AlertThreshold,
  AlertEvent,
  PerformanceConfig
} from './base/PerformanceMonitor';

export {
  MetricsCollector,
  MetricPoint,
  TimeSeriesMetric,
  AggregatedMetric,
  AnomalyPoint,
  CapacityPrediction,
  MetricsSnapshot,
  RequestMetrics,
  QueueMetricsData,
  MetricsCollectorConfig,
  AnomalyDetector
} from './base/MetricsCollector';

// Python Integration
export {
  PythonBridge,
  PythonMetric,
  PythonServiceHealth,
  PythonPerformanceData,
  PythonDistributedTracingSpan,
  PythonLogEntry,
  PythonServiceDependency
} from './interfaces/PythonBridge';

// Constitutional Monitoring Components
export {
  ConstitutionalPerformanceMonitor,
  ConstitutionalMetrics,
  ConstitutionalValidationResult,
  ConstitutionalConfig
} from './constitutional/ConstitutionalPerformanceMonitor';

export {
  ConstitutionalHealthMonitor,
  ConstitutionalHealthStatus,
  ConstitutionalHealthConfig,
  ConstitutionalThresholds
} from './constitutional/ConstitutionalHealthMonitor';

export {
  AlertManager,
  AlertRule,
  AlertChannel,
  AlertManagerConfig
} from './constitutional/AlertManager';

export {
  DashboardManager,
  DashboardLayout,
  DashboardWidget,
  DashboardConfig
} from './constitutional/DashboardManager';

// Monitoring Configuration and Constants
export interface MonitoringConfig {
  performance: {
    targetLatencyP95: number;
    maxRetries: number;
    timeoutMs: number;
  };
  metrics: {
    collectionInterval: number;
    retentionPeriod: number;
    anomalyDetectionEnabled: boolean;
  };
  constitutional: {
    enabled: boolean;
    validationLevel: 'basic' | 'standard' | 'strict';
    complianceThreshold: number;
  };
  alerts: {
    enabled: boolean;
    channels: string[];
    rateLimitMs: number;
  };
  python: {
    bridgeEnabled: boolean;
    scriptPath: string;
    maxRetries: number;
  };
  exporters: {
    prometheus: {
      enabled: boolean;
      port: number;
      endpoint: string;
    };
    cloudwatch: {
      enabled: boolean;
      region: string;
      namespace: string;
      batchSize: number;
    };
    datadog: {
      enabled: boolean;
      apiKey: string;
      tags: string[];
      prefix: string;
    };
  };
}

export const DEFAULT_MONITORING_CONFIG: MonitoringConfig = {
  performance: {
    targetLatencyP95: 75,
    maxRetries: 3,
    timeoutMs: 30000
  },
  metrics: {
    collectionInterval: 10000,
    retentionPeriod: 7 * 24 * 60 * 60 * 1000, // 7 days
    anomalyDetectionEnabled: true
  },
  constitutional: {
    enabled: true,
    validationLevel: 'standard',
    complianceThreshold: 0.95
  },
  alerts: {
    enabled: true,
    channels: ['email', 'slack'],
    rateLimitMs: 60000
  },
  python: {
    bridgeEnabled: true,
    scriptPath: './monitoring/python_bridge.py',
    maxRetries: 5
  },
  exporters: {
    prometheus: {
      enabled: true,
      port: 9090,
      endpoint: '/metrics'
    },
    cloudwatch: {
      enabled: false,
      region: 'us-east-1',
      namespace: 'AIVillage',
      batchSize: 20
    },
    datadog: {
      enabled: false,
      apiKey: '',
      tags: ['service:aivillage'],
      prefix: 'aivillage.'
    }
  }
};

// Monitoring Factory for creating integrated monitoring instances
export class MonitoringFactory {
  private static instance?: MonitoringFactory;
  private pythonBridge?: PythonBridge;
  private performanceMonitor?: PerformanceMonitor;
  private metricsCollector?: MetricsCollector;

  private constructor(private config: MonitoringConfig) {}

  public static getInstance(config: MonitoringConfig = DEFAULT_MONITORING_CONFIG): MonitoringFactory {
    if (!MonitoringFactory.instance) {
      MonitoringFactory.instance = new MonitoringFactory(config);
    }
    return MonitoringFactory.instance;
  }

  /**
   * Create performance monitor with configuration
   */
  public createPerformanceMonitor(): PerformanceMonitor {
    if (!this.performanceMonitor) {
      this.performanceMonitor = new PerformanceMonitor({
        circuitBreakerThreshold: this.config.performance.targetLatencyP95,
        resourceMonitoringInterval: this.config.metrics.collectionInterval
      });
    }
    return this.performanceMonitor;
  }

  /**
   * Create metrics collector with configuration
   */
  public createMetricsCollector(): MetricsCollector {
    if (!this.metricsCollector) {
      this.metricsCollector = new MetricsCollector({
        collectionInterval: this.config.metrics.collectionInterval,
        retentionPeriod: this.config.metrics.retentionPeriod,
        anomalyDetectionEnabled: this.config.metrics.anomalyDetectionEnabled
      });
    }
    return this.metricsCollector;
  }

  /**
   * Create Python bridge for existing infrastructure integration
   */
  public createPythonBridge(): PythonBridge | null {
    if (!this.config.python.bridgeEnabled) {
      return null;
    }

    if (!this.pythonBridge) {
      this.pythonBridge = new PythonBridge(this.config.python.scriptPath);
    }
    return this.pythonBridge;
  }

  /**
   * Create constitutional performance monitor
   */
  public createConstitutionalPerformanceMonitor(): ConstitutionalPerformanceMonitor {
    const baseMonitor = this.createPerformanceMonitor();
    return new ConstitutionalPerformanceMonitor({
      validationLevel: this.config.constitutional.validationLevel,
      complianceThreshold: this.config.constitutional.complianceThreshold,
      performanceConfig: {
        circuitBreakerThreshold: this.config.performance.targetLatencyP95
      }
    });
  }

  /**
   * Create constitutional health monitor
   */
  public createConstitutionalHealthMonitor(): ConstitutionalHealthMonitor {
    return new ConstitutionalHealthMonitor({
      complianceThreshold: this.config.constitutional.complianceThreshold,
      validationLevel: this.config.constitutional.validationLevel,
      pythonBridge: this.createPythonBridge()
    });
  }

  /**
   * Get monitoring configuration
   */
  public getConfig(): MonitoringConfig {
    return { ...this.config };
  }

  /**
   * Update monitoring configuration
   */
  public updateConfig(newConfig: Partial<MonitoringConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Health check for all monitoring components
   */
  public async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    components: {
      performanceMonitor: boolean;
      metricsCollector: boolean;
      pythonBridge: boolean;
      constitutional: boolean;
    };
    details: Record<string, any>;
  }> {
    const components = {
      performanceMonitor: !!this.performanceMonitor,
      metricsCollector: !!this.metricsCollector,
      pythonBridge: this.pythonBridge?.isConnectedToPython() || false,
      constitutional: this.config.constitutional.enabled
    };

    const details: Record<string, any> = {};

    if (this.pythonBridge) {
      details.pythonBridge = this.pythonBridge.getBridgeStatus();
    }

    if (this.performanceMonitor) {
      details.performanceMonitor = {
        circuitBreakerActive: this.performanceMonitor.isCircuitBreakerActive(),
        summary: this.performanceMonitor.getPerformanceSummary()
      };
    }

    if (this.metricsCollector) {
      details.metricsCollector = {
        stats: this.metricsCollector.getRequestTimingStats(),
        anomalies: this.metricsCollector.getAnomalyAlerts().length
      };
    }

    const healthyComponents = Object.values(components).filter(Boolean).length;
    const totalComponents = Object.keys(components).length;

    let status: 'healthy' | 'degraded' | 'unhealthy';
    if (healthyComponents === totalComponents) {
      status = 'healthy';
    } else if (healthyComponents >= totalComponents * 0.5) {
      status = 'degraded';
    } else {
      status = 'unhealthy';
    }

    return {
      status,
      components,
      details
    };
  }

  /**
   * Shutdown all monitoring components
   */
  public async shutdown(): Promise<void> {
    if (this.performanceMonitor) {
      this.performanceMonitor.destroy();
    }

    if (this.metricsCollector) {
      this.metricsCollector.destroy();
    }

    if (this.pythonBridge) {
      this.pythonBridge.disconnect();
    }

    MonitoringFactory.instance = undefined;
  }
}

// Utility functions for quick monitoring setup

/**
 * Create a fully configured monitoring system
 */
export function createMonitoringSystem(config?: Partial<MonitoringConfig>) {
  const finalConfig = { ...DEFAULT_MONITORING_CONFIG, ...config };
  const factory = MonitoringFactory.getInstance(finalConfig);

  return {
    performanceMonitor: factory.createPerformanceMonitor(),
    metricsCollector: factory.createMetricsCollector(),
    pythonBridge: factory.createPythonBridge(),
    constitutionalPerformanceMonitor: factory.createConstitutionalPerformanceMonitor(),
    constitutionalHealthMonitor: factory.createConstitutionalHealthMonitor(),
    factory,
    healthCheck: () => factory.healthCheck(),
    shutdown: () => factory.shutdown()
  };
}

/**
 * Quick setup for basic monitoring
 */
export function setupBasicMonitoring() {
  return createMonitoringSystem({
    constitutional: { enabled: false },
    python: { bridgeEnabled: false },
    exporters: {
      prometheus: { enabled: false, port: 9090, endpoint: '/metrics' },
      cloudwatch: { enabled: false, region: 'us-east-1', namespace: 'AIVillage', batchSize: 20 },
      datadog: { enabled: false, apiKey: '', tags: [], prefix: '' }
    }
  });
}

/**
 * Quick setup for production monitoring with all features
 */
export function setupProductionMonitoring() {
  return createMonitoringSystem({
    constitutional: {
      enabled: true,
      validationLevel: 'strict',
      complianceThreshold: 0.99
    },
    python: {
      bridgeEnabled: true,
      scriptPath: './monitoring/python_bridge.py',
      maxRetries: 5
    },
    metrics: {
      collectionInterval: 5000,
      retentionPeriod: 30 * 24 * 60 * 60 * 1000, // 30 days
      anomalyDetectionEnabled: true
    }
  });
}

// Export the monitoring factory as default
export default MonitoringFactory;