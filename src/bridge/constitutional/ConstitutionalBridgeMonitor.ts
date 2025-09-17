/**
 * BridgeMonitor - Integration layer for comprehensive monitoring
 * Coordinates all monitoring components and provides unified interface
 */

import { EventEmitter } from 'events';
import PerformanceMonitor, { PerformanceMetrics } from '../../monitoring/base/PerformanceMonitor';
import MetricsCollector from '../../monitoring/base/MetricsCollector';
import AlertManager from '../../monitoring/constitutional/AlertManager';
import PrometheusExporter from '../../monitoring/exporters/PrometheusExporter';
import CloudWatchExporter from '../../monitoring/exporters/CloudWatchExporter';
import DataDogExporter from '../../monitoring/exporters/DataDogExporter';
import DashboardManager from '../../monitoring/constitutional/DashboardManager';
import { MonitoringConfig, DEFAULT_MONITORING_CONFIG } from '../../monitoring/index';

export interface BridgeMonitorConfig extends MonitoringConfig {
  enabledExporters: string[];
  dashboards: {
    enabled: boolean;
    defaultLayouts: string[];
  };
}

export interface MonitoringContext {
  requestId: string;
  method: string;
  endpoint: string;
  userAgent?: string;
  clientId?: string;
  metadata?: Record<string, any>;
}

export class BridgeMonitor extends EventEmitter {
  private config: BridgeMonitorConfig;
  private performanceMonitor: PerformanceMonitor;
  private metricsCollector: MetricsCollector;
  private alertManager: AlertManager;
  private dashboardManager: DashboardManager;
  
  // Exporters
  private prometheusExporter?: PrometheusExporter;
  private cloudWatchExporter?: CloudWatchExporter;
  private dataDogExporter?: DataDogExporter;
  
  // State tracking
  private isStarted = false;
  private startTime = Date.now();
  
  constructor(config: Partial<BridgeMonitorConfig> = {}) {
    super();
    
    this.config = {
      ...DEFAULT_MONITORING_CONFIG,
      enabledExporters: ['prometheus'],
      dashboards: {
        enabled: true,
        defaultLayouts: ['main-performance', 'network-analysis']
      },
      ...config
    };
    
    this.initializeComponents();
    this.setupEventHandlers();
  }
  
  /**
   * Start monitoring all components
   */
  public async start(): Promise<void> {
    if (this.isStarted) return;
    
    console.log('[BridgeMonitor] Starting comprehensive monitoring system...');
    
    // Initialize exporters
    await this.initializeExporters();
    
    // Start periodic health checks
    this.startHealthChecks();
    
    this.isStarted = true;
    this.emit('started', { timestamp: Date.now() });
    
    console.log('[BridgeMonitor] Monitoring system started successfully');
  }
  
  /**
   * Stop monitoring and clean up resources
   */
  public async stop(): Promise<void> {
    if (!this.isStarted) return;
    
    console.log('[BridgeMonitor] Stopping monitoring system...');
    
    // Stop exporters
    this.prometheusExporter?.stop();
    this.cloudWatchExporter?.stop();
    this.dataDogExporter?.stop();
    
    // Clean up components
    this.metricsCollector.destroy();
    this.dashboardManager.destroy();
    
    this.isStarted = false;
    this.emit('stopped', { timestamp: Date.now() });
    
    console.log('[BridgeMonitor] Monitoring system stopped');
  }
  
  /**
   * Start monitoring a request
   */
  public startRequest(context: MonitoringContext): void {
    if (!this.isStarted) return;
    
    // Start performance timing
    this.performanceMonitor.startTiming(context.requestId, context.metadata);
    
    // Start metrics collection
    this.metricsCollector.startRequest(
      context.requestId,
      context.method,
      context.endpoint,
      {
        userAgent: context.userAgent,
        clientId: context.clientId,
        ...context.metadata
      }
    );
    
    this.emit('requestStarted', context);
  }
  
  /**
   * Mark a timing stage in request processing
   */
  public markStage(requestId: string, stage: string): void {
    if (!this.isStarted) return;
    
    this.performanceMonitor.markStage(requestId, stage);
    this.metricsCollector.markStage(requestId, stage as any);
  }
  
  /**
   * Complete request monitoring
   */
  public completeRequest(
    requestId: string, 
    success: boolean = true, 
    statusCode: number = 200,
    responseSize?: number
  ): PerformanceMetrics | null {
    if (!this.isStarted) return null;
    
    // Complete performance monitoring
    const performanceMetrics = this.performanceMonitor.endTiming(requestId, success);
    
    // Complete metrics collection
    const requestMetrics = this.metricsCollector.completeRequest(
      requestId, 
      success ? undefined : new Error(`HTTP ${statusCode}`)
    );
    
    if (requestMetrics && responseSize) {
      this.metricsCollector.updateMetadata(requestId, { responseSize });
    }
    
    // Record in exporters
    if (requestMetrics) {
      const duration = requestMetrics.endTime! - requestMetrics.startTime;
      
      this.prometheusExporter?.recordRequest(
        duration,
        requestMetrics.method,
        requestMetrics.endpoint,
        statusCode,
        requestMetrics.metadata.payloadSize
      );
      
      this.cloudWatchExporter?.recordRequest(
        duration,
        requestMetrics.method,
        requestMetrics.endpoint,
        statusCode
      );
      
      this.dataDogExporter?.recordRequest(
        duration,
        requestMetrics.method,
        requestMetrics.endpoint,
        statusCode,
        requestMetrics.metadata.userAgent
      );
      
      if (responseSize) {
        this.prometheusExporter?.recordResponse(
          responseSize,
          requestMetrics.method,
          requestMetrics.endpoint,
          statusCode
        );
      }
    }
    
    this.emit('requestCompleted', { requestId, success, statusCode, metrics: performanceMetrics });
    
    return performanceMetrics;
  }
  
  /**
   * Record queue metrics
   */
  public recordQueueMetrics(queueName: string, depth: number, waitTime: number, processingTime: number): void {
    if (!this.isStarted) return;
    
    this.performanceMonitor.recordQueueMetrics(depth, waitTime, processingTime);
    this.metricsCollector.recordQueueMetrics({
      queueName,
      depth,
      waitTime,
      processingTime,
      throughput: 0, // Would be calculated
      backlogSize: depth,
      workers: 1,
      activeWorkers: 1
    });
  }
  
  /**
   * Record custom business metric
   */
  public recordCustomMetric(name: string, value: number, tags: Record<string, string> = {}): void {
    if (!this.isStarted) return;
    
    const tagArray = Object.entries(tags).map(([key, val]) => `${key}:${val}`);
    
    this.prometheusExporter?.setGauge(name, value, tags);
    this.cloudWatchExporter?.recordBusinessMetric(name, value, 'Count', tags);
    this.dataDogExporter?.addGauge(name, value, undefined, tagArray);
  }
  
  /**
   * Get current performance metrics
   */
  public getCurrentMetrics(): PerformanceMetrics {
    return this.performanceMonitor.getCurrentMetrics();
  }
  
  /**
   * Get aggregated metrics for monitoring
   */
  public getAggregatedMetrics(): PerformanceMetrics {
    return this.metricsCollector.getAggregatedMetrics();
  }
  
  /**
   * Get monitoring dashboard data
   */
  public getDashboardData(layoutId: string = 'main-performance'): any {
    return this.dashboardManager.getDashboardData(layoutId);
  }
  
  /**
   * Get system health status
   */
  public getHealthStatus(): any {
    const metrics = this.getCurrentMetrics();
    this.dashboardManager.addMetricsData(metrics);
    return this.dashboardManager.performHealthCheck();
  }
  
  /**
   * Get monitoring statistics
   */
  public getStats(): any {
    return {
      uptime: Date.now() - this.startTime,
      performanceMonitor: {
        circuitBreakerActive: this.performanceMonitor.isCircuitBreakerActive(),
        summary: this.performanceMonitor.getPerformanceSummary()
      },
      metricsCollector: {
        summary: this.metricsCollector.getRequestTimingStats()
      },
      alertManager: {
        stats: this.alertManager.getAlertStats(),
        activeAlerts: this.alertManager.getActiveAlerts().length
      },
      exporters: {
        prometheus: this.prometheusExporter?.getMetricsSummary(),
        cloudwatch: this.cloudWatchExporter?.getBufferStatus(),
        datadog: this.dataDogExporter?.getBufferStatus()
      }
    };
  }
  
  /**
   * Test alert system
   */
  public testAlert(alertType: string = 'test'): void {
    const testMetrics = this.getCurrentMetrics();
    
    // Trigger test alert by temporarily exceeding threshold
    testMetrics.latency.p95 = 150; // Above 100ms threshold
    
    this.alertManager.evaluateMetrics(testMetrics);
    this.emit('alertTest', { type: alertType, timestamp: Date.now() });
  }
  
  // Private methods
  
  private initializeComponents(): void {
    // Initialize core monitoring components
    this.performanceMonitor = new PerformanceMonitor();
    this.metricsCollector = new MetricsCollector();
    this.alertManager = new AlertManager();
    this.dashboardManager = new DashboardManager();
  }
  
  private setupEventHandlers(): void {
    // Performance monitor events
    this.performanceMonitor.on('metrics', (metrics: PerformanceMetrics) => {
      this.handleMetricsUpdate(metrics);
    });
    
    this.performanceMonitor.on('circuitBreakerActivated', (event) => {
      this.emit('circuitBreakerActivated', event);
      this.dataDogExporter?.recordCircuitBreakerEvent('open', event.reason);
    });
    
    this.performanceMonitor.on('circuitBreakerReset', (event) => {
      this.emit('circuitBreakerReset', event);
      this.dataDogExporter?.recordCircuitBreakerEvent('closed', 'Manual reset');
    });
    
    // Alert manager events
    this.alertManager.on('alertTriggered', (alert) => {
      this.emit('alertTriggered', alert);
      
      this.dataDogExporter?.addEvent(
        alert.message,
        `Alert triggered: ${alert.ruleId}`,
        alert.severity as any,
        [`rule:${alert.ruleId}`, `severity:${alert.severity}`]
      );
    });
    
    this.alertManager.on('circuitBreakerOpened', (event) => {
      this.emit('circuitBreakerOpened', event);
    });
    
    // Metrics collector events
    this.metricsCollector.on('requestCompleted', (request) => {
      this.emit('requestMetricsCompleted', request);
    });
    
    // Dashboard manager events
    this.dashboardManager.on('dashboardRefresh', (layoutId) => {
      this.emit('dashboardRefresh', layoutId);
    });
  }
  
  private handleMetricsUpdate(metrics: PerformanceMetrics): void {
    // Update exporters
    this.prometheusExporter?.updateMetrics(metrics);
    this.cloudWatchExporter?.updateMetrics(metrics);
    this.dataDogExporter?.updateMetrics(metrics);
    
    // Update dashboard
    this.dashboardManager.addMetricsData(metrics);
    
    // Check alerts
    this.alertManager.evaluateMetrics(metrics);
    
    this.emit('metricsUpdated', metrics);
  }
  
  private async initializeExporters(): Promise<void> {
    // Initialize Prometheus exporter
    if (this.config.enabledExporters.includes('prometheus') && this.config.exporters.prometheus.enabled) {
      this.prometheusExporter = new PrometheusExporter({
        port: this.config.exporters.prometheus.port,
        endpoint: this.config.exporters.prometheus.endpoint,
        labels: { service: 'aivillage-bridge' }
      });
      console.log('[BridgeMonitor] Prometheus exporter initialized');
    }
    
    // Initialize CloudWatch exporter
    if (this.config.enabledExporters.includes('cloudwatch') && this.config.exporters.cloudwatch.enabled) {
      this.cloudWatchExporter = new CloudWatchExporter({
        region: this.config.exporters.cloudwatch.region,
        namespace: this.config.exporters.cloudwatch.namespace,
        batchSize: this.config.exporters.cloudwatch.batchSize,
        dimensions: [{ Name: 'Service', Value: 'AIVillage-Bridge' }]
      });
      console.log('[BridgeMonitor] CloudWatch exporter initialized');
    }
    
    // Initialize DataDog exporter
    if (this.config.enabledExporters.includes('datadog') && this.config.exporters.datadog.enabled) {
      this.dataDogExporter = new DataDogExporter({
        apiKey: this.config.exporters.datadog.apiKey,
        tags: this.config.exporters.datadog.tags,
        prefix: this.config.exporters.datadog.prefix
      });
      console.log('[BridgeMonitor] DataDog exporter initialized');
    }
  }
  
  private startHealthChecks(): void {
    // Perform health checks every 30 seconds
    setInterval(() => {
      const health = this.getHealthStatus();
      this.emit('healthCheck', health);
      
      // Record health score as metric
      const healthScore = health.status === 'healthy' ? 100 : 
        health.status === 'degraded' ? 50 : 0;
      
      this.recordCustomMetric('system.health_score', healthScore);
    }, 30000);
  }
}

export default BridgeMonitor;