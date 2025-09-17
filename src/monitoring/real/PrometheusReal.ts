/**
 * PrometheusReal - Production-ready Prometheus metrics exporter
 *
 * Replaces mock implementation with actual Prometheus client
 * providing real metrics collection and /metrics endpoint
 */

import * as express from 'express';
import { Registry, collectDefaultMetrics, Histogram, Counter, Gauge, Summary } from 'prom-client';
import { EventEmitter } from 'events';
import { PerformanceMetrics } from '../base/PerformanceMonitor';

export interface PrometheusConfig {
  port: number;
  endpoint: string;
  prefix: string;
  includeDefaultMetrics: boolean;
  defaultLabels?: Record<string, string>;
  grafanaUrl?: string;
  pushGatewayUrl?: string;
}

export class PrometheusReal extends EventEmitter {
  private config: PrometheusConfig;
  private registry: Registry;
  private app: express.Application;
  private server: any;

  // Core metrics
  private httpRequestDuration: Histogram<string>;
  private httpRequestsTotal: Counter<string>;
  private httpRequestsErrors: Counter<string>;
  private activeRequests: Gauge<string>;

  // BetaNet specific metrics
  private betanetLatency: Histogram<string>;
  private betanetTranslations: Counter<string>;
  private constitutionalValidations: Counter<string>;
  private privacyTierRequests: Counter<string>;

  // Performance metrics
  private p95Latency: Gauge<string>;
  private averageLatency: Gauge<string>;
  private throughput: Gauge<string>;

  // Circuit breaker metrics
  private circuitBreakerState: Gauge<string>;
  private circuitBreakerTrips: Counter<string>;

  // Fog computing metrics
  private fogNodesActive: Gauge<string>;
  private fogJobsProcessed: Counter<string>;
  private fogLatency: Histogram<string>;

  constructor(config: PrometheusConfig) {
    super();
    this.config = {
      port: 9090,
      endpoint: '/metrics',
      prefix: 'aivillage_',
      includeDefaultMetrics: true,
      ...config
    };

    this.registry = new Registry();
    this.app = express();

    // Set default labels if provided
    if (this.config.defaultLabels) {
      this.registry.setDefaultLabels(this.config.defaultLabels);
    }

    this.initializeMetrics();
    this.setupEndpoint();
  }

  private initializeMetrics(): void {
    // Collect default Node.js metrics if enabled
    if (this.config.includeDefaultMetrics) {
      collectDefaultMetrics({
        register: this.registry,
        prefix: this.config.prefix
      });
    }

    // HTTP request duration histogram
    this.httpRequestDuration = new Histogram({
      name: `${this.config.prefix}http_request_duration_seconds`,
      help: 'Duration of HTTP requests in seconds',
      labelNames: ['method', 'route', 'status_code'],
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5, 5],
      registers: [this.registry]
    });

    // HTTP requests total counter
    this.httpRequestsTotal = new Counter({
      name: `${this.config.prefix}http_requests_total`,
      help: 'Total number of HTTP requests',
      labelNames: ['method', 'route', 'status_code'],
      registers: [this.registry]
    });

    // HTTP request errors counter
    this.httpRequestsErrors = new Counter({
      name: `${this.config.prefix}http_requests_errors_total`,
      help: 'Total number of HTTP request errors',
      labelNames: ['method', 'route', 'error_type'],
      registers: [this.registry]
    });

    // Active requests gauge
    this.activeRequests = new Gauge({
      name: `${this.config.prefix}active_requests`,
      help: 'Number of active requests',
      labelNames: ['service'],
      registers: [this.registry]
    });

    // BetaNet latency histogram
    this.betanetLatency = new Histogram({
      name: `${this.config.prefix}betanet_latency_seconds`,
      help: 'BetaNet protocol translation latency',
      labelNames: ['operation', 'constitutional_tier'],
      buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
      registers: [this.registry]
    });

    // BetaNet translations counter
    this.betanetTranslations = new Counter({
      name: `${this.config.prefix}betanet_translations_total`,
      help: 'Total number of BetaNet translations',
      labelNames: ['direction', 'status'],
      registers: [this.registry]
    });

    // Constitutional validations counter
    this.constitutionalValidations = new Counter({
      name: `${this.config.prefix}constitutional_validations_total`,
      help: 'Total number of constitutional validations',
      labelNames: ['tier', 'result'],
      registers: [this.registry]
    });

    // Privacy tier requests counter
    this.privacyTierRequests = new Counter({
      name: `${this.config.prefix}privacy_tier_requests_total`,
      help: 'Requests by privacy tier',
      labelNames: ['tier'],
      registers: [this.registry]
    });

    // P95 latency gauge
    this.p95Latency = new Gauge({
      name: `${this.config.prefix}p95_latency_milliseconds`,
      help: 'P95 latency in milliseconds',
      labelNames: ['service'],
      registers: [this.registry]
    });

    // Average latency gauge
    this.averageLatency = new Gauge({
      name: `${this.config.prefix}average_latency_milliseconds`,
      help: 'Average latency in milliseconds',
      labelNames: ['service'],
      registers: [this.registry]
    });

    // Throughput gauge
    this.throughput = new Gauge({
      name: `${this.config.prefix}throughput_requests_per_second`,
      help: 'Throughput in requests per second',
      labelNames: ['service'],
      registers: [this.registry]
    });

    // Circuit breaker state gauge
    this.circuitBreakerState = new Gauge({
      name: `${this.config.prefix}circuit_breaker_state`,
      help: 'Circuit breaker state (0=closed, 1=open, 2=half-open)',
      labelNames: ['service'],
      registers: [this.registry]
    });

    // Circuit breaker trips counter
    this.circuitBreakerTrips = new Counter({
      name: `${this.config.prefix}circuit_breaker_trips_total`,
      help: 'Total number of circuit breaker trips',
      labelNames: ['service', 'reason'],
      registers: [this.registry]
    });

    // Fog computing metrics
    this.fogNodesActive = new Gauge({
      name: `${this.config.prefix}fog_nodes_active`,
      help: 'Number of active fog nodes',
      labelNames: ['region', 'type'],
      registers: [this.registry]
    });

    this.fogJobsProcessed = new Counter({
      name: `${this.config.prefix}fog_jobs_processed_total`,
      help: 'Total fog computing jobs processed',
      labelNames: ['node_type', 'status'],
      registers: [this.registry]
    });

    this.fogLatency = new Histogram({
      name: `${this.config.prefix}fog_latency_seconds`,
      help: 'Fog computing job latency',
      labelNames: ['node_type', 'operation'],
      buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
      registers: [this.registry]
    });
  }

  private setupEndpoint(): void {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({ status: 'healthy', timestamp: Date.now() });
    });

    // Metrics endpoint
    this.app.get(this.config.endpoint, async (req, res) => {
      try {
        const metrics = await this.registry.metrics();
        res.set('Content-Type', this.registry.contentType);
        res.send(metrics);
      } catch (error) {
        console.error('Error generating metrics:', error);
        res.status(500).json({ error: 'Failed to generate metrics' });
      }
    });

    // Grafana dashboard configuration endpoint
    if (this.config.grafanaUrl) {
      this.app.get('/grafana/dashboard', (req, res) => {
        res.json(this.generateGrafanaDashboard());
      });
    }
  }

  /**
   * Start the Prometheus metrics server
   */
  public async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = this.app.listen(this.config.port, () => {
        console.log(`Prometheus metrics server listening on port ${this.config.port}`);
        console.log(`Metrics available at http://localhost:${this.config.port}${this.config.endpoint}`);
        this.emit('started', { port: this.config.port });
        resolve();
      });

      this.server.on('error', (error: Error) => {
        console.error('Prometheus server error:', error);
        reject(error);
      });
    });
  }

  /**
   * Stop the Prometheus metrics server
   */
  public async stop(): Promise<void> {
    return new Promise((resolve) => {
      if (this.server) {
        this.server.close(() => {
          console.log('Prometheus metrics server stopped');
          this.emit('stopped');
          resolve();
        });
      } else {
        resolve();
      }
    });
  }

  /**
   * Record an HTTP request
   */
  public recordHttpRequest(
    method: string,
    route: string,
    statusCode: number,
    duration: number
  ): void {
    const labels = { method, route, status_code: statusCode.toString() };

    this.httpRequestDuration.observe(labels, duration / 1000); // Convert to seconds
    this.httpRequestsTotal.inc(labels);

    if (statusCode >= 400) {
      this.httpRequestsErrors.inc({
        method,
        route,
        error_type: statusCode >= 500 ? 'server_error' : 'client_error'
      });
    }
  }

  /**
   * Record BetaNet translation
   */
  public recordBetaNetTranslation(
    direction: 'to_betanet' | 'from_betanet',
    tier: string,
    duration: number,
    success: boolean
  ): void {
    this.betanetLatency.observe(
      { operation: direction, constitutional_tier: tier },
      duration / 1000
    );

    this.betanetTranslations.inc({
      direction,
      status: success ? 'success' : 'failure'
    });
  }

  /**
   * Record constitutional validation
   */
  public recordConstitutionalValidation(
    tier: string,
    passed: boolean
  ): void {
    this.constitutionalValidations.inc({
      tier,
      result: passed ? 'passed' : 'failed'
    });
  }

  /**
   * Record privacy tier request
   */
  public recordPrivacyTierRequest(tier: string): void {
    this.privacyTierRequests.inc({ tier });
  }

  /**
   * Update performance metrics
   */
  public updatePerformanceMetrics(metrics: PerformanceMetrics): void {
    this.p95Latency.set({ service: 'bridge' }, metrics.p95);
    this.averageLatency.set({ service: 'bridge' }, metrics.avg);

    // Calculate throughput (assuming metrics are updated every second)
    const throughput = 1000 / metrics.avg; // Requests per second
    this.throughput.set({ service: 'bridge' }, throughput);
  }

  /**
   * Update circuit breaker state
   */
  public updateCircuitBreakerState(
    state: 'closed' | 'open' | 'half-open',
    service: string = 'bridge'
  ): void {
    const stateValue = state === 'closed' ? 0 : state === 'open' ? 1 : 2;
    this.circuitBreakerState.set({ service }, stateValue);
  }

  /**
   * Record circuit breaker trip
   */
  public recordCircuitBreakerTrip(
    reason: string,
    service: string = 'bridge'
  ): void {
    this.circuitBreakerTrips.inc({ service, reason });
  }

  /**
   * Update active requests
   */
  public updateActiveRequests(count: number, service: string = 'bridge'): void {
    this.activeRequests.set({ service }, count);
  }

  /**
   * Update fog computing metrics
   */
  public updateFogMetrics(metrics: {
    activeNodes: number;
    nodeType: string;
    region?: string;
  }): void {
    this.fogNodesActive.set(
      { region: metrics.region || 'default', type: metrics.nodeType },
      metrics.activeNodes
    );
  }

  /**
   * Record fog job
   */
  public recordFogJob(
    nodeType: string,
    duration: number,
    success: boolean
  ): void {
    this.fogJobsProcessed.inc({
      node_type: nodeType,
      status: success ? 'success' : 'failure'
    });

    this.fogLatency.observe(
      { node_type: nodeType, operation: 'job_execution' },
      duration / 1000
    );
  }

  /**
   * Generate Grafana dashboard configuration
   */
  private generateGrafanaDashboard(): any {
    return {
      dashboard: {
        title: 'AIVillage Bridge Metrics',
        panels: [
          {
            title: 'Request Rate',
            targets: [{
              expr: `rate(${this.config.prefix}http_requests_total[5m])`
            }]
          },
          {
            title: 'P95 Latency',
            targets: [{
              expr: `${this.config.prefix}p95_latency_milliseconds`
            }],
            alert: {
              condition: `${this.config.prefix}p95_latency_milliseconds > 75`,
              message: 'P95 latency exceeds 75ms target'
            }
          },
          {
            title: 'Circuit Breaker State',
            targets: [{
              expr: `${this.config.prefix}circuit_breaker_state`
            }]
          },
          {
            title: 'BetaNet Translation Latency',
            targets: [{
              expr: `histogram_quantile(0.95, ${this.config.prefix}betanet_latency_seconds)`
            }]
          },
          {
            title: 'Constitutional Validations',
            targets: [{
              expr: `rate(${this.config.prefix}constitutional_validations_total[5m])`
            }]
          },
          {
            title: 'Active Fog Nodes',
            targets: [{
              expr: `${this.config.prefix}fog_nodes_active`
            }]
          }
        ]
      }
    };
  }

  /**
   * Push metrics to Prometheus Push Gateway (for batch jobs)
   */
  public async pushMetrics(): Promise<void> {
    if (!this.config.pushGatewayUrl) {
      throw new Error('Push Gateway URL not configured');
    }

    const gateway = new (require('prom-client').Pushgateway)(
      this.config.pushGatewayUrl
    );

    try {
      await gateway.push({ jobName: 'aivillage_bridge' });
      console.log('Metrics pushed to Push Gateway');
    } catch (error) {
      console.error('Failed to push metrics:', error);
      throw error;
    }
  }

  /**
   * Get current metrics as JSON
   */
  public async getMetricsJson(): Promise<any> {
    const metrics = await this.registry.getMetricsAsJSON();
    return metrics;
  }

  /**
   * Reset all metrics
   */
  public resetMetrics(): void {
    this.registry.resetMetrics();
    console.log('All metrics reset');
  }
}

export default PrometheusReal;