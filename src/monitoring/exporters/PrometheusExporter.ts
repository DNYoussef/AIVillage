/**
 * PrometheusExporter - Export metrics to Prometheus monitoring system
 * Provides histogram, counter, and gauge metrics with proper labeling
 */

import { EventEmitter } from 'events';
import { PerformanceMetrics } from '../base/PerformanceMonitor';

export interface PrometheusMetric {
  name: string;
  help: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
  value?: number;
  labels?: Record<string, string>;
  buckets?: number[]; // For histograms
  observations?: number[]; // For histograms/summaries
}

export interface PrometheusConfig {
  prefix: string;
  port: number;
  endpoint: string;
  enableDefaultMetrics: boolean;
  labels: Record<string, string>;
  buckets: {
    latency: number[];
    requestSize: number[];
    responseSize: number[];
  };
}

export class PrometheusExporter extends EventEmitter {
  private config: PrometheusConfig;
  private metrics = new Map<string, PrometheusMetric>();
  private server?: any; // HTTP server for metrics endpoint
  
  // Metric collections
  private latencyHistogram: number[] = [];
  private requestCounter = 0;
  private errorCounter = 0;
  private activeConnections = 0;
  
  // Default buckets for histograms
  private readonly DEFAULT_LATENCY_BUCKETS = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
  ];
  
  private readonly DEFAULT_SIZE_BUCKETS = [
    100, 1000, 10000, 100000, 1000000, 10000000 // bytes
  ];
  
  constructor(config: Partial<PrometheusConfig> = {}) {
    super();
    
    this.config = {
      prefix: 'aivillage',
      port: 9090,
      endpoint: '/metrics',
      enableDefaultMetrics: true,
      labels: {},
      buckets: {
        latency: this.DEFAULT_LATENCY_BUCKETS,
        requestSize: this.DEFAULT_SIZE_BUCKETS,
        responseSize: this.DEFAULT_SIZE_BUCKETS
      },
      ...config
    };
    
    this.initializeMetrics();
    this.startMetricsServer();
  }
  
  /**
   * Update metrics from performance data
   */
  public updateMetrics(metrics: PerformanceMetrics): void {
    const timestamp = Date.now();
    
    // Update latency metrics
    this.updateLatencyMetrics(metrics.latency);
    
    // Update throughput metrics
    this.updateThroughputMetrics(metrics.throughput);
    
    // Update resource metrics
    this.updateResourceMetrics(metrics.resources);
    
    // Update queue metrics
    this.updateQueueMetrics(metrics.queueMetrics);
    
    // Update protocol overhead metrics
    this.updateProtocolMetrics(metrics.protocolOverhead);
    
    // Update network breakdown metrics
    this.updateNetworkMetrics(metrics.networkBreakdown);
    
    this.emit('metricsUpdated', { timestamp, metrics });
  }
  
  /**
   * Record request metrics
   */
  public recordRequest(duration: number, method: string, endpoint: string, statusCode: number, size?: number): void {
    const labels = {
      method,
      endpoint,
      status_code: statusCode.toString(),
      ...this.config.labels
    };
    
    // Update request histogram
    this.observeHistogram('request_duration_seconds', duration / 1000, labels);
    
    // Update request counter
    this.incrementCounter('requests_total', labels);
    
    // Update error counter if needed
    if (statusCode >= 400) {
      this.incrementCounter('requests_errors_total', labels);
    }
    
    // Update request size if provided
    if (size !== undefined) {
      this.observeHistogram('request_size_bytes', size, labels);
    }
  }
  
  /**
   * Record response metrics
   */
  public recordResponse(size: number, method: string, endpoint: string, statusCode: number): void {
    const labels = {
      method,
      endpoint,
      status_code: statusCode.toString(),
      ...this.config.labels
    };
    
    this.observeHistogram('response_size_bytes', size, labels);
  }
  
  /**
   * Set gauge value
   */
  public setGauge(name: string, value: number, labels: Record<string, string> = {}): void {
    const metricName = this.getMetricName(name);
    const labelString = this.formatLabels({ ...labels, ...this.config.labels });
    const key = `${metricName}${labelString}`;
    
    this.metrics.set(key, {
      name: metricName,
      help: `Gauge metric: ${name}`,
      type: 'gauge',
      value,
      labels: { ...labels, ...this.config.labels }
    });
  }
  
  /**
   * Increment counter
   */
  public incrementCounter(name: string, labels: Record<string, string> = {}, value: number = 1): void {
    const metricName = this.getMetricName(name);
    const labelString = this.formatLabels({ ...labels, ...this.config.labels });
    const key = `${metricName}${labelString}`;
    
    const existing = this.metrics.get(key);
    const currentValue = existing?.value || 0;
    
    this.metrics.set(key, {
      name: metricName,
      help: `Counter metric: ${name}`,
      type: 'counter',
      value: currentValue + value,
      labels: { ...labels, ...this.config.labels }
    });
  }
  
  /**
   * Observe histogram value
   */
  public observeHistogram(name: string, value: number, labels: Record<string, string> = {}): void {
    const metricName = this.getMetricName(name);
    const labelString = this.formatLabels({ ...labels, ...this.config.labels });
    const key = `${metricName}${labelString}`;
    
    const existing = this.metrics.get(key);
    const observations = existing?.observations || [];
    observations.push(value);
    
    // Determine buckets based on metric name
    let buckets = this.DEFAULT_LATENCY_BUCKETS;
    if (name.includes('size')) {
      buckets = this.DEFAULT_SIZE_BUCKETS;
    }
    
    this.metrics.set(key, {
      name: metricName,
      help: `Histogram metric: ${name}`,
      type: 'histogram',
      labels: { ...labels, ...this.config.labels },
      buckets,
      observations
    });
  }
  
  /**
   * Get metrics in Prometheus format
   */
  public getMetricsOutput(): string {
    const lines: string[] = [];
    const seenMetrics = new Set<string>();
    
    for (const metric of this.metrics.values()) {
      // Add HELP and TYPE comments only once per metric name
      if (!seenMetrics.has(metric.name)) {
        lines.push(`# HELP ${metric.name} ${metric.help}`);
        lines.push(`# TYPE ${metric.name} ${metric.type}`);
        seenMetrics.add(metric.name);
      }
      
      if (metric.type === 'histogram') {
        lines.push(...this.formatHistogram(metric));
      } else {
        lines.push(this.formatMetric(metric));
      }
    }
    
    return lines.join('\n') + '\n';
  }
  
  /**
   * Get metrics summary for debugging
   */
  public getMetricsSummary(): {
    totalMetrics: number;
    metricTypes: Record<string, number>;
    latestValues: Record<string, number>;
  } {
    const metricTypes: Record<string, number> = {};
    const latestValues: Record<string, number> = {};
    
    for (const metric of this.metrics.values()) {
      metricTypes[metric.type] = (metricTypes[metric.type] || 0) + 1;
      
      if (metric.value !== undefined) {
        latestValues[metric.name] = metric.value;
      }
    }
    
    return {
      totalMetrics: this.metrics.size,
      metricTypes,
      latestValues
    };
  }
  
  /**
   * Clear all metrics
   */
  public clearMetrics(): void {
    this.metrics.clear();
    this.emit('metricsCleared');
  }
  
  /**
   * Stop the metrics server
   */
  public stop(): void {
    if (this.server) {
      this.server.close();
      this.server = null;
    }
    this.removeAllListeners();
  }
  
  // Private methods
  
  private initializeMetrics(): void {
    // Initialize basic counters and gauges
    this.setGauge('up', 1);
    this.incrementCounter('requests_total', {}, 0);
    this.incrementCounter('requests_errors_total', {}, 0);
    
    // Initialize histograms with empty observations
    this.observeHistogram('request_duration_seconds', 0);
    this.observeHistogram('request_size_bytes', 0);
    this.observeHistogram('response_size_bytes', 0);
  }
  
  private updateLatencyMetrics(latency: PerformanceMetrics['latency']): void {
    this.setGauge('latency_p50_seconds', latency.p50 / 1000);
    this.setGauge('latency_p95_seconds', latency.p95 / 1000);
    this.setGauge('latency_p99_seconds', latency.p99 / 1000);
    this.setGauge('latency_avg_seconds', latency.avg / 1000);
    this.setGauge('latency_min_seconds', latency.min / 1000);
    this.setGauge('latency_max_seconds', latency.max / 1000);
  }
  
  private updateThroughputMetrics(throughput: PerformanceMetrics['throughput']): void {
    this.setGauge('requests_per_second', throughput.requestsPerSecond);
    this.setGauge('total_requests', throughput.totalRequests);
    this.setGauge('successful_requests', throughput.successfulRequests);
    this.setGauge('failed_requests', throughput.failedRequests);
    
    // Calculate error rate
    const errorRate = throughput.totalRequests > 0 ? 
      (throughput.failedRequests / throughput.totalRequests) * 100 : 0;
    this.setGauge('error_rate_percent', errorRate);
  }
  
  private updateResourceMetrics(resources: PerformanceMetrics['resources']): void {
    this.setGauge('cpu_usage_percent', resources.cpuUsage);
    this.setGauge('memory_usage_percent', resources.memoryUsage);
    this.setGauge('network_bandwidth_bytes', resources.networkBandwidth);
    this.setGauge('active_connections', resources.activeConnections);
  }
  
  private updateQueueMetrics(queue: PerformanceMetrics['queueMetrics']): void {
    this.setGauge('queue_depth', queue.depth);
    this.setGauge('queue_wait_time_seconds', queue.waitTime / 1000);
    this.setGauge('queue_processing_time_seconds', queue.processingTime / 1000);
    this.setGauge('queue_backlog_size', queue.backlogSize);
  }
  
  private updateProtocolMetrics(protocol: PerformanceMetrics['protocolOverhead']): void {
    this.setGauge('protocol_translation_latency_seconds', protocol.translationLatency / 1000);
    this.setGauge('protocol_serialization_time_seconds', protocol.serializationTime / 1000);
    this.setGauge('protocol_deserialization_time_seconds', protocol.deserializationTime / 1000);
    this.setGauge('protocol_payload_size_increase_percent', protocol.payloadSizeIncrease);
  }
  
  private updateNetworkMetrics(network: PerformanceMetrics['networkBreakdown']): void {
    this.setGauge('network_dns_lookup_seconds', network.dnsLookup / 1000);
    this.setGauge('network_tcp_connect_seconds', network.tcpConnect / 1000);
    this.setGauge('network_tls_handshake_seconds', network.tlsHandshake / 1000);
    this.setGauge('network_request_send_seconds', network.requestSend / 1000);
    this.setGauge('network_response_receive_seconds', network.responseReceive / 1000);
    this.setGauge('network_total_time_seconds', network.totalNetworkTime / 1000);
  }
  
  private getMetricName(name: string): string {
    return `${this.config.prefix}_${name}`;
  }
  
  private formatLabels(labels: Record<string, string>): string {
    const labelPairs = Object.entries(labels);
    if (labelPairs.length === 0) return '';
    
    const formatted = labelPairs
      .map(([key, value]) => `${key}="${value}"`)
      .join(',');
    
    return `{${formatted}}`;
  }
  
  private formatMetric(metric: PrometheusMetric): string {
    const labelString = this.formatLabels(metric.labels || {});
    return `${metric.name}${labelString} ${metric.value || 0}`;
  }
  
  private formatHistogram(metric: PrometheusMetric): string[] {
    const lines: string[] = [];
    const observations = metric.observations || [];
    const buckets = metric.buckets || this.DEFAULT_LATENCY_BUCKETS;
    const labelPrefix = metric.labels ? this.formatLabels(metric.labels).slice(0, -1) + ',' : '{';
    
    // Count observations in each bucket
    let cumulativeCount = 0;
    for (const bucket of buckets) {
      const count = observations.filter(obs => obs <= bucket).length;
      const bucketLabels = `${labelPrefix}le="${bucket}"}`;
      lines.push(`${metric.name}_bucket${bucketLabels} ${count}`);
      cumulativeCount = count;
    }
    
    // Add +Inf bucket
    const infLabels = `${labelPrefix}le="+Inf"}`;
    lines.push(`${metric.name}_bucket${infLabels} ${observations.length}`);
    
    // Add count and sum
    const baseLabels = metric.labels ? this.formatLabels(metric.labels) : '';
    lines.push(`${metric.name}_count${baseLabels} ${observations.length}`);
    
    const sum = observations.reduce((total, obs) => total + obs, 0);
    lines.push(`${metric.name}_sum${baseLabels} ${sum}`);
    
    return lines;
  }
  
  private startMetricsServer(): void {
    try {
      // Note: In a real implementation, you would use a proper HTTP server
      // This is a simplified version showing the structure
      console.log(`Prometheus metrics server would start on port ${this.config.port}${this.config.endpoint}`);
      
      // Simulated server setup
      this.server = {
        close: () => console.log('Prometheus metrics server stopped')
      };
      
      this.emit('serverStarted', { port: this.config.port, endpoint: this.config.endpoint });
    } catch (error) {
      console.error('Failed to start Prometheus metrics server:', error);
      this.emit('serverError', error);
    }
  }
}

export default PrometheusExporter;