/**
 * DataDogExporter - Export metrics to DataDog monitoring platform
 * Handles metric submission, tagging, and distribution metrics
 */

import { EventEmitter } from 'events';
import { PerformanceMetrics } from '../base/PerformanceMonitor';

export interface DataDogMetric {
  metric: string;
  points: Array<[number, number]>; // [timestamp, value]
  type: 'gauge' | 'count' | 'rate' | 'histogram' | 'distribution';
  tags?: string[];
  host?: string;
  interval?: number;
}

export interface DataDogEvent {
  title: string;
  text: string;
  date_happened?: number;
  priority?: 'normal' | 'low';
  alert_type?: 'error' | 'warning' | 'info' | 'success';
  tags?: string[];
  host?: string;
}

export interface DataDogConfig {
  apiKey: string;
  appKey?: string;
  host: string;
  tags: string[];
  prefix: string;
  flushIntervalMs: number;
  batchSize: number;
  enableDistributions: boolean;
  retryConfig: {
    maxRetries: number;
    retryDelayMs: number;
    backoffMultiplier: number;
  };
}

export interface MetricBuffer {
  timestamp: number;
  metrics: DataDogMetric[];
  events: DataDogEvent[];
  retryCount: number;
}

export class DataDogExporter extends EventEmitter {
  private config: DataDogConfig;
  private metricBuffer: DataDogMetric[] = [];
  private eventBuffer: DataDogEvent[] = [];
  private flushTimer?: NodeJS.Timeout;
  private pendingBuffers: MetricBuffer[] = [];
  
  // Distribution tracking for histogram metrics
  private distributionBuffers = new Map<string, number[]>();
  
  constructor(config: Partial<DataDogConfig> = {}) {
    super();
    
    this.config = {
      apiKey: process.env.DATADOG_API_KEY || '',
      appKey: process.env.DATADOG_APP_KEY,
      host: process.env.HOSTNAME || 'aivillage-bridge',
      tags: ['service:aivillage-bridge', 'env:production'],
      prefix: 'aivillage.bridge',
      flushIntervalMs: 60000, // 1 minute
      batchSize: 100,
      enableDistributions: true,
      retryConfig: {
        maxRetries: 3,
        retryDelayMs: 1000,
        backoffMultiplier: 2
      },
      ...config
    };
    
    this.startFlushTimer();
  }
  
  /**
   * Update metrics from performance data
   */
  public updateMetrics(metrics: PerformanceMetrics): void {
    const timestamp = Math.floor(metrics.timestamp / 1000); // DataDog expects seconds
    const baseTags = [...this.config.tags];
    
    // Latency metrics
    this.addGauge('latency.p50', metrics.latency.p50, timestamp, baseTags);
    this.addGauge('latency.p95', metrics.latency.p95, timestamp, baseTags);
    this.addGauge('latency.p99', metrics.latency.p99, timestamp, baseTags);
    this.addGauge('latency.avg', metrics.latency.avg, timestamp, baseTags);
    this.addGauge('latency.min', metrics.latency.min, timestamp, baseTags);
    this.addGauge('latency.max', metrics.latency.max, timestamp, baseTags);
    
    // Throughput metrics
    this.addRate('throughput.requests_per_second', metrics.throughput.requestsPerSecond, timestamp, baseTags);
    this.addCount('throughput.total_requests', metrics.throughput.totalRequests, timestamp, baseTags);
    this.addCount('throughput.successful_requests', metrics.throughput.successfulRequests, timestamp, baseTags);
    this.addCount('throughput.failed_requests', metrics.throughput.failedRequests, timestamp, baseTags);
    
    // Error rate calculation
    const errorRate = metrics.throughput.totalRequests > 0 ? 
      (metrics.throughput.failedRequests / metrics.throughput.totalRequests) * 100 : 0;
    this.addGauge('error_rate', errorRate, timestamp, baseTags);
    
    // Resource utilization
    this.addGauge('resources.cpu_usage', metrics.resources.cpuUsage, timestamp, baseTags);
    this.addGauge('resources.memory_usage', metrics.resources.memoryUsage, timestamp, baseTags);
    this.addRate('resources.network_bandwidth', metrics.resources.networkBandwidth, timestamp, baseTags);
    this.addGauge('resources.active_connections', metrics.resources.activeConnections, timestamp, baseTags);
    
    // Queue metrics
    this.addGauge('queue.depth', metrics.queueMetrics.depth, timestamp, baseTags);
    this.addGauge('queue.wait_time', metrics.queueMetrics.waitTime, timestamp, baseTags);
    this.addGauge('queue.processing_time', metrics.queueMetrics.processingTime, timestamp, baseTags);
    this.addGauge('queue.backlog_size', metrics.queueMetrics.backlogSize, timestamp, baseTags);
    
    // Protocol overhead metrics
    this.addGauge('protocol.translation_latency', metrics.protocolOverhead.translationLatency, timestamp, baseTags);
    this.addGauge('protocol.serialization_time', metrics.protocolOverhead.serializationTime, timestamp, baseTags);
    this.addGauge('protocol.deserialization_time', metrics.protocolOverhead.deserializationTime, timestamp, baseTags);
    this.addGauge('protocol.payload_size_increase', metrics.protocolOverhead.payloadSizeIncrease, timestamp, baseTags);
    
    // Network breakdown metrics
    this.addGauge('network.dns_lookup', metrics.networkBreakdown.dnsLookup, timestamp, baseTags);
    this.addGauge('network.tcp_connect', metrics.networkBreakdown.tcpConnect, timestamp, baseTags);
    this.addGauge('network.tls_handshake', metrics.networkBreakdown.tlsHandshake, timestamp, baseTags);
    this.addGauge('network.request_send', metrics.networkBreakdown.requestSend, timestamp, baseTags);
    this.addGauge('network.response_receive', metrics.networkBreakdown.responseReceive, timestamp, baseTags);
    this.addGauge('network.total_time', metrics.networkBreakdown.totalNetworkTime, timestamp, baseTags);
    
    this.emit('metricsUpdated', { timestamp: metrics.timestamp, metricCount: this.metricBuffer.length });
  }
  
  /**
   * Add gauge metric
   */
  public addGauge(name: string, value: number, timestamp?: number, tags: string[] = []): void {
    this.addMetric(name, value, 'gauge', timestamp, tags);
  }
  
  /**
   * Add count metric
   */
  public addCount(name: string, value: number, timestamp?: number, tags: string[] = []): void {
    this.addMetric(name, value, 'count', timestamp, tags);
  }
  
  /**
   * Add rate metric
   */
  public addRate(name: string, value: number, timestamp?: number, tags: string[] = []): void {
    this.addMetric(name, value, 'rate', timestamp, tags);
  }
  
  /**
   * Add histogram/distribution metric
   */
  public addHistogram(name: string, value: number, timestamp?: number, tags: string[] = []): void {
    if (this.config.enableDistributions) {
      this.addDistribution(name, value, timestamp, tags);
    } else {
      this.addMetric(name, value, 'histogram', timestamp, tags);
    }
  }
  
  /**
   * Add distribution metric (DataDog specific)
   */
  public addDistribution(name: string, value: number, timestamp?: number, tags: string[] = []): void {
    const key = `${name}:${tags.join(',')}`;
    if (!this.distributionBuffers.has(key)) {
      this.distributionBuffers.set(key, []);
    }
    this.distributionBuffers.get(key)!.push(value);
    
    // Add as distribution metric when buffer is flushed
    this.addMetric(name, value, 'distribution', timestamp, tags);
  }
  
  /**
   * Record request with detailed tags
   */
  public recordRequest(
    duration: number,
    method: string,
    endpoint: string,
    statusCode: number,
    userAgent?: string
  ): void {
    const timestamp = Math.floor(Date.now() / 1000);
    const tags = [
      `method:${method}`,
      `endpoint:${endpoint}`,
      `status_code:${statusCode}`,
      ...this.config.tags
    ];
    
    if (userAgent) {
      tags.push(`user_agent:${userAgent}`);
    }
    
    // Record latency as histogram for percentile calculation
    this.addHistogram('request.duration', duration, timestamp, tags);
    
    // Record request count
    this.addCount('request.count', 1, timestamp, tags);
    
    // Record errors separately
    if (statusCode >= 400) {
      this.addCount('request.errors', 1, timestamp, tags);
      
      // Create error event
      this.addEvent(
        'Request Error',
        `HTTP ${statusCode} error for ${method} ${endpoint}`,
        'error',
        tags
      );
    }
  }
  
  /**
   * Add custom event
   */
  public addEvent(
    title: string,
    text: string,
    alertType: 'error' | 'warning' | 'info' | 'success' = 'info',
    tags: string[] = []
  ): void {
    const event: DataDogEvent = {
      title,
      text,
      date_happened: Math.floor(Date.now() / 1000),
      alert_type: alertType,
      priority: alertType === 'error' ? 'normal' : 'low',
      tags: [...this.config.tags, ...tags],
      host: this.config.host
    };
    
    this.eventBuffer.push(event);
    this.emit('eventAdded', event);
  }
  
  /**
   * Record circuit breaker event
   */
  public recordCircuitBreakerEvent(state: string, reason: string): void {
    this.addEvent(
      'Circuit Breaker State Change',
      `Circuit breaker ${state}: ${reason}`,
      state === 'open' ? 'error' : 'info',
      [`circuit_breaker_state:${state}`]
    );
    
    this.addGauge('circuit_breaker.state', state === 'open' ? 1 : 0, undefined, [`state:${state}`]);
  }
  
  /**
   * Flush metrics and events to DataDog immediately
   */
  public async flushMetrics(): Promise<void> {
    if (this.metricBuffer.length === 0 && this.eventBuffer.length === 0) return;
    
    const buffer: MetricBuffer = {
      timestamp: Date.now(),
      metrics: [...this.metricBuffer],
      events: [...this.eventBuffer],
      retryCount: 0
    };
    
    this.metricBuffer = [];
    this.eventBuffer = [];
    
    await this.sendBuffer(buffer);
  }
  
  /**
   * Get buffer status
   */
  public getBufferStatus(): {
    bufferedMetrics: number;
    bufferedEvents: number;
    pendingBuffers: number;
    distributionKeys: string[];
  } {
    return {
      bufferedMetrics: this.metricBuffer.length,
      bufferedEvents: this.eventBuffer.length,
      pendingBuffers: this.pendingBuffers.length,
      distributionKeys: Array.from(this.distributionBuffers.keys())
    };
  }
  
  /**
   * Create DataDog dashboard JSON
   */
  public createDashboardConfig(dashboardTitle: string): any {
    return {
      title: dashboardTitle,
      description: 'AIVillage Bridge Performance Dashboard',
      widgets: [
        {
          definition: {
            type: 'timeseries',
            requests: [
              {
                q: `avg:${this.config.prefix}.latency.p95{*}`,
                display_type: 'line',
                style: { palette: 'dog_classic', line_type: 'solid', line_width: 'normal' }
              },
              {
                q: `avg:${this.config.prefix}.latency.p99{*}`,
                display_type: 'line',
                style: { palette: 'dog_classic', line_type: 'solid', line_width: 'normal' }
              }
            ],
            title: 'Response Latency (P95/P99)',
            yaxis: { scale: 'linear', min: 'auto', max: 'auto' }
          }
        },
        {
          definition: {
            type: 'query_value',
            requests: [
              {
                q: `avg:${this.config.prefix}.error_rate{*}`,
                aggregator: 'avg'
              }
            ],
            title: 'Error Rate (%)',
            precision: 2
          }
        },
        {
          definition: {
            type: 'timeseries',
            requests: [
              {
                q: `sum:${this.config.prefix}.throughput.requests_per_second{*}`,
                display_type: 'bars'
              }
            ],
            title: 'Requests per Second'
          }
        },
        {
          definition: {
            type: 'timeseries',
            requests: [
              {
                q: `avg:${this.config.prefix}.resources.cpu_usage{*}`,
                display_type: 'line'
              },
              {
                q: `avg:${this.config.prefix}.resources.memory_usage{*}`,
                display_type: 'line'
              }
            ],
            title: 'Resource Utilization'
          }
        },
        {
          definition: {
            type: 'heatmap',
            requests: [
              {
                q: `avg:${this.config.prefix}.request.duration{*}`
              }
            ],
            title: 'Request Duration Heatmap'
          }
        }
      ],
      layout_type: 'ordered',
      is_read_only: false,
      notify_list: [],
      template_variables: [
        {
          name: 'env',
          prefix: 'env',
          default: 'production'
        },
        {
          name: 'service',
          prefix: 'service',
          default: 'aivillage-bridge'
        }
      ]
    };
  }
  
  /**
   * Stop the exporter and clean up resources
   */
  public stop(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = undefined;
    }
    
    // Flush any remaining metrics
    this.flushMetrics();
    
    this.removeAllListeners();
  }
  
  // Private methods
  
  private addMetric(
    name: string,
    value: number,
    type: DataDogMetric['type'],
    timestamp?: number,
    tags: string[] = []
  ): void {
    const metric: DataDogMetric = {
      metric: `${this.config.prefix}.${name}`,
      points: [[timestamp || Math.floor(Date.now() / 1000), value]],
      type,
      tags: [...this.config.tags, ...tags],
      host: this.config.host
    };
    
    this.metricBuffer.push(metric);
    
    // Flush if buffer is full
    if (this.metricBuffer.length >= this.config.batchSize) {
      this.flushMetrics();
    }
  }
  
  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      this.flushMetrics();
    }, this.config.flushIntervalMs);
  }
  
  private async sendBuffer(buffer: MetricBuffer): Promise<void> {
    try {
      // In a real implementation, this would use DataDog API
      // const response = await fetch('https://api.datadoghq.com/api/v1/series', {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //     'DD-API-KEY': this.config.apiKey
      //   },
      //   body: JSON.stringify({ series: buffer.metrics })
      // });
      
      console.log(`[DataDog] Sending ${buffer.metrics.length} metrics and ${buffer.events.length} events`);
      
      // Simulate API call delay
      await this.sleep(150);
      
      this.emit('bufferSent', {
        metricCount: buffer.metrics.length,
        eventCount: buffer.events.length,
        timestamp: buffer.timestamp
      });
      
    } catch (error) {
      console.error('[DataDog] Failed to send buffer:', error);
      
      if (buffer.retryCount < this.config.retryConfig.maxRetries) {
        buffer.retryCount++;
        this.pendingBuffers.push(buffer);
        
        // Retry with exponential backoff
        const delay = this.config.retryConfig.retryDelayMs * 
          Math.pow(this.config.retryConfig.backoffMultiplier, buffer.retryCount - 1);
        
        setTimeout(() => {
          const retryBuffer = this.pendingBuffers.shift();
          if (retryBuffer) {
            this.sendBuffer(retryBuffer);
          }
        }, delay);
        
        this.emit('bufferRetry', { buffer, error, retryCount: buffer.retryCount });
      } else {
        this.emit('bufferFailed', { buffer, error });
      }
    }
  }
  
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export default DataDogExporter;