/**
 * CloudWatchExporter - Export metrics to AWS CloudWatch
 * Handles metric data aggregation, batching, and namespace organization
 */

import { EventEmitter } from 'events';
import { PerformanceMetrics } from '../base/PerformanceMonitor';

export interface CloudWatchMetric {
  MetricName: string;
  Value: number;
  Unit: CloudWatchUnit;
  Timestamp: Date;
  Dimensions?: CloudWatchDimension[];
  StatisticValues?: {
    SampleCount: number;
    Sum: number;
    Minimum: number;
    Maximum: number;
  };
}

export interface CloudWatchDimension {
  Name: string;
  Value: string;
}

export type CloudWatchUnit = 
  | 'Seconds' | 'Microseconds' | 'Milliseconds'
  | 'Bytes' | 'Kilobytes' | 'Megabytes' | 'Gigabytes'
  | 'Bits' | 'Kilobits' | 'Megabits' | 'Gigabits'
  | 'Percent' | 'Count' | 'Count/Second'
  | 'Bytes/Second' | 'Kilobytes/Second' | 'Megabytes/Second' | 'Gigabytes/Second'
  | 'Bits/Second' | 'Kilobits/Second' | 'Megabits/Second' | 'Gigabits/Second'
  | 'None';

export interface CloudWatchConfig {
  region: string;
  namespace: string;
  batchSize: number;
  flushIntervalMs: number;
  dimensions: CloudWatchDimension[];
  enableDetailedMetrics: boolean;
  retryConfig: {
    maxRetries: number;
    retryDelayMs: number;
    backoffMultiplier: number;
  };
}

export interface MetricBatch {
  timestamp: number;
  metrics: CloudWatchMetric[];
  retryCount: number;
}

export class CloudWatchExporter extends EventEmitter {
  private config: CloudWatchConfig;
  private metricBuffer: CloudWatchMetric[] = [];
  private flushTimer?: NodeJS.Timeout;
  private pendingBatches: MetricBatch[] = [];
  
  // Aggregation windows for statistical metrics
  private latencyBuffer: number[] = [];
  private requestCounts = new Map<string, number>();
  private errorCounts = new Map<string, number>();
  
  constructor(config: Partial<CloudWatchConfig> = {}) {
    super();
    
    this.config = {
      region: process.env.AWS_REGION || 'us-east-1',
      namespace: 'AIVillage/Bridge',
      batchSize: 20, // CloudWatch limit
      flushIntervalMs: 60000, // 1 minute
      dimensions: [],
      enableDetailedMetrics: true,
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
    const timestamp = new Date(metrics.timestamp);
    const baseDimensions = [
      ...this.config.dimensions,
      { Name: 'Service', Value: 'AIVillage-Bridge' }
    ];
    
    // Latency metrics
    this.addMetric('Latency.P50', metrics.latency.p50, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Latency.P95', metrics.latency.p95, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Latency.P99', metrics.latency.p99, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Latency.Average', metrics.latency.avg, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Latency.Min', metrics.latency.min, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Latency.Max', metrics.latency.max, 'Milliseconds', timestamp, baseDimensions);
    
    // Throughput metrics
    this.addMetric('Throughput.RequestsPerSecond', metrics.throughput.requestsPerSecond, 'Count/Second', timestamp, baseDimensions);
    this.addMetric('Throughput.TotalRequests', metrics.throughput.totalRequests, 'Count', timestamp, baseDimensions);
    this.addMetric('Throughput.SuccessfulRequests', metrics.throughput.successfulRequests, 'Count', timestamp, baseDimensions);
    this.addMetric('Throughput.FailedRequests', metrics.throughput.failedRequests, 'Count', timestamp, baseDimensions);
    
    // Error rate calculation
    const errorRate = metrics.throughput.totalRequests > 0 ? 
      (metrics.throughput.failedRequests / metrics.throughput.totalRequests) * 100 : 0;
    this.addMetric('ErrorRate', errorRate, 'Percent', timestamp, baseDimensions);
    
    // Resource utilization
    this.addMetric('Resources.CPUUsage', metrics.resources.cpuUsage, 'Percent', timestamp, baseDimensions);
    this.addMetric('Resources.MemoryUsage', metrics.resources.memoryUsage, 'Percent', timestamp, baseDimensions);
    this.addMetric('Resources.NetworkBandwidth', metrics.resources.networkBandwidth, 'Bytes/Second', timestamp, baseDimensions);
    this.addMetric('Resources.ActiveConnections', metrics.resources.activeConnections, 'Count', timestamp, baseDimensions);
    
    // Queue metrics
    this.addMetric('Queue.Depth', metrics.queueMetrics.depth, 'Count', timestamp, baseDimensions);
    this.addMetric('Queue.WaitTime', metrics.queueMetrics.waitTime, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Queue.ProcessingTime', metrics.queueMetrics.processingTime, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Queue.BacklogSize', metrics.queueMetrics.backlogSize, 'Count', timestamp, baseDimensions);
    
    // Protocol overhead metrics
    this.addMetric('Protocol.TranslationLatency', metrics.protocolOverhead.translationLatency, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Protocol.SerializationTime', metrics.protocolOverhead.serializationTime, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Protocol.DeserializationTime', metrics.protocolOverhead.deserializationTime, 'Milliseconds', timestamp, baseDimensions);
    this.addMetric('Protocol.PayloadSizeIncrease', metrics.protocolOverhead.payloadSizeIncrease, 'Percent', timestamp, baseDimensions);
    
    // Network breakdown metrics
    if (this.config.enableDetailedMetrics) {
      this.addMetric('Network.DNSLookup', metrics.networkBreakdown.dnsLookup, 'Milliseconds', timestamp, baseDimensions);
      this.addMetric('Network.TCPConnect', metrics.networkBreakdown.tcpConnect, 'Milliseconds', timestamp, baseDimensions);
      this.addMetric('Network.TLSHandshake', metrics.networkBreakdown.tlsHandshake, 'Milliseconds', timestamp, baseDimensions);
      this.addMetric('Network.RequestSend', metrics.networkBreakdown.requestSend, 'Milliseconds', timestamp, baseDimensions);
      this.addMetric('Network.ResponseReceive', metrics.networkBreakdown.responseReceive, 'Milliseconds', timestamp, baseDimensions);
      this.addMetric('Network.TotalTime', metrics.networkBreakdown.totalNetworkTime, 'Milliseconds', timestamp, baseDimensions);
    }
    
    this.emit('metricsUpdated', { timestamp: metrics.timestamp, metricCount: this.metricBuffer.length });
  }
  
  /**
   * Add custom metric
   */
  public addMetric(
    name: string, 
    value: number, 
    unit: CloudWatchUnit, 
    timestamp: Date = new Date(),
    dimensions: CloudWatchDimension[] = []
  ): void {
    const metric: CloudWatchMetric = {
      MetricName: name,
      Value: value,
      Unit: unit,
      Timestamp: timestamp,
      Dimensions: [...this.config.dimensions, ...dimensions]
    };
    
    this.metricBuffer.push(metric);
    
    // Flush if buffer is full
    if (this.metricBuffer.length >= this.config.batchSize) {
      this.flushMetrics();
    }
  }
  
  /**
   * Add statistical metric with aggregated values
   */
  public addStatisticalMetric(
    name: string,
    values: number[],
    unit: CloudWatchUnit,
    timestamp: Date = new Date(),
    dimensions: CloudWatchDimension[] = []
  ): void {
    if (values.length === 0) return;
    
    const sum = values.reduce((total, value) => total + value, 0);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    const metric: CloudWatchMetric = {
      MetricName: name,
      Value: 0, // Not used when StatisticValues is provided
      Unit: unit,
      Timestamp: timestamp,
      Dimensions: [...this.config.dimensions, ...dimensions],
      StatisticValues: {
        SampleCount: values.length,
        Sum: sum,
        Minimum: min,
        Maximum: max
      }
    };
    
    this.metricBuffer.push(metric);
    
    if (this.metricBuffer.length >= this.config.batchSize) {
      this.flushMetrics();
    }
  }
  
  /**
   * Record request with detailed dimensions
   */
  public recordRequest(
    duration: number,
    method: string,
    endpoint: string,
    statusCode: number,
    region?: string
  ): void {
    const timestamp = new Date();
    const dimensions: CloudWatchDimension[] = [
      { Name: 'Method', Value: method },
      { Name: 'Endpoint', Value: endpoint },
      { Name: 'StatusCode', Value: statusCode.toString() }
    ];
    
    if (region) {
      dimensions.push({ Name: 'Region', Value: region });
    }
    
    // Record latency
    this.addMetric('Request.Duration', duration, 'Milliseconds', timestamp, dimensions);
    
    // Record request count
    this.addMetric('Request.Count', 1, 'Count', timestamp, dimensions);
    
    // Record errors separately
    if (statusCode >= 400) {
      this.addMetric('Request.Errors', 1, 'Count', timestamp, dimensions);
    }
  }
  
  /**
   * Record custom business metrics
   */
  public recordBusinessMetric(
    metricName: string,
    value: number,
    unit: CloudWatchUnit,
    dimensions: Record<string, string> = {}
  ): void {
    const cloudWatchDimensions: CloudWatchDimension[] = Object.entries(dimensions)
      .map(([name, value]) => ({ Name: name, Value: value }));
    
    this.addMetric(metricName, value, unit, new Date(), cloudWatchDimensions);
  }
  
  /**
   * Flush metrics to CloudWatch immediately
   */
  public async flushMetrics(): Promise<void> {
    if (this.metricBuffer.length === 0) return;
    
    const batch: MetricBatch = {
      timestamp: Date.now(),
      metrics: [...this.metricBuffer],
      retryCount: 0
    };
    
    this.metricBuffer = [];
    await this.sendBatch(batch);
  }
  
  /**
   * Get buffer status
   */
  public getBufferStatus(): {
    bufferedMetrics: number;
    pendingBatches: number;
    lastFlush: number;
  } {
    return {
      bufferedMetrics: this.metricBuffer.length,
      pendingBatches: this.pendingBatches.length,
      lastFlush: this.getLastFlushTime()
    };
  }
  
  /**
   * Create CloudWatch dashboard JSON
   */
  public createDashboardConfig(dashboardName: string): any {
    return {
      widgets: [
        {
          type: 'metric',
          properties: {
            metrics: [
              [this.config.namespace, 'Latency.P95'],
              [this.config.namespace, 'Latency.P99'],
              [this.config.namespace, 'Latency.Average']
            ],
            period: 300,
            stat: 'Average',
            region: this.config.region,
            title: 'Response Latency',
            yAxis: {
              left: {
                min: 0
              }
            }
          }
        },
        {
          type: 'metric',
          properties: {
            metrics: [
              [this.config.namespace, 'Throughput.RequestsPerSecond'],
              [this.config.namespace, 'ErrorRate']
            ],
            period: 300,
            stat: 'Average',
            region: this.config.region,
            title: 'Throughput and Error Rate'
          }
        },
        {
          type: 'metric',
          properties: {
            metrics: [
              [this.config.namespace, 'Resources.CPUUsage'],
              [this.config.namespace, 'Resources.MemoryUsage']
            ],
            period: 300,
            stat: 'Average',
            region: this.config.region,
            title: 'Resource Utilization'
          }
        },
        {
          type: 'metric',
          properties: {
            metrics: [
              [this.config.namespace, 'Queue.Depth'],
              [this.config.namespace, 'Queue.WaitTime']
            ],
            period: 300,
            stat: 'Average',
            region: this.config.region,
            title: 'Queue Metrics'
          }
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
  
  private startFlushTimer(): void {
    this.flushTimer = setInterval(() => {
      this.flushMetrics();
    }, this.config.flushIntervalMs);
  }
  
  private async sendBatch(batch: MetricBatch): Promise<void> {
    try {
      // In a real implementation, this would use AWS SDK
      // const cloudwatch = new AWS.CloudWatch({ region: this.config.region });
      // await cloudwatch.putMetricData({
      //   Namespace: this.config.namespace,
      //   MetricData: batch.metrics
      // }).promise();
      
      console.log(`[CloudWatch] Sending ${batch.metrics.length} metrics to namespace ${this.config.namespace}`);
      
      // Simulate API call delay
      await this.sleep(100);
      
      this.emit('batchSent', {
        metricCount: batch.metrics.length,
        namespace: this.config.namespace,
        timestamp: batch.timestamp
      });
      
    } catch (error) {
      console.error('[CloudWatch] Failed to send metrics batch:', error);
      
      if (batch.retryCount < this.config.retryConfig.maxRetries) {
        batch.retryCount++;
        this.pendingBatches.push(batch);
        
        // Retry with exponential backoff
        const delay = this.config.retryConfig.retryDelayMs * 
          Math.pow(this.config.retryConfig.backoffMultiplier, batch.retryCount - 1);
        
        setTimeout(() => {
          const retryBatch = this.pendingBatches.shift();
          if (retryBatch) {
            this.sendBatch(retryBatch);
          }
        }, delay);
        
        this.emit('batchRetry', { batch, error, retryCount: batch.retryCount });
      } else {
        this.emit('batchFailed', { batch, error });
      }
    }
  }
  
  private getLastFlushTime(): number {
    // In a real implementation, this would track the actual last flush time
    return Date.now();
  }
  
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

export default CloudWatchExporter;