/**
 * AIVillage Bridge - Metrics Collector
 * Comprehensive system metrics aggregation and analysis
 */

export interface MetricPoint {
  timestamp: Date;
  value: number;
  tags?: Record<string, string>;
  metadata?: Record<string, any>;
}

export interface TimeSeriesMetric {
  name: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
  description: string;
  unit: string;
  points: MetricPoint[];
  retentionPeriod: number; // in milliseconds
}

export interface AggregatedMetric {
  name: string;
  timeRange: { start: Date; end: Date };
  aggregations: {
    avg: number;
    min: number;
    max: number;
    sum: number;
    count: number;
    p50: number;
    p95: number;
    p99: number;
  };
  trend: 'increasing' | 'decreasing' | 'stable';
  anomalies: AnomalyPoint[];
}

export interface AnomalyPoint {
  timestamp: Date;
  value: number;
  type: 'spike' | 'drop' | 'pattern_break';
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export interface CapacityPrediction {
  metric: string;
  currentUsage: number;
  predictedCapacity: number;
  timeToCapacity: number; // milliseconds
  confidence: number; // 0-1
  recommendations: string[];
}

export interface MetricsSnapshot {
  timestamp: Date;
  systemMetrics: {
    cpu: {
      usage: number;
      cores: number;
      loadAverage: number[];
    };
    memory: {
      total: number;
      used: number;
      free: number;
      heapUsed: number;
      heapTotal: number;
    };
    network: {
      bytesIn: number;
      bytesOut: number;
      connectionsActive: number;
      connectionsTotal: number;
    };
    disk: {
      total: number;
      used: number;
      free: number;
      ioRead: number;
      ioWrite: number;
    };
  };
  applicationMetrics: {
    requests: {
      total: number;
      rate: number;
      errorRate: number;
      avgResponseTime: number;
    };
    aivillage: {
      agentsActive: number;
      conversationsActive: number;
      messagesProcessed: number;
      validationRate: number;
    };
    betanet: {
      nodesConnected: number;
      messagesSent: number;
      messagesReceived: number;
      syncLatency: number;
    };
    constitutional: {
      validationsPerformed: number;
      violationsDetected: number;
      mitigationsApplied: number;
      complianceScore: number;
    };
  };
}

export class MetricsCollector {
  private metrics = new Map<string, TimeSeriesMetric>();
  private collectors = new Map<string, () => Promise<number>>();
  private collectionInterval = 10000; // 10 seconds
  private intervalId?: NodeJS.Timeout;
  private retentionPeriod = 7 * 24 * 60 * 60 * 1000; // 7 days default
  private anomalyDetectors = new Map<string, AnomalyDetector>();

  constructor() {
    this.initializeSystemMetrics();
  }

  /**
   * Register a metric collector
   */
  registerMetric(
    name: string,
    type: TimeSeriesMetric['type'],
    description: string,
    unit: string,
    collector: () => Promise<number>,
    retentionPeriod?: number
  ): void {
    this.metrics.set(name, {
      name,
      type,
      description,
      unit,
      points: [],
      retentionPeriod: retentionPeriod || this.retentionPeriod
    });

    this.collectors.set(name, collector);
    this.anomalyDetectors.set(name, new AnomalyDetector());
  }

  /**
   * Record a metric point manually
   */
  recordMetric(
    name: string,
    value: number,
    tags?: Record<string, string>,
    metadata?: Record<string, any>
  ): void {
    const metric = this.metrics.get(name);
    if (!metric) {
      console.warn(`Metric ${name} not registered`);
      return;
    }

    const point: MetricPoint = {
      timestamp: new Date(),
      value,
      tags,
      metadata
    };

    metric.points.push(point);
    this.cleanupOldPoints(metric);

    // Check for anomalies
    const detector = this.anomalyDetectors.get(name);
    if (detector) {
      detector.addPoint(point);
    }
  }

  /**
   * Start metrics collection
   */
  start(): void {
    if (this.intervalId) {
      this.stop();
    }

    this.intervalId = setInterval(async () => {
      await this.collectAllMetrics();
    }, this.collectionInterval);

    // Perform initial collection
    this.collectAllMetrics();
  }

  /**
   * Stop metrics collection
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  /**
   * Get current metrics snapshot
   */
  async getSnapshot(): Promise<MetricsSnapshot> {
    const systemMetrics = await this.collectSystemMetrics();
    const applicationMetrics = await this.collectApplicationMetrics();

    return {
      timestamp: new Date(),
      systemMetrics,
      applicationMetrics
    };
  }

  /**
   * Get aggregated metrics for time range
   */
  getAggregatedMetrics(
    metricNames: string[],
    startTime: Date,
    endTime: Date
  ): AggregatedMetric[] {
    return metricNames.map(name => {
      const metric = this.metrics.get(name);
      if (!metric) {
        throw new Error(`Metric ${name} not found`);
      }

      const filteredPoints = metric.points.filter(
        point => point.timestamp >= startTime && point.timestamp <= endTime
      );

      return this.calculateAggregations(name, filteredPoints, startTime, endTime);
    });
  }

  /**
   * Get capacity predictions
   */
  getCapacityPredictions(): CapacityPrediction[] {
    const predictions: CapacityPrediction[] = [];
    const capacityMetrics = [
      'system.memory.usage',
      'system.cpu.usage',
      'system.disk.usage',
      'application.requests.rate'
    ];

    for (const metricName of capacityMetrics) {
      const metric = this.metrics.get(metricName);
      if (!metric || metric.points.length < 10) continue;

      const prediction = this.predictCapacity(metric);
      if (prediction) {
        predictions.push(prediction);
      }
    }

    return predictions;
  }

  /**
   * Get historical trends
   */
  getHistoricalTrends(
    metricName: string,
    periods: number = 7
  ): {
    daily: AggregatedMetric[];
    weekly: AggregatedMetric[];
    trend: 'increasing' | 'decreasing' | 'stable';
  } {
    const metric = this.metrics.get(metricName);
    if (!metric) {
      throw new Error(`Metric ${metricName} not found`);
    }

    const now = new Date();
    const dailyTrends: AggregatedMetric[] = [];
    const weeklyTrends: AggregatedMetric[] = [];

    // Daily trends
    for (let i = 0; i < periods; i++) {
      const start = new Date(now.getTime() - (i + 1) * 24 * 60 * 60 * 1000);
      const end = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);

      const points = metric.points.filter(
        point => point.timestamp >= start && point.timestamp <= end
      );

      if (points.length > 0) {
        dailyTrends.push(this.calculateAggregations(metricName, points, start, end));
      }
    }

    // Weekly trends
    for (let i = 0; i < Math.floor(periods / 7); i++) {
      const start = new Date(now.getTime() - (i + 1) * 7 * 24 * 60 * 60 * 1000);
      const end = new Date(now.getTime() - i * 7 * 24 * 60 * 60 * 1000);

      const points = metric.points.filter(
        point => point.timestamp >= start && point.timestamp <= end
      );

      if (points.length > 0) {
        weeklyTrends.push(this.calculateAggregations(metricName, points, start, end));
      }
    }

    // Calculate overall trend
    const overallTrend = this.calculateTrend(dailyTrends.map(t => t.aggregations.avg));

    return {
      daily: dailyTrends.reverse(),
      weekly: weeklyTrends.reverse(),
      trend: overallTrend
    };
  }

  /**
   * Export metrics to external monitoring system
   */
  exportMetrics(format: 'prometheus' | 'json' | 'csv'): string {
    switch (format) {
      case 'prometheus':
        return this.exportPrometheusFormat();
      case 'json':
        return this.exportJsonFormat();
      case 'csv':
        return this.exportCsvFormat();
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  }

  /**
   * Get anomaly alerts
   */
  getAnomalyAlerts(): Array<{
    metric: string;
    anomalies: AnomalyPoint[];
    severity: 'low' | 'medium' | 'high';
  }> {
    const alerts: Array<{
      metric: string;
      anomalies: AnomalyPoint[];
      severity: 'low' | 'medium' | 'high';
    }> = [];

    for (const [metricName, detector] of this.anomalyDetectors) {
      const anomalies = detector.getRecentAnomalies();
      if (anomalies.length > 0) {
        const maxSeverity = anomalies.reduce((max, anomaly) => {
          if (anomaly.severity === 'high') return 'high';
          if (anomaly.severity === 'medium' && max !== 'high') return 'medium';
          return max;
        }, 'low' as 'low' | 'medium' | 'high');

        alerts.push({
          metric: metricName,
          anomalies,
          severity: maxSeverity
        });
      }
    }

    return alerts.sort((a, b) => {
      const severityOrder = { high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }

  /**
   * Collect all registered metrics
   */
  private async collectAllMetrics(): Promise<void> {
    const collectionPromises = Array.from(this.collectors.entries()).map(
      async ([name, collector]) => {
        try {
          const value = await collector();
          this.recordMetric(name, value);
        } catch (error) {
          console.error(`Failed to collect metric ${name}:`, error);
        }
      }
    );

    await Promise.allSettled(collectionPromises);
  }

  /**
   * Initialize system metrics collectors
   */
  private initializeSystemMetrics(): void {
    // CPU metrics
    this.registerMetric(
      'system.cpu.usage',
      'gauge',
      'CPU usage percentage',
      'percent',
      async () => {
        const usage = process.cpuUsage();
        return (usage.user + usage.system) / 1000000; // Convert to percentage
      }
    );

    // Memory metrics
    this.registerMetric(
      'system.memory.usage',
      'gauge',
      'Memory usage percentage',
      'percent',
      async () => {
        const usage = process.memoryUsage();
        return (usage.heapUsed / usage.heapTotal) * 100;
      }
    );

    // Request metrics
    this.registerMetric(
      'application.requests.total',
      'counter',
      'Total number of requests',
      'count',
      async () => {
        // Implementation would depend on request tracking
        return 0;
      }
    );

    // AIVillage specific metrics
    this.registerMetric(
      'aivillage.agents.active',
      'gauge',
      'Number of active AI agents',
      'count',
      async () => {
        // Implementation would depend on agent tracking
        return 0;
      }
    );

    // BetaNet metrics
    this.registerMetric(
      'betanet.nodes.connected',
      'gauge',
      'Number of connected BetaNet nodes',
      'count',
      async () => {
        // Implementation would depend on network monitoring
        return 0;
      }
    );

    // Constitutional AI metrics
    this.registerMetric(
      'constitutional.compliance.score',
      'gauge',
      'Constitutional compliance score',
      'score',
      async () => {
        // Implementation would depend on compliance monitoring
        return 0;
      }
    );
  }

  /**
   * Collect system metrics
   */
  private async collectSystemMetrics(): Promise<MetricsSnapshot['systemMetrics']> {
    const memUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();

    return {
      cpu: {
        usage: (cpuUsage.user + cpuUsage.system) / 1000000,
        cores: require('os').cpus().length,
        loadAverage: require('os').loadavg()
      },
      memory: {
        total: require('os').totalmem(),
        used: require('os').totalmem() - require('os').freemem(),
        free: require('os').freemem(),
        heapUsed: memUsage.heapUsed,
        heapTotal: memUsage.heapTotal
      },
      network: {
        bytesIn: 0, // Would need network monitoring
        bytesOut: 0,
        connectionsActive: 0,
        connectionsTotal: 0
      },
      disk: {
        total: 0, // Would need disk monitoring
        used: 0,
        free: 0,
        ioRead: 0,
        ioWrite: 0
      }
    };
  }

  /**
   * Collect application metrics
   */
  private async collectApplicationMetrics(): Promise<MetricsSnapshot['applicationMetrics']> {
    return {
      requests: {
        total: 0,
        rate: 0,
        errorRate: 0,
        avgResponseTime: 0
      },
      aivillage: {
        agentsActive: 0,
        conversationsActive: 0,
        messagesProcessed: 0,
        validationRate: 0
      },
      betanet: {
        nodesConnected: 0,
        messagesSent: 0,
        messagesReceived: 0,
        syncLatency: 0
      },
      constitutional: {
        validationsPerformed: 0,
        violationsDetected: 0,
        mitigationsApplied: 0,
        complianceScore: 0
      }
    };
  }

  /**
   * Calculate aggregations for metric points
   */
  private calculateAggregations(
    name: string,
    points: MetricPoint[],
    startTime: Date,
    endTime: Date
  ): AggregatedMetric {
    if (points.length === 0) {
      throw new Error(`No data points for metric ${name} in time range`);
    }

    const values = points.map(p => p.value).sort((a, b) => a - b);
    const sum = values.reduce((acc, val) => acc + val, 0);
    const avg = sum / values.length;

    const p50Index = Math.floor(values.length * 0.5);
    const p95Index = Math.floor(values.length * 0.95);
    const p99Index = Math.floor(values.length * 0.99);

    const trend = this.calculateTrend(values);
    const anomalies = this.detectAnomalies(points);

    return {
      name,
      timeRange: { start: startTime, end: endTime },
      aggregations: {
        avg,
        min: values[0],
        max: values[values.length - 1],
        sum,
        count: values.length,
        p50: values[p50Index],
        p95: values[p95Index],
        p99: values[p99Index]
      },
      trend,
      anomalies
    };
  }

  /**
   * Calculate trend from values
   */
  private calculateTrend(values: number[]): 'increasing' | 'decreasing' | 'stable' {
    if (values.length < 2) return 'stable';

    const firstHalf = values.slice(0, Math.floor(values.length / 2));
    const secondHalf = values.slice(Math.floor(values.length / 2));

    const firstAvg = firstHalf.reduce((acc, val) => acc + val, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((acc, val) => acc + val, 0) / secondHalf.length;

    const changePercent = ((secondAvg - firstAvg) / firstAvg) * 100;

    if (Math.abs(changePercent) < 5) return 'stable';
    return changePercent > 0 ? 'increasing' : 'decreasing';
  }

  /**
   * Detect anomalies in metric points
   */
  private detectAnomalies(points: MetricPoint[]): AnomalyPoint[] {
    const anomalies: AnomalyPoint[] = [];

    if (points.length < 10) return anomalies;

    const values = points.map(p => p.value);
    const avg = values.reduce((acc, val) => acc + val, 0) / values.length;
    const stdDev = Math.sqrt(
      values.reduce((acc, val) => acc + Math.pow(val - avg, 2), 0) / values.length
    );

    const threshold = 2 * stdDev;

    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      const deviation = Math.abs(point.value - avg);

      if (deviation > threshold) {
        anomalies.push({
          timestamp: point.timestamp,
          value: point.value,
          type: point.value > avg ? 'spike' : 'drop',
          severity: deviation > 3 * stdDev ? 'high' : 'medium',
          description: `Value ${point.value} deviates from average ${avg.toFixed(2)} by ${deviation.toFixed(2)}`
        });
      }
    }

    return anomalies;
  }

  /**
   * Predict capacity based on historical data
   */
  private predictCapacity(metric: TimeSeriesMetric): CapacityPrediction | null {
    if (metric.points.length < 20) return null;

    const recentPoints = metric.points.slice(-20);
    const values = recentPoints.map(p => p.value);
    const trend = this.calculateTrend(values);

    if (trend !== 'increasing') return null;

    const currentUsage = values[values.length - 1];
    const growthRate = this.calculateGrowthRate(values);

    // Assume 100% as capacity for percentage metrics, or extrapolate from max
    const maxCapacity = metric.unit === 'percent' ? 100 : Math.max(...values) * 1.5;

    const remainingCapacity = maxCapacity - currentUsage;
    const timeToCapacity = remainingCapacity / growthRate;

    return {
      metric: metric.name,
      currentUsage,
      predictedCapacity: maxCapacity,
      timeToCapacity,
      confidence: Math.min(0.9, values.length / 50), // Higher confidence with more data
      recommendations: this.generateCapacityRecommendations(metric.name, timeToCapacity)
    };
  }

  /**
   * Calculate growth rate from values
   */
  private calculateGrowthRate(values: number[]): number {
    if (values.length < 2) return 0;

    const differences = [];
    for (let i = 1; i < values.length; i++) {
      differences.push(values[i] - values[i - 1]);
    }

    return differences.reduce((acc, diff) => acc + diff, 0) / differences.length;
  }

  /**
   * Generate capacity recommendations
   */
  private generateCapacityRecommendations(
    metricName: string,
    timeToCapacity: number
  ): string[] {
    const recommendations: string[] = [];
    const daysToCapacity = timeToCapacity / (24 * 60 * 60 * 1000);

    if (daysToCapacity < 7) {
      recommendations.push('Immediate scaling required');
      recommendations.push('Consider emergency capacity addition');
    } else if (daysToCapacity < 30) {
      recommendations.push('Plan capacity increase within 2 weeks');
      recommendations.push('Monitor growth rate closely');
    } else {
      recommendations.push('Include in next capacity planning cycle');
      recommendations.push('Continue monitoring trends');
    }

    // Metric-specific recommendations
    if (metricName.includes('memory')) {
      recommendations.push('Consider memory optimization');
      recommendations.push('Review memory leaks');
    } else if (metricName.includes('cpu')) {
      recommendations.push('Consider CPU optimization');
      recommendations.push('Review performance bottlenecks');
    } else if (metricName.includes('disk')) {
      recommendations.push('Plan storage expansion');
      recommendations.push('Review data retention policies');
    }

    return recommendations;
  }

  /**
   * Clean up old metric points based on retention policy
   */
  private cleanupOldPoints(metric: TimeSeriesMetric): void {
    const cutoffTime = new Date(Date.now() - metric.retentionPeriod);
    metric.points = metric.points.filter(point => point.timestamp >= cutoffTime);
  }

  /**
   * Export metrics in Prometheus format
   */
  private exportPrometheusFormat(): string {
    let output = '';

    for (const [name, metric] of this.metrics) {
      output += `# HELP ${name} ${metric.description}\n`;
      output += `# TYPE ${name} ${metric.type}\n`;

      for (const point of metric.points.slice(-1)) { // Latest point only
        const labels = point.tags ?
          Object.entries(point.tags).map(([k, v]) => `${k}="${v}"`).join(',') :
          '';

        output += `${name}{${labels}} ${point.value} ${point.timestamp.getTime()}\n`;
      }
    }

    return output;
  }

  /**
   * Export metrics in JSON format
   */
  private exportJsonFormat(): string {
    const data = Object.fromEntries(this.metrics);
    return JSON.stringify(data, null, 2);
  }

  /**
   * Export metrics in CSV format
   */
  private exportCsvFormat(): string {
    let csv = 'metric,timestamp,value,tags,metadata\n';

    for (const [name, metric] of this.metrics) {
      for (const point of metric.points) {
        const tags = point.tags ? JSON.stringify(point.tags) : '';
        const metadata = point.metadata ? JSON.stringify(point.metadata) : '';

        csv += `${name},${point.timestamp.toISOString()},${point.value},"${tags}","${metadata}"\n`;
      }
    }

    return csv;
  }
}

/**
 * Anomaly detector for time series data
 */
class AnomalyDetector {
  private points: MetricPoint[] = [];
  private window = 50; // Number of points to consider for baseline
  private threshold = 2.5; // Standard deviations for anomaly detection

  addPoint(point: MetricPoint): void {
    this.points.push(point);

    // Keep only recent points
    if (this.points.length > this.window * 2) {
      this.points = this.points.slice(-this.window);
    }
  }

  getRecentAnomalies(hours: number = 1): AnomalyPoint[] {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    const recentPoints = this.points.filter(p => p.timestamp >= cutoff);

    if (recentPoints.length < 10) return [];

    const values = recentPoints.map(p => p.value);
    const avg = values.reduce((acc, val) => acc + val, 0) / values.length;
    const stdDev = Math.sqrt(
      values.reduce((acc, val) => acc + Math.pow(val - avg, 2), 0) / values.length
    );

    const anomalies: AnomalyPoint[] = [];

    for (const point of recentPoints) {
      const deviation = Math.abs(point.value - avg);

      if (deviation > this.threshold * stdDev) {
        anomalies.push({
          timestamp: point.timestamp,
          value: point.value,
          type: point.value > avg ? 'spike' : 'drop',
          severity: deviation > 3 * stdDev ? 'high' : 'medium',
          description: `Anomaly detected: value ${point.value} deviates from baseline by ${deviation.toFixed(2)}`
        });
      }
    }

    return anomalies;
  }
}

export default MetricsCollector;