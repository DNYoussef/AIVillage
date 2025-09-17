/**
 * Constitutional Dashboard Manager - Dashboard management with constitutional metrics
 * Provides constitutional compliance dashboards and monitoring interfaces
 */

import { EventEmitter } from 'events';

export interface DashboardLayout {
  id: string;
  name: string;
  description: string;
  widgets: DashboardWidget[];
  constitutionalFocus: boolean;
  refreshInterval: number;
}

export interface DashboardWidget {
  id: string;
  type: 'chart' | 'metric' | 'table' | 'status' | 'constitutional-score' | 'compliance-trend';
  title: string;
  dataSource: string;
  config: Record<string, any>;
  constitutionalContext?: {
    showComplianceScore: boolean;
    showEthicalRisk: boolean;
    showGovernanceStatus: boolean;
  };
}

export interface DashboardConfig {
  defaultRefreshInterval: number;
  constitutionalDashboardsEnabled: boolean;
  realTimeUpdates: boolean;
  historicalDataRetention: number;
}

export class DashboardManager extends EventEmitter {
  private config: DashboardConfig;
  private layouts = new Map<string, DashboardLayout>();
  private metricsData: any[] = [];
  private refreshTimers = new Map<string, NodeJS.Timeout>();

  constructor(config?: Partial<DashboardConfig>) {
    super();

    this.config = {
      defaultRefreshInterval: 10000,
      constitutionalDashboardsEnabled: true,
      realTimeUpdates: true,
      historicalDataRetention: 24 * 60 * 60 * 1000, // 24 hours
      ...config
    };

    this.initializeDefaultLayouts();
  }

  /**
   * Add dashboard layout
   */
  public addLayout(layout: DashboardLayout): void {
    this.layouts.set(layout.id, layout);
    this.startLayoutRefresh(layout);
  }

  /**
   * Get dashboard data for layout
   */
  public getDashboardData(layoutId: string): any {
    const layout = this.layouts.get(layoutId);
    if (!layout) {
      throw new Error(`Dashboard layout ${layoutId} not found`);
    }

    return {
      layout,
      data: this.generateLayoutData(layout),
      timestamp: Date.now(),
      constitutionalSummary: layout.constitutionalFocus ? this.getConstitutionalSummary() : null
    };
  }

  /**
   * Add metrics data point
   */
  public addMetricsData(metrics: any): void {
    this.metricsData.push({
      ...metrics,
      timestamp: Date.now()
    });

    // Maintain data retention
    const cutoffTime = Date.now() - this.config.historicalDataRetention;
    this.metricsData = this.metricsData.filter(data => data.timestamp >= cutoffTime);

    // Emit real-time updates if enabled
    if (this.config.realTimeUpdates) {
      this.emit('metricsUpdate', metrics);
    }
  }

  /**
   * Perform health check
   */
  public performHealthCheck(): any {
    const recentMetrics = this.getRecentMetrics(5); // Last 5 data points

    if (recentMetrics.length === 0) {
      return {
        status: 'unknown',
        message: 'No metrics data available',
        constitutional: {
          complianceStatus: 'unknown',
          ethicalRisk: 'unknown'
        }
      };
    }

    const latestMetrics = recentMetrics[recentMetrics.length - 1];

    // Base health assessment
    let status = 'healthy';
    let message = 'All systems operational';

    if (latestMetrics.latency?.p95 > 100) {
      status = 'degraded';
      message = 'High latency detected';
    }

    if (latestMetrics.resources?.cpuUsage > 80) {
      status = 'degraded';
      message = 'High resource usage';
    }

    // Constitutional health assessment
    const constitutional = this.assessConstitutionalHealth(latestMetrics);

    if (constitutional.complianceStatus === 'non-compliant') {
      status = 'unhealthy';
      message = 'Constitutional compliance failure';
    }

    return {
      status,
      message,
      metrics: latestMetrics,
      constitutional,
      timestamp: Date.now()
    };
  }

  /**
   * Get constitutional summary
   */
  public getConstitutionalSummary(): any {
    const recentMetrics = this.getRecentMetrics(10);

    if (recentMetrics.length === 0) {
      return {
        overallScore: 0,
        trend: 'unknown',
        alerts: 0,
        lastValidation: null
      };
    }

    const constitutionalMetrics = recentMetrics
      .filter(m => m.constitutional)
      .map(m => m.constitutional);

    if (constitutionalMetrics.length === 0) {
      return {
        overallScore: 0,
        trend: 'unknown',
        alerts: 0,
        lastValidation: null
      };
    }

    const scores = constitutionalMetrics.map(m => m.complianceScore || 0);
    const overallScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;

    // Calculate trend
    const midpoint = Math.floor(scores.length / 2);
    const firstHalf = scores.slice(0, midpoint);
    const secondHalf = scores.slice(midpoint);

    const firstAvg = firstHalf.reduce((sum, score) => sum + score, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((sum, score) => sum + score, 0) / secondHalf.length;

    let trend = 'stable';
    if (secondAvg > firstAvg + 0.05) trend = 'improving';
    else if (firstAvg > secondAvg + 0.05) trend = 'declining';

    const alerts = constitutionalMetrics.filter(m =>
      m.ethicalRiskLevel === 'high' || m.ethicalRiskLevel === 'critical'
    ).length;

    const latestMetric = constitutionalMetrics[constitutionalMetrics.length - 1];

    return {
      overallScore,
      trend,
      alerts,
      lastValidation: latestMetric?.lastValidationTime || null,
      ethicalRiskLevel: latestMetric?.ethicalRiskLevel || 'unknown',
      violationsDetected: latestMetric?.violationsDetected || 0
    };
  }

  /**
   * Export dashboard data
   */
  public exportDashboardData(layoutId: string, format: 'json' | 'csv'): string {
    const dashboardData = this.getDashboardData(layoutId);

    if (format === 'json') {
      return JSON.stringify(dashboardData, null, 2);
    } else {
      return this.convertToCsv(dashboardData);
    }
  }

  private initializeDefaultLayouts(): void {
    const layouts: DashboardLayout[] = [
      {
        id: 'main-performance',
        name: 'Main Performance Dashboard',
        description: 'Primary performance and health monitoring',
        constitutionalFocus: false,
        refreshInterval: this.config.defaultRefreshInterval,
        widgets: [
          {
            id: 'latency-chart',
            type: 'chart',
            title: 'Response Latency',
            dataSource: 'latency.p95',
            config: { chartType: 'line', timeRange: '1h' }
          },
          {
            id: 'throughput-metric',
            type: 'metric',
            title: 'Throughput',
            dataSource: 'throughput.requestsPerSecond',
            config: { unit: 'req/s' }
          },
          {
            id: 'health-status',
            type: 'status',
            title: 'System Health',
            dataSource: 'health',
            config: {}
          }
        ]
      },
      {
        id: 'constitutional-compliance',
        name: 'Constitutional Compliance Dashboard',
        description: 'Constitutional AI compliance and ethical monitoring',
        constitutionalFocus: true,
        refreshInterval: this.config.defaultRefreshInterval,
        widgets: [
          {
            id: 'compliance-score',
            type: 'constitutional-score',
            title: 'Compliance Score',
            dataSource: 'constitutional.complianceScore',
            config: { threshold: 0.95 },
            constitutionalContext: {
              showComplianceScore: true,
              showEthicalRisk: true,
              showGovernanceStatus: true
            }
          },
          {
            id: 'ethical-risk',
            type: 'status',
            title: 'Ethical Risk Level',
            dataSource: 'constitutional.ethicalRiskLevel',
            config: { colorMapping: {
              low: 'green',
              medium: 'yellow',
              high: 'orange',
              critical: 'red'
            }},
            constitutionalContext: {
              showComplianceScore: false,
              showEthicalRisk: true,
              showGovernanceStatus: false
            }
          },
          {
            id: 'compliance-trend',
            type: 'compliance-trend',
            title: 'Compliance Trend',
            dataSource: 'constitutional.complianceScore',
            config: { timeRange: '24h', showPrediction: true },
            constitutionalContext: {
              showComplianceScore: true,
              showEthicalRisk: false,
              showGovernanceStatus: false
            }
          },
          {
            id: 'violations-table',
            type: 'table',
            title: 'Recent Violations',
            dataSource: 'constitutional.violations',
            config: {
              columns: ['timestamp', 'type', 'severity', 'description'],
              maxRows: 10
            },
            constitutionalContext: {
              showComplianceScore: false,
              showEthicalRisk: true,
              showGovernanceStatus: true
            }
          }
        ]
      },
      {
        id: 'governance-overview',
        name: 'Governance Overview',
        description: 'AI governance and audit trail monitoring',
        constitutionalFocus: true,
        refreshInterval: this.config.defaultRefreshInterval * 2, // Slower refresh
        widgets: [
          {
            id: 'governance-score',
            type: 'metric',
            title: 'Governance Score',
            dataSource: 'governance.governanceScore',
            config: { unit: '%', threshold: 90 }
          },
          {
            id: 'audit-trail',
            type: 'status',
            title: 'Audit Trail Status',
            dataSource: 'governance.auditTrailComplete',
            config: {}
          },
          {
            id: 'policy-violations',
            type: 'metric',
            title: 'Policy Violations',
            dataSource: 'governance.policyViolations',
            config: { unit: 'count', alertThreshold: 5 }
          }
        ]
      }
    ];

    layouts.forEach(layout => this.addLayout(layout));
  }

  private generateLayoutData(layout: DashboardLayout): any {
    const data: Record<string, any> = {};

    for (const widget of layout.widgets) {
      data[widget.id] = this.generateWidgetData(widget);
    }

    return data;
  }

  private generateWidgetData(widget: DashboardWidget): any {
    const recentMetrics = this.getRecentMetrics(100); // Last 100 data points

    if (recentMetrics.length === 0) {
      return { error: 'No data available' };
    }

    switch (widget.type) {
      case 'chart':
        return this.generateChartData(widget, recentMetrics);
      case 'metric':
        return this.generateMetricData(widget, recentMetrics);
      case 'table':
        return this.generateTableData(widget, recentMetrics);
      case 'status':
        return this.generateStatusData(widget, recentMetrics);
      case 'constitutional-score':
        return this.generateConstitutionalScoreData(widget, recentMetrics);
      case 'compliance-trend':
        return this.generateComplianceTrendData(widget, recentMetrics);
      default:
        return { error: 'Unknown widget type' };
    }
  }

  private generateChartData(widget: DashboardWidget, metrics: any[]): any {
    const dataPoints = metrics.map(metric => {
      const value = this.getNestedValue(metric, widget.dataSource);
      return {
        timestamp: metric.timestamp,
        value: typeof value === 'number' ? value : 0
      };
    });

    return {
      type: 'chart',
      data: dataPoints,
      config: widget.config
    };
  }

  private generateMetricData(widget: DashboardWidget, metrics: any[]): any {
    const latestMetric = metrics[metrics.length - 1];
    const value = this.getNestedValue(latestMetric, widget.dataSource);

    return {
      type: 'metric',
      value: typeof value === 'number' ? value : 0,
      unit: widget.config.unit || '',
      threshold: widget.config.threshold,
      timestamp: latestMetric.timestamp
    };
  }

  private generateTableData(widget: DashboardWidget, metrics: any[]): any {
    const data = metrics.slice(-widget.config.maxRows || 10).map(metric => {
      const row: Record<string, any> = {};

      for (const column of widget.config.columns || []) {
        row[column] = this.getNestedValue(metric, column);
      }

      return row;
    });

    return {
      type: 'table',
      data,
      columns: widget.config.columns || []
    };
  }

  private generateStatusData(widget: DashboardWidget, metrics: any[]): any {
    const latestMetric = metrics[metrics.length - 1];
    const value = this.getNestedValue(latestMetric, widget.dataSource);

    let status = 'unknown';
    let color = 'gray';

    if (widget.config.colorMapping && typeof value === 'string') {
      color = widget.config.colorMapping[value] || 'gray';
      status = value;
    } else if (typeof value === 'boolean') {
      status = value ? 'healthy' : 'unhealthy';
      color = value ? 'green' : 'red';
    }

    return {
      type: 'status',
      status,
      color,
      timestamp: latestMetric.timestamp
    };
  }

  private generateConstitutionalScoreData(widget: DashboardWidget, metrics: any[]): any {
    const constitutionalMetrics = metrics
      .filter(m => m.constitutional)
      .map(m => m.constitutional);

    if (constitutionalMetrics.length === 0) {
      return {
        type: 'constitutional-score',
        score: 0,
        status: 'unknown',
        trend: 'unknown'
      };
    }

    const latestMetric = constitutionalMetrics[constitutionalMetrics.length - 1];
    const score = latestMetric.complianceScore || 0;
    const threshold = widget.config.threshold || 0.95;

    let status = 'healthy';
    if (score < threshold * 0.8) status = 'critical';
    else if (score < threshold) status = 'warning';

    // Calculate trend
    const recentScores = constitutionalMetrics.slice(-10).map(m => m.complianceScore || 0);
    const trend = this.calculateTrend(recentScores);

    return {
      type: 'constitutional-score',
      score,
      status,
      trend,
      threshold,
      ethicalRiskLevel: latestMetric.ethicalRiskLevel,
      violationsCount: latestMetric.violationsDetected || 0,
      timestamp: metrics[metrics.length - 1].timestamp
    };
  }

  private generateComplianceTrendData(widget: DashboardWidget, metrics: any[]): any {
    const constitutionalMetrics = metrics
      .filter(m => m.constitutional)
      .map(m => ({
        timestamp: m.timestamp,
        complianceScore: m.constitutional.complianceScore || 0,
        ethicalRiskLevel: m.constitutional.ethicalRiskLevel
      }));

    if (constitutionalMetrics.length === 0) {
      return {
        type: 'compliance-trend',
        data: [],
        trend: 'unknown',
        prediction: null
      };
    }

    const scores = constitutionalMetrics.map(m => m.complianceScore);
    const trend = this.calculateTrend(scores);

    // Simple prediction (would be more sophisticated in practice)
    let prediction = null;
    if (widget.config.showPrediction && scores.length >= 5) {
      const recentScores = scores.slice(-5);
      const avgChange = recentScores.reduce((sum, score, index) => {
        if (index === 0) return sum;
        return sum + (score - recentScores[index - 1]);
      }, 0) / (recentScores.length - 1);

      prediction = {
        nextScore: Math.max(0, Math.min(1, scores[scores.length - 1] + avgChange)),
        confidence: 0.7 // Placeholder confidence
      };
    }

    return {
      type: 'compliance-trend',
      data: constitutionalMetrics,
      trend,
      prediction
    };
  }

  private getRecentMetrics(count: number): any[] {
    return this.metricsData.slice(-count);
  }

  private getNestedValue(obj: any, path: string): any {
    const parts = path.split('.');
    let value = obj;

    for (const part of parts) {
      value = value?.[part];
    }

    return value;
  }

  private assessConstitutionalHealth(metrics: any): any {
    if (!metrics.constitutional) {
      return {
        complianceStatus: 'unknown',
        ethicalRisk: 'unknown',
        governanceStatus: 'unknown'
      };
    }

    const constitutional = metrics.constitutional;

    let complianceStatus = 'compliant';
    if (constitutional.complianceScore < 0.8) complianceStatus = 'non-compliant';
    else if (constitutional.complianceScore < 0.95) complianceStatus = 'at-risk';

    return {
      complianceStatus,
      ethicalRisk: constitutional.ethicalRiskLevel || 'unknown',
      governanceStatus: metrics.governance?.auditTrailComplete ? 'compliant' : 'incomplete'
    };
  }

  private calculateTrend(values: number[]): 'improving' | 'declining' | 'stable' {
    if (values.length < 2) return 'stable';

    const midpoint = Math.floor(values.length / 2);
    const firstHalf = values.slice(0, midpoint);
    const secondHalf = values.slice(midpoint);

    const firstAvg = firstHalf.reduce((sum, val) => sum + val, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((sum, val) => sum + val, 0) / secondHalf.length;

    const changeThreshold = 0.05; // 5% change threshold

    if (secondAvg > firstAvg + changeThreshold) return 'improving';
    else if (firstAvg > secondAvg + changeThreshold) return 'declining';
    else return 'stable';
  }

  private startLayoutRefresh(layout: DashboardLayout): void {
    const timer = setInterval(() => {
      this.emit('dashboardRefresh', layout.id);
    }, layout.refreshInterval);

    this.refreshTimers.set(layout.id, timer);
  }

  private convertToCsv(data: any): string {
    // Simple CSV conversion - would be more sophisticated in practice
    const headers = Object.keys(data.data);
    let csv = headers.join(',') + '\n';

    // Add timestamp row
    csv += `${data.timestamp},${headers.map(h =>
      typeof data.data[h].value !== 'undefined' ? data.data[h].value : data.data[h].status || ''
    ).join(',')}\n`;

    return csv;
  }

  /**
   * Cleanup dashboard resources
   */
  public destroy(): void {
    // Clear all refresh timers
    for (const timer of this.refreshTimers.values()) {
      clearInterval(timer);
    }

    this.removeAllListeners();
    this.layouts.clear();
    this.refreshTimers.clear();
    this.metricsData = [];
  }
}

export default DashboardManager;