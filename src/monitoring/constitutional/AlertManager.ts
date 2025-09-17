/**
 * Constitutional Alert Manager - Alert management with constitutional compliance
 * Handles alerts, notifications, and escalations with constitutional context
 */

import { EventEmitter } from 'events';

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  condition: string;
  threshold: number;
  severity: 'info' | 'warning' | 'error' | 'critical';
  enabled: boolean;
  constitutionalContext?: {
    requiresEthicalReview: boolean;
    complianceImplications: string[];
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
  };
}

export interface AlertChannel {
  id: string;
  type: 'email' | 'slack' | 'webhook' | 'sms' | 'pagerduty';
  name: string;
  config: Record<string, any>;
  enabled: boolean;
  constitutionalFilters?: {
    minComplianceScore: number;
    ethicalRiskLevels: string[];
    requiresHumanReview: boolean;
  };
}

export interface AlertManagerConfig {
  maxAlertsPerMinute: number;
  alertRetentionDays: number;
  constitutionalReviewRequired: boolean;
  escalationPolicy: {
    enabled: boolean;
    levels: Array<{
      timeoutMinutes: number;
      channels: string[];
      constitutionalOverride?: boolean;
    }>;
  };
}

export class AlertManager extends EventEmitter {
  private config: AlertManagerConfig;
  private rules = new Map<string, AlertRule>();
  private channels = new Map<string, AlertChannel>();
  private activeAlerts = new Map<string, any>();
  private alertHistory: any[] = [];
  private rateLimitCounter = new Map<string, number>();

  constructor(config: AlertManagerConfig) {
    super();
    this.config = config;
    this.initializeDefaultRules();
    this.initializeDefaultChannels();
    this.startRateLimitReset();
  }

  /**
   * Add alert rule
   */
  public addRule(rule: AlertRule): void {
    this.rules.set(rule.id, rule);
  }

  /**
   * Add alert channel
   */
  public addChannel(channel: AlertChannel): void {
    this.channels.set(channel.id, channel);
  }

  /**
   * Evaluate metrics against rules
   */
  public evaluateMetrics(metrics: any): void {
    for (const [ruleId, rule] of this.rules) {
      if (!rule.enabled) continue;

      if (this.evaluateRule(rule, metrics)) {
        this.triggerAlert(rule, metrics);
      }
    }
  }

  /**
   * Get active alerts
   */
  public getActiveAlerts(): any[] {
    return Array.from(this.activeAlerts.values());
  }

  /**
   * Get alert statistics
   */
  public getAlertStats(): any {
    return {
      active: this.activeAlerts.size,
      total: this.alertHistory.length,
      rateLimit: Object.fromEntries(this.rateLimitCounter)
    };
  }

  private initializeDefaultRules(): void {
    const defaultRules: AlertRule[] = [
      {
        id: 'high-latency',
        name: 'High Latency',
        description: 'P95 latency exceeded threshold',
        condition: 'latency.p95 > threshold',
        threshold: 100,
        severity: 'warning',
        enabled: true
      },
      {
        id: 'constitutional-compliance',
        name: 'Constitutional Compliance Failure',
        description: 'Constitutional compliance score below threshold',
        condition: 'constitutional.complianceScore < threshold',
        threshold: 0.95,
        severity: 'critical',
        enabled: true,
        constitutionalContext: {
          requiresEthicalReview: true,
          complianceImplications: ['regulatory', 'ethical', 'safety'],
          riskLevel: 'critical'
        }
      }
    ];

    defaultRules.forEach(rule => this.addRule(rule));
  }

  private initializeDefaultChannels(): void {
    const defaultChannels: AlertChannel[] = [
      {
        id: 'console',
        type: 'webhook',
        name: 'Console Logger',
        config: { url: 'console' },
        enabled: true
      },
      {
        id: 'constitutional-review',
        type: 'webhook',
        name: 'Constitutional Review Board',
        config: { url: 'constitutional-review-endpoint' },
        enabled: true,
        constitutionalFilters: {
          minComplianceScore: 0.8,
          ethicalRiskLevels: ['high', 'critical'],
          requiresHumanReview: true
        }
      }
    ];

    defaultChannels.forEach(channel => this.addChannel(channel));
  }

  private evaluateRule(rule: AlertRule, metrics: any): boolean {
    // Simple evaluation - in practice this would be more sophisticated
    const value = this.getMetricValue(metrics, rule.condition.split(' ')[0]);
    return value > rule.threshold;
  }

  private getMetricValue(metrics: any, path: string): number {
    const parts = path.split('.');
    let value: any = metrics;

    for (const part of parts) {
      value = value?.[part];
    }

    return typeof value === 'number' ? value : 0;
  }

  private triggerAlert(rule: AlertRule, metrics: any): void {
    const alertId = `${rule.id}-${Date.now()}`;

    const alert = {
      id: alertId,
      ruleId: rule.id,
      rule,
      metrics,
      timestamp: Date.now(),
      status: 'active'
    };

    this.activeAlerts.set(alertId, alert);
    this.alertHistory.push(alert);

    // Emit alert event
    this.emit('alertTriggered', alert);

    // Send to channels
    this.sendAlertToChannels(alert);
  }

  private sendAlertToChannels(alert: any): void {
    for (const [channelId, channel] of this.channels) {
      if (!channel.enabled) continue;

      // Check constitutional filters
      if (channel.constitutionalFilters && alert.rule.constitutionalContext) {
        if (!this.passesConstitutionalFilters(alert, channel.constitutionalFilters)) {
          continue;
        }
      }

      // Check rate limiting
      if (this.isRateLimited(channelId)) {
        continue;
      }

      this.sendToChannel(channel, alert);
      this.incrementRateLimit(channelId);
    }
  }

  private passesConstitutionalFilters(alert: any, filters: any): boolean {
    const context = alert.rule.constitutionalContext;
    if (!context) return true;

    // Check compliance score
    const complianceScore = alert.metrics?.constitutional?.complianceScore || 0;
    if (complianceScore < filters.minComplianceScore) return false;

    // Check ethical risk level
    const riskLevel = context.riskLevel;
    if (filters.ethicalRiskLevels && !filters.ethicalRiskLevels.includes(riskLevel)) {
      return false;
    }

    return true;
  }

  private isRateLimited(channelId: string): boolean {
    const count = this.rateLimitCounter.get(channelId) || 0;
    return count >= this.config.maxAlertsPerMinute;
  }

  private incrementRateLimit(channelId: string): void {
    const current = this.rateLimitCounter.get(channelId) || 0;
    this.rateLimitCounter.set(channelId, current + 1);
  }

  private sendToChannel(channel: AlertChannel, alert: any): void {
    if (channel.config.url === 'console') {
      console.warn(`ALERT [${alert.rule.severity.toUpperCase()}]: ${alert.rule.name}`, {
        description: alert.rule.description,
        timestamp: new Date(alert.timestamp).toISOString(),
        constitutionalContext: alert.rule.constitutionalContext
      });
    } else {
      // In a real implementation, this would send to actual channels
      console.log(`Sending alert to ${channel.name}:`, alert);
    }
  }

  private startRateLimitReset(): void {
    setInterval(() => {
      this.rateLimitCounter.clear();
    }, 60000); // Reset every minute
  }

  public destroy(): void {
    this.removeAllListeners();
    this.rules.clear();
    this.channels.clear();
    this.activeAlerts.clear();
  }
}

export default AlertManager;