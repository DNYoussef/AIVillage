/**
 * AIVillage Bridge - Health Monitor
 * Comprehensive component health monitoring with graceful degradation
 */

export interface HealthStatus {
  component: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  lastCheck: Date;
  responseTime: number;
  errorCount: number;
  dependencies: string[];
  metadata?: Record<string, any>;
}

export interface HealthThresholds {
  responseTimeWarning: number;
  responseTimeCritical: number;
  errorRateWarning: number;
  errorRateCritical: number;
  consecutiveFailures: number;
}

export interface DegradationPolicy {
  component: string;
  triggers: string[];
  actions: DegradationAction[];
}

export interface DegradationAction {
  type: 'disable_feature' | 'fallback_mode' | 'circuit_breaker' | 'throttle';
  target: string;
  parameters: Record<string, any>;
}

export class HealthMonitor {
  private healthStatus = new Map<string, HealthStatus>();
  private healthChecks = new Map<string, () => Promise<HealthStatus>>();
  private thresholds: HealthThresholds;
  private degradationPolicies: DegradationPolicy[] = [];
  private checkInterval = 30000; // 30 seconds
  private intervalId?: NodeJS.Timeout;
  private circuitBreakers = new Map<string, CircuitBreaker>();

  constructor(thresholds: HealthThresholds) {
    this.thresholds = thresholds;
    this.initializeDefaultPolicies();
  }

  /**
   * Register a health check for a component
   */
  registerHealthCheck(
    component: string,
    healthCheck: () => Promise<HealthStatus>
  ): void {
    this.healthChecks.set(component, healthCheck);
    this.initializeComponentHealth(component);
  }

  /**
   * Start continuous health monitoring
   */
  start(): void {
    if (this.intervalId) {
      this.stop();
    }

    this.intervalId = setInterval(async () => {
      await this.performHealthChecks();
    }, this.checkInterval);

    // Perform initial check
    this.performHealthChecks();
  }

  /**
   * Stop health monitoring
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  /**
   * Get current health status for all components
   */
  getOverallHealth(): {
    status: 'healthy' | 'degraded' | 'unhealthy';
    components: HealthStatus[];
    summary: {
      healthy: number;
      degraded: number;
      unhealthy: number;
      unknown: number;
    };
  } {
    const components = Array.from(this.healthStatus.values());
    const summary = {
      healthy: components.filter(c => c.status === 'healthy').length,
      degraded: components.filter(c => c.status === 'degraded').length,
      unhealthy: components.filter(c => c.status === 'unhealthy').length,
      unknown: components.filter(c => c.status === 'unknown').length
    };

    let overallStatus: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (summary.unhealthy > 0) {
      overallStatus = 'unhealthy';
    } else if (summary.degraded > 0) {
      overallStatus = 'degraded';
    }

    return {
      status: overallStatus,
      components,
      summary
    };
  }

  /**
   * Get health status for specific component
   */
  getComponentHealth(component: string): HealthStatus | undefined {
    return this.healthStatus.get(component);
  }

  /**
   * Check if component is ready to serve requests
   */
  isReady(component: string): boolean {
    const health = this.healthStatus.get(component);
    if (!health) return false;

    return health.status === 'healthy' || health.status === 'degraded';
  }

  /**
   * Check if component is alive (basic liveness probe)
   */
  isAlive(component: string): boolean {
    const health = this.healthStatus.get(component);
    if (!health) return false;

    const timeSinceLastCheck = Date.now() - health.lastCheck.getTime();
    return timeSinceLastCheck < this.checkInterval * 2; // Allow some buffer
  }

  /**
   * Trigger graceful degradation for component
   */
  async triggerDegradation(component: string, reason: string): Promise<void> {
    const policy = this.degradationPolicies.find(p => p.component === component);
    if (!policy) return;

    console.warn(`Triggering degradation for ${component}: ${reason}`);

    for (const action of policy.actions) {
      await this.executeDegradationAction(action, component);
    }

    // Update component status
    const currentHealth = this.healthStatus.get(component);
    if (currentHealth) {
      currentHealth.status = 'degraded';
      currentHealth.metadata = {
        ...currentHealth.metadata,
        degradationReason: reason,
        degradationTime: new Date().toISOString()
      };
    }
  }

  /**
   * Register degradation policy
   */
  registerDegradationPolicy(policy: DegradationPolicy): void {
    const existingIndex = this.degradationPolicies.findIndex(
      p => p.component === policy.component
    );

    if (existingIndex >= 0) {
      this.degradationPolicies[existingIndex] = policy;
    } else {
      this.degradationPolicies.push(policy);
    }
  }

  /**
   * Perform health checks for all registered components
   */
  private async performHealthChecks(): Promise<void> {
    const checkPromises = Array.from(this.healthChecks.entries()).map(
      async ([component, healthCheck]) => {
        try {
          const status = await this.executeHealthCheck(component, healthCheck);
          this.healthStatus.set(component, status);
          await this.evaluateComponentHealth(status);
        } catch (error) {
          console.error(`Health check failed for ${component}:`, error);
          await this.handleHealthCheckFailure(component, error as Error);
        }
      }
    );

    await Promise.allSettled(checkPromises);
  }

  /**
   * Execute individual health check with timeout and error handling
   */
  private async executeHealthCheck(
    component: string,
    healthCheck: () => Promise<HealthStatus>
  ): Promise<HealthStatus> {
    const startTime = Date.now();

    try {
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Health check timeout')), 10000);
      });

      const status = await Promise.race([healthCheck(), timeoutPromise]);
      status.responseTime = Date.now() - startTime;
      status.lastCheck = new Date();

      return status;
    } catch (error) {
      return {
        component,
        status: 'unhealthy',
        lastCheck: new Date(),
        responseTime: Date.now() - startTime,
        errorCount: (this.healthStatus.get(component)?.errorCount || 0) + 1,
        dependencies: [],
        metadata: { error: (error as Error).message }
      };
    }
  }

  /**
   * Evaluate component health and trigger actions if needed
   */
  private async evaluateComponentHealth(status: HealthStatus): Promise<void> {
    const circuitBreaker = this.circuitBreakers.get(status.component);

    // Update circuit breaker
    if (circuitBreaker) {
      if (status.status === 'healthy') {
        circuitBreaker.recordSuccess();
      } else {
        circuitBreaker.recordFailure();
      }
    }

    // Check thresholds
    if (status.responseTime > this.thresholds.responseTimeCritical ||
        status.errorCount > this.thresholds.consecutiveFailures) {
      await this.triggerDegradation(
        status.component,
        `Threshold exceeded: responseTime=${status.responseTime}, errors=${status.errorCount}`
      );
    }

    // Check dependencies
    if (status.dependencies.length > 0) {
      const unhealthyDeps = status.dependencies.filter(dep => {
        const depHealth = this.healthStatus.get(dep);
        return !depHealth || depHealth.status === 'unhealthy';
      });

      if (unhealthyDeps.length > 0) {
        await this.triggerDegradation(
          status.component,
          `Unhealthy dependencies: ${unhealthyDeps.join(', ')}`
        );
      }
    }
  }

  /**
   * Handle health check failures
   */
  private async handleHealthCheckFailure(
    component: string,
    error: Error
  ): Promise<void> {
    const currentHealth = this.healthStatus.get(component);
    const errorCount = (currentHealth?.errorCount || 0) + 1;

    const failureStatus: HealthStatus = {
      component,
      status: 'unhealthy',
      lastCheck: new Date(),
      responseTime: 0,
      errorCount,
      dependencies: currentHealth?.dependencies || [],
      metadata: { error: error.message }
    };

    this.healthStatus.set(component, failureStatus);

    if (errorCount >= this.thresholds.consecutiveFailures) {
      await this.triggerDegradation(component, `Consecutive failures: ${errorCount}`);
    }
  }

  /**
   * Execute degradation action
   */
  private async executeDegradationAction(
    action: DegradationAction,
    component: string
  ): Promise<void> {
    switch (action.type) {
      case 'circuit_breaker':
        this.activateCircuitBreaker(component);
        break;
      case 'throttle':
        await this.activateThrottling(action.target, action.parameters);
        break;
      case 'disable_feature':
        await this.disableFeature(action.target);
        break;
      case 'fallback_mode':
        await this.activateFallbackMode(action.target, action.parameters);
        break;
    }
  }

  /**
   * Initialize component health status
   */
  private initializeComponentHealth(component: string): void {
    if (!this.healthStatus.has(component)) {
      this.healthStatus.set(component, {
        component,
        status: 'unknown',
        lastCheck: new Date(),
        responseTime: 0,
        errorCount: 0,
        dependencies: []
      });
    }

    // Initialize circuit breaker
    this.circuitBreakers.set(component, new CircuitBreaker({
      failureThreshold: 5,
      resetTimeout: 60000,
      monitoringPeriod: 10000
    }));
  }

  /**
   * Initialize default degradation policies
   */
  private initializeDefaultPolicies(): void {
    this.degradationPolicies = [
      {
        component: 'aivillage-core',
        triggers: ['response_time_critical', 'error_rate_high'],
        actions: [
          { type: 'circuit_breaker', target: 'aivillage-core', parameters: {} },
          { type: 'fallback_mode', target: 'basic_mode', parameters: { features: ['essential'] } }
        ]
      },
      {
        component: 'betanet-transport',
        triggers: ['connection_failed', 'timeout'],
        actions: [
          { type: 'throttle', target: 'message_rate', parameters: { limit: 10 } },
          { type: 'fallback_mode', target: 'offline_mode', parameters: {} }
        ]
      },
      {
        component: 'constitutional-ai',
        triggers: ['validation_timeout', 'high_error_rate'],
        actions: [
          { type: 'disable_feature', target: 'advanced_validation', parameters: {} },
          { type: 'fallback_mode', target: 'basic_validation', parameters: {} }
        ]
      }
    ];
  }

  /**
   * Activate circuit breaker for component
   */
  private activateCircuitBreaker(component: string): void {
    const circuitBreaker = this.circuitBreakers.get(component);
    if (circuitBreaker) {
      circuitBreaker.open();
      console.warn(`Circuit breaker activated for ${component}`);
    }
  }

  /**
   * Activate throttling for target
   */
  private async activateThrottling(
    target: string,
    parameters: Record<string, any>
  ): Promise<void> {
    // Implementation would depend on specific throttling mechanism
    console.warn(`Throttling activated for ${target}:`, parameters);
  }

  /**
   * Disable feature
   */
  private async disableFeature(target: string): Promise<void> {
    // Implementation would depend on feature management system
    console.warn(`Feature disabled: ${target}`);
  }

  /**
   * Activate fallback mode
   */
  private async activateFallbackMode(
    target: string,
    parameters: Record<string, any>
  ): Promise<void> {
    // Implementation would depend on fallback system
    console.warn(`Fallback mode activated for ${target}:`, parameters);
  }
}

/**
 * Circuit Breaker implementation for component protection
 */
class CircuitBreaker {
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  private failureCount = 0;
  private lastFailureTime?: Date;
  private config: {
    failureThreshold: number;
    resetTimeout: number;
    monitoringPeriod: number;
  };

  constructor(config: {
    failureThreshold: number;
    resetTimeout: number;
    monitoringPeriod: number;
  }) {
    this.config = config;
  }

  recordSuccess(): void {
    this.failureCount = 0;
    this.state = 'closed';
  }

  recordFailure(): void {
    this.failureCount++;
    this.lastFailureTime = new Date();

    if (this.failureCount >= this.config.failureThreshold) {
      this.state = 'open';
    }
  }

  canExecute(): boolean {
    if (this.state === 'closed') return true;
    if (this.state === 'open') {
      if (this.shouldAttemptReset()) {
        this.state = 'half-open';
        return true;
      }
      return false;
    }
    return true; // half-open state
  }

  open(): void {
    this.state = 'open';
    this.lastFailureTime = new Date();
  }

  private shouldAttemptReset(): boolean {
    if (!this.lastFailureTime) return false;

    const timeSinceLastFailure = Date.now() - this.lastFailureTime.getTime();
    return timeSinceLastFailure >= this.config.resetTimeout;
  }
}

export default HealthMonitor;