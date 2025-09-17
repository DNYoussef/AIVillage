import { describe, it, expect, beforeEach, afterEach, beforeAll, afterAll } from '@jest/globals';
import { ConstitutionalBetaNetAdapter } from '../../../src/bridge/ConstitutionalBetaNetAdapter';
import { PerformanceMonitor } from '../../../src/bridge/PerformanceMonitor';
import { ThroughputTester } from '../../helpers/ThroughputTester';
import { LoadGenerator } from '../../helpers/LoadGenerator';
import { MetricsCollector } from '../../helpers/MetricsCollector';

describe('Throughput Testing', () => {
  let adapter: ConstitutionalBetaNetAdapter;
  let performanceMonitor: PerformanceMonitor;
  let throughputTester: ThroughputTester;
  let loadGenerator: LoadGenerator;
  let metricsCollector: MetricsCollector;

  // Throughput targets
  const THROUGHPUT_TARGETS = {
    sustainedRPS: 250,     // Sustained requests per second
    peakRPS: 500,          // Peak requests per second (short burst)
    concurrentUsers: 100,   // Concurrent user sessions
    batchSize: 50,         // Maximum batch operation size
    queueDepth: 1000       // Maximum request queue depth
  };

  beforeAll(async () => {
    throughputTester = new ThroughputTester();
    loadGenerator = new LoadGenerator();
    metricsCollector = new MetricsCollector();

    await throughputTester.initialize();
    await loadGenerator.initialize();
  });

  afterAll(async () => {
    await throughputTester.cleanup();
    await loadGenerator.cleanup();
  });

  beforeEach(async () => {
    performanceMonitor = new PerformanceMonitor({
      enableThroughputTracking: true,
      enableQueueMetrics: true,
      metricsInterval: 1000
    });

    adapter = new ConstitutionalBetaNetAdapter({
      performanceMonitor,
      enableLoadBalancing: true,
      enableRequestQueuing: true,
      maxConcurrentRequests: 200,
      queueSize: 2000
    });

    await adapter.initialize();
    metricsCollector.reset();
  });

  afterEach(async () => {
    await adapter.cleanup();
    await performanceMonitor.cleanup();
  });

  describe('sustained throughput', () => {
    it('should maintain 250+ RPS for 10 minutes', async () => {
      const testConfig = {
        targetRPS: THROUGHPUT_TARGETS.sustainedRPS,
        duration: 600000, // 10 minutes
        requestMix: [
          { type: 'agent_create', weight: 0.2, template: { method: 'POST', path: '/api/v1/agents' } },
          { type: 'agent_interact', weight: 0.6, template: { method: 'POST', path: '/api/v1/agents/{id}/interact' } },
          { type: 'agent_status', weight: 0.2, template: { method: 'GET', path: '/api/v1/agents/{id}/status' } }
        ]
      };

      const results = await throughputTester.runSustainedThroughputTest(testConfig);

      // Verify sustained throughput targets
      expect(results.averageRPS).toBeGreaterThan(THROUGHPUT_TARGETS.sustainedRPS);
      expect(results.minRPS).toBeGreaterThan(THROUGHPUT_TARGETS.sustainedRPS * 0.9); // Allow 10% dip
      expect(results.totalRequests).toBeGreaterThan(THROUGHPUT_TARGETS.sustainedRPS * 600); // 600 seconds

      // Verify success rate remained high
      expect(results.successRate).toBeGreaterThan(0.995); // >99.5% success rate

      // Verify performance was stable throughout test
      const rpsVariance = results.rpsVariance;
      expect(rpsVariance).toBeLessThan(0.1); // <10% coefficient of variation

      // Verify no memory leaks during sustained load
      expect(results.memoryGrowth).toBeLessThan(100 * 1024 * 1024); // <100MB growth
    }, 650000); // 11 minutes timeout

    it('should handle request rate increases gracefully', async () => {
      const rampConfig = {
        startRPS: 50,
        endRPS: 400,
        rampDuration: 300000, // 5 minutes
        sustainDuration: 120000, // 2 minutes at peak
        stepSize: 25
      };

      const results = await throughputTester.runRampUpTest(rampConfig);

      // Verify system scaled smoothly
      expect(results.peakRPS).toBeGreaterThan(350); // Should handle at least 350 RPS
      expect(results.scalingEfficiency).toBeGreaterThan(0.9); // >90% efficiency

      // Verify no sudden drops in throughput during ramp
      const throughputSteps = results.stepResults;
      throughputSteps.forEach((step, index) => {
        if (index > 0) {
          const previousStep = throughputSteps[index - 1];
          const throughputRatio = step.actualRPS / previousStep.actualRPS;
          expect(throughputRatio).toBeGreaterThan(0.95); // No >5% drops
        }
      });

      // Verify system maintained performance at peak
      const peakPeriod = results.peakPeriodMetrics;
      expect(peakPeriod.averageLatency).toBeLessThan(100); // <100ms during peak
      expect(peakPeriod.successRate).toBeGreaterThan(0.99); // >99% success at peak
    }, 450000); // 7.5 minutes timeout

    it('should maintain throughput with varying payload sizes', async () => {
      const payloadVariations = [
        { name: 'small', size: 1024, weight: 0.4 },     // 1KB
        { name: 'medium', size: 10240, weight: 0.4 },   // 10KB
        { name: 'large', size: 102400, weight: 0.2 }    // 100KB
      ];

      const variationConfig = {
        targetRPS: 200,
        duration: 180000, // 3 minutes
        payloadVariations
      };

      const results = await throughputTester.runPayloadVariationTest(variationConfig);

      // Verify overall throughput target was met
      expect(results.averageRPS).toBeGreaterThan(180); // Allow some degradation for large payloads

      // Verify throughput by payload size
      expect(results.throughputBySize.small).toBeGreaterThan(300); // Small payloads should be fast
      expect(results.throughputBySize.medium).toBeGreaterThan(200); // Medium payloads moderate speed
      expect(results.throughputBySize.large).toBeGreaterThan(100); // Large payloads slower but acceptable

      // Verify payload processing efficiency
      const efficiency = results.throughputBySize.large / results.throughputBySize.small;
      expect(efficiency).toBeGreaterThan(0.2); // Large payloads shouldn't be <20% efficiency
    }, 200000); // 3.5 minutes timeout

    it('should handle batch operations efficiently', async () => {
      const batchSizes = [1, 5, 10, 25, 50];
      const batchResults = {};

      for (const batchSize of batchSizes) {
        const batchConfig = {
          batchSize,
          targetRPS: 50, // 50 batch requests per second
          duration: 60000 // 1 minute per batch size
        };

        const result = await throughputTester.runBatchThroughputTest(batchConfig);
        batchResults[batchSize] = result;

        // Individual batch should complete successfully
        expect(result.successRate).toBeGreaterThan(0.99);
        expect(result.averageRPS).toBeGreaterThan(45); // Allow small variance
      }

      // Verify batch efficiency scaling
      const effectiveThroughputs = batchSizes.map(size =>
        batchResults[size].averageRPS * size // Actual items processed per second
      );

      // Larger batches should process more items per second overall
      expect(effectiveThroughputs[4]).toBeGreaterThan(effectiveThroughputs[0] * 20); // 50x batch should be >20x throughput
      expect(effectiveThroughputs[3]).toBeGreaterThan(effectiveThroughputs[2] * 2); // 25x batch should be >2x of 10x batch

      // Verify batch latency remains reasonable
      const latencies = batchSizes.map(size => batchResults[size].averageLatency);
      expect(latencies[4]).toBeLessThan(latencies[0] * 10); // 50x batch shouldn't be 10x slower
    }, 350000); // 6 minutes timeout
  });

  describe('peak throughput', () => {
    it('should handle 500+ RPS peak load for 30 seconds', async () => {
      const peakConfig = {
        targetRPS: THROUGHPUT_TARGETS.peakRPS,
        duration: 30000, // 30 seconds
        warmupDuration: 10000, // 10 second warmup
        cooldownDuration: 10000 // 10 second cooldown
      };

      const results = await throughputTester.runPeakLoadTest(peakConfig);

      // Verify peak performance targets
      expect(results.peakRPS).toBeGreaterThan(THROUGHPUT_TARGETS.peakRPS);
      expect(results.averageRPS).toBeGreaterThan(THROUGHPUT_TARGETS.peakRPS * 0.95);

      // Verify system handled peak gracefully
      expect(results.successRate).toBeGreaterThan(0.98); // >98% success during peak
      expect(results.errorSpikes).toBe(0); // No sudden error spikes

      // Verify quick recovery after peak
      expect(results.recoveryTime).toBeLessThan(5000); // <5 seconds to return to baseline
      expect(results.postPeakPerformance).toBeGreaterThan(0.95); // 95% of baseline after peak
    });

    it('should handle burst traffic patterns', async () => {
      const burstConfig = {
        baselineRPS: 100,
        burstRPS: 800,
        burstDuration: 5000, // 5 second bursts
        burstInterval: 30000, // Every 30 seconds
        totalDuration: 300000, // 5 minutes
        burstCount: 10
      };

      const results = await throughputTester.runBurstTrafficTest(burstConfig);

      // Verify burst handling
      expect(results.averageBurstRPS).toBeGreaterThan(600); // Should handle most of burst traffic
      expect(results.burstSuccessRate).toBeGreaterThan(0.95); // >95% success during bursts

      // Verify baseline performance maintained between bursts
      expect(results.baselinePerformance).toBeGreaterThan(0.98); // 98% of target between bursts

      // Verify system stability across multiple bursts
      const burstMetrics = results.individualBursts;
      burstMetrics.forEach((burst, index) => {
        expect(burst.peakRPS).toBeGreaterThan(500); // Each burst should achieve good throughput

        // Later bursts shouldn't degrade (no cumulative stress)
        if (index > 0) {
          const previousBurst = burstMetrics[index - 1];
          expect(burst.peakRPS / previousBurst.peakRPS).toBeGreaterThan(0.95);
        }
      });
    }, 350000); // 6 minutes timeout

    it('should demonstrate horizontal scaling benefits', async () => {
      const scalingConfigs = [
        { instances: 1, targetRPS: 150 },
        { instances: 2, targetRPS: 280 },
        { instances: 4, targetRPS: 500 }
      ];

      const scalingResults = {};

      for (const config of scalingConfigs) {
        adapter.configureInstanceCount(config.instances);

        const result = await throughputTester.runScalingTest({
          targetRPS: config.targetRPS,
          duration: 120000 // 2 minutes
        });

        scalingResults[config.instances] = result;

        // Each configuration should meet its target
        expect(result.averageRPS).toBeGreaterThan(config.targetRPS * 0.9);
      }

      // Verify scaling efficiency
      const throughputs = [1, 2, 4].map(instances => scalingResults[instances].averageRPS);

      // 2 instances should provide >1.7x throughput
      expect(throughputs[1] / throughputs[0]).toBeGreaterThan(1.7);

      // 4 instances should provide >3x throughput
      expect(throughputs[2] / throughputs[0]).toBeGreaterThan(3.0);

      // Verify latency remains stable with scaling
      const latencies = [1, 2, 4].map(instances => scalingResults[instances].averageLatency);
      latencies.forEach(latency => {
        expect(latency).toBeLessThan(80); // All configurations should maintain <80ms latency
      });
    }, 400000); // 7 minutes timeout
  });

  describe('concurrent user simulation', () => {
    it('should support 100+ concurrent user sessions', async () => {
      const userConfig = {
        concurrentUsers: THROUGHPUT_TARGETS.concurrentUsers,
        sessionDuration: 300000, // 5 minutes per session
        actionsPerMinute: 12, // 1 action every 5 seconds
        userBehaviorPatterns: [
          { pattern: 'researcher', weight: 0.4, actions: ['search', 'analyze', 'document'] },
          { pattern: 'developer', weight: 0.3, actions: ['create', 'test', 'deploy'] },
          { pattern: 'reviewer', weight: 0.3, actions: ['review', 'approve', 'feedback'] }
        ]
      };

      const results = await throughputTester.runConcurrentUserTest(userConfig);

      // Verify concurrent user support
      expect(results.peakConcurrentUsers).toBeGreaterThan(THROUGHPUT_TARGETS.concurrentUsers);
      expect(results.averageConcurrentUsers).toBeGreaterThan(THROUGHPUT_TARGETS.concurrentUsers * 0.9);

      // Verify user experience quality
      expect(results.averageUserSessionLatency).toBeLessThan(100); // <100ms average response time
      expect(results.userSessionSuccessRate).toBeGreaterThan(0.99); // >99% success rate per user

      // Verify session stability
      expect(results.sessionDropouts).toBeLessThan(5); // <5% session failures
      expect(results.sessionTimeouts).toBeLessThan(1); // <1% timeouts

      // Verify fair resource allocation across users
      const userLatencyVariance = results.userLatencyDistribution.variance;
      expect(userLatencyVariance).toBeLessThan(0.2); // Low variance indicates fair allocation
    }, 350000); // 6 minutes timeout

    it('should handle user session scaling gracefully', async () => {
      const scalingSteps = [10, 25, 50, 75, 100, 125, 150];
      const sessionResults = {};

      for (const userCount of scalingSteps) {
        const config = {
          concurrentUsers: userCount,
          rampUpTime: 30000, // 30 seconds to reach target
          testDuration: 60000, // 1 minute at target
          actionsPerMinute: 10
        };

        const result = await throughputTester.runUserScalingTest(config);
        sessionResults[userCount] = result;

        // Each step should complete successfully
        expect(result.successfulUsers).toBeGreaterThan(userCount * 0.95);
      }

      // Verify graceful degradation at scale
      const latencies = scalingSteps.map(count => sessionResults[count].averageLatency);
      const throughputs = scalingSteps.map(count => sessionResults[count].totalActionsPerSecond);

      // Latency should increase sub-linearly with user count
      const latencyGrowthRates = latencies.slice(1).map((lat, i) => lat / latencies[i]);
      const userGrowthRates = scalingSteps.slice(1).map((count, i) => count / scalingSteps[i]);

      latencyGrowthRates.forEach((latGrowth, i) => {
        expect(latGrowth).toBeLessThan(userGrowthRates[i]); // Sublinear latency growth
      });

      // Throughput should scale reasonably with user count
      expect(throughputs[6] / throughputs[0]).toBeGreaterThan(10); // 150 users should provide >10x throughput vs 10 users
    }, 900000); // 15 minutes timeout

    it('should maintain session state consistency under load', async () => {
      const consistencyConfig = {
        concurrentUsers: 75,
        duration: 180000, // 3 minutes
        stateModifyingActions: 0.4, // 40% of actions modify state
        stateReadActions: 0.6, // 60% read state
        enableStateValidation: true
      };

      const results = await throughputTester.runSessionConsistencyTest(consistencyConfig);

      // Verify state consistency
      expect(results.stateInconsistencies).toBe(0); // No state corruption
      expect(results.readAfterWriteFailures).toBe(0); // No read-after-write failures
      expect(results.staleReadCount).toBeLessThan(results.totalReads * 0.01); // <1% stale reads

      // Verify performance with state management overhead
      expect(results.averageLatency).toBeLessThan(120); // <120ms with state overhead
      expect(results.successRate).toBeGreaterThan(0.99); // >99% success with state operations

      // Verify session isolation
      expect(results.crossSessionContamination).toBe(0); // No session data leakage
    }, 200000); // 3.5 minutes timeout
  });

  describe('queue and backpressure handling', () => {
    it('should manage request queues effectively', async () => {
      const queueConfig = {
        maxQueueDepth: THROUGHPUT_TARGETS.queueDepth,
        arrivalRate: 600, // 600 RPS arrival rate
        serviceRate: 400, // 400 RPS processing rate (create backlog)
        testDuration: 120000 // 2 minutes
      };

      const results = await throughputTester.runQueueManagementTest(queueConfig);

      // Verify queue handling
      expect(results.maxQueueDepth).toBeLessThan(THROUGHPUT_TARGETS.queueDepth);
      expect(results.queueOverflows).toBe(0); // No dropped requests
      expect(results.averageQueueTime).toBeLessThan(1000); // <1 second average queue time

      // Verify backpressure mechanisms
      expect(results.backpressureActivations).toBeGreaterThan(0); // Backpressure should activate
      expect(results.clientThrottlingRate).toBeGreaterThan(0.1); // Some throttling should occur

      // Verify queue recovery
      expect(results.queueDrainTime).toBeLessThan(30000); // <30 seconds to drain queue
    });

    it('should implement fair queuing across request types', async () => {
      const fairnessConfig = {
        requestTypes: [
          { type: 'high_priority', weight: 0.2, arrivalRate: 100 },
          { type: 'medium_priority', weight: 0.5, arrivalRate: 200 },
          { type: 'low_priority', weight: 0.3, arrivalRate: 150 }
        ],
        duration: 180000, // 3 minutes
        enablePriorityQueuing: true
      };

      const results = await throughputTester.runFairQueuingTest(fairnessConfig);

      // Verify priority handling
      expect(results.highPriorityLatency).toBeLessThan(results.mediumPriorityLatency);
      expect(results.mediumPriorityLatency).toBeLessThan(results.lowPriorityLatency);

      // Verify fairness (no starvation)
      expect(results.lowPriorityCompletionRate).toBeGreaterThan(0.9); // Low priority still gets >90% completion
      expect(results.maxStarvationTime).toBeLessThan(5000); // No request waits >5 seconds

      // Verify overall system efficiency
      expect(results.overallThroughput).toBeGreaterThan(400); // System should handle total load efficiently
    });

    it('should handle queue overflow scenarios gracefully', async () => {
      const overflowConfig = {
        queueSize: 100, // Intentionally small queue
        arrivalRate: 1000, // High arrival rate to cause overflow
        serviceRate: 200, // Low service rate
        duration: 60000, // 1 minute
        overflowStrategy: 'shed_load'
      };

      const results = await throughputTester.runQueueOverflowTest(overflowConfig);

      // Verify graceful overflow handling
      expect(results.queueOverflows).toBeGreaterThan(0); // Should experience overflows
      expect(results.systemStability).toBe(true); // System should remain stable
      expect(results.shedLoadStrategy).toBe('oldest_first'); // Should shed oldest requests

      // Verify critical requests still processed
      expect(results.highPriorityDropRate).toBeLessThan(0.1); // <10% high priority drops
      expect(results.systemRecoveryTime).toBeLessThan(10000); // <10 seconds to recover

      // Verify error responses are proper
      expect(results.overflowErrorRate).toBeGreaterThan(0.3); // Should return proper errors for overflows
      expect(results.malformedResponses).toBe(0); // No corrupted responses
    });
  });

  describe('throughput optimization validation', () => {
    it('should demonstrate connection pooling efficiency', async () => {
      const poolingTests = [
        { poolSize: 10, label: 'small_pool' },
        { poolSize: 50, label: 'medium_pool' },
        { poolSize: 100, label: 'large_pool' }
      ];

      const poolingResults = {};

      for (const test of poolingTests) {
        adapter.configureConnectionPool({ size: test.poolSize });

        const result = await throughputTester.runConnectionPoolTest({
          targetRPS: 300,
          duration: 90000, // 1.5 minutes
          connectionHoldTime: 1000
        });

        poolingResults[test.label] = result;
      }

      // Verify optimal pool size provides best throughput
      const throughputs = Object.values(poolingResults).map((r: any) => r.averageRPS);
      const maxThroughput = Math.max(...throughputs);
      const optimalResult = Object.values(poolingResults).find((r: any) => r.averageRPS === maxThroughput);

      expect(maxThroughput).toBeGreaterThan(280); // Should exceed target with optimal pooling
      expect((optimalResult as any).connectionUtilization).toBeGreaterThan(0.7); // Good utilization
      expect((optimalResult as any).connectionWaitTime).toBeLessThan(10); // Low wait times
    });

    it('should validate request pipelining benefits', async () => {
      const pipelineConfig = {
        targetRPS: 400,
        duration: 120000, // 2 minutes
        requestPatterns: [
          { pattern: 'sequential', pipelining: false },
          { pattern: 'pipelined', pipelining: true, pipelineDepth: 5 }
        ]
      };

      const sequentialResult = await throughputTester.runSequentialTest(pipelineConfig);
      const pipelinedResult = await throughputTester.runPipelinedTest(pipelineConfig);

      // Verify pipelining improvement
      expect(pipelinedResult.averageRPS).toBeGreaterThan(sequentialResult.averageRPS * 1.3); // >30% improvement
      expect(pipelinedResult.resourceUtilization).toBeGreaterThan(sequentialResult.resourceUtilization);

      // Verify pipelining maintains correctness
      expect(pipelinedResult.successRate).toBeGreaterThan(0.99);
      expect(pipelinedResult.orderingViolations).toBe(0); // No request reordering issues
    });

    it('should validate caching impact on throughput', async () => {
      const cachingConfig = {
        requestMix: [
          { type: 'cacheable', weight: 0.7, cacheHitRatio: 0.8 },
          { type: 'non_cacheable', weight: 0.3, cacheHitRatio: 0.0 }
        ],
        targetRPS: 300,
        duration: 180000 // 3 minutes
      };

      adapter.enableCaching({ hitRatio: 0.8 });
      const cachedResult = await throughputTester.runCachingTest(cachingConfig);

      adapter.disableCaching();
      const uncachedResult = await throughputTester.runCachingTest(cachingConfig);

      // Verify caching improvement
      expect(cachedResult.averageRPS).toBeGreaterThan(uncachedResult.averageRPS * 1.5); // >50% improvement
      expect(cachedResult.cacheHitLatency).toBeLessThan(10); // <10ms for cache hits
      expect(cachedResult.actualCacheHitRatio).toBeGreaterThan(0.75); // Achieved expected hit ratio

      // Verify cache doesn't impact correctness
      expect(cachedResult.dataConsistencyIssues).toBe(0);
      expect(cachedResult.staleCacheReads).toBeLessThan(cachedResult.totalCacheHits * 0.01); // <1% stale reads
    });
  });
});