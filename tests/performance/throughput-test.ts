/**
 * Throughput Performance Tests
 * Measures system throughput under various loads
 */

import { performance } from 'perf_hooks';
import { BridgeOrchestrator } from '../../src/bridge/BridgeOrchestrator';
import { ConstitutionalBridgeRequest } from '../../src/bridge/types';

interface ThroughputMetrics {
  totalRequests: number;
  duration: number;
  throughput: number; // requests per second
  successRate: number;
  errorCount: number;
  avgLatency: number;
  concurrency: number;
}

class ThroughputTest {
  private orchestrator: BridgeOrchestrator;
  private metrics: ThroughputMetrics[] = [];

  constructor() {
    this.orchestrator = new BridgeOrchestrator({
      betaNetEndpoint: 'http://localhost:8080',
      defaultPrivacyTier: 'Silver',
      enableConstitutionalValidation: true,
      complianceThreshold: 0.8,
      performance: {
        targetP95Latency: 75,
        circuitBreakerEnabled: true,
        maxConcurrentRequests: 200,
        requestTimeout: 5000
      }
    });
  }

  async initialize(): Promise<void> {
    await this.orchestrator.initialize();
    console.log('System initialized for throughput testing');
  }

  async testThroughput(
    requestCount: number,
    concurrency: number
  ): Promise<ThroughputMetrics> {
    console.log(`\nTesting throughput: ${requestCount} requests with concurrency ${concurrency}`);

    const startTime = performance.now();
    let successCount = 0;
    let errorCount = 0;
    const latencies: number[] = [];

    // Create batches based on concurrency
    const batches: number[][] = [];
    for (let i = 0; i < requestCount; i += concurrency) {
      const batch = [];
      for (let j = 0; j < concurrency && i + j < requestCount; j++) {
        batch.push(i + j);
      }
      batches.push(batch);
    }

    // Process batches
    for (const batch of batches) {
      const promises = batch.map(async (index) => {
        const requestStart = performance.now();
        try {
          const request = this.createTestRequest(`throughput-${index}`);
          await this.orchestrator.processRequest(request);
          successCount++;
          const latency = performance.now() - requestStart;
          latencies.push(latency);
        } catch (error) {
          errorCount++;
        }
      });

      await Promise.all(promises);
    }

    const duration = (performance.now() - startTime) / 1000; // Convert to seconds
    const throughput = successCount / duration;
    const avgLatency = latencies.reduce((sum, l) => sum + l, 0) / latencies.length;

    const metrics: ThroughputMetrics = {
      totalRequests: requestCount,
      duration,
      throughput,
      successRate: successCount / requestCount,
      errorCount,
      avgLatency,
      concurrency
    };

    this.metrics.push(metrics);
    return metrics;
  }

  async runConcurrencyTest(): Promise<void> {
    const concurrencyLevels = [1, 10, 25, 50, 100, 200];
    const requestsPerLevel = 1000;

    console.log('\n=== CONCURRENCY THROUGHPUT TEST ===');

    for (const concurrency of concurrencyLevels) {
      const metrics = await this.testThroughput(requestsPerLevel, concurrency);
      this.printMetrics(metrics);

      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    this.printSummary();
  }

  async runSustainedLoadTest(durationSeconds: number = 60): Promise<void> {
    console.log(`\n=== SUSTAINED LOAD TEST (${durationSeconds}s) ===`);

    const targetRPS = 100; // Target requests per second
    const batchSize = 10;
    const startTime = performance.now();

    let totalRequests = 0;
    let successCount = 0;
    let errorCount = 0;
    const latencies: number[] = [];

    while ((performance.now() - startTime) / 1000 < durationSeconds) {
      const batchStart = performance.now();

      // Send batch of requests
      const promises = [];
      for (let i = 0; i < batchSize; i++) {
        const request = this.createTestRequest(`sustained-${totalRequests + i}`);

        promises.push(
          this.orchestrator.processRequest(request)
            .then(() => {
              successCount++;
              const latency = performance.now() - batchStart;
              latencies.push(latency);
            })
            .catch(() => errorCount++)
        );
      }

      await Promise.all(promises);
      totalRequests += batchSize;

      // Maintain target rate
      const batchDuration = performance.now() - batchStart;
      const targetBatchTime = (batchSize / targetRPS) * 1000; // ms
      const sleepTime = Math.max(0, targetBatchTime - batchDuration);

      if (sleepTime > 0) {
        await new Promise(resolve => setTimeout(resolve, sleepTime));
      }

      // Progress indicator every 10 seconds
      const elapsed = (performance.now() - startTime) / 1000;
      if (Math.floor(elapsed) % 10 === 0 && Math.floor(elapsed) !== Math.floor(elapsed - 0.1)) {
        console.log(`Progress: ${elapsed.toFixed(0)}s, Requests: ${totalRequests}, Success: ${successCount}`);
      }
    }

    const actualDuration = (performance.now() - startTime) / 1000;
    const avgLatency = latencies.reduce((sum, l) => sum + l, 0) / latencies.length;

    const metrics: ThroughputMetrics = {
      totalRequests,
      duration: actualDuration,
      throughput: successCount / actualDuration,
      successRate: successCount / totalRequests,
      errorCount,
      avgLatency,
      concurrency: batchSize
    };

    console.log('\n--- Sustained Load Results ---');
    this.printMetrics(metrics);

    // Check if we met target RPS
    const achievedRPS = totalRequests / actualDuration;
    console.log(`\nTarget RPS: ${targetRPS}`);
    console.log(`Achieved RPS: ${achievedRPS.toFixed(1)}`);
    console.log(`RPS Target Met: ${achievedRPS >= targetRPS * 0.9 ? 'PASS' : 'FAIL'}`);
  }

  async runBurstTest(): Promise<void> {
    console.log('\n=== BURST THROUGHPUT TEST ===');

    const burstSizes = [100, 500, 1000, 2000];

    for (const burstSize of burstSizes) {
      console.log(`\n--- Burst of ${burstSize} requests ---`);

      const startTime = performance.now();
      let successCount = 0;
      let errorCount = 0;

      // Send all requests at once
      const promises = [];
      for (let i = 0; i < burstSize; i++) {
        const request = this.createTestRequest(`burst-${i}`);
        promises.push(
          this.orchestrator.processRequest(request)
            .then(() => successCount++)
            .catch(() => errorCount++)
        );
      }

      await Promise.all(promises);

      const duration = (performance.now() - startTime) / 1000;
      const throughput = successCount / duration;

      console.log(`Duration: ${duration.toFixed(2)}s`);
      console.log(`Throughput: ${throughput.toFixed(1)} req/s`);
      console.log(`Success Rate: ${(successCount / burstSize * 100).toFixed(1)}%`);
      console.log(`Errors: ${errorCount}`);

      // Cooldown between bursts
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  async runProtocolThroughputTest(): Promise<void> {
    console.log('\n=== PROTOCOL THROUGHPUT COMPARISON ===');

    const protocols = ['betanet', 'bitchat', 'p2p', 'fog'] as const;
    const requestsPerProtocol = 500;
    const concurrency = 50;

    for (const protocol of protocols) {
      console.log(`\n--- ${protocol.toUpperCase()} Protocol ---`);

      const startTime = performance.now();
      let successCount = 0;

      const promises = [];
      for (let i = 0; i < requestsPerProtocol; i++) {
        const request: ConstitutionalBridgeRequest = {
          id: `${protocol}-throughput-${i}`,
          protocol,
          privacyTier: 'Silver',
          data: { test: i },
          userContext: {},
          timestamp: Date.now()
        };

        promises.push(
          this.orchestrator.processRequest(request)
            .then(() => successCount++)
            .catch(() => {})
        );

        // Batch requests
        if (promises.length >= concurrency || i === requestsPerProtocol - 1) {
          await Promise.all(promises);
          promises.length = 0;
        }
      }

      const duration = (performance.now() - startTime) / 1000;
      const throughput = successCount / duration;

      console.log(`Throughput: ${throughput.toFixed(1)} req/s`);
      console.log(`Success Rate: ${(successCount / requestsPerProtocol * 100).toFixed(1)}%`);
    }
  }

  private createTestRequest(id: string): ConstitutionalBridgeRequest {
    return {
      id,
      protocol: 'betanet',
      privacyTier: 'Silver',
      data: {
        type: 'throughput-test',
        content: `Test request ${id}`,
        timestamp: Date.now()
      },
      userContext: {
        userId: 'throughput-user',
        trustScore: 0.8
      },
      timestamp: Date.now()
    };
  }

  private printMetrics(metrics: ThroughputMetrics): void {
    console.log(`\nConcurrency: ${metrics.concurrency}`);
    console.log(`Total Requests: ${metrics.totalRequests}`);
    console.log(`Duration: ${metrics.duration.toFixed(2)}s`);
    console.log(`Throughput: ${metrics.throughput.toFixed(1)} req/s`);
    console.log(`Success Rate: ${(metrics.successRate * 100).toFixed(1)}%`);
    console.log(`Error Count: ${metrics.errorCount}`);
    console.log(`Avg Latency: ${metrics.avgLatency.toFixed(2)}ms`);
  }

  private printSummary(): void {
    console.log('\n=== THROUGHPUT TEST SUMMARY ===');

    const maxThroughput = Math.max(...this.metrics.map(m => m.throughput));
    const optimalConcurrency = this.metrics.find(m => m.throughput === maxThroughput)?.concurrency;

    console.log(`\nMax Throughput: ${maxThroughput.toFixed(1)} req/s`);
    console.log(`Optimal Concurrency: ${optimalConcurrency}`);

    // Check if we meet minimum throughput target (100 req/s)
    const targetMet = maxThroughput >= 100;
    console.log(`\nThroughput Target (>=100 req/s): ${targetMet ? 'PASS' : 'FAIL'}`);
  }

  async shutdown(): Promise<void> {
    await this.orchestrator.shutdown();
  }
}

// Main execution
async function main() {
  const test = new ThroughputTest();

  try {
    await test.initialize();

    // Run concurrency test
    await test.runConcurrencyTest();

    // Run sustained load test
    await test.runSustainedLoadTest(30); // 30 seconds

    // Run burst test
    await test.runBurstTest();

    // Run protocol comparison
    await test.runProtocolThroughputTest();

    await test.shutdown();
  } catch (error) {
    console.error('Throughput test failed:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { ThroughputTest, ThroughputMetrics };