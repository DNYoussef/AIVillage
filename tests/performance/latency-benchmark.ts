/**
 * Latency Benchmark Tests
 * Validates P95 latency < 75ms target
 */

import { performance } from 'perf_hooks';
import { BridgeOrchestrator } from '../../src/bridge/BridgeOrchestrator';
import { ConstitutionalBridgeRequest } from '../../src/bridge/types';

interface LatencyMetrics {
  avg: number;
  median: number;
  p50: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
}

class LatencyBenchmark {
  private orchestrator: BridgeOrchestrator;
  private latencies: number[] = [];

  constructor() {
    this.orchestrator = new BridgeOrchestrator({
      betaNetEndpoint: 'http://localhost:8080',
      defaultPrivacyTier: 'Silver',
      enableConstitutionalValidation: true,
      complianceThreshold: 0.8,
      performance: {
        targetP95Latency: 75,
        circuitBreakerEnabled: true,
        maxConcurrentRequests: 100,
        requestTimeout: 5000
      }
    });
  }

  async initialize(): Promise<void> {
    await this.orchestrator.initialize();
    await this.warmup();
  }

  private async warmup(): Promise<void> {
    console.log('Warming up system...');
    
    // Send 20 warmup requests
    for (let i = 0; i < 20; i++) {
      const request = this.createTestRequest(`warmup-${i}`);
      await this.orchestrator.processRequest(request);
    }
  }

  async runBenchmark(iterations: number = 1000): Promise<LatencyMetrics> {
    console.log(`Running latency benchmark with ${iterations} iterations...`);
    this.latencies = [];

    for (let i = 0; i < iterations; i++) {
      const latency = await this.measureRequestLatency(i);
      this.latencies.push(latency);

      // Progress indicator
      if ((i + 1) % 100 === 0) {
        console.log(`Progress: ${i + 1}/${iterations}`);
      }
    }

    return this.calculateMetrics();
  }

  private async measureRequestLatency(index: number): Promise<number> {
    const request = this.createTestRequest(`bench-${index}`);
    
    const start = performance.now();
    try {
      await this.orchestrator.processRequest(request);
    } catch (error) {
      // Count errors but continue
      console.error(`Request ${index} failed:`, error);
    }
    const end = performance.now();

    return end - start;
  }

  private createTestRequest(id: string): ConstitutionalBridgeRequest {
    return {
      id,
      protocol: 'betanet',
      privacyTier: 'Silver',
      data: {
        type: 'query',
        content: `Test request ${id}`,
        timestamp: Date.now()
      },
      userContext: {
        userId: 'bench-user',
        trustScore: 0.8
      },
      timestamp: Date.now()
    };
  }

  private calculateMetrics(): LatencyMetrics {
    const sorted = [...this.latencies].sort((a, b) => a - b);
    const len = sorted.length;

    return {
      avg: this.latencies.reduce((sum, val) => sum + val, 0) / len,
      median: sorted[Math.floor(len / 2)],
      p50: sorted[Math.floor(len * 0.50)],
      p95: sorted[Math.floor(len * 0.95)],
      p99: sorted[Math.floor(len * 0.99)],
      min: sorted[0],
      max: sorted[len - 1]
    };
  }

  async runProtocolComparison(): Promise<void> {
    const protocols = ['betanet', 'bitchat', 'p2p', 'fog'] as const;
    const results: Record<string, LatencyMetrics> = {};

    for (const protocol of protocols) {
      console.log(`\nBenchmarking ${protocol} protocol...`);
      this.latencies = [];

      for (let i = 0; i < 200; i++) {
        const request: ConstitutionalBridgeRequest = {
          id: `${protocol}-${i}`,
          protocol,
          privacyTier: 'Silver',
          data: { test: i },
          userContext: {},
          timestamp: Date.now()
        };

        const latency = await this.measureRequestLatency(i);
        this.latencies.push(latency);
      }

      results[protocol] = this.calculateMetrics();
    }

    this.printProtocolComparison(results);
  }

  async runPrivacyTierComparison(): Promise<void> {
    const tiers = ['Bronze', 'Silver', 'Gold', 'Platinum'] as const;
    const results: Record<string, LatencyMetrics> = {};

    for (const tier of tiers) {
      console.log(`\nBenchmarking ${tier} privacy tier...`);
      this.latencies = [];

      for (let i = 0; i < 200; i++) {
        const request: ConstitutionalBridgeRequest = {
          id: `${tier}-${i}`,
          protocol: 'betanet',
          privacyTier: tier,
          data: {
            userId: `user-${i}`,
            email: `test${i}@example.com`,
            sensitive: tier === 'Platinum'
          },
          userContext: {
            trustScore: tier === 'Bronze' ? 0.5 : 0.9
          },
          timestamp: Date.now()
        };

        const latency = await this.measureRequestLatency(i);
        this.latencies.push(latency);
      }

      results[tier] = this.calculateMetrics();
    }

    this.printTierComparison(results);
  }

  private printMetrics(metrics: LatencyMetrics): void {
    console.log('\n=== LATENCY METRICS ===' );
    console.log(`Average: ${metrics.avg.toFixed(2)}ms`);
    console.log(`Median: ${metrics.median.toFixed(2)}ms`);
    console.log(`P50: ${metrics.p50.toFixed(2)}ms`);
    console.log(`P95: ${metrics.p95.toFixed(2)}ms`);
    console.log(`P99: ${metrics.p99.toFixed(2)}ms`);
    console.log(`Min: ${metrics.min.toFixed(2)}ms`);
    console.log(`Max: ${metrics.max.toFixed(2)}ms`);

    // Check P95 target
    const targetMet = metrics.p95 < 75;
    console.log(`\nP95 Target (<75ms): ${targetMet ? 'PASS' : 'FAIL'}`);
    
    if (!targetMet) {
      console.error(`P95 latency ${metrics.p95.toFixed(2)}ms exceeds 75ms target!`);
    }
  }

  private printProtocolComparison(results: Record<string, LatencyMetrics>): void {
    console.log('\n=== PROTOCOL COMPARISON ===');
    
    for (const [protocol, metrics] of Object.entries(results)) {
      console.log(`\n${protocol.toUpperCase()}:`);
      console.log(`  Avg: ${metrics.avg.toFixed(2)}ms`);
      console.log(`  P95: ${metrics.p95.toFixed(2)}ms`);
      console.log(`  P99: ${metrics.p99.toFixed(2)}ms`);
    }
  }

  private printTierComparison(results: Record<string, LatencyMetrics>): void {
    console.log('\n=== PRIVACY TIER COMPARISON ===');
    
    for (const [tier, metrics] of Object.entries(results)) {
      console.log(`\n${tier}:`);
      console.log(`  Avg: ${metrics.avg.toFixed(2)}ms`);
      console.log(`  P95: ${metrics.p95.toFixed(2)}ms`);
      console.log(`  Overhead: ${(metrics.avg - results['Platinum'].avg).toFixed(2)}ms`);
    }
  }

  async shutdown(): Promise<void> {
    await this.orchestrator.shutdown();
  }
}

// Main execution
async function main() {
  const benchmark = new LatencyBenchmark();

  try {
    await benchmark.initialize();

    // Run main benchmark
    const metrics = await benchmark.runBenchmark(1000);
    benchmark['printMetrics'](metrics);

    // Run protocol comparison
    await benchmark.runProtocolComparison();

    // Run privacy tier comparison  
    await benchmark.runPrivacyTierComparison();

    await benchmark.shutdown();
  } catch (error) {
    console.error('Benchmark failed:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { LatencyBenchmark, LatencyMetrics };