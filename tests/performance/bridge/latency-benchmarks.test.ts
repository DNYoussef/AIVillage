import { describe, it, expect, beforeEach, afterEach, beforeAll, afterAll } from '@jest/globals';
import { ConstitutionalBetaNetAdapter } from '../../../src/bridge/ConstitutionalBetaNetAdapter';
import { PerformanceMonitor } from '../../../src/bridge/PerformanceMonitor';
import { BenchmarkRunner } from '../../helpers/BenchmarkRunner';
import { StatisticalAnalyzer } from '../../helpers/StatisticalAnalyzer';
import { LatencyProfiler } from '../../helpers/LatencyProfiler';

describe('Latency Benchmarks (<75ms p95)', () => {
  let adapter: ConstitutionalBetaNetAdapter;
  let performanceMonitor: PerformanceMonitor;
  let benchmarkRunner: BenchmarkRunner;
  let analyzer: StatisticalAnalyzer;
  let profiler: LatencyProfiler;

  // Performance targets
  const LATENCY_TARGETS = {
    p50: 25,  // 50th percentile: 25ms
    p75: 45,  // 75th percentile: 45ms
    p90: 60,  // 90th percentile: 60ms
    p95: 75,  // 95th percentile: 75ms (SLA requirement)
    p99: 150, // 99th percentile: 150ms
    mean: 30  // Average: 30ms
  };

  beforeAll(async () => {
    benchmarkRunner = new BenchmarkRunner({
      warmupRounds: 100,
      measurementRounds: 1000,
      enableProfiling: true
    });

    analyzer = new StatisticalAnalyzer();
    profiler = new LatencyProfiler();

    await benchmarkRunner.initialize();
  });

  afterAll(async () => {
    await benchmarkRunner.cleanup();
  });

  beforeEach(async () => {
    performanceMonitor = new PerformanceMonitor({
      enableHighResolutionTiming: true,
      enableLatencyProfiling: true
    });

    adapter = new ConstitutionalBetaNetAdapter({
      performanceMonitor,
      enableOptimizations: true,
      enableCaching: true,
      enableConnectionPooling: true
    });

    await adapter.initialize();
    profiler.reset();
  });

  afterEach(async () => {
    await adapter.cleanup();
    await performanceMonitor.cleanup();
  });

  describe('request translation latency', () => {
    it('should translate simple requests within latency targets', async () => {
      const simpleRequest = {
        method: 'GET',
        path: '/api/v1/agents/status',
        headers: { 'Content-Type': 'application/json' },
        body: {}
      };

      const benchmark = await benchmarkRunner.runBenchmark(
        'simple_request_translation',
        () => adapter.translateRequest(simpleRequest),
        { iterations: 1000 }
      );

      // Verify latency targets
      expect(benchmark.percentiles.p50).toBeLessThan(LATENCY_TARGETS.p50);
      expect(benchmark.percentiles.p75).toBeLessThan(LATENCY_TARGETS.p75);
      expect(benchmark.percentiles.p90).toBeLessThan(LATENCY_TARGETS.p90);
      expect(benchmark.percentiles.p95).toBeLessThan(LATENCY_TARGETS.p95);
      expect(benchmark.percentiles.p99).toBeLessThan(LATENCY_TARGETS.p99);
      expect(benchmark.mean).toBeLessThan(LATENCY_TARGETS.mean);

      // Verify consistency (low standard deviation)
      expect(benchmark.standardDeviation).toBeLessThan(benchmark.mean * 0.5);

      // Verify no significant outliers
      expect(benchmark.outliers.length).toBeLessThan(benchmark.totalSamples * 0.01);
    });

    it('should translate complex requests within latency targets', async () => {
      const complexRequest = {
        method: 'POST',
        path: '/api/v1/agents/batch',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer complex-token-with-many-claims',
          'X-Request-ID': 'complex-request-123456789'
        },
        body: {
          operations: Array(10).fill(null).map((_, i) => ({
            action: 'create',
            data: {
              name: `Agent ${i}`,
              type: 'researcher',
              capabilities: ['web_search', 'document_analysis', 'data_synthesis'],
              configuration: {
                maxTokens: 4000,
                temperature: 0.7,
                model: 'claude-3-sonnet',
                systemPrompt: 'You are a helpful research assistant...'
              }
            }
          }))
        }
      };

      const benchmark = await benchmarkRunner.runBenchmark(
        'complex_request_translation',
        () => adapter.translateRequest(complexRequest),
        { iterations: 500 }
      );

      // Complex requests should still meet most targets (with some allowance)
      expect(benchmark.percentiles.p50).toBeLessThan(LATENCY_TARGETS.p50 * 1.5);
      expect(benchmark.percentiles.p75).toBeLessThan(LATENCY_TARGETS.p75 * 1.5);
      expect(benchmark.percentiles.p90).toBeLessThan(LATENCY_TARGETS.p90 * 1.3);
      expect(benchmark.percentiles.p95).toBeLessThan(LATENCY_TARGETS.p95); // Strict requirement
      expect(benchmark.percentiles.p99).toBeLessThan(LATENCY_TARGETS.p99 * 1.2);
    });

    it('should maintain latency under concurrent load', async () => {
      const request = {
        method: 'POST',
        path: '/api/v1/agents',
        body: { name: 'Concurrent Agent', type: 'assistant' }
      };

      const concurrencyLevels = [1, 5, 10, 25, 50];
      const results = {};

      for (const concurrency of concurrencyLevels) {
        const benchmark = await benchmarkRunner.runConcurrentBenchmark(
          `translation_concurrency_${concurrency}`,
          () => adapter.translateRequest(request),
          { concurrency, iterations: 200 }
        );

        results[concurrency] = benchmark;

        // Verify latency doesn't degrade significantly with concurrency
        expect(benchmark.percentiles.p95).toBeLessThan(LATENCY_TARGETS.p95 * 1.5);
      }

      // Verify latency scaling is sub-linear
      const latencies = concurrencyLevels.map(c => results[c].percentiles.p95);
      const latencyGrowthRates = latencies.slice(1).map((l, i) => l / latencies[i]);

      // Growth rate should be less than concurrency growth rate
      latencyGrowthRates.forEach((rate, i) => {
        const concurrencyGrowthRate = concurrencyLevels[i + 1] / concurrencyLevels[i];
        expect(rate).toBeLessThan(concurrencyGrowthRate);
      });
    });

    it('should optimize latency through caching', async () => {
      const cacheableRequest = {
        method: 'GET',
        path: '/api/v1/agents/types',
        headers: { 'Cache-Control': 'max-age=300' }
      };

      // First request (cache miss)
      const missLatencies = await benchmarkRunner.measureLatencies(
        () => adapter.translateRequest(cacheableRequest),
        { iterations: 100 }
      );

      // Subsequent requests (cache hits)
      const hitLatencies = await benchmarkRunner.measureLatencies(
        () => adapter.translateRequest(cacheableRequest),
        { iterations: 100 }
      );

      const missP95 = analyzer.calculatePercentile(missLatencies, 95);
      const hitP95 = analyzer.calculatePercentile(hitLatencies, 95);

      // Cache hits should be significantly faster
      expect(hitP95).toBeLessThan(missP95 * 0.3); // 70% improvement
      expect(hitP95).toBeLessThan(10); // <10ms for cache hits
    });
  });

  describe('response translation latency', () => {
    it('should translate responses within latency targets', async () => {
      const aiVillageResponse = {
        success: true,
        data: {
          agentId: 'agent-123',
          name: 'Test Agent',
          status: 'active',
          capabilities: ['research', 'analysis'],
          metadata: { created: Date.now(), version: '1.0.0' }
        },
        performance: { processingTime: 150, tokensUsed: 75 }
      };

      const benchmark = await benchmarkRunner.runBenchmark(
        'response_translation',
        () => adapter.translateResponse(aiVillageResponse),
        { iterations: 1000 }
      );

      // Response translation should be even faster than request translation
      expect(benchmark.percentiles.p95).toBeLessThan(LATENCY_TARGETS.p95 * 0.8);
      expect(benchmark.mean).toBeLessThan(LATENCY_TARGETS.mean * 0.8);
    });

    it('should handle large response payloads efficiently', async () => {
      const largeResponse = {
        success: true,
        data: {
          agents: Array(1000).fill(null).map((_, i) => ({
            id: `agent-${i}`,
            name: `Agent ${i}`,
            status: 'active',
            capabilities: ['research', 'analysis', 'synthesis'],
            metadata: {
              created: Date.now() - (i * 1000),
              lastActive: Date.now() - (i * 100),
              version: '1.0.0',
              description: 'A'.repeat(200) // 200 character description
            }
          })),
          pagination: { total: 1000, page: 1, pageSize: 1000 }
        }
      };

      const benchmark = await benchmarkRunner.runBenchmark(
        'large_response_translation',
        () => adapter.translateResponse(largeResponse),
        { iterations: 100 }
      );

      // Large responses should still meet reasonable targets
      expect(benchmark.percentiles.p95).toBeLessThan(LATENCY_TARGETS.p95 * 2);
      expect(benchmark.mean).toBeLessThan(LATENCY_TARGETS.mean * 1.5);
    });

    it('should optimize serialization performance', async () => {
      const responses = [
        { size: 'small', data: { id: 1, name: 'Small' } },
        { size: 'medium', data: { agents: Array(100).fill({ id: 1, name: 'Agent' }) } },
        { size: 'large', data: { agents: Array(1000).fill({ id: 1, name: 'Agent' }) } },
        { size: 'xlarge', data: { agents: Array(5000).fill({ id: 1, name: 'Agent' }) } }
      ];

      const serializationBenchmarks = {};

      for (const response of responses) {
        const benchmark = await benchmarkRunner.runBenchmark(
          `serialization_${response.size}`,
          () => adapter.translateResponse({ success: true, data: response.data }),
          { iterations: 200 }
        );

        serializationBenchmarks[response.size] = benchmark;
      }

      // Verify serialization time scales sub-linearly with data size
      const times = ['small', 'medium', 'large', 'xlarge'].map(
        size => serializationBenchmarks[size].mean
      );

      expect(times[1] / times[0]).toBeLessThan(50); // Medium shouldn't be 100x slower
      expect(times[2] / times[1]).toBeLessThan(5);  // Large shouldn't be 10x slower
      expect(times[3] / times[2]).toBeLessThan(3);  // XLarge shouldn't be 5x slower
    });
  });

  describe('end-to-end latency', () => {
    it('should complete full request cycles within SLA', async () => {
      const e2eRequest = {
        method: 'POST',
        path: '/api/v1/agents',
        headers: { 'Content-Type': 'application/json' },
        body: { name: 'E2E Agent', type: 'assistant' }
      };

      const benchmark = await benchmarkRunner.runBenchmark(
        'end_to_end_latency',
        () => adapter.processRequest(e2eRequest),
        { iterations: 500 }
      );

      // End-to-end should meet stricter targets for user experience
      expect(benchmark.percentiles.p95).toBeLessThan(LATENCY_TARGETS.p95);
      expect(benchmark.percentiles.p99).toBeLessThan(LATENCY_TARGETS.p99);
      expect(benchmark.mean).toBeLessThan(LATENCY_TARGETS.mean * 1.2);

      // Verify low jitter
      const jitter = benchmark.standardDeviation / benchmark.mean;
      expect(jitter).toBeLessThan(0.3); // <30% coefficient of variation
    });

    it('should profile latency breakdown by component', async () => {
      const request = {
        method: 'POST',
        path: '/api/v1/agents/complex',
        body: {
          name: 'Profiled Agent',
          configuration: { detailed: true },
          metadata: { profiling: true }
        }
      };

      profiler.startProfiling();

      await benchmarkRunner.runBenchmark(
        'latency_profiling',
        () => adapter.processRequest(request),
        { iterations: 100 }
      );

      const profile = profiler.getProfile();

      // Verify each component meets its latency budget
      expect(profile.components.requestTranslation.p95).toBeLessThan(20);
      expect(profile.components.privacyValidation.p95).toBeLessThan(15);
      expect(profile.components.constitutionalCheck.p95).toBeLessThan(25);
      expect(profile.components.responseTranslation.p95).toBeLessThan(15);

      // Verify total latency is sum of components (no hidden overhead)
      const componentSum = Object.values(profile.components)
        .reduce((sum, component: any) => sum + component.mean, 0);
      expect(profile.total.mean).toBeLessThan(componentSum * 1.1); // <10% overhead
    });

    it('should maintain consistent latency across request patterns', async () => {
      const requestPatterns = [
        { name: 'simple_get', method: 'GET', path: '/api/v1/agents/list' },
        { name: 'simple_post', method: 'POST', path: '/api/v1/agents', body: { name: 'Test' } },
        { name: 'update_request', method: 'PUT', path: '/api/v1/agents/123', body: { status: 'active' } },
        { name: 'delete_request', method: 'DELETE', path: '/api/v1/agents/123' },
        { name: 'batch_request', method: 'POST', path: '/api/v1/agents/batch', body: { operations: [] } }
      ];

      const patternBenchmarks = {};

      for (const pattern of requestPatterns) {
        const benchmark = await benchmarkRunner.runBenchmark(
          `pattern_${pattern.name}`,
          () => adapter.processRequest(pattern),
          { iterations: 200 }
        );

        patternBenchmarks[pattern.name] = benchmark;
      }

      // All patterns should meet the same latency targets
      Object.values(patternBenchmarks).forEach((benchmark: any) => {
        expect(benchmark.percentiles.p95).toBeLessThan(LATENCY_TARGETS.p95);
      });

      // Verify pattern latencies are reasonably consistent
      const p95Latencies = Object.values(patternBenchmarks).map((b: any) => b.percentiles.p95);
      const latencyRange = Math.max(...p95Latencies) - Math.min(...p95Latencies);
      expect(latencyRange).toBeLessThan(LATENCY_TARGETS.p95 * 0.5); // <50% of target range
    });
  });

  describe('latency optimization validation', () => {
    it('should demonstrate connection pooling benefits', async () => {
      const request = {
        method: 'GET',
        path: '/api/v1/agents/pool-test',
        body: {}
      };

      // Test without connection pooling
      adapter.disableConnectionPooling();
      const nopoolBenchmark = await benchmarkRunner.runBenchmark(
        'no_connection_pooling',
        () => adapter.processRequest(request),
        { iterations: 100 }
      );

      // Test with connection pooling
      adapter.enableConnectionPooling();
      const poolBenchmark = await benchmarkRunner.runBenchmark(
        'with_connection_pooling',
        () => adapter.processRequest(request),
        { iterations: 100 }
      );

      // Connection pooling should provide significant improvement
      expect(poolBenchmark.percentiles.p95).toBeLessThan(nopoolBenchmark.percentiles.p95 * 0.8);
      expect(poolBenchmark.mean).toBeLessThan(nopoolBenchmark.mean * 0.8);
    });

    it('should demonstrate request batching efficiency', async () => {
      const singleRequests = Array(10).fill(null).map((_, i) => ({
        method: 'POST',
        path: '/api/v1/agents',
        body: { name: `Single Agent ${i}` }
      }));

      const batchRequest = {
        method: 'POST',
        path: '/api/v1/agents/batch',
        body: {
          operations: singleRequests.map(req => ({
            action: 'create',
            data: req.body
          }))
        }
      };

      // Measure individual requests
      const individualLatencies = [];
      for (const request of singleRequests) {
        const start = performance.now();
        await adapter.processRequest(request);
        individualLatencies.push(performance.now() - start);
      }
      const totalIndividualTime = individualLatencies.reduce((sum, time) => sum + time, 0);

      // Measure batch request
      const batchLatencies = await benchmarkRunner.measureLatencies(
        () => adapter.processRequest(batchRequest),
        { iterations: 10 }
      );
      const avgBatchTime = analyzer.calculateMean(batchLatencies);

      // Batching should be significantly more efficient
      expect(avgBatchTime).toBeLessThan(totalIndividualTime * 0.6);
    });

    it('should validate compression benefits for large payloads', async () => {
      const largePayload = {
        method: 'POST',
        path: '/api/v1/agents/large-data',
        body: {
          data: Array(1000).fill('x'.repeat(1000)).join(''), // 1MB of compressible data
          metadata: { compression: 'test' }
        }
      };

      // Test without compression
      adapter.disableCompression();
      const noCompressionBenchmark = await benchmarkRunner.runBenchmark(
        'no_compression',
        () => adapter.processRequest(largePayload),
        { iterations: 50 }
      );

      // Test with compression
      adapter.enableCompression();
      const compressionBenchmark = await benchmarkRunner.runBenchmark(
        'with_compression',
        () => adapter.processRequest(largePayload),
        { iterations: 50 }
      );

      // Compression should improve latency for large payloads
      expect(compressionBenchmark.percentiles.p95).toBeLessThan(noCompressionBenchmark.percentiles.p95 * 0.9);
    });

    it('should validate async processing benefits', async () => {
      const asyncRequest = {
        method: 'POST',
        path: '/api/v1/agents/async-test',
        body: { name: 'Async Agent', processAsync: true }
      };

      const syncRequest = {
        method: 'POST',
        path: '/api/v1/agents/sync-test',
        body: { name: 'Sync Agent', processAsync: false }
      };

      // Compare sync vs async processing
      const syncBenchmark = await benchmarkRunner.runBenchmark(
        'sync_processing',
        () => adapter.processRequest(syncRequest),
        { iterations: 100 }
      );

      const asyncBenchmark = await benchmarkRunner.runBenchmark(
        'async_processing',
        () => adapter.processRequest(asyncRequest),
        { iterations: 100 }
      );

      // Async should have better latency characteristics under load
      expect(asyncBenchmark.percentiles.p99).toBeLessThan(syncBenchmark.percentiles.p99 * 0.8);
      expect(asyncBenchmark.standardDeviation).toBeLessThan(syncBenchmark.standardDeviation);
    });
  });

  describe('latency regression testing', () => {
    it('should detect latency regressions compared to baseline', async () => {
      // Establish baseline
      const baselineRequest = {
        method: 'GET',
        path: '/api/v1/baseline-test',
        body: {}
      };

      const baseline = await benchmarkRunner.runBenchmark(
        'baseline_latency',
        () => adapter.processRequest(baselineRequest),
        { iterations: 200 }
      );

      // Store baseline for future comparisons
      benchmarkRunner.storeBaseline('request_processing', baseline);

      // Simulate code change that might affect performance
      adapter.enableVerboseLogging();

      // Measure current performance
      const current = await benchmarkRunner.runBenchmark(
        'current_latency',
        () => adapter.processRequest(baselineRequest),
        { iterations: 200 }
      );

      // Check for regression
      const regression = analyzer.detectRegression(baseline, current, {
        significanceThreshold: 0.05,
        effectSizeThreshold: 0.1
      });

      if (regression.detected) {
        console.warn(`Latency regression detected:
          Baseline P95: ${baseline.percentiles.p95}ms
          Current P95: ${current.percentiles.p95}ms
          Regression: ${regression.percentageIncrease}%`);
      }

      // Ensure no significant regression
      expect(regression.percentageIncrease).toBeLessThan(10); // <10% regression allowed
    });

    it('should track latency trends over time', async () => {
      const trendRequest = {
        method: 'GET',
        path: '/api/v1/trend-test',
        body: {}
      };

      const measurements = [];

      // Collect measurements over simulated time periods
      for (let period = 0; period < 10; period++) {
        const benchmark = await benchmarkRunner.runBenchmark(
          `trend_period_${period}`,
          () => adapter.processRequest(trendRequest),
          { iterations: 50 }
        );

        measurements.push({
          period,
          p95: benchmark.percentiles.p95,
          mean: benchmark.mean,
          timestamp: Date.now()
        });

        // Simulate gradual load increase
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      // Analyze trend
      const trend = analyzer.analyzeTrend(measurements.map(m => m.p95));

      // Latency should remain stable (no significant upward trend)
      expect(trend.slope).toBeLessThan(1); // <1ms increase per period
      expect(trend.correlation).toBeLessThan(0.5); // Weak correlation with time
    });
  });
});