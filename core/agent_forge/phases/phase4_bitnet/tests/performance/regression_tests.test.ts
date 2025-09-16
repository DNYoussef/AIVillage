/**
 * Performance Regression Tests
 * Phase 4 - Ensuring performance targets are maintained
 */

import { BitNetLayer } from '../../../src/phase4_bitnet/BitNetLayer';
import { PerformanceBaseline } from '../../../src/phase4_bitnet/utils/PerformanceBaseline';
import { RegressionDetector } from '../../../src/phase4_bitnet/utils/RegressionDetector';

describe('Performance Regression Tests', () => {
  let performanceBaseline: PerformanceBaseline;
  let regressionDetector: RegressionDetector;

  beforeAll(async () => {
    performanceBaseline = new PerformanceBaseline();
    regressionDetector = new RegressionDetector();

    // Load existing baselines or create new ones
    await performanceBaseline.loadBaselines('./tests/phase4_bitnet/baselines');
  });

  beforeEach(() => {
    // Warm up V8 JIT compiler
    const warmupLayer = new BitNetLayer({
      inputSize: 64,
      outputSize: 32,
      quantizationBits: 1,
      useBinaryActivations: true
    });

    const warmupInput = new Float32Array(64).fill(0.5);
    for (let i = 0; i < 10; i++) {
      warmupLayer.forward(warmupInput);
    }
  });

  describe('Forward Pass Performance', () => {
    it('should maintain forward pass latency under 10ms for standard layers', async () => {
      const testConfigs = [
        { inputSize: 256, outputSize: 128, maxLatency: 5 },
        { inputSize: 512, outputSize: 256, maxLatency: 8 },
        { inputSize: 1024, outputSize: 512, maxLatency: 10 }
      ];

      for (const config of testConfigs) {
        const bitNetLayer = new BitNetLayer({
          inputSize: config.inputSize,
          outputSize: config.outputSize,
          quantizationBits: 1,
          useBinaryActivations: true
        });

        const input = new Float32Array(config.inputSize).fill(0.5);
        const measurements: number[] = [];

        // Run multiple measurements
        for (let i = 0; i < 100; i++) {
          const startTime = performance.now();
          await bitNetLayer.forward(input);
          const endTime = performance.now();
          measurements.push(endTime - startTime);
        }

        // Calculate statistics
        const avgLatency = measurements.reduce((a, b) => a + b) / measurements.length;
        const p95Latency = measurements.sort((a, b) => a - b)[Math.floor(measurements.length * 0.95)];

        expect(avgLatency).toBeLessThan(config.maxLatency);
        expect(p95Latency).toBeLessThan(config.maxLatency * 1.5);

        // Check against baseline
        const baselineKey = `forward_${config.inputSize}_${config.outputSize}`;
        await performanceBaseline.compareWithBaseline(baselineKey, avgLatency, 0.1); // 10% tolerance
      }
    });

    it('should scale linearly with input size', async () => {
      const inputSizes = [128, 256, 512, 1024];
      const outputSize = 256;
      const latencies: Array<{ size: number; latency: number }> = [];

      for (const inputSize of inputSizes) {
        const bitNetLayer = new BitNetLayer({
          inputSize,
          outputSize,
          quantizationBits: 1,
          useBinaryActivations: true
        });

        const input = new Float32Array(inputSize).fill(0.5);
        const measurements: number[] = [];

        for (let i = 0; i < 50; i++) {
          const startTime = performance.now();
          await bitNetLayer.forward(input);
          const endTime = performance.now();
          measurements.push(endTime - startTime);
        }

        const avgLatency = measurements.reduce((a, b) => a + b) / measurements.length;
        latencies.push({ size: inputSize, latency: avgLatency });
      }

      // Check for linear scaling
      for (let i = 1; i < latencies.length; i++) {
        const ratio = latencies[i].latency / latencies[i-1].latency;
        const sizeRatio = latencies[i].size / latencies[i-1].size;

        // Performance should scale roughly linearly (within 20% tolerance)
        expect(Math.abs(ratio - sizeRatio) / sizeRatio).toBeLessThan(0.2);
      }
    });
  });

  describe('Batch Processing Performance', () => {
    it('should achieve high throughput for batch processing', async () => {
      const batchSizes = [1, 8, 16, 32, 64];
      const inputSize = 512;
      const outputSize = 256;

      const bitNetLayer = new BitNetLayer({
        inputSize,
        outputSize,
        quantizationBits: 1,
        useBinaryActivations: true
      });

      const throughputResults: Array<{ batchSize: number; throughput: number }> = [];

      for (const batchSize of batchSizes) {
        const batchInput = new Float32Array(batchSize * inputSize);
        for (let i = 0; i < batchInput.length; i++) {
          batchInput[i] = Math.random();
        }

        const iterations = 20;
        const startTime = performance.now();

        for (let i = 0; i < iterations; i++) {
          await bitNetLayer.forwardBatch(batchInput, batchSize);
        }

        const endTime = performance.now();
        const totalTime = (endTime - startTime) / 1000; // Convert to seconds
        const totalSamples = iterations * batchSize;
        const throughput = totalSamples / totalTime; // Samples per second

        throughputResults.push({ batchSize, throughput });

        // Larger batches should achieve higher throughput
        if (batchSize > 1) {
          const baselineThroughput = throughputResults.find(r => r.batchSize === 1)?.throughput || 0;
          expect(throughput).toBeGreaterThan(baselineThroughput * 0.8); // At least 80% of single-sample throughput
        }
      }

      console.table(throughputResults);
    });

    it('should maintain low latency for small batches', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 256,
        outputSize: 128,
        quantizationBits: 1,
        useBinaryActivations: true
      });

      const smallBatch = new Float32Array(4 * 256); // Batch size of 4
      for (let i = 0; i < smallBatch.length; i++) {
        smallBatch[i] = Math.random();
      }

      const measurements: number[] = [];

      for (let i = 0; i < 100; i++) {
        const startTime = performance.now();
        await bitNetLayer.forwardBatch(smallBatch, 4);
        const endTime = performance.now();
        measurements.push(endTime - startTime);
      }

      const avgLatency = measurements.reduce((a, b) => a + b) / measurements.length;
      const p99Latency = measurements.sort((a, b) => a - b)[Math.floor(measurements.length * 0.99)];

      expect(avgLatency).toBeLessThan(15); // Average under 15ms
      expect(p99Latency).toBeLessThan(25); // P99 under 25ms
    });
  });

  describe('Memory Access Performance', () => {
    it('should minimize cache misses during operations', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 1024,
        outputSize: 512,
        quantizationBits: 1,
        useBinaryActivations: true,
        optimizeMemoryLayout: true
      });

      const input = new Float32Array(1024);
      for (let i = 0; i < input.length; i++) {
        input[i] = Math.sin(i * 0.01); // Predictable pattern
      }

      // Measure cache efficiency through repeated access patterns
      const repetitions = 1000;
      const startTime = performance.now();

      for (let i = 0; i < repetitions; i++) {
        await bitNetLayer.forward(input);
      }

      const endTime = performance.now();
      const avgTime = (endTime - startTime) / repetitions;

      // Should benefit from cache locality
      expect(avgTime).toBeLessThan(5); // Less than 5ms per operation
    });

    it('should optimize for sequential memory access patterns', async () => {
      const layerSizes = [256, 512, 1024];
      const accessPatterns: Array<{ size: number; time: number }> = [];

      for (const size of layerSizes) {
        const bitNetLayer = new BitNetLayer({
          inputSize: size,
          outputSize: size / 2,
          quantizationBits: 1,
          useBinaryActivations: true
        });

        const input = new Float32Array(size);

        // Sequential access pattern
        for (let i = 0; i < input.length; i++) {
          input[i] = i % 256;
        }

        const iterations = 100;
        const startTime = performance.now();

        for (let i = 0; i < iterations; i++) {
          await bitNetLayer.forward(input);
        }

        const endTime = performance.now();
        const avgTime = (endTime - startTime) / iterations;

        accessPatterns.push({ size, time: avgTime });
      }

      // Memory access should scale sub-linearly due to cache benefits
      for (let i = 1; i < accessPatterns.length; i++) {
        const timeRatio = accessPatterns[i].time / accessPatterns[i-1].time;
        const sizeRatio = accessPatterns[i].size / accessPatterns[i-1].size;

        expect(timeRatio).toBeLessThan(sizeRatio * 0.9); // Should be faster than linear scaling
      }
    });
  });

  describe('Quantization Performance', () => {
    it('should quantize weights efficiently', async () => {
      const weightSizes = [1000, 10000, 100000, 1000000];
      const quantizationTimes: Array<{ size: number; time: number }> = [];

      for (const size of weightSizes) {
        const weights = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          weights[i] = Math.random() - 0.5;
        }

        const bitNetLayer = new BitNetLayer({
          inputSize: Math.sqrt(size),
          outputSize: Math.sqrt(size),
          quantizationBits: 1,
          useBinaryActivations: true
        });

        const startTime = performance.now();
        await bitNetLayer.quantizeWeights(weights);
        const endTime = performance.now();

        const quantizationTime = endTime - startTime;
        quantizationTimes.push({ size, time: quantizationTime });

        // Quantization should complete quickly
        const timePerWeight = quantizationTime / size;
        expect(timePerWeight).toBeLessThan(0.001); // Less than 1μs per weight
      }

      // Quantization should scale linearly
      for (let i = 1; i < quantizationTimes.length; i++) {
        const timeRatio = quantizationTimes[i].time / quantizationTimes[i-1].time;
        const sizeRatio = quantizationTimes[i].size / quantizationTimes[i-1].size;

        expect(Math.abs(timeRatio - sizeRatio) / sizeRatio).toBeLessThan(0.3); // Within 30% of linear
      }
    });
  });

  describe('Regression Detection', () => {
    it('should detect performance regressions automatically', async () => {
      const testConfig = {
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 1
      };

      const bitNetLayer = new BitNetLayer({
        ...testConfig,
        useBinaryActivations: true
      });

      const input = new Float32Array(testConfig.inputSize).fill(0.5);
      const measurements: number[] = [];

      // Collect performance measurements
      for (let i = 0; i < 50; i++) {
        const startTime = performance.now();
        await bitNetLayer.forward(input);
        const endTime = performance.now();
        measurements.push(endTime - startTime);
      }

      const avgPerformance = measurements.reduce((a, b) => a + b) / measurements.length;
      const testKey = `regression_test_${testConfig.inputSize}_${testConfig.outputSize}`;

      // Compare with historical data
      const regressionResult = await regressionDetector.detectRegression(
        testKey,
        avgPerformance,
        measurements
      );

      expect(regressionResult.isRegression).toBe(false);
      expect(regressionResult.confidenceLevel).toBeGreaterThan(0.95);

      // Store baseline for future comparisons
      await performanceBaseline.updateBaseline(testKey, {
        mean: avgPerformance,
        std: calculateStandardDeviation(measurements),
        measurements: measurements.slice(-10) // Keep last 10 measurements
      });
    });

    it('should track performance trends over time', async () => {
      const trendTracker = new PerformanceTrendTracker();
      const testConfig = {
        inputSize: 256,
        outputSize: 128,
        quantizationBits: 1
      };

      // Simulate multiple test runs over time
      const runs = 10;
      for (let run = 0; run < runs; run++) {
        const bitNetLayer = new BitNetLayer({
          ...testConfig,
          useBinaryActivations: true
        });

        const input = new Float32Array(testConfig.inputSize).fill(0.5);
        const startTime = performance.now();
        await bitNetLayer.forward(input);
        const endTime = performance.now();

        const measurement = endTime - startTime;
        trendTracker.addMeasurement('forward_pass_trend', measurement, Date.now());
      }

      const trend = trendTracker.analyzeTrend('forward_pass_trend');

      // Performance should be stable or improving
      expect(trend.slope).toBeLessThanOrEqual(0.1); // No significant degradation
      expect(trend.rSquared).toBeGreaterThan(0.1); // Some correlation with time
    });
  });

  describe('Hardware-Specific Performance', () => {
    it('should optimize for available CPU features', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 1,
        useBinaryActivations: true,
        optimizeForHardware: true
      });

      const input = new Float32Array(512).fill(0.5);
      const measurements: number[] = [];

      // Test with hardware optimization
      for (let i = 0; i < 50; i++) {
        const startTime = performance.now();
        await bitNetLayer.forward(input);
        const endTime = performance.now();
        measurements.push(endTime - startTime);
      }

      const optimizedPerformance = measurements.reduce((a, b) => a + b) / measurements.length;

      // Create layer without optimization
      const unoptimizedLayer = new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 1,
        useBinaryActivations: true,
        optimizeForHardware: false
      });

      const unoptimizedMeasurements: number[] = [];

      for (let i = 0; i < 50; i++) {
        const startTime = performance.now();
        await unoptimizedLayer.forward(input);
        const endTime = performance.now();
        unoptimizedMeasurements.push(endTime - startTime);
      }

      const unoptimizedPerformance = unoptimizedMeasurements.reduce((a, b) => a + b) / unoptimizedMeasurements.length;

      // Optimized version should be faster (or at least not slower)
      expect(optimizedPerformance).toBeLessThanOrEqual(unoptimizedPerformance * 1.1);
    });
  });
});

// Helper functions and classes
function calculateStandardDeviation(values: number[]): number {
  const mean = values.reduce((a, b) => a + b) / values.length;
  const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
  const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b) / squaredDiffs.length;
  return Math.sqrt(avgSquaredDiff);
}

class PerformanceTrendTracker {
  private trends: Map<string, Array<{ value: number; timestamp: number }>> = new Map();

  addMeasurement(key: string, value: number, timestamp: number) {
    if (!this.trends.has(key)) {
      this.trends.set(key, []);
    }
    this.trends.get(key)!.push({ value, timestamp });
  }

  analyzeTrend(key: string) {
    const data = this.trends.get(key) || [];
    if (data.length < 2) {
      return { slope: 0, rSquared: 0 };
    }

    // Simple linear regression
    const n = data.length;
    const sumX = data.reduce((sum, point) => sum + point.timestamp, 0);
    const sumY = data.reduce((sum, point) => sum + point.value, 0);
    const sumXY = data.reduce((sum, point) => sum + point.timestamp * point.value, 0);
    const sumXX = data.reduce((sum, point) => sum + point.timestamp * point.timestamp, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);

    // Calculate R-squared
    const meanY = sumY / n;
    const totalSumSquares = data.reduce((sum, point) => sum + Math.pow(point.value - meanY, 2), 0);
    const residualSumSquares = data.reduce((sum, point) => {
      const predicted = slope * point.timestamp + (sumY - slope * sumX) / n;
      return sum + Math.pow(point.value - predicted, 2);
    }, 0);

    const rSquared = 1 - (residualSumSquares / totalSumSquares);

    return { slope, rSquared };
  }
}