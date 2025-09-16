/**
 * Memory Usage Benchmarking Tests
 * Phase 4 - Validating 8x memory reduction and efficient memory usage
 */

import { BitNetLayer } from '../../../src/phase4_bitnet/BitNetLayer';
import { MemoryProfiler } from '../../../src/phase4_bitnet/utils/MemoryProfiler';
import { ModelComparison } from '../../../src/phase4_bitnet/utils/ModelComparison';

describe('Memory Benchmarks', () => {
  let memoryProfiler: MemoryProfiler;
  let modelComparison: ModelComparison;

  beforeEach(() => {
    memoryProfiler = new MemoryProfiler();
    modelComparison = new ModelComparison();
  });

  afterEach(async () => {
    // Force garbage collection after each test
    if (global.gc) {
      global.gc();
    }
    await new Promise(resolve => setTimeout(resolve, 100));
  });

  describe('BitNet vs FP32 Memory Comparison', () => {
    it('should achieve 8x memory reduction for weights', () => {
      const inputSize = 1024;
      const outputSize = 512;

      // FP32 baseline
      const fp32WeightSize = inputSize * outputSize * 4; // 4 bytes per float32

      // BitNet layer
      const bitNetLayer = new BitNetLayer({
        inputSize,
        outputSize,
        quantizationBits: 1,
        useBinaryActivations: true
      });

      const bitNetWeightSize = bitNetLayer.getWeightMemoryUsage();

      expect(bitNetWeightSize).toBeLessThanOrEqual(fp32WeightSize / 8);
      expect(bitNetWeightSize / fp32WeightSize).toBeLessThan(0.125); // Less than 1/8
    });

    it('should demonstrate memory efficiency across different layer sizes', () => {
      const layerConfigs = [
        { input: 256, output: 128 },
        { input: 512, output: 256 },
        { input: 1024, output: 512 },
        { input: 2048, output: 1024 }
      ];

      const results = layerConfigs.map(config => {
        const fp32Size = config.input * config.output * 4;

        const bitNetLayer = new BitNetLayer({
          inputSize: config.input,
          outputSize: config.output,
          quantizationBits: 1,
          useBinaryActivations: true
        });

        const bitNetSize = bitNetLayer.getWeightMemoryUsage();
        const reductionRatio = fp32Size / bitNetSize;

        return {
          config,
          fp32Size,
          bitNetSize,
          reductionRatio
        };
      });

      // All configurations should achieve at least 8x reduction
      results.forEach(result => {
        expect(result.reductionRatio).toBeGreaterThanOrEqual(8);
      });

      // Log results for analysis
      console.table(results);
    });
  });

  describe('Dynamic Memory Usage', () => {
    it('should maintain low memory footprint during inference', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 1,
        useBinaryActivations: true
      });

      const initialMemory = memoryProfiler.getCurrentMemoryUsage();

      // Run multiple inference passes
      for (let i = 0; i < 100; i++) {
        const input = new Float32Array(512).fill(Math.random());
        await bitNetLayer.forward(input);
      }

      const finalMemory = memoryProfiler.getCurrentMemoryUsage();
      const memoryGrowth = finalMemory.heapUsed - initialMemory.heapUsed;

      // Memory growth should be minimal (< 10MB)
      expect(memoryGrowth).toBeLessThan(10 * 1024 * 1024);
    });

    it('should release temporary memory after batch processing', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 1024,
        outputSize: 512,
        quantizationBits: 1,
        useBinaryActivations: true
      });

      const batchSize = 64;
      const inputSize = 1024;
      const largeBatch = new Float32Array(batchSize * inputSize);

      // Fill with random data
      for (let i = 0; i < largeBatch.length; i++) {
        largeBatch[i] = Math.random();
      }

      const beforeBatch = memoryProfiler.getCurrentMemoryUsage();
      await bitNetLayer.forwardBatch(largeBatch, batchSize);

      // Force garbage collection
      if (global.gc) {
        global.gc();
      }
      await new Promise(resolve => setTimeout(resolve, 100));

      const afterBatch = memoryProfiler.getCurrentMemoryUsage();
      const memoryIncrease = afterBatch.heapUsed - beforeBatch.heapUsed;

      // Memory increase should be minimal after GC
      expect(memoryIncrease).toBeLessThan(5 * 1024 * 1024); // < 5MB
    });
  });

  describe('Memory Profiling Under Load', () => {
    it('should handle concurrent operations without memory leaks', async () => {
      const numLayers = 10;
      const layers = Array.from({ length: numLayers }, () =>
        new BitNetLayer({
          inputSize: 256,
          outputSize: 128,
          quantizationBits: 1,
          useBinaryActivations: true
        })
      );

      const initialMemory = memoryProfiler.getCurrentMemoryUsage();

      // Run concurrent operations
      const promises = layers.map(async (layer, index) => {
        for (let i = 0; i < 50; i++) {
          const input = new Float32Array(256).fill(Math.random());
          await layer.forward(input);
        }
      });

      await Promise.all(promises);

      // Force garbage collection
      if (global.gc) {
        global.gc();
      }
      await new Promise(resolve => setTimeout(resolve, 200));

      const finalMemory = memoryProfiler.getCurrentMemoryUsage();
      const memoryGrowth = finalMemory.heapUsed - initialMemory.heapUsed;

      // Memory growth should be proportional to number of layers, not operations
      const expectedMaxGrowth = numLayers * 1024 * 1024; // 1MB per layer
      expect(memoryGrowth).toBeLessThan(expectedMaxGrowth);
    });

    it('should maintain stable memory usage over extended periods', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 1,
        useBinaryActivations: true
      });

      const memoryReadings: number[] = [];
      const numIterations = 200;

      for (let i = 0; i < numIterations; i++) {
        const input = new Float32Array(512).fill(Math.random());
        await bitNetLayer.forward(input);

        if (i % 20 === 0) {
          // Force GC periodically
          if (global.gc) {
            global.gc();
          }
          memoryReadings.push(memoryProfiler.getCurrentMemoryUsage().heapUsed);
        }
      }

      // Memory usage should stabilize (slope should be near zero)
      const firstHalf = memoryReadings.slice(0, Math.floor(memoryReadings.length / 2));
      const secondHalf = memoryReadings.slice(Math.floor(memoryReadings.length / 2));

      const firstHalfAvg = firstHalf.reduce((a, b) => a + b) / firstHalf.length;
      const secondHalfAvg = secondHalf.reduce((a, b) => a + b) / secondHalf.length;

      const memoryGrowthRate = (secondHalfAvg - firstHalfAvg) / firstHalfAvg;

      // Memory growth rate should be less than 10%
      expect(Math.abs(memoryGrowthRate)).toBeLessThan(0.1);
    });
  });

  describe('Model Size Comparison', () => {
    it('should compare model sizes with equivalent FP32 networks', async () => {
      const networkConfigs = [
        { layers: [512, 256, 128], name: 'Small Network' },
        { layers: [1024, 512, 256, 128], name: 'Medium Network' },
        { layers: [2048, 1024, 512, 256, 128], name: 'Large Network' }
      ];

      for (const config of networkConfigs) {
        const comparison = await modelComparison.compareNetworkSizes(
          config.layers,
          'fp32',
          'bitnet'
        );

        expect(comparison.bitnetSize).toBeLessThan(comparison.fp32Size / 6); // At least 6x reduction
        expect(comparison.compressionRatio).toBeGreaterThan(6);

        console.log(`${config.name} - Compression Ratio: ${comparison.compressionRatio.toFixed(2)}x`);
      }
    });

    it('should validate memory usage against theoretical calculations', () => {
      const layerConfig = {
        inputSize: 1024,
        outputSize: 512,
        quantizationBits: 1
      };

      const bitNetLayer = new BitNetLayer({
        ...layerConfig,
        useBinaryActivations: true
      });

      // Theoretical calculation
      const theoreticalSize = Math.ceil(
        (layerConfig.inputSize * layerConfig.outputSize * layerConfig.quantizationBits) / 8
      );

      const actualSize = bitNetLayer.getWeightMemoryUsage();

      // Actual size should be close to theoretical (within 10% overhead)
      const overhead = (actualSize - theoreticalSize) / theoreticalSize;
      expect(overhead).toBeLessThan(0.1);
    });
  });

  describe('Memory Optimization Strategies', () => {
    it('should optimize memory layout for cache efficiency', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 1024,
        outputSize: 512,
        quantizationBits: 1,
        useBinaryActivations: true,
        optimizeMemoryLayout: true
      });

      const input = new Float32Array(1024).fill(1.0);

      // Measure cache performance through repeated access
      const iterations = 1000;
      const startTime = performance.now();

      for (let i = 0; i < iterations; i++) {
        await bitNetLayer.forward(input);
      }

      const endTime = performance.now();
      const avgTime = (endTime - startTime) / iterations;

      // Should complete each forward pass in reasonable time
      expect(avgTime).toBeLessThan(10); // Less than 10ms per forward pass
    });

    it('should use memory pools for frequent allocations', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 1,
        useBinaryActivations: true,
        useMemoryPool: true
      });

      const numOperations = 100;
      const allocations = memoryProfiler.trackAllocations(async () => {
        for (let i = 0; i < numOperations; i++) {
          const input = new Float32Array(512).fill(Math.random());
          await bitNetLayer.forward(input);
        }
      });

      // Memory pool should reduce allocation count
      expect(allocations.count).toBeLessThan(numOperations / 2);
      expect(allocations.totalSize).toBeLessThan(numOperations * 512 * 4);
    });
  });
});

// Test utilities for memory profiling
class MemoryProfilerMock {
  getCurrentMemoryUsage() {
    return process.memoryUsage();
  }

  trackAllocations(operation: () => Promise<void>) {
    // Mock implementation for tracking allocations
    const initialMemory = this.getCurrentMemoryUsage();

    return operation().then(() => {
      const finalMemory = this.getCurrentMemoryUsage();
      return {
        count: Math.floor(Math.random() * 50), // Mock allocation count
        totalSize: finalMemory.heapUsed - initialMemory.heapUsed
      };
    });
  }
}