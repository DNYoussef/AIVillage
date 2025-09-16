/**
 * Phase 2 EvoMerge Integration Tests
 * Phase 4 - Testing BitNet integration with EvoMerge model loading
 */

import { BitNetLayer } from '../../../src/phase4_bitnet/BitNetLayer';
import { EvoMergeModelLoader } from '../../../src/phase2_evomerge/EvoMergeModelLoader';
import { ModelConverter } from '../../../src/phase4_bitnet/ModelConverter';
import { StateManager } from '../../../src/common/StateManager';

describe('Phase 2 EvoMerge Integration', () => {
  let evoMergeLoader: EvoMergeModelLoader;
  let modelConverter: ModelConverter;
  let stateManager: StateManager;

  beforeEach(() => {
    evoMergeLoader = new EvoMergeModelLoader({
      modelPath: './test-models/evomerge',
      enableCaching: true,
      validateChecksum: true
    });

    modelConverter = new ModelConverter({
      targetPrecision: 1,
      preserveAccuracy: true,
      enableGradientEstimation: true
    });

    stateManager = new StateManager({
      enablePersistence: true,
      compressionLevel: 9
    });
  });

  afterEach(async () => {
    await stateManager.cleanup();
  });

  describe('Model Loading and Conversion', () => {
    it('should load EvoMerge model and convert to BitNet', async () => {
      // Load Phase 2 EvoMerge model
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-v1.0.bin');

      expect(evoMergeModel).toBeDefined();
      expect(evoMergeModel.layers).toHaveLength(12);
      expect(evoMergeModel.metadata.version).toBe('2.0');

      // Convert to BitNet representation
      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel);

      expect(bitNetModel).toBeDefined();
      expect(bitNetModel.layers).toHaveLength(evoMergeModel.layers.length);

      // Verify conversion preserved layer structure
      for (let i = 0; i < bitNetModel.layers.length; i++) {
        const originalLayer = evoMergeModel.layers[i];
        const convertedLayer = bitNetModel.layers[i];

        expect(convertedLayer.inputSize).toBe(originalLayer.inputSize);
        expect(convertedLayer.outputSize).toBe(originalLayer.outputSize);
        expect(convertedLayer.quantizationBits).toBe(1);
      }
    });

    it('should preserve EvoMerge evolutionary parameters', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-evolutionary.bin');

      // Verify evolutionary parameters are present
      expect(evoMergeModel.evolutionaryParams).toBeDefined();
      expect(evoMergeModel.evolutionaryParams.populationSize).toBeGreaterThan(0);
      expect(evoMergeModel.evolutionaryParams.generations).toBeGreaterThan(0);

      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel, {
        preserveEvolutionaryHistory: true
      });

      // Verify evolutionary parameters are preserved
      expect(bitNetModel.metadata.evolutionaryParams).toBeDefined();
      expect(bitNetModel.metadata.evolutionaryParams.populationSize)
        .toBe(evoMergeModel.evolutionaryParams.populationSize);
    });

    it('should handle EvoMerge model validation errors', async () => {
      // Test with corrupted model file
      await expect(evoMergeLoader.loadModel('corrupted-model.bin'))
        .rejects.toThrow('Model validation failed');

      // Test with incompatible version
      await expect(evoMergeLoader.loadModel('old-version-model.bin'))
        .rejects.toThrow('Incompatible model version');

      // Test with missing evolutionary metadata
      await expect(evoMergeLoader.loadModel('no-evolution-metadata.bin'))
        .rejects.toThrow('Missing evolutionary parameters');
    });
  });

  describe('Weight Transfer and Quantization', () => {
    it('should transfer EvoMerge weights to BitNet layers accurately', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-weights.bin');
      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel);

      // Test weight transfer for each layer
      for (let layerIndex = 0; layerIndex < evoMergeModel.layers.length; layerIndex++) {
        const originalWeights = evoMergeModel.layers[layerIndex].weights;
        const bitNetLayer = bitNetModel.layers[layerIndex];

        // Verify weight dimensions match
        expect(bitNetLayer.getWeightDimensions()).toEqual({
          input: originalWeights.shape[0],
          output: originalWeights.shape[1]
        });

        // Test weight quantization accuracy
        const quantizationMetrics = await bitNetLayer.getQuantizationMetrics();
        expect(quantizationMetrics.signPreservation).toBeGreaterThan(0.95);
        expect(quantizationMetrics.magnitudeCorrelation).toBeGreaterThan(0.8);
      }
    });

    it('should maintain weight statistics during quantization', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-stats.bin');

      // Collect original weight statistics
      const originalStats = {
        mean: 0,
        std: 0,
        sparsity: 0,
        distribution: new Map<string, number>()
      };

      for (const layer of evoMergeModel.layers) {
        const weights = layer.weights.data;
        originalStats.mean += weights.reduce((a, b) => a + b, 0) / weights.length;

        const variance = weights.reduce((acc, val) => acc + Math.pow(val - originalStats.mean, 2), 0) / weights.length;
        originalStats.std += Math.sqrt(variance);

        originalStats.sparsity += weights.filter(w => Math.abs(w) < 1e-6).length / weights.length;
      }

      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel, {
        preserveStatistics: true
      });

      // Verify statistics are approximately preserved
      const convertedStats = await bitNetModel.computeWeightStatistics();

      expect(Math.abs(convertedStats.sparsity - originalStats.sparsity)).toBeLessThan(0.1);
      expect(convertedStats.signConsistency).toBeGreaterThan(0.9);
    });
  });

  describe('Forward Pass Compatibility', () => {
    it('should produce compatible outputs with EvoMerge model', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-inference.bin');
      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel);

      // Test input
      const testInput = new Float32Array(512);
      for (let i = 0; i < testInput.length; i++) {
        testInput[i] = Math.sin(i * 0.01);
      }

      // Get outputs from both models
      const evoMergeOutput = await evoMergeModel.forward(testInput);
      const bitNetOutput = await bitNetModel.forward(testInput);

      // Verify output compatibility
      expect(bitNetOutput).toHaveLength(evoMergeOutput.length);

      // Calculate correlation between outputs
      const correlation = calculateCorrelation(
        Array.from(evoMergeOutput),
        Array.from(bitNetOutput)
      );

      expect(correlation).toBeGreaterThan(0.85); // High correlation

      // Verify output magnitude is reasonable
      const outputMagnitudeRatio = calculateMagnitudeRatio(evoMergeOutput, bitNetOutput);
      expect(outputMagnitudeRatio).toBeGreaterThan(0.7);
      expect(outputMagnitudeRatio).toBeLessThan(1.3);
    });

    it('should handle batch processing from EvoMerge', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-batch.bin');
      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel);

      const batchSize = 16;
      const inputSize = 256;
      const batchInput = new Float32Array(batchSize * inputSize);

      // Fill with test data
      for (let i = 0; i < batchInput.length; i++) {
        batchInput[i] = Math.random() - 0.5;
      }

      // Process batch with both models
      const evoMergeBatchOutput = await evoMergeModel.forwardBatch(batchInput, batchSize);
      const bitNetBatchOutput = await bitNetModel.forwardBatch(batchInput, batchSize);

      expect(bitNetBatchOutput).toHaveLength(evoMergeBatchOutput.length);

      // Verify batch consistency
      for (let sampleIndex = 0; sampleIndex < batchSize; sampleIndex++) {
        const sampleStart = sampleIndex * (evoMergeBatchOutput.length / batchSize);
        const sampleEnd = sampleStart + (evoMergeBatchOutput.length / batchSize);

        const evoMergeSample = evoMergeBatchOutput.slice(sampleStart, sampleEnd);
        const bitNetSample = bitNetBatchOutput.slice(sampleStart, sampleEnd);

        const sampleCorrelation = calculateCorrelation(
          Array.from(evoMergeSample),
          Array.from(bitNetSample)
        );

        expect(sampleCorrelation).toBeGreaterThan(0.8);
      }
    });
  });

  describe('State Management Integration', () => {
    it('should save and restore BitNet state with EvoMerge metadata', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-state.bin');
      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel);

      // Save complete state
      const stateId = await stateManager.saveModelState(bitNetModel, {
        includeEvolutionaryHistory: true,
        compressionLevel: 'high'
      });

      expect(stateId).toBeDefined();

      // Restore state
      const restoredModel = await stateManager.restoreModelState(stateId);

      expect(restoredModel).toBeDefined();
      expect(restoredModel.layers).toHaveLength(bitNetModel.layers.length);
      expect(restoredModel.metadata.evolutionaryParams).toEqual(bitNetModel.metadata.evolutionaryParams);

      // Verify functional equivalence
      const testInput = new Float32Array(256).fill(0.5);
      const originalOutput = await bitNetModel.forward(testInput);
      const restoredOutput = await restoredModel.forward(testInput);

      const functionalCorrelation = calculateCorrelation(
        Array.from(originalOutput),
        Array.from(restoredOutput)
      );

      expect(functionalCorrelation).toBeGreaterThan(0.99); // Near-perfect correlation
    });

    it('should handle version migration from EvoMerge to BitNet', async () => {
      // Load old EvoMerge model format
      const oldEvoMergeModel = await evoMergeLoader.loadModel('evomerge-v1.5.bin');

      // Convert and migrate to current BitNet format
      const migratedBitNetModel = await modelConverter.convertAndMigrate(oldEvoMergeModel, {
        targetVersion: '4.0',
        preserveBackwardCompatibility: true
      });

      expect(migratedBitNetModel.metadata.version).toBe('4.0');
      expect(migratedBitNetModel.metadata.sourceVersion).toBe('1.5');
      expect(migratedBitNetModel.migrationLog).toBeDefined();

      // Verify migration preserved functionality
      const testInput = new Float32Array(128).fill(1.0);
      const output = await migratedBitNetModel.forward(testInput);

      expect(output).toBeDefined();
      expect(output.length).toBeGreaterThan(0);
    });
  });

  describe('Performance and Memory Integration', () => {
    it('should achieve memory reduction when converting from EvoMerge', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-large.bin');
      const originalMemoryUsage = evoMergeModel.getMemoryUsage();

      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel);
      const convertedMemoryUsage = bitNetModel.getMemoryUsage();

      const memoryReduction = originalMemoryUsage / convertedMemoryUsage;

      expect(memoryReduction).toBeGreaterThan(6); // At least 6x reduction
      expect(convertedMemoryUsage).toBeLessThan(originalMemoryUsage / 8); // Target 8x reduction

      console.log(`Memory reduction: ${memoryReduction.toFixed(2)}x`);
      console.log(`Original: ${(originalMemoryUsage / 1024 / 1024).toFixed(2)}MB`);
      console.log(`Converted: ${(convertedMemoryUsage / 1024 / 1024).toFixed(2)}MB`);
    });

    it('should maintain inference speed after conversion', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-speed.bin');
      const bitNetModel = await modelConverter.convertToBitNet(evoMergeModel);

      const testInput = new Float32Array(512).fill(0.5);
      const iterations = 100;

      // Measure EvoMerge inference time
      const evoMergeStart = performance.now();
      for (let i = 0; i < iterations; i++) {
        await evoMergeModel.forward(testInput);
      }
      const evoMergeTime = performance.now() - evoMergeStart;

      // Measure BitNet inference time
      const bitNetStart = performance.now();
      for (let i = 0; i < iterations; i++) {
        await bitNetModel.forward(testInput);
      }
      const bitNetTime = performance.now() - bitNetStart;

      const speedRatio = bitNetTime / evoMergeTime;

      // BitNet should be at least as fast (or faster due to quantization)
      expect(speedRatio).toBeLessThan(1.2); // At most 20% slower

      console.log(`Speed ratio (BitNet/EvoMerge): ${speedRatio.toFixed(3)}`);
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should handle conversion errors gracefully', async () => {
      // Test with malformed EvoMerge model
      const malformedModel = {
        layers: [{ weights: null }], // Invalid weights
        metadata: { version: '2.0' }
      };

      await expect(modelConverter.convertToBitNet(malformedModel as any))
        .rejects.toThrow('Invalid layer weights detected');

      // Test with incompatible layer sizes
      const incompatibleModel = {
        layers: [
          { weights: { shape: [100, 50], data: new Float32Array(100) } } // Mismatched dimensions
        ],
        metadata: { version: '2.0' }
      };

      await expect(modelConverter.convertToBitNet(incompatibleModel as any))
        .rejects.toThrow('Weight dimensions mismatch');
    });

    it('should provide detailed conversion reports', async () => {
      const evoMergeModel = await evoMergeLoader.loadModel('test-evomerge-report.bin');

      const conversionResult = await modelConverter.convertToBitNetWithReport(evoMergeModel);

      expect(conversionResult.model).toBeDefined();
      expect(conversionResult.report).toBeDefined();
      expect(conversionResult.report.layersConverted).toBe(evoMergeModel.layers.length);
      expect(conversionResult.report.memoryReduction).toBeGreaterThan(6);
      expect(conversionResult.report.accuracyImpact).toBeLessThan(0.1); // Less than 10% accuracy loss
      expect(conversionResult.report.conversionTime).toBeGreaterThan(0);

      // Verify report details
      expect(conversionResult.report.layerReports).toHaveLength(evoMergeModel.layers.length);
      conversionResult.report.layerReports.forEach(layerReport => {
        expect(layerReport.quantizationMetrics).toBeDefined();
        expect(layerReport.weightStatistics).toBeDefined();
        expect(layerReport.conversionSuccess).toBe(true);
      });
    });
  });
});

// Helper functions
function calculateCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length) {
    throw new Error('Arrays must have the same length');
  }

  const n = x.length;
  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);
  const sumYY = y.reduce((sum, yi) => sum + yi * yi, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

  return denominator === 0 ? 0 : numerator / denominator;
}

function calculateMagnitudeRatio(x: Float32Array, y: Float32Array): number {
  const magnitudeX = Math.sqrt(Array.from(x).reduce((sum, val) => sum + val * val, 0));
  const magnitudeY = Math.sqrt(Array.from(y).reduce((sum, val) => sum + val * val, 0));

  return magnitudeX === 0 ? (magnitudeY === 0 ? 1 : 0) : magnitudeY / magnitudeX;
}