/**
 * Quantization Engine Unit Tests
 * Phase 4 - Testing quantization accuracy and precision
 */

import { QuantizationEngine } from '../../../src/phase4_bitnet/QuantizationEngine';
import { QuantizationStrategy } from '../../../src/phase4_bitnet/types';

describe('QuantizationEngine', () => {
  let quantizationEngine: QuantizationEngine;

  beforeEach(() => {
    quantizationEngine = new QuantizationEngine({
      strategy: QuantizationStrategy.SIGN_BASED,
      calibrationSamples: 1000,
      enableDynamicRange: true
    });
  });

  describe('Weight Quantization', () => {
    it('should quantize weights to target bit precision', async () => {
      const weights = new Float32Array([0.7, -0.3, 0.1, -0.8, 0.0]);

      const quantized = await quantizationEngine.quantizeWeights(weights, 1);

      expect(quantized).toHaveLength(weights.length);
      expect(Array.from(quantized)).toEqual([1, -1, 1, -1, 1]); // Sign-based quantization
    });

    it('should preserve weight magnitude distribution', async () => {
      const weights = generateGaussianWeights(10000, 0, 1);

      const quantized = await quantizationEngine.quantizeWeights(weights, 1);

      const originalPositiveRatio = weights.filter(w => w > 0).length / weights.length;
      const quantizedPositiveRatio = Array.from(quantized).filter(w => w > 0).length / quantized.length;

      expect(Math.abs(originalPositiveRatio - quantizedPositiveRatio)).toBeLessThan(0.1);
    });

    it('should handle different quantization strategies', async () => {
      const weights = new Float32Array([0.7, -0.3, 0.1, -0.8]);

      // Test uniform quantization
      quantizationEngine.setStrategy(QuantizationStrategy.UNIFORM);
      const uniformQuantized = await quantizationEngine.quantizeWeights(weights, 2);

      // Test asymmetric quantization
      quantizationEngine.setStrategy(QuantizationStrategy.ASYMMETRIC);
      const asymmetricQuantized = await quantizationEngine.quantizeWeights(weights, 2);

      expect(uniformQuantized).not.toEqual(asymmetricQuantized);
    });

    it('should compute quantization error metrics', async () => {
      const weights = new Float32Array([0.5, -0.3, 0.8, -0.1]);

      const result = await quantizationEngine.quantizeWithMetrics(weights, 1);

      expect(result.quantized).toBeDefined();
      expect(result.metrics.mse).toBeGreaterThan(0);
      expect(result.metrics.snr).toBeGreaterThan(0);
      expect(result.metrics.quantizationNoise).toBeGreaterThan(0);
    });
  });

  describe('Activation Quantization', () => {
    it('should quantize activations dynamically', async () => {
      const activations = new Float32Array([2.5, -1.2, 0.8, -3.1, 1.5]);

      const quantized = await quantizationEngine.quantizeActivations(activations, 8);

      expect(quantized).toHaveLength(activations.length);
      expect(Math.max(...Array.from(quantized))).toBeLessThanOrEqual(127);
      expect(Math.min(...Array.from(quantized))).toBeGreaterThanOrEqual(-128);
    });

    it('should adapt to activation range', async () => {
      const smallRange = new Float32Array([0.1, 0.2, 0.3, 0.4]);
      const largeRange = new Float32Array([10, 20, 30, 40]);

      const smallQuantized = await quantizationEngine.quantizeActivations(smallRange, 8);
      const largeQuantized = await quantizationEngine.quantizeActivations(largeRange, 8);

      // Both should use full quantization range
      const smallSpan = Math.max(...Array.from(smallQuantized)) - Math.min(...Array.from(smallQuantized));
      const largeSpan = Math.max(...Array.from(largeQuantized)) - Math.min(...Array.from(largeQuantized));

      expect(smallSpan).toBeGreaterThan(100);
      expect(largeSpan).toBeGreaterThan(100);
    });
  });

  describe('Calibration', () => {
    it('should calibrate quantization parameters', async () => {
      const calibrationData = generateCalibrationData(1000, 512);

      await quantizationEngine.calibrate(calibrationData);

      const params = quantizationEngine.getCalibrationParams();
      expect(params.scale).toBeGreaterThan(0);
      expect(params.zeroPoint).toBeDefined();
      expect(params.clippingRange.min).toBeLessThan(params.clippingRange.max);
    });

    it('should improve quantization accuracy after calibration', async () => {
      const testData = new Float32Array([0.5, -0.3, 0.8, -0.1]);
      const calibrationData = generateCalibrationData(100, 4);

      // Before calibration
      const beforeResult = await quantizationEngine.quantizeWithMetrics(testData, 8);

      // After calibration
      await quantizationEngine.calibrate(calibrationData);
      const afterResult = await quantizationEngine.quantizeWithMetrics(testData, 8);

      expect(afterResult.metrics.mse).toBeLessThan(beforeResult.metrics.mse);
    });

    it('should handle insufficient calibration data', async () => {
      const insufficientData = [new Float32Array([1, 2])]; // Too small

      await expect(quantizationEngine.calibrate(insufficientData))
        .rejects.toThrow('Insufficient calibration data');
    });
  });

  describe('Performance Optimization', () => {
    it('should quantize large tensors efficiently', async () => {
      const largeWeights = new Float32Array(1000000); // 1M weights
      for (let i = 0; i < largeWeights.length; i++) {
        largeWeights[i] = Math.random() - 0.5;
      }

      const startTime = performance.now();
      await quantizationEngine.quantizeWeights(largeWeights, 1);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(1000); // Should complete in under 1 second
    });

    it('should use vectorized operations when available', async () => {
      const weights = new Float32Array(10000);
      for (let i = 0; i < weights.length; i++) {
        weights[i] = Math.random() - 0.5;
      }

      // Enable vectorization
      quantizationEngine.enableVectorization(true);
      const vectorizedTime = await measureQuantizationTime(weights);

      // Disable vectorization
      quantizationEngine.enableVectorization(false);
      const scalarTime = await measureQuantizationTime(weights);

      expect(vectorizedTime).toBeLessThan(scalarTime * 0.8); // At least 20% faster
    });
  });

  describe('Dequantization', () => {
    it('should dequantize weights correctly', async () => {
      const originalWeights = new Float32Array([0.5, -0.3, 0.8, -0.1]);

      const quantized = await quantizationEngine.quantizeWeights(originalWeights, 8);
      const dequantized = await quantizationEngine.dequantizeWeights(quantized, 8);

      // Should be close to original values
      for (let i = 0; i < originalWeights.length; i++) {
        expect(Math.abs(dequantized[i] - originalWeights[i])).toBeLessThan(0.1);
      }
    });

    it('should maintain value range after round-trip', async () => {
      const weights = generateUniformWeights(1000, -2, 2);

      const quantized = await quantizationEngine.quantizeWeights(weights, 8);
      const dequantized = await quantizationEngine.dequantizeWeights(quantized, 8);

      const originalRange = Math.max(...weights) - Math.min(...weights);
      const dequantizedRange = Math.max(...dequantized) - Math.min(...dequantized);

      expect(Math.abs(originalRange - dequantizedRange) / originalRange).toBeLessThan(0.1);
    });
  });

  describe('Error Handling', () => {
    it('should handle NaN and Infinity values', async () => {
      const invalidWeights = new Float32Array([1, NaN, Infinity, -Infinity, 0]);

      await expect(quantizationEngine.quantizeWeights(invalidWeights, 8))
        .rejects.toThrow('Invalid weight values detected');
    });

    it('should validate bit precision range', async () => {
      const weights = new Float32Array([0.5, -0.3]);

      await expect(quantizationEngine.quantizeWeights(weights, 0))
        .rejects.toThrow('Bit precision must be between 1 and 32');

      await expect(quantizationEngine.quantizeWeights(weights, 33))
        .rejects.toThrow('Bit precision must be between 1 and 32');
    });

    it('should handle empty weight arrays', async () => {
      const emptyWeights = new Float32Array(0);

      await expect(quantizationEngine.quantizeWeights(emptyWeights, 8))
        .rejects.toThrow('Weight array cannot be empty');
    });
  });

  // Helper functions
  async function measureQuantizationTime(weights: Float32Array): Promise<number> {
    const start = performance.now();
    await quantizationEngine.quantizeWeights(weights, 8);
    return performance.now() - start;
  }
});

function generateGaussianWeights(size: number, mean: number, std: number): Float32Array {
  const weights = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    // Box-Muller transform for Gaussian distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    weights[i] = mean + std * z;
  }
  return weights;
}

function generateUniformWeights(size: number, min: number, max: number): Float32Array {
  const weights = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    weights[i] = min + (max - min) * Math.random();
  }
  return weights;
}

function generateCalibrationData(numSamples: number, sampleSize: number): Float32Array[] {
  const data: Float32Array[] = [];
  for (let i = 0; i < numSamples; i++) {
    data.push(generateGaussianWeights(sampleSize, 0, 1));
  }
  return data;
}