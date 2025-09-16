/**
 * BitNet Layer Unit Tests
 * Phase 4 - Comprehensive testing for BitNet layer functionality
 */

import { BitNetLayer } from '../../../src/phase4_bitnet/BitNetLayer';
import { QuantizationEngine } from '../../../src/phase4_bitnet/QuantizationEngine';
import { TensorOperations } from '../../../src/phase4_bitnet/TensorOperations';

describe('BitNetLayer', () => {
  let bitNetLayer: BitNetLayer;
  let mockQuantization: jest.Mocked<QuantizationEngine>;
  let mockTensorOps: jest.Mocked<TensorOperations>;

  beforeEach(() => {
    mockQuantization = {
      quantizeWeights: jest.fn(),
      dequantizeWeights: jest.fn(),
      quantizeActivations: jest.fn(),
      calibrateQuantization: jest.fn(),
      getQuantizationMetrics: jest.fn()
    } as any;

    mockTensorOps = {
      matmul: jest.fn(),
      conv2d: jest.fn(),
      batchNorm: jest.fn(),
      relu: jest.fn(),
      dropout: jest.fn()
    } as any;

    bitNetLayer = new BitNetLayer({
      inputSize: 512,
      outputSize: 256,
      quantizationBits: 1,
      useBinaryActivations: true
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Weight Quantization', () => {
    it('should quantize weights to 1-bit precision', async () => {
      const weights = new Float32Array([0.5, -0.3, 0.8, -0.1]);
      mockQuantization.quantizeWeights.mockResolvedValue(new Int8Array([1, -1, 1, -1]));

      const quantizedWeights = await bitNetLayer.quantizeWeights(weights);

      expect(mockQuantization.quantizeWeights).toHaveBeenCalledWith(weights, 1);
      expect(quantizedWeights).toEqual(new Int8Array([1, -1, 1, -1]));
    });

    it('should handle weight quantization errors gracefully', async () => {
      const weights = new Float32Array([NaN, Infinity, -Infinity]);
      mockQuantization.quantizeWeights.mockRejectedValue(new Error('Invalid weights'));

      await expect(bitNetLayer.quantizeWeights(weights)).rejects.toThrow('Invalid weights');
    });

    it('should preserve weight distribution statistics', async () => {
      const weights = generateRandomWeights(1000);
      mockQuantization.quantizeWeights.mockImplementation(async (w) => {
        return new Int8Array(w.map(val => val > 0 ? 1 : -1));
      });

      const quantizedWeights = await bitNetLayer.quantizeWeights(weights);
      const originalPositive = weights.filter(w => w > 0).length;
      const quantizedPositive = Array.from(quantizedWeights).filter(w => w > 0).length;

      expect(Math.abs(originalPositive - quantizedPositive)).toBeLessThan(weights.length * 0.1);
    });
  });

  describe('Forward Pass', () => {
    it('should perform forward pass with quantized weights', async () => {
      const input = new Float32Array([1, 2, 3, 4]);
      const expectedOutput = new Float32Array([0.5, 0.3]);

      mockTensorOps.matmul.mockResolvedValue(expectedOutput);
      bitNetLayer.setWeights(new Int8Array([1, -1, 1, -1]));

      const output = await bitNetLayer.forward(input);

      expect(mockTensorOps.matmul).toHaveBeenCalledWith(input, expect.any(Int8Array));
      expect(output).toEqual(expectedOutput);
    });

    it('should apply activation functions correctly', async () => {
      const input = new Float32Array([-1, 0, 1, 2]);
      const preActivation = new Float32Array([-0.5, 0, 0.5, 1]);
      const expectedOutput = new Float32Array([0, 0, 0.5, 1]);

      mockTensorOps.matmul.mockResolvedValue(preActivation);
      mockTensorOps.relu.mockResolvedValue(expectedOutput);

      const output = await bitNetLayer.forward(input);

      expect(mockTensorOps.relu).toHaveBeenCalledWith(preActivation);
      expect(output).toEqual(expectedOutput);
    });

    it('should handle batch processing efficiently', async () => {
      const batchSize = 32;
      const inputSize = 512;
      const batchInput = new Float32Array(batchSize * inputSize);

      for (let i = 0; i < batchInput.length; i++) {
        batchInput[i] = Math.random();
      }

      const startTime = performance.now();
      await bitNetLayer.forwardBatch(batchInput, batchSize);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(100); // Should complete in under 100ms
    });
  });

  describe('Memory Efficiency', () => {
    it('should achieve 8x memory reduction compared to FP32', () => {
      const fp32Size = 512 * 256 * 4; // 4 bytes per float32
      const bitnetSize = bitNetLayer.getMemoryUsage();

      expect(bitnetSize).toBeLessThanOrEqual(fp32Size / 8);
    });

    it('should minimize memory allocations during forward pass', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      const input = new Float32Array(512);

      await bitNetLayer.forward(input);

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      expect(memoryIncrease).toBeLessThan(1024 * 1024); // Less than 1MB increase
    });

    it('should release memory properly after operations', async () => {
      const input = new Float32Array(512);

      for (let i = 0; i < 100; i++) {
        await bitNetLayer.forward(input);
      }

      global.gc?.(); // Force garbage collection if available
      const memoryAfterGC = process.memoryUsage().heapUsed;

      expect(memoryAfterGC).toBeLessThan(50 * 1024 * 1024); // Less than 50MB
    });
  });

  describe('Gradient Computation', () => {
    it('should compute gradients for quantized weights', async () => {
      const outputGradients = new Float32Array([0.1, 0.2, 0.3]);
      const expectedWeightGradients = new Float32Array([0.05, 0.1, 0.15, 0.2]);

      const gradients = await bitNetLayer.computeGradients(outputGradients);

      expect(gradients.weightGradients).toHaveLength(expectedWeightGradients.length);
      expect(gradients.inputGradients).toBeDefined();
    });

    it('should handle straight-through estimator for quantization', async () => {
      const outputGradients = new Float32Array([1, 0, -1]);

      const gradients = await bitNetLayer.computeGradients(outputGradients);

      // Verify that gradients are passed through despite quantization
      expect(gradients.weightGradients.every(g => !isNaN(g))).toBe(true);
    });
  });

  describe('Configuration Validation', () => {
    it('should validate input and output dimensions', () => {
      expect(() => new BitNetLayer({
        inputSize: 0,
        outputSize: 256,
        quantizationBits: 1,
        useBinaryActivations: true
      })).toThrow('Input size must be positive');

      expect(() => new BitNetLayer({
        inputSize: 512,
        outputSize: 0,
        quantizationBits: 1,
        useBinaryActivations: true
      })).toThrow('Output size must be positive');
    });

    it('should validate quantization bits parameter', () => {
      expect(() => new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 0,
        useBinaryActivations: true
      })).toThrow('Quantization bits must be between 1 and 8');

      expect(() => new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 9,
        useBinaryActivations: true
      })).toThrow('Quantization bits must be between 1 and 8');
    });
  });

  describe('Serialization and Deserialization', () => {
    it('should serialize layer state correctly', () => {
      const weights = new Int8Array([1, -1, 1, -1]);
      bitNetLayer.setWeights(weights);

      const serialized = bitNetLayer.serialize();

      expect(serialized.config.inputSize).toBe(512);
      expect(serialized.config.outputSize).toBe(256);
      expect(serialized.weights).toEqual(Array.from(weights));
    });

    it('should deserialize layer state correctly', () => {
      const serializedData = {
        config: {
          inputSize: 512,
          outputSize: 256,
          quantizationBits: 1,
          useBinaryActivations: true
        },
        weights: [1, -1, 1, -1]
      };

      const deserializedLayer = BitNetLayer.deserialize(serializedData);

      expect(deserializedLayer.getConfig().inputSize).toBe(512);
      expect(deserializedLayer.getConfig().outputSize).toBe(256);
    });
  });
});

function generateRandomWeights(size: number): Float32Array {
  const weights = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    weights[i] = (Math.random() - 0.5) * 2; // Random values between -1 and 1
  }
  return weights;
}