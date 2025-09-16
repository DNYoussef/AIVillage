/**
 * Phase 5 Training Pipeline Compatibility Tests
 * Phase 4 - Testing BitNet compatibility with future training pipeline
 */

import { BitNetLayer } from '../../../src/phase4_bitnet/BitNetLayer';
import { TrainingPipelineAdapter } from '../../../src/phase4_bitnet/TrainingPipelineAdapter';
import { GradientEstimator } from '../../../src/phase4_bitnet/GradientEstimator';
import { BackpropagationEngine } from '../../../src/phase5_training/BackpropagationEngine';

describe('Phase 5 Training Pipeline Compatibility', () => {
  let trainingAdapter: TrainingPipelineAdapter;
  let gradientEstimator: GradientEstimator;
  let backpropEngine: BackpropagationEngine;

  beforeEach(() => {
    trainingAdapter = new TrainingPipelineAdapter({
      quantizationBits: 1,
      enableStraightThroughEstimator: true,
      gradientClipping: true,
      adaptiveLearningRate: true
    });

    gradientEstimator = new GradientEstimator({
      estimationMethod: 'straight_through',
      enableNoisyGradients: false,
      gradientAccumulation: true
    });

    backpropEngine = new BackpropagationEngine({
      precision: 'mixed',
      enableQuantizedGradients: true,
      optimizerType: 'adamw'
    });
  });

  describe('Gradient Flow Compatibility', () => {
    it('should support gradient flow through quantized layers', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 256,
        outputSize: 128,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true
      });

      // Forward pass
      const input = new Float32Array(256).fill(0.5);
      const output = await bitNetLayer.forward(input);

      // Backward pass simulation
      const outputGradients = new Float32Array(128).fill(0.1);
      const gradients = await bitNetLayer.backward(outputGradients);

      expect(gradients.inputGradients).toBeDefined();
      expect(gradients.weightGradients).toBeDefined();
      expect(gradients.inputGradients).toHaveLength(256);
      expect(gradients.weightGradients).toHaveLength(256 * 128);

      // Verify gradients are not zero (straight-through estimator)
      const nonZeroInputGrads = Array.from(gradients.inputGradients).filter(g => Math.abs(g) > 1e-6);
      const nonZeroWeightGrads = Array.from(gradients.weightGradients).filter(g => Math.abs(g) > 1e-6);

      expect(nonZeroInputGrads.length).toBeGreaterThan(0);
      expect(nonZeroWeightGrads.length).toBeGreaterThan(0);
    });

    it('should implement straight-through estimator correctly', async () => {
      const weights = new Float32Array([0.7, -0.3, 0.1, -0.8]);
      const outputGradients = new Float32Array([0.5, -0.2]);

      const estimatedGradients = await gradientEstimator.estimateWeightGradients(
        weights,
        outputGradients,
        'straight_through'
      );

      expect(estimatedGradients).toHaveLength(weights.length);

      // In straight-through estimator, gradients should flow through unchanged
      // for the quantization operation
      const hasNonZeroGradients = Array.from(estimatedGradients).some(g => Math.abs(g) > 1e-6);
      expect(hasNonZeroGradients).toBe(true);

      // Gradients should be reasonable in magnitude
      const maxGradient = Math.max(...Array.from(estimatedGradients).map(Math.abs));
      expect(maxGradient).toBeLessThan(10); // Should not explode
    });

    it('should handle gradient clipping in quantized networks', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 128,
        outputSize: 64,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true,
        gradientClipping: { enabled: true, maxNorm: 1.0 }
      });

      // Create large gradients that should be clipped
      const largeOutputGradients = new Float32Array(64).fill(10.0);

      const gradients = await bitNetLayer.backward(largeOutputGradients);

      // Check that gradients were clipped
      const gradientNorm = Math.sqrt(
        Array.from(gradients.weightGradients)
          .reduce((sum, g) => sum + g * g, 0)
      );

      expect(gradientNorm).toBeLessThanOrEqual(1.1); // Allow small tolerance
    });
  });

  describe('Training Loop Integration', () => {
    it('should integrate with Phase 5 training loop', async () => {
      const networkLayers = [
        new BitNetLayer({ inputSize: 128, outputSize: 64, quantizationBits: 1, useBinaryActivations: true }),
        new BitNetLayer({ inputSize: 64, outputSize: 32, quantizationBits: 1, useBinaryActivations: true }),
        new BitNetLayer({ inputSize: 32, outputSize: 16, quantizationBits: 1, useBinaryActivations: true })
      ];

      // Simulate training data
      const batchSize = 8;
      const trainingBatch = Array.from({ length: batchSize }, () => ({
        input: new Float32Array(128).fill(Math.random()),
        target: new Float32Array(16).fill(Math.random())
      }));

      // Training step simulation
      for (const sample of trainingBatch) {
        // Forward pass through all layers
        let activation = sample.input;
        const activations: Float32Array[] = [activation];

        for (const layer of networkLayers) {
          activation = await layer.forward(activation);
          activations.push(activation);
        }

        // Backward pass
        let gradients = new Float32Array(activation.length);

        // Calculate loss gradients (simplified MSE)
        for (let i = 0; i < activation.length; i++) {
          gradients[i] = 2 * (activation[i] - sample.target[i]);
        }

        // Backpropagate through layers
        for (let i = networkLayers.length - 1; i >= 0; i--) {
          const layerGradients = await networkLayers[i].backward(gradients);
          gradients = layerGradients.inputGradients;

          // Update weights (simplified)
          await networkLayers[i].updateWeights(layerGradients.weightGradients, 0.001);
        }
      }

      // Verify network can still perform inference after training
      const testInput = new Float32Array(128).fill(0.5);
      let testOutput = testInput;

      for (const layer of networkLayers) {
        testOutput = await layer.forward(testOutput);
      }

      expect(testOutput).toBeDefined();
      expect(testOutput).toHaveLength(16);
    });

    it('should support mixed precision training', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 256,
        outputSize: 128,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableMixedPrecision: true
      });

      // Configure mixed precision training
      await trainingAdapter.configureMixedPrecision(bitNetLayer, {
        forwardPrecision: 'int8',
        backwardPrecision: 'fp16',
        gradientPrecision: 'fp32'
      });

      const input = new Float32Array(256).fill(0.5);
      const output = await bitNetLayer.forward(input);

      // Verify forward pass uses quantized precision
      expect(bitNetLayer.getActivationPrecision()).toBe('int8');

      const outputGradients = new Float32Array(128).fill(0.1);
      const gradients = await bitNetLayer.backward(outputGradients);

      // Verify gradients maintain higher precision
      expect(bitNetLayer.getGradientPrecision()).toBe('fp32');
      expect(gradients.weightGradients.constructor.name).toBe('Float32Array');
    });
  });

  describe('Optimizer Integration', () => {
    it('should work with Adam optimizer', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 128,
        outputSize: 64,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true
      });

      const adamOptimizer = {
        learningRate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        m: new Float32Array(128 * 64), // First moment
        v: new Float32Array(128 * 64), // Second moment
        t: 0 // Time step
      };

      // Simulate multiple optimization steps
      for (let step = 0; step < 5; step++) {
        const input = new Float32Array(128).fill(Math.random());
        const output = await bitNetLayer.forward(input);

        const outputGradients = new Float32Array(64).fill(Math.random() - 0.5);
        const gradients = await bitNetLayer.backward(outputGradients);

        // Adam update
        adamOptimizer.t++;
        const lr_t = adamOptimizer.learningRate *
          Math.sqrt(1 - Math.pow(adamOptimizer.beta2, adamOptimizer.t)) /
          (1 - Math.pow(adamOptimizer.beta1, adamOptimizer.t));

        for (let i = 0; i < gradients.weightGradients.length; i++) {
          const grad = gradients.weightGradients[i];

          adamOptimizer.m[i] = adamOptimizer.beta1 * adamOptimizer.m[i] + (1 - adamOptimizer.beta1) * grad;
          adamOptimizer.v[i] = adamOptimizer.beta2 * adamOptimizer.v[i] + (1 - adamOptimizer.beta2) * grad * grad;

          const update = lr_t * adamOptimizer.m[i] / (Math.sqrt(adamOptimizer.v[i]) + adamOptimizer.epsilon);

          // Apply update (would typically update quantized weights)
          expect(Math.abs(update)).toBeLessThan(1.0); // Reasonable update magnitude
        }
      }
    });

    it('should handle learning rate scheduling', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 64,
        outputSize: 32,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true
      });

      const scheduler = {
        initialLR: 0.01,
        decay: 0.95,
        step: 0
      };

      const learningRates: number[] = [];

      // Simulate training with learning rate decay
      for (let epoch = 0; epoch < 10; epoch++) {
        const currentLR = scheduler.initialLR * Math.pow(scheduler.decay, epoch);
        learningRates.push(currentLR);

        const input = new Float32Array(64).fill(Math.random());
        const output = await bitNetLayer.forward(input);

        const outputGradients = new Float32Array(32).fill(Math.random() - 0.5);
        const gradients = await bitNetLayer.backward(outputGradients);

        // Apply updates with current learning rate
        await bitNetLayer.updateWeights(gradients.weightGradients, currentLR);
      }

      // Verify learning rate decay
      expect(learningRates[0]).toBe(0.01);
      expect(learningRates[9]).toBeLessThan(learningRates[0]);
      expect(learningRates[9]).toBeCloseTo(0.01 * Math.pow(0.95, 9), 5);
    });
  });

  describe('Regularization Techniques', () => {
    it('should support dropout during training', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 128,
        outputSize: 64,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableDropout: true,
        dropoutRate: 0.5
      });

      const input = new Float32Array(128).fill(1.0);

      // During training (dropout enabled)
      await bitNetLayer.setTraining(true);
      const trainingOutput1 = await bitNetLayer.forward(input);
      const trainingOutput2 = await bitNetLayer.forward(input);

      // Outputs should be different due to dropout
      const outputDifference = Array.from(trainingOutput1)
        .reduce((sum, val, i) => sum + Math.abs(val - trainingOutput2[i]), 0);
      expect(outputDifference).toBeGreaterThan(0);

      // During inference (dropout disabled)
      await bitNetLayer.setTraining(false);
      const inferenceOutput1 = await bitNetLayer.forward(input);
      const inferenceOutput2 = await bitNetLayer.forward(input);

      // Outputs should be identical during inference
      const inferenceDifference = Array.from(inferenceOutput1)
        .reduce((sum, val, i) => sum + Math.abs(val - inferenceOutput2[i]), 0);
      expect(inferenceDifference).toBe(0);
    });

    it('should implement weight decay for quantized weights', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 64,
        outputSize: 32,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true,
        weightDecay: 0.001
      });

      const initialWeights = await bitNetLayer.getWeights();

      // Simulate training steps with weight decay
      for (let step = 0; step < 10; step++) {
        const input = new Float32Array(64).fill(Math.random());
        const output = await bitNetLayer.forward(input);

        const outputGradients = new Float32Array(32).fill(Math.random() - 0.5);
        const gradients = await bitNetLayer.backward(outputGradients);

        // Apply weight decay
        await bitNetLayer.applyWeightDecay(0.001);
        await bitNetLayer.updateWeights(gradients.weightGradients, 0.01);
      }

      const finalWeights = await bitNetLayer.getWeights();

      // Verify weights have changed (training occurred)
      const weightChange = Array.from(initialWeights)
        .reduce((sum, w, i) => sum + Math.abs(w - finalWeights[i]), 0);
      expect(weightChange).toBeGreaterThan(0);
    });
  });

  describe('Batch Training Capabilities', () => {
    it('should support efficient batch training', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 128,
        outputSize: 64,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true
      });

      const batchSize = 16;
      const batchInput = new Float32Array(batchSize * 128);
      const batchTargets = new Float32Array(batchSize * 64);

      // Fill with random data
      for (let i = 0; i < batchInput.length; i++) {
        batchInput[i] = Math.random() - 0.5;
      }
      for (let i = 0; i < batchTargets.length; i++) {
        batchTargets[i] = Math.random() - 0.5;
      }

      // Batch forward pass
      const batchOutput = await bitNetLayer.forwardBatch(batchInput, batchSize);
      expect(batchOutput).toHaveLength(batchSize * 64);

      // Batch backward pass
      const batchOutputGradients = new Float32Array(batchSize * 64);
      for (let i = 0; i < batchOutput.length; i++) {
        batchOutputGradients[i] = 2 * (batchOutput[i] - batchTargets[i]);
      }

      const batchGradients = await bitNetLayer.backwardBatch(batchOutputGradients, batchSize);

      expect(batchGradients.inputGradients).toHaveLength(batchSize * 128);
      expect(batchGradients.weightGradients).toHaveLength(128 * 64);

      // Verify gradient accumulation across batch
      const avgGradientMagnitude = Array.from(batchGradients.weightGradients)
        .reduce((sum, g) => sum + Math.abs(g), 0) / batchGradients.weightGradients.length;
      expect(avgGradientMagnitude).toBeGreaterThan(0);
    });

    it('should handle variable batch sizes', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 64,
        outputSize: 32,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true
      });

      const batchSizes = [1, 4, 8, 16, 32];

      for (const batchSize of batchSizes) {
        const batchInput = new Float32Array(batchSize * 64);
        for (let i = 0; i < batchInput.length; i++) {
          batchInput[i] = Math.random();
        }

        const batchOutput = await bitNetLayer.forwardBatch(batchInput, batchSize);
        expect(batchOutput).toHaveLength(batchSize * 32);

        const batchGradients = new Float32Array(batchSize * 32).fill(0.1);
        const gradients = await bitNetLayer.backwardBatch(batchGradients, batchSize);

        expect(gradients.inputGradients).toHaveLength(batchSize * 64);
        expect(gradients.weightGradients).toHaveLength(64 * 32);
      }
    });
  });

  describe('Training Stability', () => {
    it('should maintain numerical stability during training', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 256,
        outputSize: 128,
        quantizationBits: 1,
        useBinaryActivations: true,
        enableGradients: true,
        gradientClipping: { enabled: true, maxNorm: 1.0 }
      });

      const losses: number[] = [];

      // Extended training simulation
      for (let step = 0; step < 100; step++) {
        const input = new Float32Array(256);
        for (let i = 0; i < input.length; i++) {
          input[i] = Math.random() - 0.5;
        }

        const target = new Float32Array(128);
        for (let i = 0; i < target.length; i++) {
          target[i] = Math.random() - 0.5;
        }

        const output = await bitNetLayer.forward(input);

        // Calculate MSE loss
        let loss = 0;
        const gradients = new Float32Array(128);
        for (let i = 0; i < output.length; i++) {
          const diff = output[i] - target[i];
          loss += diff * diff;
          gradients[i] = 2 * diff;
        }
        loss /= output.length;
        losses.push(loss);

        const layerGradients = await bitNetLayer.backward(gradients);
        await bitNetLayer.updateWeights(layerGradients.weightGradients, 0.001);

        // Check for numerical instability
        expect(isFinite(loss)).toBe(true);
        expect(loss).toBeGreaterThan(0);

        // Verify gradients are finite
        const hasInvalidGradients = Array.from(layerGradients.weightGradients)
          .some(g => !isFinite(g));
        expect(hasInvalidGradients).toBe(false);
      }

      // Check training progress (loss should generally decrease or stabilize)
      const earlyLoss = losses.slice(0, 20).reduce((a, b) => a + b) / 20;
      const lateLoss = losses.slice(-20).reduce((a, b) => a + b) / 20;

      // Loss should not explode
      expect(lateLoss).toBeLessThan(earlyLoss * 2);
    });
  });
});

// Test utilities
class MockBackpropagationEngine {
  async computeGradients(output: Float32Array, target: Float32Array): Promise<Float32Array> {
    const gradients = new Float32Array(output.length);
    for (let i = 0; i < output.length; i++) {
      gradients[i] = 2 * (output[i] - target[i]);
    }
    return gradients;
  }

  async updateWeights(weights: Float32Array, gradients: Float32Array, learningRate: number): Promise<void> {
    for (let i = 0; i < weights.length; i++) {
      weights[i] -= learningRate * gradients[i];
    }
  }
}