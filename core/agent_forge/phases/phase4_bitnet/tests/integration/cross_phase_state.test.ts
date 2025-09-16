/**
 * Cross-Phase State Management Tests
 * Phase 4 - Testing state consistency across all phases
 */

import { BitNetLayer } from '../../../src/phase4_bitnet/BitNetLayer';
import { StateManager } from '../../../src/common/StateManager';
import { PhaseCoordinator } from '../../../src/common/PhaseCoordinator';
import { ModelRegistry } from '../../../src/common/ModelRegistry';

describe('Cross-Phase State Management', () => {
  let stateManager: StateManager;
  let phaseCoordinator: PhaseCoordinator;
  let modelRegistry: ModelRegistry;

  beforeEach(() => {
    stateManager = new StateManager({
      enablePersistence: true,
      compressionLevel: 'high',
      encryptionEnabled: true,
      backupRetention: 7
    });

    phaseCoordinator = new PhaseCoordinator({
      enablePhaseValidation: true,
      allowPhaseRollback: true,
      trackDependencies: true
    });

    modelRegistry = new ModelRegistry({
      enableVersioning: true,
      trackProvenance: true,
      validateIntegrity: true
    });
  });

  afterEach(async () => {
    await stateManager.cleanup();
    await phaseCoordinator.reset();
    await modelRegistry.cleanup();
  });

  describe('Phase Transition State Management', () => {
    it('should maintain state consistency during phase transitions', async () => {
      // Simulate Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 progression
      const phases = ['phase1_base', 'phase2_evomerge', 'phase3_quietstar', 'phase4_bitnet'];

      let currentState = await stateManager.initializePhase(phases[0], {
        modelArchitecture: 'transformer',
        precision: 'fp32',
        layers: 12
      });

      expect(currentState.phaseId).toBe(phases[0]);
      expect(currentState.metadata.precision).toBe('fp32');

      // Phase 1 -> Phase 2 (EvoMerge)
      const phase2State = await phaseCoordinator.transitionPhase(
        currentState,
        phases[1],
        {
          enableEvolution: true,
          populationSize: 50,
          generations: 10
        }
      );

      expect(phase2State.phaseId).toBe(phases[1]);
      expect(phase2State.parentPhase).toBe(phases[0]);
      expect(phase2State.metadata.evolutionaryParams).toBeDefined();

      // Phase 2 -> Phase 3 (Quiet-STaR)
      const phase3State = await phaseCoordinator.transitionPhase(
        phase2State,
        phases[2],
        {
          enableQuietReasoning: true,
          reasoningTokens: ['<think>', '</think>'],
          maxReasoningLength: 256
        }
      );

      expect(phase3State.phaseId).toBe(phases[2]);
      expect(phase3State.parentPhase).toBe(phases[1]);
      expect(phase3State.metadata.reasoningConfig).toBeDefined();

      // Phase 3 -> Phase 4 (BitNet)
      const phase4State = await phaseCoordinator.transitionPhase(
        phase3State,
        phases[3],
        {
          quantizationBits: 1,
          useBinaryActivations: true,
          preserveReasoning: true
        }
      );

      expect(phase4State.phaseId).toBe(phases[3]);
      expect(phase4State.parentPhase).toBe(phases[2]);
      expect(phase4State.metadata.quantizationConfig).toBeDefined();

      // Verify state lineage
      const stateLineage = await stateManager.getStateLineage(phase4State.stateId);
      expect(stateLineage).toHaveLength(4);
      expect(stateLineage.map(s => s.phaseId)).toEqual(phases);

      // Verify data integrity across phases
      const integrityCheck = await stateManager.validateStateIntegrity(phase4State.stateId);
      expect(integrityCheck.isValid).toBe(true);
      expect(integrityCheck.checksumValid).toBe(true);
    });

    it('should handle phase rollback correctly', async () => {
      // Create progression through phases
      const baseState = await stateManager.initializePhase('phase2_evomerge', {
        modelArchitecture: 'transformer',
        evolutionaryParams: { generations: 5 }
      });

      const quietStarState = await phaseCoordinator.transitionPhase(
        baseState,
        'phase3_quietstar',
        { enableQuietReasoning: true }
      );

      const bitNetState = await phaseCoordinator.transitionPhase(
        quietStarState,
        'phase4_bitnet',
        { quantizationBits: 1 }
      );

      // Rollback to Phase 3
      const rolledBackState = await phaseCoordinator.rollbackToPhase(
        bitNetState.stateId,
        'phase3_quietstar'
      );

      expect(rolledBackState.phaseId).toBe('phase3_quietstar');
      expect(rolledBackState.stateId).toBe(quietStarState.stateId);

      // Verify rollback didn't corrupt data
      const postRollbackValidation = await stateManager.validateStateIntegrity(rolledBackState.stateId);
      expect(postRollbackValidation.isValid).toBe(true);

      // Verify can re-transition to Phase 4
      const newBitNetState = await phaseCoordinator.transitionPhase(
        rolledBackState,
        'phase4_bitnet',
        { quantizationBits: 1, preserveReasoning: true }
      );

      expect(newBitNetState.phaseId).toBe('phase4_bitnet');
      expect(newBitNetState.parentPhase).toBe('phase3_quietstar');
    });
  });

  describe('Model State Persistence', () => {
    it('should persist and restore BitNet model state correctly', async () => {
      const bitNetLayer = new BitNetLayer({
        inputSize: 512,
        outputSize: 256,
        quantizationBits: 1,
        useBinaryActivations: true
      });

      // Train model briefly to establish state
      const trainingData = Array.from({ length: 10 }, () => ({
        input: new Float32Array(512).fill(Math.random()),
        target: new Float32Array(256).fill(Math.random())
      }));

      for (const sample of trainingData) {
        const output = await bitNetLayer.forward(sample.input);
        const gradients = new Float32Array(256);
        for (let i = 0; i < output.length; i++) {
          gradients[i] = 2 * (output[i] - sample.target[i]);
        }
        const layerGradients = await bitNetLayer.backward(gradients);
        await bitNetLayer.updateWeights(layerGradients.weightGradients, 0.001);
      }

      // Save model state
      const modelState = {
        layer: bitNetLayer,
        metadata: {
          phaseId: 'phase4_bitnet',
          trainingSteps: 10,
          timestamp: Date.now()
        }
      };

      const stateId = await stateManager.saveModelState(modelState);
      expect(stateId).toBeDefined();

      // Restore model state
      const restoredState = await stateManager.restoreModelState(stateId);

      expect(restoredState.metadata.phaseId).toBe('phase4_bitnet');
      expect(restoredState.metadata.trainingSteps).toBe(10);

      // Verify functional equivalence
      const testInput = new Float32Array(512).fill(0.5);
      const originalOutput = await bitNetLayer.forward(testInput);
      const restoredOutput = await restoredState.layer.forward(testInput);

      const outputCorrelation = calculateCorrelation(
        Array.from(originalOutput),
        Array.from(restoredOutput)
      );
      expect(outputCorrelation).toBeGreaterThan(0.99);
    });

    it('should handle state migration between versions', async () => {
      // Create old version state
      const oldVersionState = {
        version: '3.0',
        phaseId: 'phase3_quietstar',
        modelData: {
          layers: [{ type: 'dense', weights: new Float32Array(1000) }],
          reasoningConfig: { tokens: ['<think>', '</think>'] }
        }
      };

      const oldStateId = await stateManager.saveModelState(oldVersionState);

      // Migrate to Phase 4 BitNet
      const migratedState = await stateManager.migrateState(
        oldStateId,
        'phase4_bitnet',
        '4.0',
        {
          quantizationBits: 1,
          preserveReasoning: true,
          enableBinaryActivations: true
        }
      );

      expect(migratedState.version).toBe('4.0');
      expect(migratedState.phaseId).toBe('phase4_bitnet');
      expect(migratedState.migrationLog).toBeDefined();
      expect(migratedState.migrationLog.sourceVersion).toBe('3.0');
      expect(migratedState.migrationLog.targetVersion).toBe('4.0');

      // Verify migration preserved reasoning configuration
      expect(migratedState.modelData.reasoningConfig).toBeDefined();
      expect(migratedState.modelData.quantizationConfig).toBeDefined();
    });
  });

  describe('Dependency Management', () => {
    it('should track and validate cross-phase dependencies', async () => {
      // Register Phase 2 model
      const phase2ModelId = await modelRegistry.registerModel({
        phaseId: 'phase2_evomerge',
        version: '2.0',
        modelType: 'transformer_evomerge',
        dependencies: []
      });

      // Register Phase 3 model with Phase 2 dependency
      const phase3ModelId = await modelRegistry.registerModel({
        phaseId: 'phase3_quietstar',
        version: '3.0',
        modelType: 'transformer_quietstar',
        dependencies: [phase2ModelId]
      });

      // Register Phase 4 model with Phase 3 dependency
      const phase4ModelId = await modelRegistry.registerModel({
        phaseId: 'phase4_bitnet',
        version: '4.0',
        modelType: 'transformer_bitnet',
        dependencies: [phase3ModelId]
      });

      // Validate dependency chain
      const dependencyChain = await modelRegistry.getDependencyChain(phase4ModelId);
      expect(dependencyChain).toHaveLength(3);
      expect(dependencyChain[0].phaseId).toBe('phase2_evomerge');
      expect(dependencyChain[1].phaseId).toBe('phase3_quietstar');
      expect(dependencyChain[2].phaseId).toBe('phase4_bitnet');

      // Verify dependency integrity
      const integrityCheck = await modelRegistry.validateDependencyIntegrity(phase4ModelId);
      expect(integrityCheck.isValid).toBe(true);
      expect(integrityCheck.missingDependencies).toHaveLength(0);
    });

    it('should detect and handle circular dependencies', async () => {
      // Attempt to create circular dependency
      const modelA = await modelRegistry.registerModel({
        phaseId: 'phase_a',
        version: '1.0',
        dependencies: []
      });

      const modelB = await modelRegistry.registerModel({
        phaseId: 'phase_b',
        version: '1.0',
        dependencies: [modelA]
      });

      // Try to make A depend on B (creating cycle)
      await expect(
        modelRegistry.addDependency(modelA, modelB)
      ).rejects.toThrow('Circular dependency detected');

      // Verify no circular dependencies exist
      const hasCycles = await modelRegistry.hasCircularDependencies();
      expect(hasCycles).toBe(false);
    });
  });

  describe('Concurrency and Race Conditions', () => {
    it('should handle concurrent state operations safely', async () => {
      const concurrentOperations = Array.from({ length: 10 }, async (_, index) => {
        const state = await stateManager.initializePhase(`test_phase_${index}`, {
          index,
          data: new Float32Array(100).fill(index)
        });

        const stateId = await stateManager.saveModelState(state);
        const restored = await stateManager.restoreModelState(stateId);

        return { index, stateId, restored };
      });

      const results = await Promise.all(concurrentOperations);

      // Verify all operations completed successfully
      expect(results).toHaveLength(10);
      results.forEach((result, index) => {
        expect(result.index).toBe(index);
        expect(result.stateId).toBeDefined();
        expect(result.restored.index).toBe(index);
      });

      // Verify no state corruption occurred
      for (const result of results) {
        const integrityCheck = await stateManager.validateStateIntegrity(result.stateId);
        expect(integrityCheck.isValid).toBe(true);
      }
    });

    it('should prevent race conditions in phase transitions', async () => {
      const baseState = await stateManager.initializePhase('base_phase', {
        data: 'initial'
      });

      // Attempt concurrent phase transitions
      const transitions = [
        phaseCoordinator.transitionPhase(baseState, 'phase_a', { config: 'a' }),
        phaseCoordinator.transitionPhase(baseState, 'phase_b', { config: 'b' }),
        phaseCoordinator.transitionPhase(baseState, 'phase_c', { config: 'c' })
      ];

      const results = await Promise.allSettled(transitions);

      // Only one transition should succeed, others should fail gracefully
      const successful = results.filter(r => r.status === 'fulfilled');
      const failed = results.filter(r => r.status === 'rejected');

      expect(successful).toHaveLength(1);
      expect(failed).toHaveLength(2);

      // Verify no data corruption
      const stateAfterTransition = await stateManager.getCurrentState(baseState.stateId);
      const integrityCheck = await stateManager.validateStateIntegrity(stateAfterTransition.stateId);
      expect(integrityCheck.isValid).toBe(true);
    });
  });

  describe('State Validation and Recovery', () => {
    it('should detect and recover from corrupted state', async () => {
      const validState = await stateManager.initializePhase('test_phase', {
        data: new Float32Array(100).fill(42)
      });

      const stateId = await stateManager.saveModelState(validState);

      // Simulate state corruption
      await stateManager.corruptState(stateId, 'simulate_disk_corruption');

      // Attempt to restore corrupted state
      const restorationResult = await stateManager.restoreModelState(stateId, {
        enableRecovery: true,
        fallbackToBackup: true
      });

      expect(restorationResult.recovered).toBe(true);
      expect(restorationResult.recoveryMethod).toBeDefined();

      // Verify recovered state integrity
      const integrityCheck = await stateManager.validateStateIntegrity(restorationResult.stateId);
      expect(integrityCheck.isValid).toBe(true);
    });

    it('should maintain backup states for critical phases', async () => {
      const criticalPhases = ['phase3_quietstar', 'phase4_bitnet'];

      for (const phase of criticalPhases) {
        const state = await stateManager.initializePhase(phase, {
          critical: true,
          data: new Float32Array(1000).fill(Math.random())
        });

        const stateId = await stateManager.saveModelState(state, {
          enableBackup: true,
          backupFrequency: 'immediate'
        });

        // Verify backup was created
        const backups = await stateManager.getBackups(stateId);
        expect(backups).toHaveLength(1);
        expect(backups[0].timestamp).toBeLessThanOrEqual(Date.now());

        // Verify backup integrity
        const backupIntegrity = await stateManager.validateBackupIntegrity(backups[0].backupId);
        expect(backupIntegrity.isValid).toBe(true);
      }
    });
  });

  describe('Performance and Scalability', () => {
    it('should handle large state objects efficiently', async () => {
      // Create large model state
      const largeModel = {
        phaseId: 'phase4_bitnet',
        layers: Array.from({ length: 100 }, (_, i) => ({
          id: i,
          weights: new Float32Array(10000).fill(Math.random()),
          quantizedWeights: new Int8Array(10000).fill(i % 256)
        })),
        metadata: {
          totalParameters: 100 * 10000,
          memoryUsage: 100 * 10000 * 4
        }
      };

      const startTime = performance.now();
      const stateId = await stateManager.saveModelState(largeModel);
      const saveTime = performance.now() - startTime;

      expect(saveTime).toBeLessThan(5000); // Should complete within 5 seconds

      const restoreStart = performance.now();
      const restoredModel = await stateManager.restoreModelState(stateId);
      const restoreTime = performance.now() - restoreStart;

      expect(restoreTime).toBeLessThan(5000); // Should restore within 5 seconds
      expect(restoredModel.layers).toHaveLength(100);
    });

    it('should optimize memory usage during state operations', async () => {
      const initialMemory = process.memoryUsage().heapUsed;

      // Perform multiple state operations
      const operations = Array.from({ length: 20 }, async (_, i) => {
        const state = await stateManager.initializePhase(`memory_test_${i}`, {
          data: new Float32Array(1000).fill(i)
        });

        const stateId = await stateManager.saveModelState(state);
        await stateManager.restoreModelState(stateId);
        await stateManager.deleteState(stateId);
      });

      await Promise.all(operations);

      // Force garbage collection
      if (global.gc) {
        global.gc();
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      // Memory increase should be minimal after cleanup
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024); // Less than 50MB
    });
  });
});

// Helper functions
function calculateCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length) return 0;

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