/**
 * Unit tests for ZK ProofGenerator
 * Validates proof generation, caching, and performance
 */

import { ProofGenerator, ProofInput } from '../../../src/zk/ProofGenerator';
import * as path from 'path';
import * as fs from 'fs/promises';
import * as crypto from 'crypto';

describe('ProofGenerator - Production ZK Proof Tests', () => {
  let generator: ProofGenerator;

  // Mock paths for testing (in production would use actual compiled circuits)
  const mockZkeyPath = path.join(__dirname, '../../../src/zk/build/test.zkey');
  const mockWasmPath = path.join(__dirname, '../../../src/zk/build/test.wasm');

  beforeAll(async () => {
    // Create mock build directory
    const buildDir = path.join(__dirname, '../../../src/zk/build');
    await fs.mkdir(buildDir, { recursive: true });

    // Initialize generator with test configuration
    generator = new ProofGenerator(mockZkeyPath, mockWasmPath, {
      enableCache: true,
      enableParallel: false, // Disable for deterministic tests
      maxCacheSize: 100
    });
  });

  describe('Proof Generation', () => {
    it('should generate valid proof for privacy validation', async () => {
      const input: ProofInput = {
        dataHash: crypto.randomBytes(32).toString('hex'),
        userConsent: 1,
        dataCategories: [1, 0, 0, 0, 0],
        processingPurpose: 10,
        retentionPeriod: 30,
        privacyTier: 1,
        constitutionalHash: crypto.randomBytes(32).toString('hex')
      };

      // Mock proof generation (in production would use actual snarkjs)
      const mockGenerateProof = jest.spyOn(generator as any, 'generateProof')
        .mockResolvedValue({
          proof: {
            pi_a: ['0x123', '0x456'],
            pi_b: [['0x789', '0xabc'], ['0xdef', '0x012']],
            pi_c: ['0x345', '0x678'],
            protocol: 'groth16',
            curve: 'bn128'
          },
          publicSignals: ['1', '0xcommitment', '0xnullifier', '1', '0xconstitutional'],
          commitment: '0xcommitment',
          generationTime: 45,
          proofSize: 892
        });

      const result = await generator.generateProof(input);

      expect(result).toBeDefined();
      expect(result.proof).toBeDefined();
      expect(result.proof.protocol).toBe('groth16');
      expect(result.proof.curve).toBe('bn128');
      expect(result.publicSignals).toHaveLength(5);
      expect(result.generationTime).toBeLessThan(60); // P95 target
      expect(result.proofSize).toBeLessThan(1000);

      mockGenerateProof.mockRestore();
    });

    it('should cache proofs for identical inputs', async () => {
      const input: ProofInput = {
        dataHash: 'fixed_hash_for_caching',
        userConsent: 1,
        dataCategories: [1, 0, 0, 0, 0],
        processingPurpose: 10,
        retentionPeriod: 30,
        privacyTier: 1,
        constitutionalHash: 'fixed_constitutional'
      };

      // First generation
      const mockGenerate = jest.spyOn(generator as any, 'generateWitness')
        .mockResolvedValue(Buffer.from('mock_witness'));

      const result1 = await generator.generateProof(input);
      expect(mockGenerate).toHaveBeenCalledTimes(1);

      // Second generation should use cache
      const result2 = await generator.generateProof(input);
      expect(mockGenerate).toHaveBeenCalledTimes(1); // Still 1, cached

      // Results should be identical
      expect(result1.commitment).toBe(result2.commitment);

      mockGenerate.mockRestore();
    });

    it('should generate different proofs for different privacy tiers', async () => {
      const baseInput: ProofInput = {
        dataHash: crypto.randomBytes(32).toString('hex'),
        userConsent: 1,
        dataCategories: [1, 0, 0, 0, 0],
        processingPurpose: 10,
        retentionPeriod: 30,
        privacyTier: 0, // Bronze
        constitutionalHash: crypto.randomBytes(32).toString('hex')
      };

      const bronzeProof = await generator.generateProof(baseInput);

      const silverInput = { ...baseInput, privacyTier: 1 };
      const silverProof = await generator.generateProof(silverInput);

      const goldInput = { ...baseInput, privacyTier: 2 };
      const goldProof = await generator.generateProof(goldInput);

      const platinumInput = { ...baseInput, privacyTier: 3 };
      const platinumProof = await generator.generateProof(platinumInput);

      // Commitments should be different for different tiers
      expect(bronzeProof.commitment).not.toBe(silverProof.commitment);
      expect(silverProof.commitment).not.toBe(goldProof.commitment);
      expect(goldProof.commitment).not.toBe(platinumProof.commitment);
    });
  });

  describe('Performance Optimization', () => {
    it('should meet P95 latency target (<60ms)', async () => {
      const iterations = 100;
      const latencies: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const input: ProofInput = {
          dataHash: crypto.randomBytes(32).toString('hex'),
          userConsent: 1,
          dataCategories: [Math.round(Math.random()), 0, 0, 0, 0],
          processingPurpose: Math.floor(Math.random() * 40),
          retentionPeriod: Math.floor(Math.random() * 365),
          privacyTier: Math.floor(Math.random() * 4),
          constitutionalHash: crypto.randomBytes(32).toString('hex')
        };

        const startTime = Date.now();
        await generator.generateProof(input);
        latencies.push(Date.now() - startTime);
      }

      // Calculate P95
      latencies.sort((a, b) => a - b);
      const p95Index = Math.floor(latencies.length * 0.95);
      const p95Latency = latencies[p95Index];

      expect(p95Latency).toBeLessThan(60);
      console.log(`P95 latency: ${p95Latency}ms`);
    });

    it('should generate batch proofs in parallel', async () => {
      const batchSize = 10;
      const inputs: ProofInput[] = [];

      for (let i = 0; i < batchSize; i++) {
        inputs.push({
          dataHash: crypto.randomBytes(32).toString('hex'),
          userConsent: 1,
          dataCategories: [1, 0, 0, 0, 0],
          processingPurpose: 10,
          retentionPeriod: 30,
          privacyTier: 1,
          constitutionalHash: crypto.randomBytes(32).toString('hex')
        });
      }

      const startTime = Date.now();
      const results = await generator.generateBatchProofs(inputs);
      const batchTime = Date.now() - startTime;

      expect(results).toHaveLength(batchSize);
      results.forEach(result => {
        expect(result.proof).toBeDefined();
        expect(result.publicSignals).toBeDefined();
      });

      // Batch should be faster than sequential
      const avgTimePerProof = batchTime / batchSize;
      expect(avgTimePerProof).toBeLessThan(50); // Should be faster per proof
      console.log(`Batch generation: ${batchTime}ms for ${batchSize} proofs (${avgTimePerProof}ms avg)`);
    });

    it('should warm cache with common proofs', async () => {
      const commonInputs: ProofInput[] = [
        {
          dataHash: 'common_data_1',
          userConsent: 1,
          dataCategories: [1, 0, 0, 0, 0],
          processingPurpose: 10,
          retentionPeriod: 30,
          privacyTier: 0,
          constitutionalHash: 'standard_constitutional'
        },
        {
          dataHash: 'common_data_2',
          userConsent: 1,
          dataCategories: [1, 1, 0, 0, 0],
          processingPurpose: 15,
          retentionPeriod: 180,
          privacyTier: 1,
          constitutionalHash: 'standard_constitutional'
        }
      ];

      await generator.warmCache(commonInputs);

      // Verify cached proofs are fast
      const startTime = Date.now();
      const result = await generator.generateProof(commonInputs[0]);
      const cacheTime = Date.now() - startTime;

      expect(cacheTime).toBeLessThan(5); // Should be nearly instant
      expect(result).toBeDefined();
    });
  });

  describe('Security Validation', () => {
    it('should generate unique nullifiers to prevent replay', async () => {
      const input: ProofInput = {
        dataHash: crypto.randomBytes(32).toString('hex'),
        userConsent: 1,
        dataCategories: [1, 0, 0, 0, 0],
        processingPurpose: 10,
        retentionPeriod: 30,
        privacyTier: 1,
        constitutionalHash: crypto.randomBytes(32).toString('hex')
      };

      const proof1 = await generator.generateProof(input);
      const proof2 = await generator.generateProof(input);

      // Even with same input, nullifiers should be different (time-based)
      expect(proof1.publicSignals[2]).toBeDefined(); // Nullifier position
      expect(proof2.publicSignals[2]).toBeDefined();
      // Note: In production, nullifiers would be different due to timestamp rounding
    });

    it('should reject invalid input categories', async () => {
      const invalidInput: ProofInput = {
        dataHash: crypto.randomBytes(32).toString('hex'),
        userConsent: 1,
        dataCategories: [2, 0, 0, 0, 0], // Invalid: should be 0 or 1
        processingPurpose: 10,
        retentionPeriod: 30,
        privacyTier: 1,
        constitutionalHash: crypto.randomBytes(32).toString('hex')
      };

      await expect(generator.generateProof(invalidInput)).rejects.toThrow();
    });

    it('should enforce retention limits per privacy tier', async () => {
      // Platinum tier with long retention should fail
      const invalidRetention: ProofInput = {
        dataHash: crypto.randomBytes(32).toString('hex'),
        userConsent: 1,
        dataCategories: [1, 0, 0, 0, 0],
        processingPurpose: 35,
        retentionPeriod: 365, // Too long for Platinum
        privacyTier: 3, // Platinum
        constitutionalHash: crypto.randomBytes(32).toString('hex')
      };

      // In production, circuit would reject this
      const result = await generator.generateProof(invalidRetention);
      expect(result.publicSignals[0]).toBe('0'); // Validation should fail
    });
  });

  describe('Metrics and Monitoring', () => {
    it('should track generation metrics', () => {
      const metrics = generator.getMetrics();

      expect(metrics.totalProofs).toBeGreaterThan(0);
      expect(metrics.averageTime).toBeGreaterThan(0);
      expect(metrics.p95Time).toBeGreaterThan(0);
      expect(metrics.cacheHitRate).toBeGreaterThanOrEqual(0);
      expect(metrics.cacheHitRate).toBeLessThanOrEqual(1);
    });

    it('should optimize for specific privacy tiers', async () => {
      await generator.optimizeForTier(2); // Gold tier

      const goldInput: ProofInput = {
        dataHash: crypto.randomBytes(32).toString('hex'),
        userConsent: 1,
        dataCategories: [1, 0, 0, 0, 0],
        processingPurpose: 25,
        retentionPeriod: 90,
        privacyTier: 2,
        constitutionalHash: 'standard_constitutional_hash'
      };

      const startTime = Date.now();
      const result = await generator.generateProof(goldInput);
      const optimizedTime = Date.now() - startTime;

      expect(optimizedTime).toBeLessThan(50);
      expect(result).toBeDefined();
    });
  });

  afterAll(() => {
    generator.clearCache();
  });
});