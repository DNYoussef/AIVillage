/**
 * Proof Verifier for Zero-Knowledge Proofs
 * Provides fast verification using snarkjs groth16
 */

import * as snarkjs from 'snarkjs';
import * as fs from 'fs/promises';
import * as path from 'path';
import { EventEmitter } from 'events';
import * as crypto from 'crypto';

export interface VerificationKey {
  protocol: string;
  curve: string;
  nPublic: number;
  vk_alpha_1: string[];
  vk_beta_2: string[][];
  vk_gamma_2: string[][];
  vk_delta_2: string[][];
  vk_alphabeta_12: string[][][];
  IC: string[][];
}

export interface VerificationResult {
  valid: boolean;
  publicSignals: string[];
  commitment: string;
  verificationTime: number;
  errors: string[];
}

export interface VerificationCache {
  proofHash: string;
  result: VerificationResult;
  timestamp: number;
  verifyCount: number;
}

export class ProofVerifier extends EventEmitter {
  private verificationKey: VerificationKey | null = null;
  private verificationCache: Map<string, VerificationCache> = new Map();
  private nullifierStore: Set<string> = new Set();
  private readonly CACHE_TTL = 600000; // 10 minutes
  private readonly MAX_CACHE_SIZE = 500;
  private verificationMetrics: {
    totalVerifications: number;
    validProofs: number;
    invalidProofs: number;
    averageTime: number;
    p95Time: number;
    cacheHits: number;
  } = {
    totalVerifications: 0,
    validProofs: 0,
    invalidProofs: 0,
    averageTime: 0,
    p95Time: 0,
    cacheHits: 0
  };

  constructor(
    private vkeyPath: string,
    private options: {
      enableCache?: boolean;
      checkNullifiers?: boolean;
      strictMode?: boolean;
      maxCacheSize?: number;
    } = {}
  ) {
    super();
    this.options = {
      enableCache: true,
      checkNullifiers: true,
      strictMode: true,
      maxCacheSize: this.MAX_CACHE_SIZE,
      ...options
    };

    this.loadVerificationKey();
  }

  /**
   * Load verification key from file
   */
  private async loadVerificationKey(): Promise<void> {
    try {
      const vkeyData = await fs.readFile(this.vkeyPath, 'utf-8');
      this.verificationKey = JSON.parse(vkeyData);
      console.log('Verification key loaded successfully');
    } catch (error) {
      console.error(`Failed to load verification key: ${error}`);
      throw new Error(`Cannot load verification key: ${error.message}`);
    }
  }

  /**
   * Verify a ZK proof
   */
  async verifyProof(
    proof: any,
    publicSignals: string[]
  ): Promise<VerificationResult> {
    const startTime = Date.now();

    // Ensure verification key is loaded
    if (!this.verificationKey) {
      await this.loadVerificationKey();
    }

    // Check cache first
    const cacheKey = this.getCacheKey(proof, publicSignals);
    if (this.options.enableCache) {
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        this.verificationMetrics.cacheHits++;
        this.emit('cacheHit', { key: cacheKey, result: cached });
        return cached;
      }
    }

    const errors: string[] = [];

    try {
      // 1. Check nullifier to prevent replay attacks
      if (this.options.checkNullifiers && publicSignals.length > 2) {
        const nullifier = publicSignals[2]; // Nullifier is third public signal
        if (this.nullifierStore.has(nullifier)) {
          errors.push('Nullifier already used - possible replay attack');
          const result = this.createResult(false, publicSignals, errors, startTime);
          this.updateMetrics(result);
          return result;
        }
      }

      // 2. Validate proof format
      const formatValidation = this.validateProofFormat(proof);
      if (!formatValidation.valid) {
        errors.push(...formatValidation.errors);
        const result = this.createResult(false, publicSignals, errors, startTime);
        this.updateMetrics(result);
        return result;
      }

      // 3. Validate public signals
      const signalValidation = this.validatePublicSignals(publicSignals);
      if (!signalValidation.valid) {
        errors.push(...signalValidation.errors);
        const result = this.createResult(false, publicSignals, errors, startTime);
        this.updateMetrics(result);
        return result;
      }

      // 4. Perform groth16 verification
      const isValid = await snarkjs.groth16.verify(
        this.verificationKey,
        publicSignals,
        proof
      );

      // 5. Additional validation in strict mode
      if (this.options.strictMode && isValid) {
        const strictValidation = await this.performStrictValidation(proof, publicSignals);
        if (!strictValidation.valid) {
          errors.push(...strictValidation.errors);
          const result = this.createResult(false, publicSignals, errors, startTime);
          this.updateMetrics(result);
          return result;
        }
      }

      // 6. Store nullifier if valid
      if (isValid && this.options.checkNullifiers && publicSignals.length > 2) {
        this.nullifierStore.add(publicSignals[2]);
      }

      const result = this.createResult(isValid, publicSignals, errors, startTime);

      // Cache the result
      if (this.options.enableCache) {
        this.addToCache(cacheKey, result);
      }

      // Update metrics
      this.updateMetrics(result);

      // Emit verification event
      this.emit('verified', {
        valid: isValid,
        time: result.verificationTime
      });

      // Check if we meet P95 target (<15ms)
      if (result.verificationTime > 15) {
        console.warn(`Verification exceeded target: ${result.verificationTime}ms > 15ms`);
      }

      return result;

    } catch (error) {
      console.error(`Proof verification failed: ${error}`);
      errors.push(`Verification error: ${error.message}`);
      const result = this.createResult(false, publicSignals, errors, startTime);
      this.updateMetrics(result);
      this.emit('error', error);
      return result;
    }
  }

  /**
   * Verify multiple proofs in batch
   */
  async verifyBatch(
    proofs: any[],
    publicSignalsBatch: string[][]
  ): Promise<VerificationResult[]> {
    const batchStartTime = Date.now();
    const results: VerificationResult[] = [];

    // Process in parallel for better performance
    const promises = proofs.map((proof, index) =>
      this.verifyProof(proof, publicSignalsBatch[index])
    );

    const batchResults = await Promise.all(promises);

    console.log(`Batch of ${proofs.length} proofs verified in ${Date.now() - batchStartTime}ms`);

    return batchResults;
  }

  /**
   * Validate proof format
   */
  private validateProofFormat(proof: any): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check required fields
    if (!proof.pi_a || !Array.isArray(proof.pi_a) || proof.pi_a.length !== 2) {
      errors.push('Invalid pi_a format');
    }

    if (!proof.pi_b || !Array.isArray(proof.pi_b) || proof.pi_b.length !== 2) {
      errors.push('Invalid pi_b format');
    }

    if (!proof.pi_c || !Array.isArray(proof.pi_c) || proof.pi_c.length !== 2) {
      errors.push('Invalid pi_c format');
    }

    if (!proof.protocol || proof.protocol !== 'groth16') {
      errors.push('Invalid protocol - expected groth16');
    }

    if (!proof.curve || proof.curve !== 'bn128') {
      errors.push('Invalid curve - expected bn128');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Validate public signals
   */
  private validatePublicSignals(publicSignals: string[]): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check signal count matches verification key
    if (this.verificationKey?.nPublic && publicSignals.length !== this.verificationKey.nPublic) {
      errors.push(`Signal count mismatch: expected ${this.verificationKey.nPublic}, got ${publicSignals.length}`);
    }

    // Validate each signal is a valid field element
    for (let i = 0; i < publicSignals.length; i++) {
      try {
        const signal = BigInt(publicSignals[i]);

        // Check if within field range
        const fieldModulus = BigInt('21888242871839275222246405745257275088548364400416034343698204186575808495617');
        if (signal >= fieldModulus || signal < 0) {
          errors.push(`Signal ${i} out of field range`);
        }
      } catch {
        errors.push(`Signal ${i} is not a valid number`);
      }
    }

    // Check expected signals structure
    // [validationResult, commitment, nullifier, privacyTier, constitutionalHash]
    if (publicSignals.length >= 5) {
      // Validation result must be 0 or 1
      const validationResult = BigInt(publicSignals[0]);
      if (validationResult !== 0n && validationResult !== 1n) {
        errors.push('Validation result must be 0 or 1');
      }

      // Privacy tier must be 0-3
      const privacyTier = BigInt(publicSignals[3]);
      if (privacyTier < 0n || privacyTier > 3n) {
        errors.push('Privacy tier must be between 0 and 3');
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Perform additional strict validation
   */
  private async performStrictValidation(
    proof: any,
    publicSignals: string[]
  ): Promise<{ valid: boolean; errors: string[] }> {
    const errors: string[] = [];

    // 1. Check proof elements are valid field elements
    try {
      const fieldModulus = BigInt('21888242871839275222246405745257275088548364400416034343698204186575808495617');

      // Validate pi_a
      for (const element of proof.pi_a) {
        const value = BigInt(element);
        if (value >= fieldModulus || value < 0) {
          errors.push('Proof element pi_a out of field range');
        }
      }

      // Validate pi_c
      for (const element of proof.pi_c) {
        const value = BigInt(element);
        if (value >= fieldModulus || value < 0) {
          errors.push('Proof element pi_c out of field range');
        }
      }

      // Validate pi_b (2x2 array)
      for (const row of proof.pi_b) {
        for (const element of row) {
          const value = BigInt(element);
          if (value >= fieldModulus || value < 0) {
            errors.push('Proof element pi_b out of field range');
          }
        }
      }
    } catch (error) {
      errors.push(`Invalid proof element format: ${error.message}`);
    }

    // 2. Verify commitment integrity
    if (publicSignals.length >= 2) {
      const commitment = publicSignals[1];

      // Check if commitment is valid hex
      const hexRegex = /^0x[0-9a-fA-F]{64}$/;
      if (!commitment.match(hexRegex)) {
        errors.push('Invalid commitment format - must be 64 hex characters');
      }

      // Verify commitment is deterministic (would be checked against circuit)
      try {
        const commitmentBN = BigInt(commitment);
        const fieldModulus = BigInt('21888242871839275222246405745257275088548364400416034343698204186575808495617');
        if (commitmentBN >= fieldModulus) {
          errors.push('Commitment value exceeds field modulus');
        }
      } catch {
        errors.push('Commitment is not a valid field element');
      }
    }

    // 3. Check temporal validity using nullifier timestamp
    if (publicSignals.length >= 3) {
      const nullifier = publicSignals[2];

      // Extract timestamp from nullifier (if encoded)
      // In production, nullifier would include timestamp for freshness
      try {
        // Check nullifier uniqueness was already handled in main verification
        // Here we could decode timestamp if it's embedded in nullifier
        const nullifierBN = BigInt(nullifier);

        // Simple freshness check - nullifier should be recent
        // In production, decode actual timestamp from nullifier structure
        if (nullifierBN === 0n) {
          errors.push('Invalid nullifier - cannot be zero');
        }
      } catch {
        errors.push('Invalid nullifier format');
      }
    }

    // 4. Verify proof structure completeness
    if (!proof.protocol || proof.protocol !== 'groth16') {
      errors.push('Invalid proof protocol - must be groth16');
    }

    if (!proof.curve || proof.curve !== 'bn128') {
      errors.push('Invalid curve - must be bn128');
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }

  /**
   * Create verification result
   */
  private createResult(
    valid: boolean,
    publicSignals: string[],
    errors: string[],
    startTime: number
  ): VerificationResult {
    return {
      valid,
      publicSignals,
      commitment: publicSignals.length >= 2 ? publicSignals[1] : '',
      verificationTime: Date.now() - startTime,
      errors
    };
  }

  /**
   * Get cache key
   */
  private getCacheKey(proof: any, publicSignals: string[]): string {
    const data = JSON.stringify({ proof, publicSignals });
    return crypto.createHash('sha256').update(data).digest('hex');
  }

  /**
   * Get from cache
   */
  private getFromCache(key: string): VerificationResult | null {
    const cached = this.verificationCache.get(key);

    if (!cached) {
      return null;
    }

    // Check TTL
    if (Date.now() - cached.timestamp > this.CACHE_TTL) {
      this.verificationCache.delete(key);
      return null;
    }

    // Update verify count
    cached.verifyCount++;

    return cached.result;
  }

  /**
   * Add to cache
   */
  private addToCache(key: string, result: VerificationResult): void {
    // Enforce cache size limit
    if (this.verificationCache.size >= this.options.maxCacheSize!) {
      // Remove oldest entry
      const oldestKey = this.verificationCache.keys().next().value;
      this.verificationCache.delete(oldestKey);
    }

    this.verificationCache.set(key, {
      proofHash: key,
      result,
      timestamp: Date.now(),
      verifyCount: 0
    });
  }

  /**
   * Update verification metrics
   */
  private updateMetrics(result: VerificationResult): void {
    this.verificationMetrics.totalVerifications++;

    if (result.valid) {
      this.verificationMetrics.validProofs++;
    } else {
      this.verificationMetrics.invalidProofs++;
    }

    // Update average time
    const prevAvg = this.verificationMetrics.averageTime;
    const n = this.verificationMetrics.totalVerifications;
    this.verificationMetrics.averageTime = (prevAvg * (n - 1) + result.verificationTime) / n;

    // Update P95 (simplified)
    if (result.verificationTime > this.verificationMetrics.p95Time) {
      this.verificationMetrics.p95Time = result.verificationTime;
    }
  }

  /**
   * Get verification metrics
   */
  getMetrics(): typeof this.verificationMetrics {
    return {
      ...this.verificationMetrics,
      cacheHitRate: this.verificationMetrics.totalVerifications > 0
        ? this.verificationMetrics.cacheHits / this.verificationMetrics.totalVerifications
        : 0,
      validityRate: this.verificationMetrics.totalVerifications > 0
        ? this.verificationMetrics.validProofs / this.verificationMetrics.totalVerifications
        : 0
    };
  }

  /**
   * Clear nullifier store (for testing)
   */
  clearNullifiers(): void {
    this.nullifierStore.clear();
    console.log('Nullifier store cleared');
  }

  /**
   * Clear verification cache
   */
  clearCache(): void {
    this.verificationCache.clear();
    console.log('Verification cache cleared');
  }

  /**
   * Export Solidity verifier contract
   */
  async exportSolidityVerifier(outputPath: string): Promise<void> {
    try {
      const template = await fs.readFile(
        path.join(__dirname, 'contracts', 'Verifier.sol'),
        'utf-8'
      );

      // Replace placeholders with actual verification key values
      let contract = template;
      contract = contract.replace('{{vk_alpha_1}}', JSON.stringify(this.verificationKey?.vk_alpha_1));
      contract = contract.replace('{{vk_beta_2}}', JSON.stringify(this.verificationKey?.vk_beta_2));
      contract = contract.replace('{{vk_gamma_2}}', JSON.stringify(this.verificationKey?.vk_gamma_2));
      contract = contract.replace('{{vk_delta_2}}', JSON.stringify(this.verificationKey?.vk_delta_2));
      contract = contract.replace('{{IC}}', JSON.stringify(this.verificationKey?.IC));

      await fs.writeFile(outputPath, contract);
      console.log(`Solidity verifier exported to ${outputPath}`);
    } catch (error) {
      console.error(`Failed to export Solidity verifier: ${error}`);
      throw error;
    }
  }
}

export default ProofVerifier;