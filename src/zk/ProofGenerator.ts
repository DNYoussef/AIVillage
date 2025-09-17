/**
 * Proof Generator for Zero-Knowledge Proofs
 * Uses snarkjs groth16 for efficient proof generation
 */

import * as snarkjs from 'snarkjs';
import * as fs from 'fs/promises';
import * as path from 'path';
import { Worker } from 'worker_threads';
import { EventEmitter } from 'events';
import * as crypto from 'crypto';

export interface ProofInput {
  // Private inputs
  dataHash: string;
  userConsent: number;
  dataCategories: number[];
  processingPurpose: number;
  retentionPeriod: number;

  // Public inputs
  privacyTier: number;
  constitutionalHash: string;
  nullifier?: string;
}

export interface Proof {
  pi_a: string[];
  pi_b: string[][];
  pi_c: string[];
  protocol: string;
  curve: string;
}

export interface ProofResult {
  proof: Proof;
  publicSignals: string[];
  commitment: string;
  generationTime: number;
  proofSize: number;
}

export interface ProofCache {
  inputHash: string;
  proof: ProofResult;
  timestamp: number;
  hitCount: number;
}

export class ProofGenerator extends EventEmitter {
  private proofCache: Map<string, ProofCache> = new Map();
  private workerPool: Worker[] = [];
  private readonly MAX_WORKERS = 4;
  private readonly CACHE_TTL = 3600000; // 1 hour
  private readonly MAX_CACHE_SIZE = 1000;
  private generationMetrics: {
    totalProofs: number;
    averageTime: number;
    p95Time: number;
    cacheHits: number;
  } = {
    totalProofs: 0,
    averageTime: 0,
    p95Time: 0,
    cacheHits: 0
  };

  constructor(
    private zkeyPath: string,
    private wasmPath: string,
    private options: {
      enableCache?: boolean;
      enableParallel?: boolean;
      maxCacheSize?: number;
      workerCount?: number;
    } = {}
  ) {
    super();
    this.options = {
      enableCache: true,
      enableParallel: true,
      maxCacheSize: this.MAX_CACHE_SIZE,
      workerCount: this.MAX_WORKERS,
      ...options
    };

    if (this.options.enableParallel) {
      this.initializeWorkerPool();
    }
  }

  /**
   * Generate a ZK proof for privacy validation
   */
  async generateProof(input: ProofInput): Promise<ProofResult> {
    const startTime = Date.now();

    // Generate nullifier if not provided
    if (!input.nullifier) {
      input.nullifier = this.generateNullifier(input);
    }

    // Check cache first
    const cacheKey = this.getCacheKey(input);
    if (this.options.enableCache) {
      const cached = this.getFromCache(cacheKey);
      if (cached) {
        this.generationMetrics.cacheHits++;
        this.emit('cacheHit', { key: cacheKey, proof: cached });
        return cached;
      }
    }

    try {
      // Prepare witness input
      const witnessInput = this.prepareWitnessInput(input);

      // Generate witness
      const witness = await this.generateWitness(witnessInput);

      // Generate proof using groth16
      const { proof, publicSignals } = await snarkjs.groth16.fullProve(
        witnessInput,
        this.wasmPath,
        this.zkeyPath
      );

      // Extract commitment from public signals
      const commitment = publicSignals[publicSignals.length - 1];

      // Calculate proof size
      const proofSize = JSON.stringify(proof).length;

      const result: ProofResult = {
        proof: this.formatProof(proof),
        publicSignals,
        commitment,
        generationTime: Date.now() - startTime,
        proofSize
      };

      // Update metrics
      this.updateMetrics(result.generationTime);

      // Cache the result
      if (this.options.enableCache) {
        this.addToCache(cacheKey, result);
      }

      // Emit generation event
      this.emit('proofGenerated', {
        input: input.privacyTier,
        time: result.generationTime,
        size: proofSize
      });

      console.log(`Proof generated in ${result.generationTime}ms (size: ${proofSize} bytes)`);

      // Check if we meet P95 target (<60ms)
      if (result.generationTime > 60) {
        console.warn(`Proof generation exceeded target: ${result.generationTime}ms > 60ms`);
      }

      return result;

    } catch (error) {
      console.error(`Proof generation failed: ${error}`);
      this.emit('error', error);
      throw new Error(`Failed to generate proof: ${error.message}`);
    }
  }

  /**
   * Generate multiple proofs in parallel
   */
  async generateBatchProofs(inputs: ProofInput[]): Promise<ProofResult[]> {
    if (!this.options.enableParallel || inputs.length === 1) {
      // Sequential generation
      const results: ProofResult[] = [];
      for (const input of inputs) {
        results.push(await this.generateProof(input));
      }
      return results;
    }

    // Parallel generation using worker pool
    const batchStartTime = Date.now();
    const chunks = this.chunkArray(inputs, this.options.workerCount!);

    const promises = chunks.map(chunk =>
      this.generateProofsInWorker(chunk)
    );

    const results = await Promise.all(promises);
    const flatResults = results.flat();

    console.log(`Batch of ${inputs.length} proofs generated in ${Date.now() - batchStartTime}ms`);

    return flatResults;
  }

  /**
   * Prepare witness input from ProofInput
   */
  private prepareWitnessInput(input: ProofInput): any {
    // Convert string hashes to field elements
    const dataHashBN = BigInt('0x' + crypto.createHash('sha256')
      .update(input.dataHash)
      .digest('hex')) % snarkjs.bn128.r;

    const constitutionalHashBN = BigInt('0x' + crypto.createHash('sha256')
      .update(input.constitutionalHash)
      .digest('hex')) % snarkjs.bn128.r;

    const nullifierBN = BigInt('0x' + crypto.createHash('sha256')
      .update(input.nullifier!)
      .digest('hex')) % snarkjs.bn128.r;

    return {
      // Private inputs
      dataHash: dataHashBN.toString(),
      userConsent: input.userConsent,
      dataCategories: input.dataCategories,
      processingPurpose: input.processingPurpose,
      retentionPeriod: input.retentionPeriod,

      // Public inputs
      privacyTier: input.privacyTier,
      constitutionalHash: constitutionalHashBN.toString(),
      nullifier: nullifierBN.toString()
    };
  }

  /**
   * Generate witness using WASM circuit
   */
  private async generateWitness(input: any): Promise<Buffer> {
    const startTime = Date.now();

    try {
      // Check if WASM file exists
      try {
        await fs.access(this.wasmPath);
      } catch {
        throw new Error(`WASM file not found at ${this.wasmPath}. Run circuit compilation first.`);
      }

      // Load WASM and generate witness
      const wasm = await fs.readFile(this.wasmPath);
      const witnessCalculator = await snarkjs.wtns.calculator(wasm);

      // Calculate witness with actual input
      const wtnsBuffer = await witnessCalculator.calculateWTNSBin(input, 0);

      const witnessTime = Date.now() - startTime;
      if (witnessTime > 20) {
        console.warn(`Witness generation slow: ${witnessTime}ms`);
      }

      return Buffer.from(wtnsBuffer);

    } catch (error) {
      throw new Error(`Witness generation failed: ${error.message}`);
    }
  }

  /**
   * Format proof for output
   */
  private formatProof(proof: any): Proof {
    return {
      pi_a: [proof.pi_a[0], proof.pi_a[1]],
      pi_b: [[proof.pi_b[0][1], proof.pi_b[0][0]], [proof.pi_b[1][1], proof.pi_b[1][0]]],
      pi_c: [proof.pi_c[0], proof.pi_c[1]],
      protocol: 'groth16',
      curve: 'bn128'
    };
  }

  /**
   * Generate deterministic nullifier
   */
  private generateNullifier(input: ProofInput): string {
    const data = JSON.stringify({
      dataHash: input.dataHash,
      purpose: input.processingPurpose,
      timestamp: Math.floor(Date.now() / 60000) // Round to minute for reuse
    });

    return crypto.createHash('sha256').update(data).digest('hex');
  }

  /**
   * Get cache key for input
   */
  private getCacheKey(input: ProofInput): string {
    const normalized = {
      ...input,
      nullifier: input.nullifier || 'default'
    };

    return crypto.createHash('sha256')
      .update(JSON.stringify(normalized))
      .digest('hex');
  }

  /**
   * Get proof from cache
   */
  private getFromCache(key: string): ProofResult | null {
    const cached = this.proofCache.get(key);

    if (!cached) {
      return null;
    }

    // Check TTL
    if (Date.now() - cached.timestamp > this.CACHE_TTL) {
      this.proofCache.delete(key);
      return null;
    }

    // Update hit count
    cached.hitCount++;

    return cached.proof;
  }

  /**
   * Add proof to cache
   */
  private addToCache(key: string, proof: ProofResult): void {
    // Enforce cache size limit
    if (this.proofCache.size >= this.options.maxCacheSize!) {
      // Remove least recently used entry (LRU eviction)
      let oldestKey = '';
      let oldestTime = Date.now();
      let lowestHitCount = Infinity;

      for (const [cacheKey, cacheEntry] of this.proofCache.entries()) {
        // Prefer removing entries with low hit count and old timestamp
        const score = cacheEntry.hitCount + ((Date.now() - cacheEntry.timestamp) / 3600000); // Age in hours
        if (score < (lowestHitCount + ((Date.now() - oldestTime) / 3600000))) {
          oldestKey = cacheKey;
          oldestTime = cacheEntry.timestamp;
          lowestHitCount = cacheEntry.hitCount;
        }
      }

      if (oldestKey) {
        this.proofCache.delete(oldestKey);
      }
    }

    this.proofCache.set(key, {
      inputHash: key,
      proof,
      timestamp: Date.now(),
      hitCount: 0
    });
  }

  /**
   * Initialize worker pool for parallel generation
   */
  private initializeWorkerPool(): void {
    // Create actual worker threads
    for (let i = 0; i < this.options.workerCount!; i++) {
      const workerPath = path.join(__dirname, 'workers', 'proof.worker.js');

      try {
        // Check if worker file exists, if not create it
        const workerCode = `
const { parentPort } = require('worker_threads');
const snarkjs = require('snarkjs');

parentPort.on('message', async (data) => {
  try {
    const { inputs, wasmPath, zkeyPath } = data;
    const results = [];

    for (const input of inputs) {
      const { proof, publicSignals } = await snarkjs.groth16.fullProve(
        input,
        wasmPath,
        zkeyPath
      );
      results.push({ proof, publicSignals });
    }

    parentPort.postMessage({ success: true, results });
  } catch (error) {
    parentPort.postMessage({ success: false, error: error.message });
  }
});`;

        // Ensure worker directory exists
        const workerDir = path.dirname(workerPath);
        await fs.mkdir(workerDir, { recursive: true });
        await fs.writeFile(workerPath, workerCode);

        const worker = new Worker(workerPath);
        worker.on('error', (err) => {
          console.error(`Worker ${i} error:`, err);
          // Remove failed worker from pool
          const index = this.workerPool.indexOf(worker);
          if (index > -1) {
            this.workerPool.splice(index, 1);
          }
        });
        worker.on('exit', (code) => {
          if (code !== 0) {
            console.error(`Worker ${i} stopped with exit code ${code}`);
            // Remove exited worker from pool
            const index = this.workerPool.indexOf(worker);
            if (index > -1) {
              this.workerPool.splice(index, 1);
            }
          }
        });

        this.workerPool.push(worker);
      } catch (error) {
        console.error(`Failed to create worker ${i}:`, error);
      }
    }

    if (this.workerPool.length > 0) {
      console.log(`Initialized ${this.workerPool.length} worker threads for parallel proof generation`);
    } else {
      console.warn('Failed to initialize worker pool, falling back to sequential processing');
      this.options.enableParallel = false;
    }
  }

  /**
   * Generate proofs in worker thread
   */
  private async generateProofsInWorker(inputs: ProofInput[]): Promise<ProofResult[]> {
    if (this.workerPool.length === 0) {
      // Fallback to sequential if no workers available
      const results: ProofResult[] = [];
      for (const input of inputs) {
        results.push(await this.generateProof(input));
      }
      return results;
    }

    // Get available worker (round-robin)
    const workerIndex = Math.floor(Math.random() * this.workerPool.length);
    const worker = this.workerPool[workerIndex];

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Worker timeout - proof generation took too long'));
      }, 30000); // 30 second timeout

      worker.once('message', (result) => {
        clearTimeout(timeout);

        if (result.success) {
          const proofResults: ProofResult[] = result.results.map((r: any) => ({
            proof: this.formatProof(r.proof),
            publicSignals: r.publicSignals,
            commitment: r.publicSignals[r.publicSignals.length - 1],
            generationTime: Date.now() - startTime,
            proofSize: JSON.stringify(r.proof).length
          }));
          resolve(proofResults);
        } else {
          reject(new Error(result.error));
        }
      });

      const startTime = Date.now();

      // Prepare witness inputs
      const witnessInputs = inputs.map(input => this.prepareWitnessInput(input));

      // Send to worker
      worker.postMessage({
        inputs: witnessInputs,
        wasmPath: this.wasmPath,
        zkeyPath: this.zkeyPath
      });
    });
  }

  /**
   * Update generation metrics
   */
  private updateMetrics(generationTime: number): void {
    this.generationMetrics.totalProofs++;

    // Update average time
    const prevAvg = this.generationMetrics.averageTime;
    const n = this.generationMetrics.totalProofs;
    this.generationMetrics.averageTime = (prevAvg * (n - 1) + generationTime) / n;

    // Update P95 (simplified - in production use proper percentile tracking)
    if (generationTime > this.generationMetrics.p95Time) {
      this.generationMetrics.p95Time = generationTime;
    }
  }

  /**
   * Chunk array for parallel processing
   */
  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  /**
   * Pre-compute common proofs for warming cache
   */
  async warmCache(commonInputs: ProofInput[]): Promise<void> {
    console.log(`Warming cache with ${commonInputs.length} common proofs...`);

    const startTime = Date.now();
    const results = await this.generateBatchProofs(commonInputs);

    console.log(`Cache warmed in ${Date.now() - startTime}ms`);
    console.log(`Cached ${results.length} proofs`);
  }

  /**
   * Get generation metrics
   */
  getMetrics(): typeof this.generationMetrics {
    return {
      ...this.generationMetrics,
      cacheHitRate: this.generationMetrics.totalProofs > 0
        ? this.generationMetrics.cacheHits / this.generationMetrics.totalProofs
        : 0
    };
  }

  /**
   * Clear proof cache
   */
  clearCache(): void {
    this.proofCache.clear();
    console.log('Proof cache cleared');
  }

  /**
   * Optimize for specific privacy tier
   */
  async optimizeForTier(tier: number): Promise<void> {
    // Pre-generate common proofs for this tier
    const commonInputs: ProofInput[] = [];

    // Generate common patterns for the tier
    for (let i = 0; i < 10; i++) {
      commonInputs.push({
        dataHash: crypto.randomBytes(32).toString('hex'),
        userConsent: 1,
        dataCategories: [1, 0, 0, 0, 0],
        processingPurpose: tier * 10 + i,
        retentionPeriod: 30 * (4 - tier),
        privacyTier: tier,
        constitutionalHash: 'standard_constitutional_hash'
      });
    }

    await this.warmCache(commonInputs);
  }
}

export default ProofGenerator;