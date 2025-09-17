/**
 * Circuit Compiler for Zero-Knowledge Proofs
 * Integrates snarkjs for circuit compilation and setup
 */

import * as snarkjs from 'snarkjs';
import * as circomlib from 'circomlib';
import * as fs from 'fs/promises';
import * as path from 'path';
import { execSync } from 'child_process';
import * as crypto from 'crypto';

export interface CircuitConfig {
  name: string;
  circuitPath: string;
  outputPath: string;
  powersOfTauPath?: string;
  maxConstraints: number;
  optimizationLevel: 'O0' | 'O1' | 'O2';
}

export interface CompilationResult {
  r1csPath: string;
  wasmPath: string;
  symPath: string;
  constraints: number;
  publicSignals: number;
  privateSignals: number;
  labels: string[];
}

export interface SetupResult {
  vkeyPath: string;
  zkeyPath: string;
  verifierContractPath: string;
  setupTime: number;
}

export class CircuitCompiler {
  private compiledCircuits: Map<string, CompilationResult> = new Map();
  private setupCache: Map<string, SetupResult> = new Map();
  private readonly PTAU_CACHE_DIR = path.join(__dirname, '../../.cache/ptau');

  constructor(private config: Partial<CircuitConfig> = {}) {
    this.config = {
      maxConstraints: 100000,
      optimizationLevel: 'O2',
      ...config
    };
  }

  /**
   * Compile a Circom circuit to R1CS, WASM, and symbols
   */
  async compileCircuit(circuitPath: string, outputDir: string): Promise<CompilationResult> {
    const startTime = Date.now();

    // Check if already compiled
    const cacheKey = `${circuitPath}:${outputDir}`;
    if (this.compiledCircuits.has(cacheKey)) {
      return this.compiledCircuits.get(cacheKey)!;
    }

    // Ensure output directory exists
    await fs.mkdir(outputDir, { recursive: true });

    const circuitName = path.basename(circuitPath, '.circom');
    const r1csPath = path.join(outputDir, `${circuitName}.r1cs`);
    const wasmPath = path.join(outputDir, `${circuitName}_js`);
    const symPath = path.join(outputDir, `${circuitName}.sym`);

    try {
      // Check if circom is installed
      try {
        execSync('circom --version', { stdio: 'pipe' });
      } catch {
        console.warn('Circom compiler not found. Please install circom first.');
        console.log('Install instructions: npm install -g circom');
        throw new Error('Circom compiler not installed');
      }

      // Compile circuit with circom compiler
      const compileCmd = `circom ${circuitPath} --r1cs --wasm --sym -o ${outputDir} --${this.config.optimizationLevel}`;

      console.log(`Compiling circuit: ${circuitName}`);

      try {
        const output = execSync(compileCmd, {
          stdio: 'pipe',
          encoding: 'utf8'
        });

        if (output) {
          console.log('Compilation output:', output);
        }
      } catch (error) {
        console.error('Circuit compilation failed:', error.message);
        if (error.stderr) {
          console.error('Compilation errors:', error.stderr.toString());
        }
        throw error;
      }

      // Read circuit info
      const r1cs = await snarkjs.r1cs.info(r1csPath);

      const result: CompilationResult = {
        r1csPath,
        wasmPath: path.join(wasmPath, `${circuitName}.wasm`),
        symPath,
        constraints: r1cs.nConstraints,
        publicSignals: r1cs.nPubInputs + r1cs.nOutputs,
        privateSignals: r1cs.nPrvInputs,
        labels: r1cs.labels || []
      };

      // Validate constraint count
      if (result.constraints > this.config.maxConstraints!) {
        throw new Error(
          `Circuit has ${result.constraints} constraints, exceeding maximum of ${this.config.maxConstraints}`
        );
      }

      // Cache compilation result
      this.compiledCircuits.set(cacheKey, result);

      console.log(`Circuit compiled in ${Date.now() - startTime}ms`);
      console.log(`Constraints: ${result.constraints}`);
      console.log(`Public signals: ${result.publicSignals}`);
      console.log(`Private signals: ${result.privateSignals}`);

      return result;

    } catch (error) {
      console.error(`Failed to compile circuit: ${error}`);
      throw new Error(`Circuit compilation failed: ${error.message}`);
    }
  }

  /**
   * Perform trusted setup using Powers of Tau
   */
  async performTrustedSetup(
    r1csPath: string,
    outputDir: string,
    entropy?: string
  ): Promise<SetupResult> {
    const startTime = Date.now();

    // Check cache
    const cacheKey = `${r1csPath}:${outputDir}`;
    if (this.setupCache.has(cacheKey)) {
      return this.setupCache.get(cacheKey)!;
    }

    const circuitName = path.basename(r1csPath, '.r1cs');
    const zkeyPath = path.join(outputDir, `${circuitName}.zkey`);
    const vkeyPath = path.join(outputDir, `${circuitName}_verification_key.json`);
    const verifierPath = path.join(outputDir, `${circuitName}_verifier.sol`);

    try {
      // Get appropriate powers of tau file
      const r1cs = await snarkjs.r1cs.info(r1csPath);
      const ptauPath = await this.getPowersOfTau(r1cs.nConstraints);

      console.log(`Starting trusted setup for ${circuitName}`);

      // Phase 1: Powers of Tau contribution (using cached ptau)
      // Phase 2: Circuit-specific setup
      const zkeyNew = path.join(outputDir, `${circuitName}_0000.zkey`);
      await snarkjs.zKey.newZKey(r1csPath, ptauPath, zkeyNew);

      // Generate deterministic entropy for reproducible setup
      // In production, this should use a proper ceremony with multiple contributors
      const ceremonyData = {
        circuit: circuitName,
        timestamp: Math.floor(Date.now() / 1000),
        contributor: 'AIVillage Constitutional System',
        round: 1
      };

      // Use provided entropy or generate secure random entropy
      const contributionEntropy = entropy || crypto.randomBytes(32).toString('hex');

      const zkeyContrib1 = path.join(outputDir, `${circuitName}_0001.zkey`);

      // First contribution
      await snarkjs.zKey.contribute(
        zkeyNew,
        zkeyContrib1,
        'AIVillage Phase 4 - Contribution 1',
        contributionEntropy
      );

      // Second contribution with independent secure entropy
      const secondEntropy = crypto.randomBytes(32).toString('hex');

      const zkeyFinal = path.join(outputDir, `${circuitName}_final.zkey`);

      await snarkjs.zKey.contribute(
        zkeyContrib1,
        zkeyFinal,
        'AIVillage Phase 4 - Contribution 2',
        secondEntropy
      );

      // Apply beacon with secure random value
      const beaconHash = crypto.randomBytes(32).toString('hex');

      await snarkjs.zKey.beacon(
        zkeyFinal,
        zkeyPath,
        'AIVillage Phase 4 Final Beacon',
        beaconHash,
        10  // Number of iterations for beacon
      );

      // Export verification key
      const vKey = await snarkjs.zKey.exportVerificationKey(zkeyFinal);
      await fs.writeFile(vkeyPath, JSON.stringify(vKey, null, 2));

      // Generate Solidity verifier contract
      const verifierCode = await snarkjs.zKey.exportSolidityVerifier(zkeyFinal);
      await fs.writeFile(verifierPath, verifierCode);

      // Rename final zkey
      await fs.rename(zkeyFinal, zkeyPath);

      // Clean up intermediate files
      await fs.unlink(zkeyNew).catch(() => {});
      const zkeyContrib1Path = path.join(outputDir, `${circuitName}_0001.zkey`);
      await fs.unlink(zkeyContrib1Path).catch(() => {});

      const result: SetupResult = {
        vkeyPath,
        zkeyPath,
        verifierContractPath: verifierPath,
        setupTime: Date.now() - startTime
      };

      // Cache setup result
      this.setupCache.set(cacheKey, result);

      console.log(`Trusted setup completed in ${result.setupTime}ms`);
      console.log(`Verification key: ${vkeyPath}`);
      console.log(`Proving key: ${zkeyPath}`);
      console.log(`Verifier contract: ${verifierPath}`);

      return result;

    } catch (error) {
      console.error(`Trusted setup failed: ${error}`);
      throw new Error(`Trusted setup failed: ${error.message}`);
    }
  }

  /**
   * Get or download appropriate Powers of Tau file
   */
  private async getPowersOfTau(constraints: number): Promise<string> {
    // Determine required Powers of Tau size
    let power = 12; // Start with 2^12 = 4096 constraints
    while ((1 << power) < constraints) {
      power++;
    }

    const ptauFilename = `powersOfTau28_hez_final_${power}.ptau`;
    const ptauPath = path.join(this.PTAU_CACHE_DIR, ptauFilename);

    // Check if already exists
    try {
      await fs.access(ptauPath);
      console.log(`Using cached Powers of Tau: ${ptauFilename}`);
      return ptauPath;
    } catch {
      // Download if not exists
      await fs.mkdir(this.PTAU_CACHE_DIR, { recursive: true });

      console.log(`Downloading Powers of Tau: ${ptauFilename}`);
      console.log('This may take a few minutes for large files...');

      const url = `https://hermez.s3-eu-west-1.amazonaws.com/${ptauFilename}`;

      try {
        // Use https module for actual download
        const https = await import('https');
        const fileStream = await fs.open(ptauPath, 'w');
        const writeStream = fileStream.createWriteStream();

        return new Promise<string>((resolve, reject) => {
          https.get(url, (response) => {
            if (response.statusCode !== 200) {
              reject(new Error(`Failed to download: ${response.statusCode}`));
              return;
            }

            const totalSize = parseInt(response.headers['content-length'] || '0', 10);
            let downloadedSize = 0;
            let lastProgress = 0;

            response.on('data', (chunk) => {
              downloadedSize += chunk.length;
              const progress = Math.floor((downloadedSize / totalSize) * 100);
              if (progress >= lastProgress + 10) {
                console.log(`Download progress: ${progress}%`);
                lastProgress = progress;
              }
            });

            response.pipe(writeStream);

            writeStream.on('finish', () => {
              writeStream.close();
              console.log(`Downloaded Powers of Tau to: ${ptauPath}`);
              resolve(ptauPath);
            });

            writeStream.on('error', (err) => {
              fs.unlink(ptauPath).catch(() => {});
              reject(new Error(`Download failed: ${err.message}`));
            });
          }).on('error', (err) => {
            reject(new Error(`Download request failed: ${err.message}`));
          });
        });
      } catch (downloadError) {
        // Fallback to curl if available
        console.log('Falling back to curl for download...');
        try {
          execSync(`curl -L -o ${ptauPath} ${url}`, { stdio: 'inherit' });
          console.log(`Downloaded Powers of Tau to: ${ptauPath}`);
          return ptauPath;
        } catch (curlError) {
          throw new Error(`Failed to download Powers of Tau file: ${downloadError.message}`);
        }
      }
    }
  }

  /**
   * Optimize circuit for performance
   */
  async optimizeCircuit(
    circuitPath: string,
    optimizations: {
      useCustomGates?: boolean;
      parallelWitness?: boolean;
      cacheIntermediates?: boolean;
    } = {}
  ): Promise<string> {
    const optimizedPath = circuitPath.replace('.circom', '_optimized.circom');
    let circuitContent = await fs.readFile(circuitPath, 'utf-8');

    // Apply optimizations
    if (optimizations.useCustomGates) {
      // Replace expensive operations with custom gates
      circuitContent = circuitContent.replace(
        /component hash = Sha256\(\);/g,
        'component hash = Poseidon(2);' // Poseidon is more efficient in circuits
      );
    }

    if (optimizations.parallelWitness) {
      // Add parallel witness generation hints
      circuitContent = `pragma circom 2.1.0;\n/* parallel witness generation */\n${circuitContent}`;
    }

    if (optimizations.cacheIntermediates) {
      // Add caching hints for intermediate values
      circuitContent = circuitContent.replace(
        /signal input/g,
        'signal input /* @cache */'
      );
    }

    await fs.writeFile(optimizedPath, circuitContent);
    return optimizedPath;
  }

  /**
   * Validate circuit constraints
   */
  async validateCircuit(r1csPath: string): Promise<{
    valid: boolean;
    issues: string[];
  }> {
    const issues: string[] = [];

    try {
      const r1cs = await snarkjs.r1cs.info(r1csPath);

      // Check constraint count
      if (r1cs.nConstraints > this.config.maxConstraints!) {
        issues.push(`Too many constraints: ${r1cs.nConstraints} > ${this.config.maxConstraints}`);
      }

      // Check for common issues
      if (r1cs.nConstraints < 10) {
        issues.push('Very few constraints - circuit may be too simple');
      }

      if (r1cs.nPubInputs + r1cs.nOutputs === 0) {
        issues.push('No public inputs or outputs');
      }

      // Validate R1CS format
      const validation = await snarkjs.r1cs.validate(r1csPath);
      if (!validation) {
        issues.push('R1CS validation failed');
      }

      return {
        valid: issues.length === 0,
        issues
      };

    } catch (error) {
      issues.push(`Validation error: ${error.message}`);
      return {
        valid: false,
        issues
      };
    }
  }

  /**
   * Generate circuit statistics for optimization
   */
  async getCircuitStats(r1csPath: string): Promise<{
    constraints: number;
    variables: number;
    publicInputs: number;
    privateInputs: number;
    outputs: number;
    labels: string[];
    estimatedProofTime: number; // ms
    estimatedVerifyTime: number; // ms
  }> {
    const r1cs = await snarkjs.r1cs.info(r1csPath);

    // Estimate proof generation time (rough approximation)
    const estimatedProofTime = Math.round(r1cs.nConstraints * 0.01 + 500); // ~10μs per constraint + overhead

    // Verification is much faster
    const estimatedVerifyTime = Math.round(10 + r1cs.nPubInputs * 0.1); // ~10ms base + 0.1ms per public input

    return {
      constraints: r1cs.nConstraints,
      variables: r1cs.nVars,
      publicInputs: r1cs.nPubInputs,
      privateInputs: r1cs.nPrvInputs,
      outputs: r1cs.nOutputs,
      labels: r1cs.labels || [],
      estimatedProofTime,
      estimatedVerifyTime
    };
  }

  /**
   * Clear compilation cache
   */
  clearCache(): void {
    this.compiledCircuits.clear();
    this.setupCache.clear();
  }
}

export default CircuitCompiler;