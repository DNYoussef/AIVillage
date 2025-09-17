/**
 * BridgeOrchestrator - Central coordinator for all bridge operations
 * Integrates constitutional components with existing AIVillage infrastructure
 */

import { EventEmitter } from 'events';
import { ConstitutionalBetaNetAdapter } from './ConstitutionalBetaNetAdapter';
import { ConstitutionalPrivacyManager, PrivacyTier } from './PrivacyManager';
import { ConstitutionalPrivacyEnforcer } from './PrivacyEnforcer';
import { AdapterFactory } from './AdapterFactory';
import {
  AIVillageRequest,
  AIVillageResponse,
  BetaNetMessage,
  HealthStatus
} from './types';

// Import ZK proof components
import { CircuitCompiler } from '../zk/CircuitCompiler';
import { ProofGenerator, ProofInput } from '../zk/ProofGenerator';
import { ProofVerifier } from '../zk/ProofVerifier';
import { PerformanceOptimizer } from '../zk/PerformanceOptimizer';
import * as path from 'path';
import * as crypto from 'crypto';

// Import monitoring components
import { ConstitutionalPerformanceMonitor } from '../monitoring/constitutional/ConstitutionalPerformanceMonitor';
import { ConstitutionalHealthMonitor } from '../monitoring/constitutional/ConstitutionalHealthMonitor';
import { AlertManager } from '../monitoring/constitutional/AlertManager';
import { DashboardManager } from '../monitoring/constitutional/DashboardManager';
import { PythonBridge } from '../monitoring/interfaces/PythonBridge';

// Import exporters
import { PrometheusExporter } from '../monitoring/exporters/PrometheusExporter';
import { CloudWatchExporter } from '../monitoring/exporters/CloudWatchExporter';
import { DataDogExporter } from '../monitoring/exporters/DataDogExporter';

export interface OrchestratorConfig {
  // Bridge configuration
  betaNetEndpoint: string;
  bitChatEndpoint?: string;
  p2pConfig?: {
    nodeId: string;
    bootstrapNodes: string[];
    maxPeers: number;
  };

  // Fog computing configuration
  fogConfig?: {
    enabled: boolean;
    nodeType: 'edge' | 'fog' | 'cloud';
    coordinatorUrl?: string;
    maxLoad: number;
  };

  // Privacy configuration
  defaultPrivacyTier: PrivacyTier;
  enableConstitutionalValidation: boolean;
  complianceThreshold: number;

  // Monitoring configuration
  monitoring: {
    enabled: boolean;
    exporters: ('prometheus' | 'cloudwatch' | 'datadog')[];
    pythonBridgeEnabled: boolean;
    dashboardEnabled: boolean;
    alertingEnabled: boolean;
  };

  // Performance configuration
  performance: {
    targetP95Latency: number;
    circuitBreakerEnabled: boolean;
    maxConcurrentRequests: number;
    requestTimeout: number;
  };

  // Zero-Knowledge Proof configuration
  zkProofs?: {
    enabled: boolean;
    circuitPath?: string;
    zkeyPath?: string;
    vkeyPath?: string;
    optimizationLevel?: 'O0' | 'O1' | 'O2';
    cacheProofs?: boolean;
    precomputeCommon?: boolean;
  };
}

export class BridgeOrchestrator extends EventEmitter {
  private config: OrchestratorConfig;

  // Core components
  private betaNetAdapter: ConstitutionalBetaNetAdapter;
  private privacyManager: ConstitutionalPrivacyManager;
  private privacyEnforcer: ConstitutionalPrivacyEnforcer;

  // Monitoring components
  private performanceMonitor?: ConstitutionalPerformanceMonitor;
  private healthMonitor?: ConstitutionalHealthMonitor;
  private alertManager?: AlertManager;
  private dashboardManager?: DashboardManager;
  private pythonBridge?: PythonBridge;

  // Metric exporters
  private prometheusExporter?: PrometheusExporter;
  private cloudWatchExporter?: CloudWatchExporter;
  private dataDogExporter?: DataDogExporter;

  // Zero-Knowledge Proof components
  private zkCompiler?: CircuitCompiler;
  private zkProofGenerator?: ProofGenerator;
  private zkProofVerifier?: ProofVerifier;
  private zkPerformanceOptimizer?: PerformanceOptimizer;
  private zkCircuitCompiled: boolean = false;
  private zkInitRetried: boolean = false;

  // State tracking
  private isInitialized = false;
  private activeRequests = new Map<string, AIVillageRequest>();
  private circuitBreakerState: 'closed' | 'open' | 'half-open' = 'closed';
  private circuitBreakerFailures = 0;
  private readonly CIRCUIT_BREAKER_THRESHOLD = 5;
  private readonly CIRCUIT_BREAKER_RESET_TIME = 30000; // 30 seconds

  constructor(config: OrchestratorConfig) {
    super();
    this.config = config;

    // Initialize core components
    this.betaNetAdapter = AdapterFactory.createAdapter({
      endpoint: config.betaNetEndpoint,
      maxRetries: 3,
      timeout: config.performance.requestTimeout
    });

    this.privacyManager = new ConstitutionalPrivacyManager({
      defaultTier: config.defaultPrivacyTier,
      encryptionEnabled: true,
      auditingEnabled: true
    });

    this.privacyEnforcer = new ConstitutionalPrivacyEnforcer({
      enforcementLevel: 'strict',
      complianceThreshold: config.complianceThreshold
    });
  }

  /**
   * Initialize the orchestrator and all components
   */
  public async initialize(): Promise<void> {
    if (this.isInitialized) {
      throw new Error('BridgeOrchestrator already initialized');
    }

    try {
      // Initialize monitoring if enabled
      if (this.config.monitoring.enabled) {
        await this.initializeMonitoring();
      }

      // Initialize fog computing if configured
      if (this.config.fogConfig?.enabled) {
        await this.initializeFogComputing();
      }

      // Initialize P2P if configured
      if (this.config.p2pConfig) {
        await this.initializeP2P();
      }

      // Initialize Zero-Knowledge Proofs if enabled
      if (this.config.zkProofs?.enabled) {
        await this.initializeZKProofs();
      }

      // Start health checks
      this.startHealthChecks();

      this.isInitialized = true;
      this.emit('initialized', { timestamp: Date.now() });

    } catch (error) {
      console.error('Failed to initialize BridgeOrchestrator:', error);
      throw new Error(`Initialization failed: ${error.message}`);
    }
  }

  /**
   * Process a request through the bridge
   */
  public async processRequest(request: AIVillageRequest): Promise<AIVillageResponse> {
    const requestId = this.generateRequestId();
    const startTime = Date.now();

    try {
      // Check circuit breaker
      if (!this.canProcessRequest()) {
        throw new Error('Circuit breaker is open - service temporarily unavailable');
      }

      // Track active request
      this.activeRequests.set(requestId, request);

      // Start performance monitoring
      if (this.performanceMonitor) {
        this.performanceMonitor.startConstitutionalTiming(requestId, {
          privacyTier: request.privacyTier || this.config.defaultPrivacyTier,
          userContext: request.userContext
        });
      }

      // Step 1: Privacy validation and classification
      const privacyValidation = await this.validatePrivacy(request);
      if (!privacyValidation.isValid) {
        throw new Error(`Privacy validation failed: ${privacyValidation.reason}`);
      }

      // Step 2: Constitutional validation
      if (this.config.enableConstitutionalValidation) {
        const constitutionalValidation = await this.validateConstitutional(request);
        if (!constitutionalValidation.isValid) {
          throw new Error(`Constitutional validation failed: ${constitutionalValidation.violations.join(', ')}`);
        }
      }

      // Step 2.5: Zero-Knowledge Proof validation
      if (this.config.zkProofs?.enabled && this.zkProofGenerator && this.zkProofVerifier) {
        const zkValidation = await this.validateWithZKProof(request);
        if (!zkValidation.valid) {
          throw new Error(`ZK proof validation failed: ${zkValidation.errors.join(', ')}`);
        }
      }

      // Step 3: Apply privacy transformations
      const securedRequest = await this.applyPrivacyTransformations(request);

      // Step 4: Route to appropriate protocol
      let response: AIVillageResponse;

      switch (request.protocol) {
        case 'betanet':
          response = await this.processBetaNetRequest(securedRequest);
          break;
        case 'bitchat':
          response = await this.processBitChatRequest(securedRequest);
          break;
        case 'p2p':
          response = await this.processP2PRequest(securedRequest);
          break;
        case 'fog':
          response = await this.processFogRequest(securedRequest);
          break;
        default:
          // Default to BetaNet
          response = await this.processBetaNetRequest(securedRequest);
      }

      // Step 5: Post-process response
      response = await this.postProcessResponse(response, request.privacyTier);

      // End performance monitoring
      if (this.performanceMonitor) {
        const metrics = this.performanceMonitor.endConstitutionalTiming(
          requestId,
          true,
          { responseSize: JSON.stringify(response).length }
        );

        // Check performance targets
        if (metrics.latency > this.config.performance.targetP95Latency) {
          this.emit('performanceWarning', {
            requestId,
            latency: metrics.latency,
            target: this.config.performance.targetP95Latency
          });
        }
      }

      // Reset circuit breaker on success
      this.circuitBreakerFailures = 0;

      return response;

    } catch (error) {
      // Track failure for circuit breaker
      this.handleRequestFailure(error);

      // End performance monitoring with error
      if (this.performanceMonitor) {
        this.performanceMonitor.endConstitutionalTiming(requestId, false, { error: error.message });
      }

      throw error;

    } finally {
      // Clean up
      this.activeRequests.delete(requestId);

      // Export metrics
      this.exportMetrics({
        requestId,
        duration: Date.now() - startTime,
        success: !error
      });
    }
  }

  /**
   * Get current health status
   */
  public async getHealthStatus(): Promise<HealthStatus> {
    const statuses: HealthStatus[] = [];

    // Check BetaNet adapter
    statuses.push(await this.betaNetAdapter.getHealthStatus());

    // Check monitoring components
    if (this.healthMonitor) {
      const monitoringHealth = await this.healthMonitor.getConstitutionalHealth();
      statuses.push({
        status: monitoringHealth.overallStatus,
        components: monitoringHealth.components,
        timestamp: monitoringHealth.timestamp
      });
    }

    // Check Python bridge
    if (this.pythonBridge) {
      const pythonHealth = await this.pythonBridge.getHealth();
      statuses.push({
        status: pythonHealth.healthy ? 'healthy' : 'unhealthy',
        components: {
          pythonBridge: pythonHealth.healthy ? 'healthy' : 'unhealthy'
        },
        timestamp: Date.now()
      });
    }

    // Aggregate health status
    const overallStatus = statuses.every(s => s.status === 'healthy') ? 'healthy' :
                         statuses.some(s => s.status === 'unhealthy') ? 'unhealthy' : 'degraded';

    return {
      status: overallStatus,
      components: Object.assign({}, ...statuses.map(s => s.components)),
      metrics: {
        activeRequests: this.activeRequests.size,
        circuitBreakerState: this.circuitBreakerState,
        uptime: process.uptime()
      },
      timestamp: Date.now()
    };
  }

  /**
   * Shutdown the orchestrator gracefully
   */
  public async shutdown(): Promise<void> {
    console.log('Shutting down BridgeOrchestrator...');

    // Stop accepting new requests
    this.circuitBreakerState = 'open';

    // Wait for active requests to complete (with timeout)
    const timeout = setTimeout(() => {
      console.warn('Force closing active requests after timeout');
      this.activeRequests.clear();
    }, 10000);

    while (this.activeRequests.size > 0) {
      await this.sleep(100);
    }
    clearTimeout(timeout);

    // Shutdown components
    if (this.performanceMonitor) {
      this.performanceMonitor.reset();
    }

    if (this.pythonBridge) {
      await this.pythonBridge.disconnect();
    }

    // Stop exporters
    if (this.prometheusExporter) {
      this.prometheusExporter.stop();
    }

    if (this.cloudWatchExporter) {
      this.cloudWatchExporter.stop();
    }

    if (this.dataDogExporter) {
      this.dataDogExporter.stop();
    }

    this.isInitialized = false;
    this.emit('shutdown', { timestamp: Date.now() });
  }

  // Private methods

  private async initializeMonitoring(): Promise<void> {
    // Initialize performance monitor
    this.performanceMonitor = new ConstitutionalPerformanceMonitor({
      constitutional: {
        validationLevel: 'strict',
        complianceThreshold: this.config.complianceThreshold
      },
      circuitBreaker: {
        enabled: this.config.performance.circuitBreakerEnabled,
        threshold: this.CIRCUIT_BREAKER_THRESHOLD,
        resetTime: this.CIRCUIT_BREAKER_RESET_TIME
      }
    });

    // Initialize health monitor
    this.healthMonitor = new ConstitutionalHealthMonitor({
      checkInterval: 30000,
      constitutional: {
        complianceThreshold: this.config.complianceThreshold
      }
    });

    // Initialize alert manager
    if (this.config.monitoring.alertingEnabled) {
      this.alertManager = new AlertManager({
        channels: ['email', 'slack'],
        constitutional: {
          enabled: true,
          severityThreshold: 'medium'
        }
      });
    }

    // Initialize dashboard
    if (this.config.monitoring.dashboardEnabled) {
      this.dashboardManager = new DashboardManager({
        constitutional: {
          enabled: true,
          widgets: ['compliance', 'privacy', 'performance']
        }
      });
    }

    // Initialize Python bridge
    if (this.config.monitoring.pythonBridgeEnabled) {
      this.pythonBridge = new PythonBridge({
        host: 'localhost',
        port: 5678,
        protocol: 'tcp'
      });
      await this.pythonBridge.connect();
    }

    // Initialize exporters
    await this.initializeExporters();
  }

  private async initializeExporters(): Promise<void> {
    const exporters = this.config.monitoring.exporters;

    if (exporters.includes('prometheus')) {
      this.prometheusExporter = new PrometheusExporter({
        port: 9090,
        prefix: 'aivillage_bridge'
      });
    }

    if (exporters.includes('cloudwatch')) {
      this.cloudWatchExporter = new CloudWatchExporter({
        namespace: 'AIVillage/Bridge',
        region: process.env.AWS_REGION || 'us-east-1'
      });
    }

    if (exporters.includes('datadog')) {
      this.dataDogExporter = new DataDogExporter({
        apiKey: process.env.DATADOG_API_KEY,
        service: 'aivillage-bridge'
      });
    }
  }

  private async initializeFogComputing(): Promise<void> {
    // Initialize fog computing connection
    console.log('Initializing fog computing with config:', this.config.fogConfig);
    // Implementation would connect to fog coordinator
  }

  private async initializeP2P(): Promise<void> {
    // Initialize P2P network
    console.log('Initializing P2P network with config:', this.config.p2pConfig);
    // Implementation would join P2P network
  }

  private startHealthChecks(): void {
    setInterval(async () => {
      const health = await this.getHealthStatus();
      this.emit('healthCheck', health);

      if (health.status === 'unhealthy' && this.alertManager) {
        await this.alertManager.sendAlert({
          severity: 'critical',
          title: 'Bridge Unhealthy',
          description: 'BridgeOrchestrator health check failed',
          constitutional: {
            validated: true,
            complianceScore: 1.0
          }
        });
      }
    }, 30000); // Every 30 seconds
  }

  private canProcessRequest(): boolean {
    if (!this.config.performance.circuitBreakerEnabled) {
      return true;
    }

    switch (this.circuitBreakerState) {
      case 'closed':
        return true;
      case 'open':
        // Check if enough time has passed to try half-open
        setTimeout(() => {
          this.circuitBreakerState = 'half-open';
        }, this.CIRCUIT_BREAKER_RESET_TIME);
        return false;
      case 'half-open':
        // Allow one request through to test
        return true;
      default:
        return false;
    }
  }

  private handleRequestFailure(error: Error): void {
    this.circuitBreakerFailures++;

    if (this.circuitBreakerFailures >= this.CIRCUIT_BREAKER_THRESHOLD) {
      this.circuitBreakerState = 'open';
      this.emit('circuitBreakerOpen', {
        failures: this.circuitBreakerFailures,
        error: error.message
      });
    }
  }

  private async validatePrivacy(request: AIVillageRequest): Promise<any> {
    return this.privacyManager.validateRequest(request);
  }

  private async validateConstitutional(request: AIVillageRequest): Promise<any> {
    return this.privacyEnforcer.validateConstitutional(request);
  }

  private async applyPrivacyTransformations(request: AIVillageRequest): Promise<AIVillageRequest> {
    return this.privacyManager.applyPrivacyTier(request, request.privacyTier || this.config.defaultPrivacyTier);
  }

  private async processBetaNetRequest(request: AIVillageRequest): Promise<AIVillageResponse> {
    const betaNetMessage = await this.betaNetAdapter.translateToBetaNet(request);
    // Process through BetaNet...
    const response = await this.betaNetAdapter.translateFromBetaNet(betaNetMessage);
    return response;
  }

  private async processBitChatRequest(request: AIVillageRequest): Promise<AIVillageResponse> {
    // BitChat processing
    console.log('Processing BitChat request');
    return { success: true, data: {}, timestamp: Date.now() };
  }

  private async processP2PRequest(request: AIVillageRequest): Promise<AIVillageResponse> {
    // P2P processing
    console.log('Processing P2P request');
    return { success: true, data: {}, timestamp: Date.now() };
  }

  private async processFogRequest(request: AIVillageRequest): Promise<AIVillageResponse> {
    // Fog computing processing
    console.log('Processing fog request');
    return { success: true, data: {}, timestamp: Date.now() };
  }

  private async postProcessResponse(response: AIVillageResponse, tier?: PrivacyTier): Promise<AIVillageResponse> {
    // Apply privacy filters to response
    if (tier) {
      response = await this.privacyManager.filterResponse(response, tier);
    }
    return response;
  }

  private exportMetrics(metrics: any): void {
    if (this.prometheusExporter) {
      this.prometheusExporter.recordRequest(
        metrics.duration,
        'POST',
        '/bridge',
        metrics.success ? 200 : 500
      );
    }

    if (this.cloudWatchExporter) {
      this.cloudWatchExporter.recordRequest(
        metrics.duration,
        'POST',
        '/bridge',
        metrics.success ? 200 : 500
      );
    }

    if (this.dataDogExporter) {
      this.dataDogExporter.recordRequest(
        metrics.duration,
        'POST',
        '/bridge',
        metrics.success ? 200 : 500
      );
    }
  }

  private generateRequestId(): string {
    return `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Initialize Zero-Knowledge Proof system
   */
  private async initializeZKProofs(): Promise<void> {
    console.log('Initializing Zero-Knowledge Proof system...');

    try {
      // Set default paths
      const zkDir = path.join(__dirname, '../zk');
      const circuitPath = this.config.zkProofs?.circuitPath ||
                         path.join(zkDir, 'circuits/privacy_validation.circom');
      const buildDir = path.join(zkDir, 'build');

      // Initialize compiler
      this.zkCompiler = new CircuitCompiler({
        maxConstraints: 100000,
        optimizationLevel: this.config.zkProofs?.optimizationLevel || 'O2'
      });

      // Compile circuit if not already compiled
      if (!this.zkCircuitCompiled) {
        console.log('Compiling privacy validation circuit...');
        const compilationResult = await this.zkCompiler.compileCircuit(circuitPath, buildDir);

        // Perform trusted setup
        console.log('Performing trusted setup...');
        const setupResult = await this.zkCompiler.performTrustedSetup(
          compilationResult.r1csPath,
          buildDir,
          crypto.randomBytes(32).toString('hex')
        );

        // Store paths
        this.config.zkProofs!.zkeyPath = setupResult.zkeyPath;
        this.config.zkProofs!.vkeyPath = setupResult.vkeyPath;

        this.zkCircuitCompiled = true;
        console.log(`Circuit compiled: ${compilationResult.constraints} constraints`);
      }

      // Initialize proof generator
      const zkeyPath = this.config.zkProofs?.zkeyPath || path.join(buildDir, 'privacy_validation.zkey');
      const wasmPath = path.join(buildDir, 'privacy_validation_js/privacy_validation.wasm');

      this.zkProofGenerator = new ProofGenerator(zkeyPath, wasmPath, {
        enableCache: this.config.zkProofs?.cacheProofs !== false,
        enableParallel: true,
        maxCacheSize: 1000,
        workerCount: 4
      });

      // Initialize proof verifier
      const vkeyPath = this.config.zkProofs?.vkeyPath || path.join(buildDir, 'privacy_validation_verification_key.json');

      this.zkProofVerifier = new ProofVerifier(vkeyPath, {
        enableCache: true,
        checkNullifiers: true,
        strictMode: true,
        maxCacheSize: 500
      });

      // Initialize performance optimizer
      this.zkPerformanceOptimizer = new PerformanceOptimizer({
        targetP95Latency: this.config.performance.targetP95Latency,
        enableParallelization: true,
        enableCaching: true,
        precomputeCommonProofs: this.config.zkProofs?.precomputeCommon !== false,
        adaptiveOptimization: true
      });

      this.zkPerformanceOptimizer.attachComponents(this.zkProofGenerator, this.zkProofVerifier);

      // Precompute common proofs if enabled
      if (this.config.zkProofs?.precomputeCommon !== false) {
        await this.zkPerformanceOptimizer.precomputeCommonProofs();
      }

      console.log('Zero-Knowledge Proof system initialized successfully');

    } catch (error) {
      console.error('Failed to initialize ZK proofs:', error);

      // Determine if this is a critical failure or can be recovered
      const errorMessage = error.message || String(error);

      if (errorMessage.includes('Circom compiler not installed')) {
        // Critical - cannot function without compiler
        throw new Error(`ZK Proof initialization failed: Circom compiler required. Install with: npm install -g circom`);
      } else if (errorMessage.includes('WASM file not found')) {
        // Try to recover by recompiling
        console.log('WASM file missing - attempting to recompile circuit...');
        this.zkCircuitCompiled = false;
        // Retry initialization once
        if (!this.zkInitRetried) {
          this.zkInitRetried = true;
          return this.initializeZKProofs();
        }
      }

      // If ZK is marked as required, throw error
      if (this.config.zkProofs?.required) {
        throw new Error(`ZK Proof initialization failed (required): ${errorMessage}`);
      }

      // Otherwise, warn and disable
      console.warn('ZK Proofs disabled due to initialization failure. System will operate without ZK validation.');
      this.config.zkProofs!.enabled = false;
      this.emit('zkProofsDegraded', { reason: errorMessage });
    }
  }

  /**
   * Validate request using Zero-Knowledge Proof
   */
  private async validateWithZKProof(request: AIVillageRequest): Promise<{
    valid: boolean;
    proof?: any;
    commitment?: string;
    errors: string[];
  }> {
    const errors: string[] = [];

    try {
      // Prepare proof input
      const proofInput: ProofInput = {
        // Private inputs
        dataHash: crypto.createHash('sha256').update(JSON.stringify(request.data)).digest('hex'),
        userConsent: request.userContext?.consent ? 1 : 0,
        dataCategories: this.extractDataCategories(request.data),
        processingPurpose: this.getProcessingPurposeCode(request.purpose),
        retentionPeriod: request.retentionDays || 30,

        // Public inputs
        privacyTier: this.privacyTierToNumber(request.privacyTier || this.config.defaultPrivacyTier),
        constitutionalHash: crypto.createHash('sha256')
          .update(JSON.stringify(this.config.complianceThreshold))
          .digest('hex')
      };

      // Generate proof with optimization
      const { proof, optimizations, latency } = await this.zkPerformanceOptimizer!
        .optimizeProofGeneration(proofInput);

      // Extract public signals from proof result
      const publicSignals = proof.publicSignals;

      // Verify the proof
      const verificationResult = await this.zkProofVerifier!.verifyProof(proof.proof, publicSignals);

      if (!verificationResult.valid) {
        errors.push(...verificationResult.errors);
        return { valid: false, errors };
      }

      // Check validation result (first public signal)
      const validationPassed = BigInt(publicSignals[0]) === 1n;
      if (!validationPassed) {
        errors.push('Privacy validation circuit rejected the request');
        return { valid: false, errors };
      }

      // Log performance metrics
      if (latency > this.config.performance.targetP95Latency) {
        console.warn(`ZK proof generation exceeded target: ${latency}ms`);
      }

      console.log(`ZK proof validated in ${latency}ms with optimizations: ${optimizations.join(', ')}`);

      return {
        valid: true,
        proof: proof.proof,
        commitment: verificationResult.commitment,
        errors: []
      };

    } catch (error) {
      console.error('ZK proof validation failed:', error);
      errors.push(`ZK proof error: ${error.message}`);
      return { valid: false, errors };
    }
  }

  /**
   * Extract data categories from request data
   */
  private extractDataCategories(data: any): number[] {
    const categories = [0, 0, 0, 0, 0]; // 5 category slots

    if (data) {
      // Category 0: Personal identifiers
      if (data.userId || data.email || data.username) categories[0] = 1;

      // Category 1: Financial data
      if (data.payment || data.creditCard || data.bankAccount) categories[1] = 1;

      // Category 2: Health data
      if (data.health || data.medical || data.diagnosis) categories[2] = 1;

      // Category 3: Location data
      if (data.location || data.gps || data.address) categories[3] = 1;

      // Category 4: Behavioral data
      if (data.behavior || data.analytics || data.tracking) categories[4] = 1;
    }

    return categories;
  }

  /**
   * Get processing purpose code
   */
  private getProcessingPurposeCode(purpose?: string): number {
    const purposeMap: { [key: string]: number } = {
      'legitimate_interest': 5,
      'contract': 15,
      'legal_obligation': 25,
      'consent_required': 35,
      'analytics': 10,
      'security': 20,
      'research': 30
    };

    return purposeMap[purpose || 'consent_required'] || 35;
  }

  /**
   * Convert privacy tier to number
   */
  private privacyTierToNumber(tier: PrivacyTier): number {
    const tierMap = {
      'Bronze': 0,
      'Silver': 1,
      'Gold': 2,
      'Platinum': 3
    };
    return tierMap[tier] || 0;
  }

  /**
   * Get ZK proof metrics
   */
  public getZKMetrics(): any {
    if (!this.zkPerformanceOptimizer) {
      return null;
    }

    return {
      performance: this.zkPerformanceOptimizer.getMetrics(),
      report: this.zkPerformanceOptimizer.generateReport()
    };
  }
}

export default BridgeOrchestrator;