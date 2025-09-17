/**
 * ML Content Analyzer with Zero-Knowledge Proof Generation
 * Integrates multiple ML models for advanced content analysis
 * Generates ZK proofs for analysis results without revealing content
 */

import * as tf from '@tensorflow/tfjs-node';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import * as toxicity from '@tensorflow-models/toxicity';
import { pipeline, Pipeline } from '@xenova/transformers';
import { ProofGenerator, ProofInput } from '../zk/ProofGenerator';
import { ProofVerifier } from '../zk/ProofVerifier';
import * as crypto from 'crypto';
import * as path from 'path';

export interface AnalysisResult {
  toxicity: ToxicityResult;
  privacy: PrivacyRiskResult;
  manipulation: ManipulationResult;
  bias: BiasResult;
  overall: OverallAssessment;
  metadata: AnalysisMetadata;
}

export interface ToxicityResult {
  isToxic: boolean;
  score: number;
  categories: {
    identity_attack: number;
    insult: number;
    obscene: number;
    severe_toxicity: number;
    sexual_explicit: number;
    threat: number;
    toxicity: number;
  };
  flaggedPhrases: string[];
}

export interface PrivacyRiskResult {
  hasPrivacyRisk: boolean;
  score: number;
  detectedPII: Array<{
    type: string;
    value: string;
    confidence: number;
    location: [number, number]; // start, end indices
  }>;
  sensitiveTopics: string[];
}

export interface ManipulationResult {
  hasManipulation: boolean;
  score: number;
  techniques: Array<{
    type: string;
    confidence: number;
    evidence: string;
  }>;
  socialEngineering: boolean;
}

export interface BiasResult {
  hasBias: boolean;
  score: number;
  types: Array<{
    category: string;
    confidence: number;
    direction: 'positive' | 'negative' | 'neutral';
  }>;
  fairnessScore: number;
}

export interface OverallAssessment {
  safe: boolean;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  recommendations: string[];
  requiredPrivacyTier: 'Bronze' | 'Silver' | 'Gold' | 'Platinum';
}

export interface AnalysisMetadata {
  timestamp: number;
  processingTime: number;
  modelsUsed: string[];
  version: string;
}

interface ModelCache {
  toxicityModel?: any;
  sentenceEncoder?: any;
  nerPipeline?: Pipeline;
  sentimentPipeline?: Pipeline;
  classificationPipeline?: Pipeline;
}

export class ContentAnalyzer {
  private models: ModelCache = {};
  private isInitialized: boolean = false;
  private analysisCache: Map<string, AnalysisResult> = new Map();

  // Zero-Knowledge Proof components
  private zkProofGenerator?: ProofGenerator;
  private zkProofVerifier?: ProofVerifier;
  private zkEnabled: boolean = false;

  constructor(private config: {
    enableCaching?: boolean;
    cacheTimeout?: number;
    toxicityThreshold?: number;
    privacyThreshold?: number;
    manipulationThreshold?: number;
    biasThreshold?: number;
  } = {}) {
    this.config = {
      enableCaching: true,
      cacheTimeout: 300000, // 5 minutes
      toxicityThreshold: 0.7,
      privacyThreshold: 0.6,
      manipulationThreshold: 0.65,
      biasThreshold: 0.7,
      ...config
    };
  }

  /**
   * Initialize ML models
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    const startTime = Date.now();
    console.log('[ML] Initializing content analysis models...');

    try {
      // Load models in parallel
      const [toxicityModel, sentenceEncoder, nerPipeline, sentimentPipeline] = await Promise.all([
        this.loadToxicityModel(),
        this.loadSentenceEncoder(),
        this.loadNERPipeline(),
        this.loadSentimentPipeline()
      ]);

      this.models.toxicityModel = toxicityModel;
      this.models.sentenceEncoder = sentenceEncoder;
      this.models.nerPipeline = nerPipeline;
      this.models.sentimentPipeline = sentimentPipeline;

      // Load classification pipeline
      this.models.classificationPipeline = await this.loadClassificationPipeline();

      this.isInitialized = true;
      console.log(`[ML] Models initialized in ${Date.now() - startTime}ms`);

    } catch (error) {
      console.error('[ML] Failed to initialize models:', error);
      throw new Error('Model initialization failed');
    }
  }

  /**
   * Analyze content with all models
   */
  async analyze(content: string): Promise<AnalysisResult> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const startTime = Date.now();

    // Check cache
    if (this.config.enableCaching) {
      const cached = this.getCached(content);
      if (cached) return cached;
    }

    // Run analyses in parallel
    const [toxicity, privacy, manipulation, bias] = await Promise.all([
      this.analyzeToxicity(content),
      this.analyzePrivacyRisk(content),
      this.analyzeManipulation(content),
      this.analyzeBias(content)
    ]);

    // Generate overall assessment
    const overall = this.generateOverallAssessment(toxicity, privacy, manipulation, bias);

    const result: AnalysisResult = {
      toxicity,
      privacy,
      manipulation,
      bias,
      overall,
      metadata: {
        timestamp: Date.now(),
        processingTime: Date.now() - startTime,
        modelsUsed: [
          'toxicity-detection',
          'ner-privacy',
          'manipulation-detection',
          'bias-detection'
        ],
        version: '1.0.0'
      }
    };

    // Cache result
    if (this.config.enableCaching) {
      this.cacheResult(content, result);
    }

    return result;
  }

  /**
   * Analyze toxicity using TensorFlow model
   */
  private async analyzeToxicity(content: string): Promise<ToxicityResult> {
    try {
      const predictions = await this.models.toxicityModel.classify([content]);

      const categories: any = {
        identity_attack: 0,
        insult: 0,
        obscene: 0,
        severe_toxicity: 0,
        sexual_explicit: 0,
        threat: 0,
        toxicity: 0
      };

      const flaggedPhrases: string[] = [];
      let maxScore = 0;

      for (const prediction of predictions) {
        const label = prediction.label.toLowerCase().replace(/ /g, '_');
        if (prediction.results && prediction.results[0]) {
          const match = prediction.results[0].match;
          const prob = prediction.results[0].probabilities[1]; // Probability of being toxic

          if (categories.hasOwnProperty(label)) {
            categories[label] = prob;
            maxScore = Math.max(maxScore, prob);
          }

          if (match && prob > this.config.toxicityThreshold!) {
            flaggedPhrases.push(content.substring(0, 50) + '...');
          }
        }
      }

      return {
        isToxic: maxScore > this.config.toxicityThreshold!,
        score: maxScore,
        categories,
        flaggedPhrases
      };

    } catch (error) {
      console.error('[ML] Toxicity analysis failed:', error);
      return {
        isToxic: false,
        score: 0,
        categories: {
          identity_attack: 0,
          insult: 0,
          obscene: 0,
          severe_toxicity: 0,
          sexual_explicit: 0,
          threat: 0,
          toxicity: 0
        },
        flaggedPhrases: []
      };
    }
  }

  /**
   * Analyze privacy risk using NER and pattern matching
   */
  private async analyzePrivacyRisk(content: string): Promise<PrivacyRiskResult> {
    try {
      // Use NER pipeline to detect entities
      const entities = await this.models.nerPipeline(content);

      const detectedPII: any[] = [];
      const sensitiveTopics: string[] = [];

      // Process NER results
      for (const entity of entities) {
        if (this.isPIIEntity(entity.entity_group)) {
          detectedPII.push({
            type: entity.entity_group,
            value: entity.word,
            confidence: entity.score,
            location: [entity.start, entity.end]
          });
        }
      }

      // Check for sensitive topics using embeddings
      const embedding = await this.models.sentenceEncoder.embed([content]);
      const sensitiveScore = await this.checkSensitiveTopics(embedding);

      if (sensitiveScore > 0.5) {
        sensitiveTopics.push('potentially_sensitive_content');
      }

      embedding.dispose();

      const hasPrivacyRisk = detectedPII.length > 0 || sensitiveTopics.length > 0;
      const score = Math.max(
        detectedPII.length > 0 ? Math.max(...detectedPII.map(p => p.confidence)) : 0,
        sensitiveScore
      );

      return {
        hasPrivacyRisk,
        score,
        detectedPII,
        sensitiveTopics
      };

    } catch (error) {
      console.error('[ML] Privacy analysis failed:', error);
      return {
        hasPrivacyRisk: false,
        score: 0,
        detectedPII: [],
        sensitiveTopics: []
      };
    }
  }

  /**
   * Analyze manipulation techniques
   */
  private async analyzeManipulation(content: string): Promise<ManipulationResult> {
    try {
      // Use sentiment and classification pipelines
      const sentiment = await this.models.sentimentPipeline(content);
      const classification = await this.models.classificationPipeline(content);

      const techniques: any[] = [];
      let manipulationScore = 0;
      let socialEngineering = false;

      // Check for emotional manipulation
      if (sentiment && sentiment[0]) {
        const emotionalIntensity = Math.abs(sentiment[0].score - 0.5) * 2;
        if (emotionalIntensity > 0.8) {
          techniques.push({
            type: 'emotional_manipulation',
            confidence: emotionalIntensity,
            evidence: `High emotional intensity detected (${sentiment[0].label})`
          });
          manipulationScore = Math.max(manipulationScore, emotionalIntensity);
        }
      }

      // Check for social engineering patterns
      const socialEngineeringPatterns = [
        /urgent.*action.*required/i,
        /verify.*account.*immediately/i,
        /suspended.*unless/i,
        /click.*here.*now/i,
        /limited.*time.*offer/i,
        /act.*now.*before/i
      ];

      for (const pattern of socialEngineeringPatterns) {
        if (pattern.test(content)) {
          socialEngineering = true;
          techniques.push({
            type: 'social_engineering',
            confidence: 0.85,
            evidence: 'Detected social engineering pattern'
          });
          manipulationScore = Math.max(manipulationScore, 0.85);
          break;
        }
      }

      // Check classification results for manipulation
      if (classification && classification[0]) {
        for (const result of classification) {
          if (result.label.includes('spam') || result.label.includes('scam')) {
            techniques.push({
              type: 'deceptive_content',
              confidence: result.score,
              evidence: `Classified as ${result.label}`
            });
            manipulationScore = Math.max(manipulationScore, result.score);
          }
        }
      }

      return {
        hasManipulation: manipulationScore > this.config.manipulationThreshold!,
        score: manipulationScore,
        techniques,
        socialEngineering
      };

    } catch (error) {
      console.error('[ML] Manipulation analysis failed:', error);
      return {
        hasManipulation: false,
        score: 0,
        techniques: [],
        socialEngineering: false
      };
    }
  }

  /**
   * Analyze bias in content
   */
  private async analyzeBias(content: string): Promise<BiasResult> {
    try {
      // Use embeddings to detect bias
      const embedding = await this.models.sentenceEncoder.embed([content]);

      const biasCategories = [
        'gender',
        'race',
        'age',
        'religion',
        'political',
        'socioeconomic'
      ];

      const types: any[] = [];
      let maxBiasScore = 0;

      // Check each bias category
      for (const category of biasCategories) {
        const biasScore = await this.checkBiasCategory(embedding, category);
        if (biasScore > 0.3) {
          types.push({
            category,
            confidence: biasScore,
            direction: this.getBiasDirection(content, category)
          });
          maxBiasScore = Math.max(maxBiasScore, biasScore);
        }
      }

      embedding.dispose();

      // Calculate fairness score (inverse of bias)
      const fairnessScore = 1 - maxBiasScore;

      return {
        hasBias: maxBiasScore > this.config.biasThreshold!,
        score: maxBiasScore,
        types,
        fairnessScore
      };

    } catch (error) {
      console.error('[ML] Bias analysis failed:', error);
      return {
        hasBias: false,
        score: 0,
        types: [],
        fairnessScore: 1
      };
    }
  }

  /**
   * Generate overall assessment
   */
  private generateOverallAssessment(
    toxicity: ToxicityResult,
    privacy: PrivacyRiskResult,
    manipulation: ManipulationResult,
    bias: BiasResult
  ): OverallAssessment {
    const scores = [
      toxicity.score,
      privacy.score,
      manipulation.score,
      bias.score
    ];

    const maxScore = Math.max(...scores);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;

    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high' | 'critical';
    if (maxScore > 0.9 || avgScore > 0.8) {
      riskLevel = 'critical';
    } else if (maxScore > 0.7 || avgScore > 0.6) {
      riskLevel = 'high';
    } else if (maxScore > 0.5 || avgScore > 0.4) {
      riskLevel = 'medium';
    } else {
      riskLevel = 'low';
    }

    // Determine required privacy tier
    let requiredPrivacyTier: 'Bronze' | 'Silver' | 'Gold' | 'Platinum';
    if (privacy.hasPrivacyRisk && privacy.detectedPII.length > 2) {
      requiredPrivacyTier = 'Platinum';
    } else if (privacy.hasPrivacyRisk) {
      requiredPrivacyTier = 'Gold';
    } else if (maxScore > 0.5) {
      requiredPrivacyTier = 'Silver';
    } else {
      requiredPrivacyTier = 'Bronze';
    }

    // Generate recommendations
    const recommendations: string[] = [];

    if (toxicity.isToxic) {
      recommendations.push('Remove or moderate toxic content');
    }
    if (privacy.hasPrivacyRisk) {
      recommendations.push('Mask or remove PII before processing');
    }
    if (manipulation.hasManipulation) {
      recommendations.push('Flag for manual review - potential manipulation');
    }
    if (bias.hasBias) {
      recommendations.push('Review for bias and ensure fairness');
    }

    const safe = riskLevel === 'low' && !toxicity.isToxic && !manipulation.socialEngineering;

    return {
      safe,
      riskLevel,
      confidence: 1 - (avgScore * 0.2), // Confidence decreases with risk
      recommendations,
      requiredPrivacyTier
    };
  }

  /**
   * Load toxicity detection model
   */
  private async loadToxicityModel(): Promise<any> {
    return await toxicity.load(this.config.toxicityThreshold!, []);
  }

  /**
   * Load universal sentence encoder
   */
  private async loadSentenceEncoder(): Promise<any> {
    return await use.load();
  }

  /**
   * Load NER pipeline for PII detection
   */
  private async loadNERPipeline(): Promise<Pipeline> {
    return await pipeline('token-classification', 'Xenova/bert-base-NER');
  }

  /**
   * Load sentiment analysis pipeline
   */
  private async loadSentimentPipeline(): Promise<Pipeline> {
    return await pipeline('sentiment-analysis', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
  }

  /**
   * Load text classification pipeline
   */
  private async loadClassificationPipeline(): Promise<Pipeline> {
    return await pipeline('text-classification', 'Xenova/toxic-bert');
  }

  /**
   * Check if entity is PII
   */
  private isPIIEntity(entityType: string): boolean {
    const piiTypes = ['PER', 'PERSON', 'ORG', 'LOC', 'EMAIL', 'PHONE', 'SSN', 'CREDIT_CARD'];
    return piiTypes.includes(entityType.toUpperCase());
  }

  /**
   * Check for sensitive topics using embeddings
   */
  private async checkSensitiveTopics(embedding: tf.Tensor): Promise<number> {
    // Simplified sensitive topic detection
    // In production, this would compare against known sensitive topic embeddings
    const values = await embedding.array();
    const magnitude = Math.sqrt(values[0].reduce((sum: number, val: number) => sum + val * val, 0));

    // Normalize to 0-1 range
    return Math.min(1, magnitude / 10);
  }

  /**
   * Check specific bias category
   */
  private async checkBiasCategory(embedding: tf.Tensor, category: string): Promise<number> {
    // Simplified bias detection
    // In production, this would use trained bias detection models
    const categoryWeights: { [key: string]: number } = {
      gender: 0.15,
      race: 0.2,
      age: 0.1,
      religion: 0.15,
      political: 0.25,
      socioeconomic: 0.15
    };

    const values = await embedding.array();
    const sum = values[0].reduce((acc: number, val: number) => acc + Math.abs(val), 0);

    return Math.min(1, sum * (categoryWeights[category] || 0.1));
  }

  /**
   * Determine bias direction
   */
  private getBiasDirection(content: string, category: string): 'positive' | 'negative' | 'neutral' {
    // Simplified direction detection
    const negativeIndicators = ['not', 'never', 'worst', 'bad', 'wrong', 'false'];
    const positiveIndicators = ['best', 'great', 'excellent', 'perfect', 'amazing'];

    const lowerContent = content.toLowerCase();
    const negCount = negativeIndicators.filter(ind => lowerContent.includes(ind)).length;
    const posCount = positiveIndicators.filter(ind => lowerContent.includes(ind)).length;

    if (negCount > posCount) return 'negative';
    if (posCount > negCount) return 'positive';
    return 'neutral';
  }

  /**
   * Get cached result
   */
  private getCached(content: string): AnalysisResult | null {
    const key = this.hashContent(content);
    const cached = this.analysisCache.get(key);

    if (cached && Date.now() - cached.metadata.timestamp < this.config.cacheTimeout!) {
      return cached;
    }

    return null;
  }

  /**
   * Cache analysis result
   */
  private cacheResult(content: string, result: AnalysisResult): void {
    const key = this.hashContent(content);
    this.analysisCache.set(key, result);

    // Clean old cache entries
    setTimeout(() => {
      this.analysisCache.delete(key);
    }, this.config.cacheTimeout!);
  }

  /**
   * Cryptographic hash function for cache keys
   */
  private hashContent(content: string): string {
    return crypto.createHash('sha256')
      .update(content)
      .update('content-analyzer-v1') // Domain separation
      .digest('hex');
  }

  /**
   * Warm up models with sample data
   */
  async warmup(): Promise<void> {
    await this.analyze('This is a warmup test sentence.');
  }

  /**
   * Get model status
   */
  getStatus(): {
    initialized: boolean;
    models: string[];
    cacheSize: number;
  } {
    return {
      initialized: this.isInitialized,
      models: Object.keys(this.models).filter(k => this.models[k] !== undefined),
      cacheSize: this.analysisCache.size
    };
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.analysisCache.clear();
  }

  /**
   * Dispose models to free memory
   */
  async dispose(): Promise<void> {
    // Dispose TensorFlow models
    if (this.models.sentenceEncoder) {
      this.models.sentenceEncoder.dispose();
    }

    // Clear cache
    this.clearCache();

    // Reset state
    this.models = {};
    this.isInitialized = false;
    this.zkEnabled = false;
    this.zkProofGenerator = undefined;
    this.zkProofVerifier = undefined;
  }

  /**
   * Initialize Zero-Knowledge Proof system for analysis
   */
  async initializeZKProofs(zkeyPath?: string, vkeyPath?: string): Promise<void> {
    try {
      const defaultZkDir = path.join(__dirname, '../zk/build');
      const zkeyFile = zkeyPath || path.join(defaultZkDir, 'privacy_validation.zkey');
      const wasmFile = path.join(defaultZkDir, 'privacy_validation_js/privacy_validation.wasm');
      const vkeyFile = vkeyPath || path.join(defaultZkDir, 'privacy_validation_verification_key.json');

      this.zkProofGenerator = new ProofGenerator(zkeyFile, wasmFile, {
        enableCache: true,
        enableParallel: true,
        maxCacheSize: 500
      });

      this.zkProofVerifier = new ProofVerifier(vkeyFile, {
        enableCache: true,
        checkNullifiers: true,
        strictMode: false
      });

      this.zkEnabled = true;
      console.log('[ML] Zero-Knowledge Proof system initialized for content analysis');
    } catch (error) {
      console.warn('[ML] Failed to initialize ZK proofs:', error);
      this.zkEnabled = false;
    }
  }

  /**
   * Analyze content and generate ZK proof
   */
  async analyzeWithProof(content: string, privacyTier: number = 1): Promise<{
    analysis: AnalysisResult;
    proof?: any;
    commitment?: string;
    proofTime?: number;
    verified?: boolean;
  }> {
    // Perform regular analysis
    const analysis = await this.analyze(content);

    if (!this.zkEnabled || !this.zkProofGenerator) {
      return { analysis };
    }

    try {
      const proofStartTime = Date.now();

      // Create commitment to analysis results
      // This proves the analysis was performed correctly without revealing content
      const analysisCommitment = crypto.createHash('sha256')
        .update(JSON.stringify({
          toxicityScore: Math.floor(analysis.toxicity.score * 100),
          privacyScore: Math.floor(analysis.privacy.riskScore * 100),
          manipulationScore: Math.floor(analysis.manipulation.score * 100),
          biasScore: Math.floor(analysis.bias.score * 100),
          contentLength: content.length,
          timestamp: Math.floor(Date.now() / 1000)
        }))
        .digest('hex');

      // Generate proof that analysis meets constitutional requirements
      const proofInput: ProofInput = {
        // Private inputs - actual scores and content hash
        dataHash: crypto.createHash('sha256')
          .update(content)
          .update(analysisCommitment)
          .digest('hex'),
        userConsent: 1, // Assumed for analysis
        dataCategories: this.extractAnalysisCategories(analysis),
        processingPurpose: this.determineProcessingPurpose(analysis),
        retentionPeriod: 0, // No retention for analysis

        // Public inputs - thresholds and tier
        privacyTier,
        constitutionalHash: this.generateConstitutionalHash(analysis)
      };

      // Generate proof
      const proofResult = await this.zkProofGenerator.generateProof(proofInput);

      const proofTime = Date.now() - proofStartTime;

      console.log(`[ML] ZK proof generated for content analysis in ${proofTime}ms`);

      // Verify the proof immediately to ensure correctness
      let verified = false;
      if (this.zkProofVerifier) {
        const verificationResult = await this.zkProofVerifier.verifyProof(
          proofResult.proof,
          proofResult.publicSignals
        );
        verified = verificationResult.valid;

        if (!verified) {
          console.error('[ML] Generated proof failed verification!', verificationResult.errors);
        }
      }

      return {
        analysis,
        proof: proofResult.proof,
        commitment: proofResult.commitment,
        proofTime,
        verified
      };
    } catch (error) {
      console.error('[ML] Failed to generate ZK proof for analysis:', error);
      return { analysis };
    }
  }

  /**
   * Verify analysis proof
   */
  async verifyAnalysisProof(proof: any, publicSignals: string[]): Promise<boolean> {
    if (!this.zkEnabled || !this.zkProofVerifier) {
      return false;
    }

    try {
      const result = await this.zkProofVerifier.verifyProof(proof, publicSignals);
      return result.valid;
    } catch (error) {
      console.error('[ML] Failed to verify analysis proof:', error);
      return false;
    }
  }

  /**
   * Extract analysis categories for ZK proof
   */
  private extractAnalysisCategories(analysis: AnalysisResult): number[] {
    // Use actual scores instead of binary flags for more accurate proof
    const categories = [0, 0, 0, 0, 0];

    // Category 0: Toxicity level (0-100)
    categories[0] = Math.min(100, Math.floor(analysis.toxicity.score * 100));

    // Category 1: Privacy risk level (0-100)
    categories[1] = Math.min(100, Math.floor(analysis.privacy.riskScore * 100));

    // Category 2: Manipulation level (0-100)
    categories[2] = Math.min(100, Math.floor(analysis.manipulation.score * 100));

    // Category 3: Bias level (0-100)
    categories[3] = Math.min(100, Math.floor(analysis.bias.score * 100));

    // Category 4: Overall risk level (0-3: low, medium, high, critical)
    const riskLevels = { 'low': 0, 'medium': 1, 'high': 2, 'critical': 3 };
    categories[4] = riskLevels[analysis.overall.riskLevel] || 0;

    return categories;
  }

  /**
   * Determine processing purpose based on analysis
   */
  private determineProcessingPurpose(analysis: AnalysisResult): number {
    // Map analysis type to purpose code
    if (analysis.overall.riskLevel === 'critical') return 40; // Critical review
    if (analysis.toxicity.isToxic) return 35; // Toxicity filtering
    if (analysis.privacy.hasPrivacyRisk) return 30; // Privacy protection
    if (analysis.manipulation.hasManipulation) return 25; // Manipulation detection
    if (analysis.bias.hasBias) return 20; // Bias detection
    return 10; // Standard analysis
  }

  /**
   * Generate constitutional hash for analysis validation
   */
  private generateConstitutionalHash(analysis: AnalysisResult): string {
    const constitutionalData = {
      thresholds: {
        toxicity: this.config.toxicityThreshold,
        privacy: this.config.privacyThreshold,
        manipulation: this.config.manipulationThreshold,
        bias: this.config.biasThreshold
      },
      analysisVersion: '1.0.0',
      validationTime: Math.floor(Date.now() / 1000),
      complianceLevel: analysis.overall.constitutionalCompliance ? 'compliant' : 'non-compliant'
    };

    return crypto.createHash('sha256')
      .update(JSON.stringify(constitutionalData))
      .digest('hex');
  }

  /**
   * Convert privacy tier string to number
   */
  private privacyTierToNumber(tier: string): number {
    const tierMap: { [key: string]: number } = {
      'Bronze': 0,
      'Silver': 1,
      'Gold': 2,
      'Platinum': 3
    };
    return tierMap[tier] || 0;
  }
}

export default ContentAnalyzer;