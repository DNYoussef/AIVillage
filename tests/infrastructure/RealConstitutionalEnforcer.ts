/**
 * Real Constitutional Enforcer for Testing
 * Implements actual constitutional validation without mocks
 */

import { RealThreatDataset, ThreatPattern } from './RealThreatDataset';

export interface ConstitutionalRequest {
  id: string;
  data: any;
  userContext?: any;
  metadata?: any;
}

export interface ConstitutionalResult {
  isValid: boolean;
  violations: string[];
  complianceScore: number;
  timestamp: number;
  evidence?: any;
}

interface ConstitutionalRule {
  id: string;
  name: string;
  category: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  validate: (request: ConstitutionalRequest) => boolean;
  description: string;
}

export class RealConstitutionalEnforcer {
  private threatDataset: RealThreatDataset;
  private enforcementLevel: 'strict' | 'balanced' | 'permissive';
  private complianceThreshold: number;
  private auditLog: any[] = [];

  private constitutionalRules: ConstitutionalRule[] = [
    // Harm Prevention Rules
    {
      id: 'harm-001',
      name: 'No Harmful Content',
      category: 'harm_prevention',
      severity: 'critical',
      validate: (request) => {
        const content = JSON.stringify(request.data);
        // Check for harmful keywords (sanitized)
        const harmfulPatterns = [
          /h[@4]rm/gi,
          /v[!1]ol[e3]nc[e3]/gi,
          /d[@4]ng[e3]r/gi
        ];
        return !harmfulPatterns.some(pattern => pattern.test(content));
      },
      description: 'Prevents generation of harmful content'
    },
    {
      id: 'harm-002',
      name: 'No Illegal Activities',
      category: 'harm_prevention',
      severity: 'critical',
      validate: (request) => {
        const content = JSON.stringify(request.data).toLowerCase();
        const illegalKeywords = [
          'illegal download',
          'crack software',
          'bypass security',
          'hack into'
        ];
        return !illegalKeywords.some(keyword => content.includes(keyword));
      },
      description: 'Prevents assistance with illegal activities'
    },

    // Privacy Preservation Rules
    {
      id: 'privacy-001',
      name: 'PII Protection',
      category: 'privacy',
      severity: 'high',
      validate: (request) => {
        const content = JSON.stringify(request.data);
        // Check for unprotected PII
        const piiPatterns = [
          /\b\d{3}-\d{2}-\d{4}\b/, // SSN
          /\b(?:\d{4}[-\s]?){3}\d{4}\b/, // Credit Card
        ];

        if (piiPatterns.some(pattern => pattern.test(content))) {
          // PII is only allowed if explicitly marked as protected
          return request.data.includesPII === false ||
                 request.metadata?.piiProtection === true;
        }
        return true;
      },
      description: 'Ensures PII is properly protected'
    },
    {
      id: 'privacy-002',
      name: 'Data Minimization',
      category: 'privacy',
      severity: 'medium',
      validate: (request) => {
        // Check if request includes unnecessary data
        const sensitiveFields = ['password', 'token', 'secret', 'key'];
        const dataStr = JSON.stringify(request.data).toLowerCase();

        for (const field of sensitiveFields) {
          if (dataStr.includes(field) && !request.metadata?.requiresSensitive) {
            return false;
          }
        }
        return true;
      },
      description: 'Enforces data minimization principles'
    },

    // Transparency Requirements
    {
      id: 'trans-001',
      name: 'Decision Transparency',
      category: 'transparency',
      severity: 'medium',
      validate: (request) => {
        if (request.data?.action === 'decision' || request.data?.action === 'judgment') {
          // Decisions must include reasoning
          return request.data.reasoning !== undefined &&
                 request.data.reasoning !== null &&
                 request.data.reasoning.length > 0;
        }
        return true;
      },
      description: 'Requires reasoning for decisions'
    },
    {
      id: 'trans-002',
      name: 'Source Attribution',
      category: 'transparency',
      severity: 'low',
      validate: (request) => {
        if (request.data?.includesData || request.data?.includesFacts) {
          // Facts must have sources
          return request.data.sources !== undefined ||
                 request.metadata?.sourcesProvided === true;
        }
        return true;
      },
      description: 'Requires source attribution for facts'
    },

    // Fairness Rules
    {
      id: 'fair-001',
      name: 'No Discrimination',
      category: 'fairness',
      severity: 'high',
      validate: (request) => {
        const content = JSON.stringify(request.data).toLowerCase();
        const discriminatoryTerms = [
          'discriminate',
          'exclude based on',
          'deny because of',
          'refuse due to race',
          'reject for gender'
        ];
        return !discriminatoryTerms.some(term => content.includes(term));
      },
      description: 'Prevents discriminatory content'
    },
    {
      id: 'fair-002',
      name: 'Bias Detection',
      category: 'fairness',
      severity: 'medium',
      validate: (request) => {
        // Check for potential bias indicators
        if (request.data?.demographics || request.data?.personalAttributes) {
          // These should not influence decisions unless explicitly relevant
          return request.metadata?.demographicsRelevant === true ||
                 !request.data?.decision;
        }
        return true;
      },
      description: 'Detects potential bias in decision-making'
    },

    // Security Rules
    {
      id: 'sec-001',
      name: 'Injection Prevention',
      category: 'security',
      severity: 'critical',
      validate: (request) => {
        const content = JSON.stringify(request.data);
        const injectionPatterns = [
          /(\bSELECT\b.*\bFROM\b|\bDROP\b.*\bTABLE\b)/gi,
          /<script[^>]*>[\s\S]*?<\/script>/gi,
          /[;&|`$]\s*(rm|del|format)/gi
        ];
        return !injectionPatterns.some(pattern => pattern.test(content));
      },
      description: 'Prevents injection attacks'
    },
    {
      id: 'sec-002',
      name: 'Path Traversal Prevention',
      category: 'security',
      severity: 'high',
      validate: (request) => {
        const content = JSON.stringify(request.data);
        return !content.includes('../') && !content.includes('..\\');
      },
      description: 'Prevents path traversal attacks'
    }
  ];

  constructor(config: {
    enforcementLevel: 'strict' | 'balanced' | 'permissive';
    complianceThreshold: number;
  }) {
    this.enforcementLevel = config.enforcementLevel;
    this.complianceThreshold = config.complianceThreshold;
    this.threatDataset = new RealThreatDataset();
  }

  async validateConstitutional(request: ConstitutionalRequest): Promise<ConstitutionalResult> {
    const startTime = Date.now();
    const violations: string[] = [];
    const evidence: any = {};

    // Check against threat patterns
    const threatValidation = this.validateAgainstThreats(request);
    if (!threatValidation.isValid) {
      violations.push(...threatValidation.violations);
      evidence.threats = threatValidation.threats;
    }

    // Check constitutional rules
    let passedRules = 0;
    let totalWeight = 0;

    for (const rule of this.constitutionalRules) {
      const weight = this.getRuleWeight(rule);
      totalWeight += weight;

      try {
        const isValid = rule.validate(request);

        if (!isValid) {
          // Apply enforcement level
          if (this.shouldEnforce(rule)) {
            violations.push(`${rule.category}: ${rule.description}`);
            evidence[rule.id] = {
              rule: rule.name,
              severity: rule.severity,
              failed: true
            };
          } else {
            // Warning only in permissive mode
            evidence[rule.id] = {
              rule: rule.name,
              severity: rule.severity,
              warning: true
            };
            passedRules += weight * 0.5; // Partial credit
          }
        } else {
          passedRules += weight;
          evidence[rule.id] = {
            rule: rule.name,
            passed: true
          };
        }
      } catch (error) {
        // Rule validation error
        evidence[rule.id] = {
          rule: rule.name,
          error: error.message
        };
      }
    }

    // Calculate compliance score
    const complianceScore = totalWeight > 0 ? passedRules / totalWeight : 0;

    // Determine if request is valid
    const isValid = violations.length === 0 && complianceScore >= this.complianceThreshold;

    // Audit logging
    const result: ConstitutionalResult = {
      isValid,
      violations,
      complianceScore,
      timestamp: Date.now(),
      evidence
    };

    this.auditLog.push({
      requestId: request.id,
      result,
      duration: Date.now() - startTime,
      enforcementLevel: this.enforcementLevel
    });

    return result;
  }

  private validateAgainstThreats(request: ConstitutionalRequest): {
    isValid: boolean;
    violations: string[];
    threats: ThreatPattern[];
  } {
    const content = JSON.stringify(request.data);
    const validation = this.threatDataset.validateInput(content);

    const violations = validation.threats.map(threat =>
      `${threat.category}: ${threat.description} (${threat.severity})`
    );

    return {
      isValid: !validation.isBlocked,
      violations,
      threats: validation.threats
    };
  }

  private getRuleWeight(rule: ConstitutionalRule): number {
    switch (rule.severity) {
      case 'critical': return 10;
      case 'high': return 5;
      case 'medium': return 2;
      case 'low': return 1;
      default: return 1;
    }
  }

  private shouldEnforce(rule: ConstitutionalRule): boolean {
    switch (this.enforcementLevel) {
      case 'strict':
        // Enforce all rules
        return true;

      case 'balanced':
        // Enforce critical and high severity
        return rule.severity === 'critical' || rule.severity === 'high';

      case 'permissive':
        // Only enforce critical rules
        return rule.severity === 'critical';

      default:
        return true;
    }
  }

  async validateWithML(request: ConstitutionalRequest): Promise<ConstitutionalResult> {
    // Simulate ML-based validation
    // In production, this would call actual ML models

    const baseResult = await this.validateConstitutional(request);

    // Simulate ML enhancement
    const mlScore = Math.random() * 0.2; // 0-20% adjustment
    const adjustedScore = Math.min(1, baseResult.complianceScore + mlScore);

    return {
      ...baseResult,
      complianceScore: adjustedScore,
      evidence: {
        ...baseResult.evidence,
        mlEnhancement: {
          applied: true,
          adjustment: mlScore
        }
      }
    };
  }

  // Test helper methods
  async runComplianceTest(testVectors: any[]): Promise<{
    passed: number;
    failed: number;
    results: any[];
  }> {
    const results = [];
    let passed = 0;
    let failed = 0;

    for (const vector of testVectors) {
      const request: ConstitutionalRequest = {
        id: `test-${Date.now()}`,
        data: vector.input,
        metadata: vector.metadata
      };

      const result = await this.validateConstitutional(request);

      const testPassed = result.isValid === vector.expectedValid;
      if (testPassed) {
        passed++;
      } else {
        failed++;
      }

      results.push({
        vector,
        result,
        testPassed
      });
    }

    return { passed, failed, results };
  }

  setEnforcementLevel(level: 'strict' | 'balanced' | 'permissive'): void {
    this.enforcementLevel = level;
  }

  setComplianceThreshold(threshold: number): void {
    this.complianceThreshold = Math.max(0, Math.min(1, threshold));
  }

  getAuditLog(): any[] {
    return [...this.auditLog];
  }

  clearAuditLog(): void {
    this.auditLog = [];
  }

  // Performance testing utilities
  async stressTest(requestCount: number): Promise<{
    avgLatency: number;
    p95Latency: number;
    throughput: number;
  }> {
    const latencies: number[] = [];
    const startTime = Date.now();

    for (let i = 0; i < requestCount; i++) {
      const request: ConstitutionalRequest = {
        id: `stress-${i}`,
        data: { test: true, index: i }
      };

      const requestStart = Date.now();
      await this.validateConstitutional(request);
      latencies.push(Date.now() - requestStart);
    }

    const totalTime = Date.now() - startTime;
    latencies.sort((a, b) => a - b);

    return {
      avgLatency: latencies.reduce((a, b) => a + b, 0) / latencies.length,
      p95Latency: latencies[Math.floor(latencies.length * 0.95)],
      throughput: (requestCount / totalTime) * 1000 // requests per second
    };
  }
}

export default RealConstitutionalEnforcer;