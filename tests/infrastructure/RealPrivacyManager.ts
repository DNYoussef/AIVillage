/**
 * Real Privacy Manager for Testing
 * Implements actual privacy validation logic without mocks
 */

import crypto from 'crypto';

export interface PrivacyRequest {
  privacyTier: 'Bronze' | 'Silver' | 'Gold' | 'Platinum';
  data: any;
  userContext: {
    userId?: string;
    trustScore?: number;
    roles?: string[];
  };
}

export interface PrivacyValidationResult {
  isValid: boolean;
  reason?: string;
  transformedData?: any;
  violations?: string[];
  auditLog?: AuditEntry[];
}

interface AuditEntry {
  timestamp: number;
  action: string;
  result: string;
  metadata: any;
}

interface PIIPattern {
  name: string;
  pattern: RegExp;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export class RealPrivacyManager {
  private piiPatterns: PIIPattern[] = [
    {
      name: 'SSN',
      pattern: /\b\d{3}-\d{2}-\d{4}\b/,
      severity: 'critical'
    },
    {
      name: 'Credit Card',
      pattern: /\b(?:\d{4}[-\s]?){3}\d{4}\b/,
      severity: 'critical'
    },
    {
      name: 'Email',
      pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/,
      severity: 'medium'
    },
    {
      name: 'Phone',
      pattern: /\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/,
      severity: 'medium'
    },
    {
      name: 'IP Address',
      pattern: /\b(?:\d{1,3}\.){3}\d{1,3}\b/,
      severity: 'low'
    }
  ];

  private tierRequirements = {
    Bronze: {
      minTrustScore: 0.3,
      maxDataSensitivity: 'low',
      allowedPII: [],
      transformations: ['heavy_masking', 'aggregation', 'deletion']
    },
    Silver: {
      minTrustScore: 0.5,
      maxDataSensitivity: 'medium',
      allowedPII: ['Email'],
      transformations: ['moderate_masking', 'generalization']
    },
    Gold: {
      minTrustScore: 0.7,
      maxDataSensitivity: 'high',
      allowedPII: ['Email', 'Phone'],
      transformations: ['light_masking']
    },
    Platinum: {
      minTrustScore: 0.9,
      maxDataSensitivity: 'critical',
      allowedPII: ['Email', 'Phone', 'SSN', 'Credit Card'],
      transformations: []
    }
  };

  private auditLog: AuditEntry[] = [];

  async validateRequest(request: PrivacyRequest): Promise<PrivacyValidationResult> {
    const startTime = Date.now();
    const violations: string[] = [];

    // Validate trust score
    const trustScore = request.userContext.trustScore || 0;
    const requiredScore = this.tierRequirements[request.privacyTier].minTrustScore;

    if (trustScore < requiredScore) {
      violations.push(
        `Trust score ${trustScore} below required ${requiredScore} for ${request.privacyTier} tier`
      );
    }

    // Detect PII in data
    const detectedPII = this.detectPII(request.data);

    // Check PII against tier allowances
    const allowedPII = this.tierRequirements[request.privacyTier].allowedPII;
    for (const pii of detectedPII) {
      if (!allowedPII.includes(pii.name)) {
        violations.push(
          `${pii.name} detected but not allowed in ${request.privacyTier} tier`
        );
      }
    }

    // Check data sensitivity
    const sensitivity = this.calculateDataSensitivity(request.data, detectedPII);
    const maxSensitivity = this.tierRequirements[request.privacyTier].maxDataSensitivity;

    if (this.compareSensitivity(sensitivity, maxSensitivity) > 0) {
      violations.push(
        `Data sensitivity ${sensitivity} exceeds maximum ${maxSensitivity} for tier`
      );
    }

    // Audit the validation
    this.auditLog.push({
      timestamp: Date.now(),
      action: 'validateRequest',
      result: violations.length === 0 ? 'approved' : 'denied',
      metadata: {
        tier: request.privacyTier,
        trustScore,
        detectedPII: detectedPII.map(p => p.name),
        violations,
        duration: Date.now() - startTime
      }
    });

    return {
      isValid: violations.length === 0,
      violations,
      reason: violations.join('; '),
      auditLog: [...this.auditLog]
    };
  }

  async applyPrivacyTier(request: PrivacyRequest, tier: string): Promise<any> {
    const transformations = this.tierRequirements[tier].transformations;
    let transformedData = JSON.parse(JSON.stringify(request.data)); // Deep clone

    for (const transformation of transformations) {
      transformedData = this.applyTransformation(transformedData, transformation);
    }

    // Verify transformation effectiveness
    const remainingPII = this.detectPII(transformedData);
    const allowedPII = this.tierRequirements[tier].allowedPII;

    for (const pii of remainingPII) {
      if (!allowedPII.includes(pii.name)) {
        // Additional masking needed
        transformedData = this.maskPII(transformedData, pii);
      }
    }

    return {
      data: transformedData,
      privacyTier: tier,
      transformationsApplied: transformations,
      timestamp: Date.now()
    };
  }

  async filterResponse(response: any, tier: string): Promise<any> {
    const filtered = JSON.parse(JSON.stringify(response)); // Deep clone

    switch (tier) {
      case 'Bronze':
        // Remove all sensitive fields
        delete filtered.data?.privateInfo;
        delete filtered.data?.internalInfo;
        delete filtered.metadata;
        break;

      case 'Silver':
        // Remove internal fields
        delete filtered.data?.internalInfo;
        // Generalize location data
        if (filtered.data?.location) {
          filtered.data.location = this.generalizeLocation(filtered.data.location);
        }
        break;

      case 'Gold':
        // Remove only internal debug info
        delete filtered.debug;
        break;

      case 'Platinum':
        // No filtering
        break;
    }

    return filtered;
  }

  private detectPII(data: any): PIIPattern[] {
    const detected: PIIPattern[] = [];
    const dataStr = JSON.stringify(data);

    for (const pattern of this.piiPatterns) {
      if (pattern.pattern.test(dataStr)) {
        detected.push(pattern);
      }
    }

    return detected;
  }

  private calculateDataSensitivity(data: any, detectedPII: PIIPattern[]): string {
    if (detectedPII.some(p => p.severity === 'critical')) {
      return 'critical';
    }
    if (detectedPII.some(p => p.severity === 'high')) {
      return 'high';
    }
    if (detectedPII.some(p => p.severity === 'medium')) {
      return 'medium';
    }
    return 'low';
  }

  private compareSensitivity(a: string, b: string): number {
    const levels = { low: 0, medium: 1, high: 2, critical: 3 };
    return levels[a] - levels[b];
  }

  private applyTransformation(data: any, transformation: string): any {
    switch (transformation) {
      case 'heavy_masking':
        return this.heavyMask(data);
      case 'moderate_masking':
        return this.moderateMask(data);
      case 'light_masking':
        return this.lightMask(data);
      case 'aggregation':
        return this.aggregate(data);
      case 'generalization':
        return this.generalize(data);
      case 'deletion':
        return this.deleteSensitive(data);
      default:
        return data;
    }
  }

  private heavyMask(data: any): any {
    if (typeof data === 'string') {
      // Replace all but first and last character
      if (data.length > 2) {
        return data[0] + '*'.repeat(data.length - 2) + data[data.length - 1];
      }
      return '***';
    }

    if (typeof data === 'object' && data !== null) {
      const masked = {};
      for (const [key, value] of Object.entries(data)) {
        if (this.isSensitiveField(key)) {
          masked[key] = '***REDACTED***';
        } else {
          masked[key] = this.heavyMask(value);
        }
      }
      return masked;
    }

    return data;
  }

  private moderateMask(data: any): any {
    if (typeof data === 'string') {
      // Mask middle portion
      if (data.length > 4) {
        const visibleChars = Math.floor(data.length / 3);
        return (
          data.substring(0, visibleChars) +
          '*'.repeat(data.length - 2 * visibleChars) +
          data.substring(data.length - visibleChars)
        );
      }
      return '***';
    }

    if (typeof data === 'object' && data !== null) {
      const masked = {};
      for (const [key, value] of Object.entries(data)) {
        masked[key] = this.isSensitiveField(key) ? '***' : this.moderateMask(value);
      }
      return masked;
    }

    return data;
  }

  private lightMask(data: any): any {
    if (typeof data === 'string' && data.includes('@')) {
      // Email masking
      const parts = data.split('@');
      if (parts.length === 2) {
        return parts[0].substring(0, 2) + '***@' + parts[1];
      }
    }

    if (typeof data === 'object' && data !== null) {
      const masked = {};
      for (const [key, value] of Object.entries(data)) {
        masked[key] = value;
      }
      return masked;
    }

    return data;
  }

  private aggregate(data: any): any {
    if (Array.isArray(data)) {
      return {
        count: data.length,
        summary: 'Aggregated data'
      };
    }

    if (typeof data === 'object' && data !== null) {
      const aggregated = {};
      for (const [key, value] of Object.entries(data)) {
        if (Array.isArray(value)) {
          aggregated[key] = { count: value.length };
        } else if (typeof value === 'number') {
          aggregated[key] = Math.round(value / 10) * 10; // Round to nearest 10
        } else {
          aggregated[key] = this.aggregate(value);
        }
      }
      return aggregated;
    }

    return data;
  }

  private generalize(data: any): any {
    if (typeof data === 'object' && data !== null) {
      const generalized = {};

      for (const [key, value] of Object.entries(data)) {
        if (key === 'location' && typeof value === 'object') {
          generalized[key] = this.generalizeLocation(value);
        } else if (key === 'age' && typeof value === 'number') {
          generalized[key] = Math.floor(value / 10) * 10 + '-' + (Math.floor(value / 10) * 10 + 9);
        } else if (key === 'timestamp' && typeof value === 'number') {
          // Generalize to day
          generalized[key] = Math.floor(value / 86400000) * 86400000;
        } else {
          generalized[key] = value;
        }
      }

      return generalized;
    }

    return data;
  }

  private generalizeLocation(location: any): any {
    if (location.lat && location.lng) {
      // Round to 1 decimal place (roughly 11km accuracy)
      return {
        lat: Math.round(location.lat * 10) / 10,
        lng: Math.round(location.lng * 10) / 10,
        accuracy: '11km'
      };
    }

    if (location.address) {
      // Keep only city and country
      return {
        city: location.city || 'Unknown',
        country: location.country || 'Unknown'
      };
    }

    return location;
  }

  private deleteSensitive(data: any): any {
    if (typeof data === 'object' && data !== null) {
      const cleaned = {};

      for (const [key, value] of Object.entries(data)) {
        if (!this.isSensitiveField(key)) {
          cleaned[key] = typeof value === 'object' ? this.deleteSensitive(value) : value;
        }
      }

      return cleaned;
    }

    return data;
  }

  private maskPII(data: any, pattern: PIIPattern): any {
    const dataStr = JSON.stringify(data);
    const masked = dataStr.replace(pattern.pattern, (match) => {
      return '*'.repeat(match.length);
    });
    return JSON.parse(masked);
  }

  private isSensitiveField(fieldName: string): boolean {
    const sensitiveFields = [
      'password', 'secret', 'token', 'key', 'ssn',
      'credit_card', 'creditCard', 'cvv', 'pin'
    ];

    return sensitiveFields.some(field =>
      fieldName.toLowerCase().includes(field.toLowerCase())
    );
  }

  getAuditLog(): AuditEntry[] {
    return [...this.auditLog];
  }

  clearAuditLog(): void {
    this.auditLog = [];
  }
}

export default RealPrivacyManager;