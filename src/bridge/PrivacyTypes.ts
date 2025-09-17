// Privacy Management Type Definitions
// Supporting types and interfaces for the Constitutional Privacy Manager

export interface PrivacyConfiguration {
  defaultTier: 'standard' | 'enhanced' | 'maximum';
  enableAutomaticClassification: boolean;
  enableKeyRotation: boolean;
  enableComplianceMonitoring: boolean;
  auditRetentionPeriod: number; // days
  maxDataRetentionPeriod: number; // days
  encryptionAlgorithms: {
    standard: string;
    enhanced: string;
    maximum: string;
  };
}

export interface DataProcessingContext {
  requestId: string;
  userId?: string;
  sessionId?: string;
  ipAddress?: string;
  userAgent?: string;
  timestamp: Date;
  purpose: string;
  legalBasis: 'consent' | 'contract' | 'legal_obligation' | 'vital_interests' | 'public_task' | 'legitimate_interests';
  dataMinimization: boolean;
  automated: boolean;
  thirdPartySharing: boolean;
  crossBorderTransfer: boolean;
}

export interface PrivacyRisk {
  id: string;
  type: 'data_breach' | 'unauthorized_access' | 'data_leak' | 'compliance_violation' | 'retention_violation';
  severity: 'low' | 'medium' | 'high' | 'critical';
  likelihood: 'rare' | 'unlikely' | 'possible' | 'likely' | 'almost_certain';
  impact: string;
  mitigation: string[];
  owner: string;
  dueDate: Date;
  status: 'open' | 'in_progress' | 'mitigated' | 'accepted';
}

export interface ComplianceFramework {
  name: 'GDPR' | 'CCPA' | 'HIPAA' | 'COPPA' | 'PIPEDA' | 'LGPD';
  version: string;
  applicableRegions: string[];
  requirements: ComplianceRequirement[];
  penalties: {
    administrative: string;
    criminal: string;
    civil: string;
  };
}

export interface ComplianceRequirement {
  id: string;
  title: string;
  description: string;
  mandatory: boolean;
  applicableDataTypes: string[];
  verificationMethod: 'automated' | 'manual' | 'audit';
  frequency: 'continuous' | 'daily' | 'weekly' | 'monthly' | 'annually';
}

export interface PrivacyMetrics {
  dataClassified: number;
  consentGranted: number;
  consentWithdrawn: number;
  dataMinimized: number;
  keysRotated: number;
  complianceViolations: number;
  auditEvents: number;
  privacyRisks: number;
  averageProcessingTime: number; // milliseconds
  encryptionCoverage: number; // percentage
}

export interface AnonymizationTechnique {
  name: 'k_anonymity' | 'l_diversity' | 't_closeness' | 'differential_privacy' | 'synthetic_data';
  parameters: Record<string, any>;
  applicableDataTypes: string[];
  privacyGuarantees: string[];
  utilityLoss: number; // percentage
}

export interface ZeroKnowledgeProtocol {
  type: 'zk_snark' | 'zk_stark' | 'bulletproof' | 'plonk';
  circuitSize: number;
  proofSize: number;
  verificationTime: number; // milliseconds
  setupTrusted: boolean;
  quantumResistant: boolean;
}

export interface PrivacyByDesignPrinciple {
  name: string;
  description: string;
  implementation: string[];
  verification: string[];
  metrics: string[];
}

export interface DataSubjectRight {
  right: 'access' | 'rectification' | 'erasure' | 'portability' | 'restriction' | 'objection';
  description: string;
  legalBasis: string[];
  timeLimit: number; // days
  exceptions: string[];
  verificationRequired: boolean;
}

export interface PrivacyImpactAssessment {
  id: string;
  projectName: string;
  dataController: string;
  dataProcessor?: string;
  dataTypes: string[];
  processingPurposes: string[];
  legalBasis: string[];
  risks: PrivacyRisk[];
  safeguards: string[];
  consultationRequired: boolean;
  approvalRequired: boolean;
  reviewDate: Date;
  status: 'draft' | 'review' | 'approved' | 'rejected';
}

export interface ConsentManagementPlatform {
  platformId: string;
  version: string;
  supportedStandards: string[]; // IAB TCF, Google CMP, etc.
  consentString: string;
  vendors: ConsentVendor[];
  purposes: ConsentPurpose[];
  legitimateInterests: string[];
  timestamp: Date;
}

export interface ConsentVendor {
  id: string;
  name: string;
  privacyPolicyUrl: string;
  purposes: number[];
  legitimateInterests: number[];
  features: number[];
  specialFeatures: number[];
}

export interface ConsentPurpose {
  id: number;
  name: string;
  description: string;
  legalBasis: 'consent' | 'legitimate_interest';
  required: boolean;
}

export interface EncryptionKeyMetadata {
  keyId: string;
  algorithm: string;
  keySize: number;
  createdAt: Date;
  rotatedAt?: Date;
  expiresAt: Date;
  usage: 'encryption' | 'signing' | 'key_exchange';
  tier: 'standard' | 'enhanced' | 'maximum';
  status: 'active' | 'rotated' | 'revoked' | 'expired';
}

export interface AuditConfiguration {
  level: 'minimal' | 'standard' | 'detailed' | 'comprehensive';
  retentionPeriod: number; // days
  anonymizeAfter: number; // days
  encryptLogs: boolean;
  realTimeMonitoring: boolean;
  alertThresholds: {
    suspiciousActivity: number;
    complianceViolations: number;
    dataBreaches: number;
  };
}

export interface PrivacyDashboard {
  userId: string;
  personalDataSummary: {
    dataTypes: string[];
    purposes: string[];
    retentionPeriods: Record<string, number>;
    thirdParties: string[];
  };
  consentStatus: {
    granted: ConsentRecord[];
    withdrawn: ConsentRecord[];
    pending: ConsentRecord[];
  };
  dataSubjectRequests: {
    pending: DataSubjectRequest[];
    completed: DataSubjectRequest[];
  };
  privacySettings: {
    marketingOptIn: boolean;
    analyticsOptIn: boolean;
    personalizationOptIn: boolean;
    dataSharing: boolean;
  };
}

export interface DataSubjectRequest {
  id: string;
  userId: string;
  type: 'access' | 'rectification' | 'erasure' | 'portability' | 'restriction' | 'objection';
  description: string;
  submittedAt: Date;
  dueDate: Date;
  status: 'submitted' | 'processing' | 'completed' | 'rejected';
  response?: string;
  attachments?: string[];
}

export interface PrivacyNotice {
  id: string;
  title: string;
  content: string;
  version: string;
  effectiveDate: Date;
  expiryDate?: Date;
  targetAudience: string[];
  languages: string[];
  channels: ('email' | 'sms' | 'push' | 'in_app' | 'website')[];
  mandatory: boolean;
  acknowledgmentRequired: boolean;
}

export interface DataFlowMapping {
  id: string;
  dataType: string;
  source: string;
  destination: string[];
  purpose: string;
  legalBasis: string;
  dataSubjects: string[];
  retentionPeriod: number;
  crossBorderTransfer: boolean;
  safeguards: string[];
  risks: string[];
}

export interface PrivacyTrainingRecord {
  employeeId: string;
  trainingModule: string;
  completedAt: Date;
  score: number;
  certificateId?: string;
  expiryDate: Date;
  refreshRequired: boolean;
}

export interface IncidentResponse {
  incidentId: string;
  type: 'data_breach' | 'unauthorized_access' | 'data_loss' | 'system_compromise';
  severity: 'low' | 'medium' | 'high' | 'critical';
  reportedAt: Date;
  detectedAt: Date;
  containedAt?: Date;
  resolvedAt?: Date;
  affectedDataSubjects: number;
  affectedRecords: number;
  notificationRequired: {
    authority: boolean;
    dataSubjects: boolean;
    deadline: Date;
  };
  rootCause: string;
  remedialActions: string[];
  lessonsLearned: string[];
}

// Export all interfaces and types
export {
  PrivacyTier,
  DataSensitivity,
  DataCategory,
  PrivacyPolicy,
  ConsentRecord,
  DataClassification,
  AuditEntry,
  EncryptionConfig,
  ZKProofConfig
} from './PrivacyManager';