/**
 * BetaNet Protocol Adapter Type Definitions
 * Comprehensive TypeScript types for protocol translation
 */

// Core Configuration Types
export interface BetaNetAdapterConfig {
  connectionPool: {
    maxConnections: number;
    maxIdleTime: number;
    cleanupInterval: number;
  };
  circuitBreaker: {
    failureThreshold: number;
    resetTimeout: number;
    monitoringPeriod: number;
  };
  performance: {
    targetLatencyP95: number;
    maxRetries: number;
    timeoutMs: number;
  };
  security: {
    encryptionEnabled: boolean;
    compressionEnabled: boolean;
    defaultSecurityLevel: SecurityLevel;
  };
  protocol: {
    version: string;
    supportedVersions: string[];
    maxFragmentSize: number;
  };
}

// Extended Message Types
export interface BetaNetHandshakeMessage extends BetaNetMessage {
  type: BetaNetMessageType.HANDSHAKE;
  payload: {
    protocolVersion: string;
    capabilities: string[];
    nodeId: string;
    publicKey?: string;
  };
}

export interface BetaNetDiscoveryMessage extends BetaNetMessage {
  type: BetaNetMessageType.DISCOVERY;
  payload: {
    serviceType: string;
    nodeCapabilities: string[];
    networkTopology: NetworkTopology;
  };
}

export interface BetaNetDataMessage extends BetaNetMessage {
  type: BetaNetMessageType.DATA_TRANSFER;
  payload: {
    data: unknown;
    format: DataFormat;
    encoding: string;
    checksum: string;
  };
}

export interface BetaNetControlMessage extends BetaNetMessage {
  type: BetaNetMessageType.CONTROL;
  payload: {
    command: ControlCommand;
    parameters: Record<string, unknown>;
    requiresAck: boolean;
  };
}

export interface BetaNetErrorMessage extends BetaNetMessage {
  type: BetaNetMessageType.ERROR;
  payload: {
    errorCode: ErrorCode;
    errorMessage: string;
    originalMessageId?: string;
    stackTrace?: string;
  };
}

// Enums and Constants
export enum NetworkTopology {
  MESH = 'mesh',
  STAR = 'star',
  RING = 'ring',
  TREE = 'tree',
  HYBRID = 'hybrid'
}

export enum DataFormat {
  JSON = 'json',
  XML = 'xml',
  BINARY = 'binary',
  PROTOBUF = 'protobuf',
  MSGPACK = 'msgpack'
}

export enum ControlCommand {
  CONNECT = 'connect',
  DISCONNECT = 'disconnect',
  PAUSE = 'pause',
  RESUME = 'resume',
  RESET = 'reset',
  STATUS = 'status',
  CONFIGURE = 'configure'
}

export enum ErrorCode {
  PROTOCOL_ERROR = 'PROTOCOL_ERROR',
  AUTHENTICATION_FAILED = 'AUTHENTICATION_FAILED',
  AUTHORIZATION_DENIED = 'AUTHORIZATION_DENIED',
  TIMEOUT = 'TIMEOUT',
  NETWORK_ERROR = 'NETWORK_ERROR',
  INVALID_MESSAGE = 'INVALID_MESSAGE',
  RESOURCE_UNAVAILABLE = 'RESOURCE_UNAVAILABLE',
  CIRCUIT_BREAKER_OPEN = 'CIRCUIT_BREAKER_OPEN'
}

// Adapter State Types
export interface AdapterState {
  status: AdapterStatus;
  activeConnections: number;
  totalRequests: number;
  totalErrors: number;
  averageLatency: number;
  circuitBreakerState: CircuitState;
  lastHealthCheck: number;
  version: string;
}

export enum AdapterStatus {
  INITIALIZING = 'initializing',
  ACTIVE = 'active',
  DEGRADED = 'degraded',
  FAILING = 'failing',
  SHUTDOWN = 'shutdown'
}

// Health Check Types
export interface HealthCheckResult {
  status: HealthStatus;
  timestamp: number;
  checks: {
    connectionPool: HealthStatus;
    circuitBreaker: HealthStatus;
    protocolStack: HealthStatus;
    performance: HealthStatus;
  };
  metrics: PerformanceSnapshot;
  errors: HealthError[];
}

export interface PerformanceSnapshot {
  latencyP50: number;
  latencyP95: number;
  latencyP99: number;
  throughput: number;
  errorRate: number;
  activeConnections: number;
  queueDepth: number;
}

export enum HealthStatus {
  HEALTHY = 'healthy',
  WARNING = 'warning',
  CRITICAL = 'critical',
  UNKNOWN = 'unknown'
}

export interface HealthError {
  code: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: number;
  component: string;
}

// Security and Authentication Types
export interface SecurityContext {
  userId?: string;
  sessionId: string;
  securityLevel: SecurityLevel;
  permissions: Permission[];
  encryptionKey?: string;
  signature?: string;
  tokenExpiry?: number;
}

export interface Permission {
  resource: string;
  actions: string[];
  conditions?: Record<string, unknown>;
}

export interface ConstitutionalValidation {
  isValid: boolean;
  violations: ConstitutionalViolation[];
  securityFlags: string[];
  riskLevel: RiskLevel;
}

export interface ConstitutionalViolation {
  rule: string;
  severity: ViolationSeverity;
  description: string;
  recommendation: string;
}

export enum ViolationSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical'
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// Protocol Layer Types
export interface LayerMetrics {
  name: string;
  processingTime: number;
  throughput: number;
  errorCount: number;
  status: LayerStatus;
}

export enum LayerStatus {
  ACTIVE = 'active',
  DEGRADED = 'degraded',
  FAILED = 'failed',
  DISABLED = 'disabled'
}

export interface ProtocolStackMetrics {
  layers: LayerMetrics[];
  totalProcessingTime: number;
  bottleneckLayer?: string;
  recommendations: string[];
}

// Routing and Discovery Types
export interface RoutingEntry {
  destination: string;
  nextHop: string;
  metric: number;
  lastUpdated: number;
  interface: string;
}

export interface NodeDiscovery {
  nodeId: string;
  address: string;
  capabilities: string[];
  lastSeen: number;
  healthStatus: HealthStatus;
  services: ServiceAdvertisement[];
}

export interface ServiceAdvertisement {
  serviceId: string;
  serviceName: string;
  version: string;
  endpoints: string[];
  metadata: Record<string, unknown>;
}

// Event Types
export interface AdapterEvent {
  type: AdapterEventType;
  timestamp: number;
  data: unknown;
  severity: EventSeverity;
}

export enum AdapterEventType {
  CONNECTION_ESTABLISHED = 'connection_established',
  CONNECTION_LOST = 'connection_lost',
  MESSAGE_TRANSLATED = 'message_translated',
  TRANSLATION_ERROR = 'translation_error',
  CIRCUIT_BREAKER_STATE_CHANGE = 'circuit_breaker_state_change',
  PERFORMANCE_THRESHOLD_EXCEEDED = 'performance_threshold_exceeded',
  SECURITY_VIOLATION = 'security_violation',
  PROTOCOL_NEGOTIATED = 'protocol_negotiated',
  SESSION_CREATED = 'session_created',
  SESSION_TERMINATED = 'session_terminated'
}

export enum EventSeverity {
  DEBUG = 'debug',
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
  CRITICAL = 'critical'
}

// Monitoring and Observability Types
export interface MonitoringConfig {
  metricsEnabled: boolean;
  metricsInterval: number;
  healthCheckInterval: number;
  eventLoggingEnabled: boolean;
  performanceThresholds: PerformanceThresholds;
}

export interface PerformanceThresholds {
  maxLatencyP95: number;
  maxErrorRate: number;
  minThroughput: number;
  maxMemoryUsage: number;
  maxCpuUsage: number;
}

// Migration and Compatibility Types
export interface ProtocolMigration {
  fromVersion: string;
  toVersion: string;
  migrationSteps: MigrationStep[];
  rollbackPlan: RollbackStep[];
  compatibility: CompatibilityInfo;
}

export interface MigrationStep {
  stepId: string;
  description: string;
  action: () => Promise<void>;
  rollback: () => Promise<void>;
  prerequisites: string[];
}

export interface RollbackStep {
  stepId: string;
  description: string;
  action: () => Promise<void>;
}

export interface CompatibilityInfo {
  backwardCompatible: boolean;
  supportedVersions: string[];
  deprecatedFeatures: string[];
  newFeatures: string[];
}

// Testing and Validation Types
export interface ProtocolTestSuite {
  testCases: ProtocolTestCase[];
  performanceTests: PerformanceTestCase[];
  securityTests: SecurityTestCase[];
  integrationTests: IntegrationTestCase[];
}

export interface ProtocolTestCase {
  id: string;
  name: string;
  description: string;
  input: BetaNetMessage;
  expectedOutput: AIVillageResponse;
  setup?: () => Promise<void>;
  teardown?: () => Promise<void>;
}

export interface PerformanceTestCase {
  id: string;
  name: string;
  description: string;
  loadProfile: LoadProfile;
  expectedMetrics: PerformanceExpectation;
}

export interface LoadProfile {
  requestsPerSecond: number;
  duration: number;
  rampUpTime: number;
  rampDownTime: number;
  messageSize: number;
}

export interface PerformanceExpectation {
  maxLatencyP95: number;
  maxErrorRate: number;
  minThroughput: number;
}

export interface SecurityTestCase {
  id: string;
  name: string;
  description: string;
  attackVector: string;
  expectedBehavior: SecurityExpectation;
}

export interface SecurityExpectation {
  shouldBlock: boolean;
  expectedErrorCode?: ErrorCode;
  alertGenerated: boolean;
}

export interface IntegrationTestCase {
  id: string;
  name: string;
  description: string;
  systems: string[];
  scenario: TestScenario;
}

export interface TestScenario {
  steps: TestStep[];
  expectedOutcome: string;
  rollbackRequired: boolean;
}

export interface TestStep {
  stepId: string;
  action: string;
  data: unknown;
  expectedResult: unknown;
}

// Utility Types
export type MessageHandler = (message: BetaNetMessage) => Promise<BetaNetMessage>;
export type ErrorHandler = (error: Error, context: unknown) => void;
export type MetricsCollector = (metrics: PerformanceMetrics) => void;
export type EventListener = (event: AdapterEvent) => void;

// Type Guards
export function isBetaNetHandshakeMessage(message: BetaNetMessage): message is BetaNetHandshakeMessage {
  return message.type === BetaNetMessageType.HANDSHAKE;
}

export function isBetaNetDiscoveryMessage(message: BetaNetMessage): message is BetaNetDiscoveryMessage {
  return message.type === BetaNetMessageType.DISCOVERY;
}

export function isBetaNetDataMessage(message: BetaNetMessage): message is BetaNetDataMessage {
  return message.type === BetaNetMessageType.DATA_TRANSFER;
}

export function isBetaNetControlMessage(message: BetaNetMessage): message is BetaNetControlMessage {
  return message.type === BetaNetMessageType.CONTROL;
}

export function isBetaNetErrorMessage(message: BetaNetMessage): message is BetaNetErrorMessage {
  return message.type === BetaNetMessageType.ERROR;
}

// Re-export main types from adapter
export {
  BetaNetMessage,
  BetaNetMetadata,
  BetaNetMessageType,
  BetaNetPriority,
  SecurityLevel,
  AIVillageRequest,
  AIVillageResponse,
  PerformanceMetrics,
  CircuitState
} from './ConstitutionalBetaNetAdapter';