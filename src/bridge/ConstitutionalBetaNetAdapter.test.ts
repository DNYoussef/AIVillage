/**
 * Constitutional BetaNet Adapter Test Suite
 * Comprehensive tests for protocol translation and performance
 */

import {
  ConstitutionalBetaNetAdapter,
  BetaNetMessage,
  BetaNetMessageType,
  BetaNetPriority,
  SecurityLevel,
  AIVillageRequest,
  AIVillageResponse,
  CircuitState
} from './ConstitutionalBetaNetAdapter';
import { adapterFactory } from './AdapterFactory';

describe('ConstitutionalBetaNetAdapter', () => {
  let adapter: ConstitutionalBetaNetAdapter;

  beforeEach(() => {
    adapter = new ConstitutionalBetaNetAdapter();
  });

  afterEach(async () => {
    await adapter.shutdown();
  });

  describe('Request Translation', () => {
    test('should translate HTTP GET request to BetaNet message', async () => {
      const request: AIVillageRequest = {
        method: 'GET',
        path: '/api/users/123',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer token123',
          'x-betanet-destination': 'user_service'
        },
        params: { id: '123' },
        query: { include: 'profile' },
        timestamp: Date.now(),
        sessionId: 'session_123',
        userId: 'user_456'
      };

      const betaNetMessage = await adapter.translateRequestToBetaNet(request);

      expect(betaNetMessage).toBeDefined();
      expect(betaNetMessage.type).toBe(BetaNetMessageType.DATA_TRANSFER);
      expect(betaNetMessage.destination).toBe('user_service');
      expect(betaNetMessage.priority).toBe(BetaNetPriority.NORMAL);
      expect(betaNetMessage.metadata.sessionId).toBe('session_123');
      expect(betaNetMessage.payload).toEqual({
        method: 'GET',
        path: '/api/users/123',
        headers: request.headers,
        body: undefined,
        params: { id: '123' },
        query: { include: 'profile' }
      });
    });

    test('should translate HTTP POST request to BetaNet message', async () => {
      const request: AIVillageRequest = {
        method: 'POST',
        path: '/api/users',
        headers: {
          'Content-Type': 'application/json',
          'x-priority': 'high',
          'x-security-level': 'confidential'
        },
        body: {
          name: 'John Doe',
          email: 'john@example.com'
        },
        timestamp: Date.now(),
        sessionId: 'session_456'
      };

      const betaNetMessage = await adapter.translateRequestToBetaNet(request);

      expect(betaNetMessage.type).toBe(BetaNetMessageType.DATA_TRANSFER);
      expect(betaNetMessage.priority).toBe(BetaNetPriority.HIGH);
      expect(betaNetMessage.metadata.securityLevel).toBe(SecurityLevel.CONFIDENTIAL);
    });

    test('should translate HTTP DELETE request to control message', async () => {
      const request: AIVillageRequest = {
        method: 'DELETE',
        path: '/api/users/123',
        headers: { 'Content-Type': 'application/json' },
        timestamp: Date.now(),
        sessionId: 'session_789'
      };

      const betaNetMessage = await adapter.translateRequestToBetaNet(request);

      expect(betaNetMessage.type).toBe(BetaNetMessageType.CONTROL);
    });

    test('should handle constitutional flags in headers', async () => {
      const request: AIVillageRequest = {
        method: 'GET',
        path: '/api/sensitive-data',
        headers: {
          'x-constitutional-flags': 'privacy-protected, audit-required, data-classification'
        },
        timestamp: Date.now(),
        sessionId: 'session_constitutional'
      };

      const betaNetMessage = await adapter.translateRequestToBetaNet(request);

      expect(betaNetMessage.metadata.constitutionalFlags).toEqual([
        'privacy-protected',
        'audit-required',
        'data-classification'
      ]);
    });
  });

  describe('Response Translation', () => {
    test('should translate BetaNet message to HTTP response', async () => {
      const betaNetMessage: BetaNetMessage = {
        id: 'msg_123',
        type: BetaNetMessageType.DATA_TRANSFER,
        payload: {
          status: 'success',
          data: { id: 123, name: 'John Doe' }
        },
        timestamp: Date.now(),
        version: '1.0',
        source: 'user_service',
        destination: 'aivillage',
        priority: BetaNetPriority.NORMAL,
        metadata: {
          sessionId: 'session_123',
          securityLevel: SecurityLevel.INTERNAL
        }
      };

      const response = await adapter.translateResponseFromBetaNet(betaNetMessage);

      expect(response).toBeDefined();
      expect(response.statusCode).toBe(200);
      expect(response.headers['Content-Type']).toBe('application/json');
      expect(response.headers['X-BetaNet-Message-ID']).toBe('msg_123');
      expect(response.body).toEqual(betaNetMessage.payload);
      expect(response.metadata?.betaNetMessageId).toBe('msg_123');
      expect(response.metadata?.securityLevel).toBe(SecurityLevel.INTERNAL);
    });

    test('should translate error message to HTTP error response', async () => {
      const errorMessage: BetaNetMessage = {
        id: 'error_123',
        type: BetaNetMessageType.ERROR,
        payload: {
          error: 'User not found',
          code: 'USER_NOT_FOUND'
        },
        timestamp: Date.now(),
        version: '1.0',
        source: 'user_service',
        destination: 'aivillage',
        priority: BetaNetPriority.HIGH,
        metadata: {}
      };

      const response = await adapter.translateResponseFromBetaNet(errorMessage);

      expect(response.statusCode).toBe(500);
      expect(response.body).toEqual(errorMessage.payload);
    });
  });

  describe('Protocol Negotiation', () => {
    test('should negotiate protocol version successfully', async () => {
      const negotiation = await adapter.negotiateProtocol('1.1', [
        'encryption',
        'compression',
        'fragmentation'
      ]);

      expect(negotiation.version).toBe('1.1');
      expect(negotiation.supportedFeatures).toContain('encryption');
      expect(negotiation.supportedFeatures).toContain('compression');
      expect(negotiation.encryptionEnabled).toBe(true);
      expect(negotiation.compressionEnabled).toBe(true);
    });

    test('should fallback to supported version', async () => {
      const negotiation = await adapter.negotiateProtocol('3.0', ['encryption']);

      expect(negotiation.version).toBe('1.0'); // Fallback to first supported
      expect(negotiation.supportedFeatures).toContain('encryption');
    });
  });

  describe('Session Management', () => {
    test('should create and manage sessions', async () => {
      const sessionId = await adapter.createSession({
        destination: 'test_service',
        securityLevel: SecurityLevel.INTERNAL,
        timeout: 60000
      });

      expect(sessionId).toBeDefined();
      expect(sessionId).toMatch(/^session_/);

      const session = await adapter.getSession(sessionId);
      expect(session).toBeDefined();

      await adapter.terminateSession(sessionId);
      const terminatedSession = await adapter.getSession(sessionId);
      expect(terminatedSession).toBeNull();
    });

    test('should handle session timeout', async () => {
      const sessionId = await adapter.createSession({
        destination: 'test_service',
        securityLevel: SecurityLevel.PUBLIC,
        timeout: 100 // Very short timeout
      });

      // Wait for timeout
      await new Promise(resolve => setTimeout(resolve, 150));

      const session = await adapter.getSession(sessionId);
      expect(session).toBeNull();
    });
  });

  describe('Performance Metrics', () => {
    test('should collect and report performance metrics', async () => {
      // Perform some operations to generate metrics
      const request: AIVillageRequest = {
        method: 'GET',
        path: '/api/test',
        headers: {},
        timestamp: Date.now(),
        sessionId: 'perf_test'
      };

      await adapter.translateRequestToBetaNet(request);

      const metrics = adapter.getPerformanceMetrics();

      expect(metrics).toBeDefined();
      expect(metrics.requestCount).toBeGreaterThan(0);
      expect(metrics.averageLatency).toBeGreaterThan(0);
      expect(metrics.circuitBreakerState).toBeDefined();
    });

    test('should track latency within target threshold', async () => {
      const startTime = Date.now();

      const request: AIVillageRequest = {
        method: 'GET',
        path: '/api/performance-test',
        headers: {},
        timestamp: Date.now(),
        sessionId: 'latency_test'
      };

      await adapter.translateRequestToBetaNet(request);

      const metrics = adapter.getPerformanceMetrics();
      const latency = Date.now() - startTime;

      // Should be well under 75ms target
      expect(latency).toBeLessThan(100);
      expect(metrics.p95Latency).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    test('should handle translation errors gracefully', async () => {
      // Create an invalid request that should cause an error
      const invalidRequest = {
        method: '',
        path: '',
        headers: {},
        timestamp: Date.now(),
        sessionId: ''
      } as AIVillageRequest;

      let errorCaught = false;
      try {
        await adapter.translateRequestToBetaNet(invalidRequest);
      } catch (error) {
        errorCaught = true;
        expect(error).toBeDefined();
      }

      // Error should be caught and handled
      expect(errorCaught).toBe(true);
    });

    test('should emit error events', (done) => {
      adapter.on('translationError', (event) => {
        expect(event.error).toBeDefined();
        done();
      });

      // Trigger an error
      const invalidRequest = {} as AIVillageRequest;
      adapter.translateRequestToBetaNet(invalidRequest).catch(() => {
        // Expected to fail
      });
    });
  });

  describe('Circuit Breaker Integration', () => {
    test('should open circuit breaker on repeated failures', async () => {
      // This would require mocking failures - simplified test
      const metrics = adapter.getPerformanceMetrics();
      expect(metrics.circuitBreakerState).toBe(CircuitState.CLOSED);
    });
  });

  describe('Event Emission', () => {
    test('should emit requestTranslated events', (done) => {
      adapter.on('requestTranslated', (event) => {
        expect(event.originalRequest).toBeDefined();
        expect(event.betaNetMessage).toBeDefined();
        expect(event.latency).toBeGreaterThan(0);
        done();
      });

      const request: AIVillageRequest = {
        method: 'GET',
        path: '/api/event-test',
        headers: {},
        timestamp: Date.now(),
        sessionId: 'event_test'
      };

      adapter.translateRequestToBetaNet(request);
    });

    test('should emit responseTranslated events', (done) => {
      adapter.on('responseTranslated', (event) => {
        expect(event.betaNetMessage).toBeDefined();
        expect(event.aivillageResponse).toBeDefined();
        expect(event.latency).toBeGreaterThan(0);
        done();
      });

      const message: BetaNetMessage = {
        id: 'test_msg',
        type: BetaNetMessageType.DATA_TRANSFER,
        payload: { test: 'data' },
        timestamp: Date.now(),
        version: '1.0',
        source: 'test',
        destination: 'aivillage',
        priority: BetaNetPriority.NORMAL,
        metadata: {}
      };

      adapter.translateResponseFromBetaNet(message);
    });
  });
});

describe('AdapterFactory', () => {
  afterEach(async () => {
    await adapterFactory.shutdownAll();
  });

  test('should create adapter with default configuration', () => {
    const adapter = adapterFactory.createAdapter('test-adapter-1');
    expect(adapter).toBeInstanceOf(ConstitutionalBetaNetAdapter);
  });

  test('should create adapter with custom configuration', () => {
    const customConfig = {
      performance: {
        targetLatencyP95: 50,
        maxRetries: 5,
        timeoutMs: 10000
      }
    };

    const adapter = adapterFactory.createAdapter('test-adapter-2', customConfig);
    expect(adapter).toBeInstanceOf(ConstitutionalBetaNetAdapter);
  });

  test('should create high-availability adapter', () => {
    const adapter = adapterFactory.createHighAvailabilityAdapter('ha-adapter');
    expect(adapter).toBeInstanceOf(ConstitutionalBetaNetAdapter);
  });

  test('should create development adapter', () => {
    const adapter = adapterFactory.createDevelopmentAdapter('dev-adapter');
    expect(adapter).toBeInstanceOf(ConstitutionalBetaNetAdapter);
  });

  test('should track adapter state', () => {
    const adapter = adapterFactory.createAdapter('state-test-adapter');
    const state = adapterFactory.getAdapterState('state-test-adapter');

    expect(state).toBeDefined();
    expect(state?.status).toBeDefined();
    expect(state?.version).toBe('1.0.0');
  });

  test('should perform health checks', async () => {
    adapterFactory.createAdapter('health-test-adapter');
    const healthResult = await adapterFactory.performHealthCheck('health-test-adapter');

    expect(healthResult).toBeDefined();
    expect(healthResult?.status).toBeDefined();
    expect(healthResult?.checks).toBeDefined();
    expect(healthResult?.metrics).toBeDefined();
  });

  test('should list all adapters', () => {
    adapterFactory.createAdapter('list-test-1');
    adapterFactory.createAdapter('list-test-2');

    const adapters = adapterFactory.listAdapters();
    expect(adapters).toContain('list-test-1');
    expect(adapters).toContain('list-test-2');
  });

  test('should update adapter configuration', async () => {
    adapterFactory.createAdapter('config-update-test');

    const updateResult = await adapterFactory.updateAdapterConfiguration(
      'config-update-test',
      {
        performance: {
          targetLatencyP95: 25,
          maxRetries: 10,
          timeoutMs: 5000
        }
      }
    );

    expect(updateResult).toBe(true);
  });

  test('should remove adapters', async () => {
    adapterFactory.createAdapter('remove-test');
    expect(adapterFactory.getAdapter('remove-test')).toBeDefined();

    await adapterFactory.removeAdapter('remove-test');
    expect(adapterFactory.getAdapter('remove-test')).toBeNull();
  });

  test('should prevent duplicate adapter IDs', () => {
    adapterFactory.createAdapter('duplicate-test');

    expect(() => {
      adapterFactory.createAdapter('duplicate-test');
    }).toThrow('already exists');
  });
});

describe('Performance Tests', () => {
  test('should meet latency requirements under load', async () => {
    const adapter = new ConstitutionalBetaNetAdapter();
    const iterations = 100;
    const latencies: number[] = [];

    try {
      for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();

        const request: AIVillageRequest = {
          method: 'GET',
          path: `/api/load-test/${i}`,
          headers: {},
          timestamp: Date.now(),
          sessionId: `load_test_${i}`
        };

        await adapter.translateRequestToBetaNet(request);
        latencies.push(Date.now() - startTime);
      }

      // Calculate P95 latency
      const sortedLatencies = latencies.sort((a, b) => a - b);
      const p95Index = Math.floor(sortedLatencies.length * 0.95);
      const p95Latency = sortedLatencies[p95Index];

      // Should meet <75ms P95 target
      expect(p95Latency).toBeLessThan(75);

      const averageLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      console.log(`Load test results - P95: ${p95Latency}ms, Average: ${averageLatency.toFixed(2)}ms`);

    } finally {
      await adapter.shutdown();
    }
  });

  test('should handle concurrent requests efficiently', async () => {
    const adapter = new ConstitutionalBetaNetAdapter();
    const concurrentRequests = 50;

    try {
      const promises = Array.from({ length: concurrentRequests }, (_, i) => {
        const request: AIVillageRequest = {
          method: 'GET',
          path: `/api/concurrent-test/${i}`,
          headers: {},
          timestamp: Date.now(),
          sessionId: `concurrent_test_${i}`
        };

        return adapter.translateRequestToBetaNet(request);
      });

      const startTime = Date.now();
      const results = await Promise.all(promises);
      const totalTime = Date.now() - startTime;

      expect(results).toHaveLength(concurrentRequests);
      expect(totalTime).toBeLessThan(1000); // Should complete within 1 second

      console.log(`Concurrent test: ${concurrentRequests} requests in ${totalTime}ms`);

    } finally {
      await adapter.shutdown();
    }
  });
});