/**
 * Load Testing Configuration
 * Uses k6 for comprehensive load testing
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const latencyTrend = new Trend('latency');
const privacyTierLatency = new Trend('privacy_tier_latency');
const protocolLatency = new Trend('protocol_latency');

// Test configuration
export const options = {
  scenarios: {
    // Smoke test
    smoke: {
      executor: 'constant-vus',
      vus: 5,
      duration: '1m',
    },

    // Load test
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 }, // Ramp up to 50 users
        { duration: '5m', target: 50 }, // Stay at 50 users
        { duration: '2m', target: 100 }, // Ramp up to 100 users
        { duration: '5m', target: 100 }, // Stay at 100 users
        { duration: '2m', target: 0 }, // Ramp down to 0
      ],
      gracefulRampDown: '30s',
      startTime: '10s', // Start after smoke test
    },

    // Stress test
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '2m', target: 300 },
        { duration: '2m', target: 400 },
        { duration: '2m', target: 0 },
      ],
      gracefulRampDown: '30s',
      startTime: '20m', // Start after load test
    },

    // Spike test
    spike: {
      executor: 'constant-vus',
      vus: 500,
      duration: '30s',
      startTime: '30m', // Start after stress test
    },

    // Soak test (endurance)
    soak: {
      executor: 'constant-vus',
      vus: 50,
      duration: '10m',
      startTime: '35m', // Start after spike test
    }
  },

  thresholds: {
    http_req_duration: ['p(95)<75'], // P95 must be < 75ms
    http_req_failed: ['rate<0.01'], // Error rate < 1%
    errors: ['rate<0.05'], // Custom error rate < 5%
  },
};

// Base URL configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';

// Test data generators
function generateBetaNetRequest() {
  return {
    jsonrpc: '2.0',
    id: `k6-${__VU}-${__ITER}`,
    method: 'processRequest',
    params: {
      protocol: 'betanet',
      privacyTier: ['Bronze', 'Silver', 'Gold', 'Platinum'][Math.floor(Math.random() * 4)],
      data: {
        type: 'load-test',
        content: `Load test from VU ${__VU} iteration ${__ITER}`,
        timestamp: Date.now(),
      },
      userContext: {
        userId: `user-${__VU}`,
        trustScore: Math.random(),
      }
    }
  };
}

function generateProtocolRequest(protocol: string) {
  return {
    jsonrpc: '2.0',
    id: `k6-${protocol}-${__VU}-${__ITER}`,
    method: 'routeProtocol',
    params: {
      protocol,
      data: {
        test: `${protocol} test`,
        vu: __VU,
        iteration: __ITER,
      }
    }
  };
}

// Main test function
export default function() {
  const scenario = __ENV.K6_SCENARIO_NAME;

  // Different behavior based on scenario
  switch(scenario) {
    case 'smoke':
      smokeTest();
      break;
    case 'load':
      loadTest();
      break;
    case 'stress':
      stressTest();
      break;
    case 'spike':
      spikeTest();
      break;
    case 'soak':
      soakTest();
      break;
    default:
      loadTest(); // Default to load test
  }
}

function smokeTest() {
  // Basic functionality test
  const request = generateBetaNetRequest();
  const startTime = Date.now();

  const response = http.post(
    `${BASE_URL}/bridge`,
    JSON.stringify(request),
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '5s',
    }
  );

  const latency = Date.now() - startTime;
  latencyTrend.add(latency);

  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response has result': (r) => {
      const body = JSON.parse(r.body as string);
      return body.result !== undefined || body.error !== undefined;
    },
    'latency < 75ms': () => latency < 75,
  });

  errorRate.add(!success);
  sleep(1);
}

function loadTest() {
  // Standard load test with mixed requests
  const requestTypes = [
    { weight: 0.4, generator: () => generateBetaNetRequest() },
    { weight: 0.2, generator: () => generateProtocolRequest('bitchat') },
    { weight: 0.2, generator: () => generateProtocolRequest('p2p') },
    { weight: 0.2, generator: () => generateProtocolRequest('fog') },
  ];

  // Select request type based on weights
  const random = Math.random();
  let accumWeight = 0;
  let request = null;

  for (const type of requestTypes) {
    accumWeight += type.weight;
    if (random <= accumWeight) {
      request = type.generator();
      break;
    }
  }

  const startTime = Date.now();
  const response = http.post(
    `${BASE_URL}/bridge`,
    JSON.stringify(request),
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '10s',
    }
  );

  const latency = Date.now() - startTime;
  latencyTrend.add(latency);

  // Track protocol-specific latency
  if (request && request.params.protocol) {
    protocolLatency.add(latency, { protocol: request.params.protocol });
  }

  check(response, {
    'status is 200': (r) => r.status === 200,
    'latency < 100ms': () => latency < 100,
  });

  sleep(Math.random() * 2); // Random delay 0-2 seconds
}

function stressTest() {
  // High load test to find breaking point
  const batchSize = 5;
  const requests = [];

  // Create batch of requests
  for (let i = 0; i < batchSize; i++) {
    requests.push(generateBetaNetRequest());
  }

  // Send requests in parallel
  const responses = http.batch(
    requests.map(req => ({
      method: 'POST',
      url: `${BASE_URL}/bridge`,
      body: JSON.stringify(req),
      params: {
        headers: { 'Content-Type': 'application/json' },
        timeout: '15s',
      },
    }))
  );

  // Check all responses
  responses.forEach(response => {
    const success = check(response, {
      'status not 503': (r) => r.status !== 503, // Service unavailable
      'status not 504': (r) => r.status !== 504, // Gateway timeout
    });

    errorRate.add(!success);
  });

  sleep(0.5);
}

function spikeTest() {
  // Sudden burst of traffic
  const request = generateBetaNetRequest();

  const response = http.post(
    `${BASE_URL}/bridge`,
    JSON.stringify(request),
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '2s', // Short timeout for spike
    }
  );

  check(response, {
    'handles spike': (r) => r.status === 200 || r.status === 503,
  });

  // No sleep - maximum pressure
}

function soakTest() {
  // Long-running endurance test
  const privacyTiers = ['Bronze', 'Silver', 'Gold', 'Platinum'];
  const tier = privacyTiers[__ITER % 4];

  const request = {
    ...generateBetaNetRequest(),
    params: {
      ...generateBetaNetRequest().params,
      privacyTier: tier,
    }
  };

  const startTime = Date.now();
  const response = http.post(
    `${BASE_URL}/bridge`,
    JSON.stringify(request),
    {
      headers: { 'Content-Type': 'application/json' },
      timeout: '10s',
    }
  );

  const latency = Date.now() - startTime;
  privacyTierLatency.add(latency, { tier });

  check(response, {
    'no memory leaks': (r) => {
      if (r.headers['X-Memory-Usage']) {
        const memoryMB = parseInt(r.headers['X-Memory-Usage']) / 1024 / 1024;
        return memoryMB < 500; // Memory should stay under 500MB
      }
      return true;
    },
    'consistent performance': () => latency < 150, // Allow higher threshold for soak
  });

  sleep(2);
}

// Setup function (runs once per VU)
export function setup() {
  console.log('Starting load test...');

  // Health check
  const healthCheck = http.get(`${BASE_URL}/health`);
  if (healthCheck.status !== 200) {
    throw new Error('System health check failed');
  }

  return {
    startTime: Date.now(),
  };
}

// Teardown function (runs once)
export function teardown(data: any) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Load test completed in ${duration}s`);
}