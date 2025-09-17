#!/usr/bin/env node

/**
 * Health Check Script for TypeScript Bridge Orchestrator
 * Validates service health, performance targets, and bridge connectivity
 */

const http = require('http');
const process = require('process');

const HEALTH_PORT = process.env.PORT || 8080;
const METRICS_PORT = process.env.METRICS_PORT || 9090;
const TARGET_P95_LATENCY = parseInt(process.env.TARGET_P95_LATENCY) || 75;
const PYTHON_BRIDGE_PORT = process.env.PYTHON_BRIDGE_PORT || 9876;

// Health check configuration
const TIMEOUT = 5000; // 5 seconds
const MAX_P95_LATENCY = TARGET_P95_LATENCY; // ms

async function checkEndpoint(port, path, timeout = TIMEOUT) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();

    const options = {
      hostname: 'localhost',
      port: port,
      path: path,
      method: 'GET',
      timeout: timeout
    };

    const req = http.request(options, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        const latency = Date.now() - startTime;
        resolve({
          statusCode: res.statusCode,
          data: data,
          latency: latency,
          healthy: res.statusCode >= 200 && res.statusCode < 300
        });
      });
    });

    req.on('error', (error) => {
      reject(new Error(`Request failed: ${error.message}`));
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new Error(`Request timeout after ${timeout}ms`));
    });

    req.end();
  });
}

async function checkMetrics() {
  try {
    const result = await checkEndpoint(METRICS_PORT, '/metrics');

    if (!result.healthy) {
      throw new Error(`Metrics endpoint unhealthy: ${result.statusCode}`);
    }

    // Parse Prometheus metrics to check P95 latency
    const metrics = result.data;
    const p95Match = metrics.match(/aivillage_p95_latency_milliseconds\s+(\d+\.?\d*)/);

    if (p95Match) {
      const p95Latency = parseFloat(p95Match[1]);
      if (p95Latency > MAX_P95_LATENCY) {
        console.warn(`WARNING: P95 latency ${p95Latency}ms exceeds target ${MAX_P95_LATENCY}ms`);
        // Don't fail health check for performance degradation, just warn
      }
    }

    return { healthy: true, p95Latency: p95Match ? parseFloat(p95Match[1]) : null };
  } catch (error) {
    throw new Error(`Metrics check failed: ${error.message}`);
  }
}

async function checkMainService() {
  try {
    const result = await checkEndpoint(HEALTH_PORT, '/health');

    if (!result.healthy) {
      throw new Error(`Main service unhealthy: ${result.statusCode}`);
    }

    // Check if response indicates healthy status
    try {
      const healthData = JSON.parse(result.data);
      if (healthData.status !== 'healthy' && healthData.status !== 'degraded') {
        throw new Error(`Service reports unhealthy status: ${healthData.status}`);
      }
    } catch (parseError) {
      // If we can't parse JSON, just check HTTP status
      console.warn('Could not parse health response as JSON, relying on HTTP status');
    }

    return { healthy: true, latency: result.latency };
  } catch (error) {
    throw new Error(`Main service check failed: ${error.message}`);
  }
}

async function checkPythonBridge() {
  try {
    // Simple TCP connection test to Python bridge
    const net = require('net');

    return new Promise((resolve, reject) => {
      const socket = new net.Socket();

      const timeout = setTimeout(() => {
        socket.destroy();
        reject(new Error('Python bridge connection timeout'));
      }, 2000);

      socket.connect(PYTHON_BRIDGE_PORT, 'localhost', () => {
        clearTimeout(timeout);
        socket.destroy();
        resolve({ healthy: true });
      });

      socket.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`Python bridge connection failed: ${error.message}`));
      });
    });
  } catch (error) {
    throw new Error(`Python bridge check failed: ${error.message}`);
  }
}

async function checkCircuitBreaker() {
  try {
    const result = await checkEndpoint(METRICS_PORT, '/metrics');

    if (result.healthy) {
      // Check if circuit breaker is open (which would indicate system issues)
      const cbMatch = result.data.match(/aivillage_circuit_breaker_state\{.*?\}\s+(\d+)/);

      if (cbMatch) {
        const cbState = parseInt(cbMatch[1]);
        if (cbState === 1) { // Open state
          console.warn('WARNING: Circuit breaker is open');
          // Don't fail health check for open circuit breaker, just warn
        }
      }
    }

    return { healthy: true };
  } catch (error) {
    // Circuit breaker check is not critical for health
    console.warn(`Circuit breaker check failed: ${error.message}`);
    return { healthy: true };
  }
}

async function runHealthCheck() {
  const checks = [];
  const results = {
    healthy: true,
    timestamp: new Date().toISOString(),
    checks: {}
  };

  try {
    // Main service health (critical)
    console.log('Checking main service health...');
    const mainResult = await checkMainService();
    results.checks.mainService = mainResult;
    console.log(`✓ Main service healthy (${mainResult.latency}ms)`);

    // Metrics endpoint (critical)
    console.log('Checking metrics endpoint...');
    const metricsResult = await checkMetrics();
    results.checks.metrics = metricsResult;
    console.log(`✓ Metrics endpoint healthy`);

    if (metricsResult.p95Latency !== null) {
      console.log(`  P95 latency: ${metricsResult.p95Latency}ms (target: ${MAX_P95_LATENCY}ms)`);
    }

    // Python bridge (critical for production)
    console.log('Checking Python bridge connectivity...');
    const pythonResult = await checkPythonBridge();
    results.checks.pythonBridge = pythonResult;
    console.log('✓ Python bridge connectivity healthy');

    // Circuit breaker state (monitoring only)
    console.log('Checking circuit breaker state...');
    const cbResult = await checkCircuitBreaker();
    results.checks.circuitBreaker = cbResult;
    console.log('✓ Circuit breaker state checked');

  } catch (error) {
    console.error(`✗ Health check failed: ${error.message}`);
    results.healthy = false;
    results.error = error.message;
    process.exit(1);
  }

  // Additional system checks
  try {
    // Memory usage check
    const memUsage = process.memoryUsage();
    const memUsageMB = Math.round(memUsage.rss / 1024 / 1024);
    results.checks.memory = { usageMB: memUsageMB };

    if (memUsageMB > 1024) { // More than 1GB
      console.warn(`WARNING: High memory usage: ${memUsageMB}MB`);
    }

    // Event loop lag check
    const start = process.hrtime.bigint();
    setImmediate(() => {
      const lag = Number(process.hrtime.bigint() - start) / 1e6; // Convert to ms
      results.checks.eventLoopLag = { lagMs: lag };

      if (lag > 100) { // More than 100ms lag
        console.warn(`WARNING: High event loop lag: ${lag}ms`);
      }
    });

  } catch (error) {
    console.warn(`System checks failed: ${error.message}`);
  }

  console.log('✓ All critical health checks passed');

  // Output results for monitoring systems
  if (process.env.NODE_ENV === 'production') {
    console.log(JSON.stringify(results));
  }

  process.exit(0);
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error(`Uncaught exception in health check: ${error.message}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(`Unhandled rejection in health check: ${reason}`);
  process.exit(1);
});

// Run the health check
runHealthCheck().catch((error) => {
  console.error(`Health check failed: ${error.message}`);
  process.exit(1);
});