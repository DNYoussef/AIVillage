import { describe, it, expect, beforeEach, afterEach, beforeAll, afterAll } from '@jest/globals';
import { ConstitutionalBetaNetAdapter } from '../../../src/bridge/ConstitutionalBetaNetAdapter';
import { PerformanceMonitor } from '../../../src/bridge/PerformanceMonitor';
import { ResourceProfiler } from '../../helpers/ResourceProfiler';
import { SystemMonitor } from '../../helpers/SystemMonitor';
import { MemoryAnalyzer } from '../../helpers/MemoryAnalyzer';
import { LoadGenerator } from '../../helpers/LoadGenerator';

describe('Resource Utilization Tests', () => {
  let adapter: ConstitutionalBetaNetAdapter;
  let performanceMonitor: PerformanceMonitor;
  let resourceProfiler: ResourceProfiler;
  let systemMonitor: SystemMonitor;
  let memoryAnalyzer: MemoryAnalyzer;
  let loadGenerator: LoadGenerator;

  // Resource utilization targets
  const RESOURCE_TARGETS = {
    maxMemoryUsage: 2 * 1024 * 1024 * 1024, // 2GB max heap
    maxCpuUsage: 85, // 85% max CPU utilization
    maxFileDescriptors: 10000, // 10k max file descriptors
    maxNetworkConnections: 5000, // 5k max network connections
    memoryLeakThreshold: 50 * 1024 * 1024, // 50MB memory leak threshold
    gcPressureThreshold: 0.3 // 30% max time in GC
  };

  beforeAll(async () => {
    resourceProfiler = new ResourceProfiler();
    systemMonitor = new SystemMonitor();
    memoryAnalyzer = new MemoryAnalyzer();
    loadGenerator = new LoadGenerator();

    await resourceProfiler.initialize();
    await systemMonitor.start();
    await loadGenerator.initialize();
  });

  afterAll(async () => {
    await resourceProfiler.cleanup();
    await systemMonitor.stop();
    await loadGenerator.cleanup();
  });

  beforeEach(async () => {
    performanceMonitor = new PerformanceMonitor({
      enableResourceMonitoring: true,
      resourceSamplingInterval: 1000
    });

    adapter = new ConstitutionalBetaNetAdapter({
      performanceMonitor,
      enableResourceOptimization: true
    });

    await adapter.initialize();
    resourceProfiler.reset();
    memoryAnalyzer.startTracking();
  });

  afterEach(async () => {
    await adapter.cleanup();
    await performanceMonitor.cleanup();
    memoryAnalyzer.stopTracking();
  });

  describe('memory utilization', () => {
    it('should maintain memory usage under 2GB during sustained load', async () => {
      const memoryTestConfig = {
        requestsPerSecond: 200,
        duration: 600000, // 10 minutes
        memoryIntensiveRequests: true
      };

      resourceProfiler.startMemoryProfiling();

      const loadTest = loadGenerator.createMemoryLoadTest(memoryTestConfig);
      const results = await loadTest.execute();

      const memoryProfile = resourceProfiler.getMemoryProfile();

      // Verify memory usage stays within limits
      expect(memoryProfile.maxHeapUsed).toBeLessThan(RESOURCE_TARGETS.maxMemoryUsage);
      expect(memoryProfile.maxRSS).toBeLessThan(RESOURCE_TARGETS.maxMemoryUsage * 1.5);

      // Verify no significant memory leaks
      const memoryGrowthRate = memoryProfile.growthRate; // bytes per minute
      expect(memoryGrowthRate).toBeLessThan(RESOURCE_TARGETS.memoryLeakThreshold / 10); // <5MB/min

      // Verify garbage collection efficiency
      expect(memoryProfile.gcPressure).toBeLessThan(RESOURCE_TARGETS.gcPressureThreshold);
      expect(memoryProfile.gcFrequency).toBeLessThan(10); // <10 GC cycles per minute

      // Verify memory usage patterns
      expect(memoryProfile.fragmentationRatio).toBeLessThan(0.3); // <30% fragmentation
      expect(memoryProfile.allocationEfficiency).toBeGreaterThan(0.8); // >80% allocation efficiency
    }, 650000); // 11 minutes timeout

    it('should handle memory pressure gracefully', async () => {
      const pressureConfig = {
        targetMemoryUsage: RESOURCE_TARGETS.maxMemoryUsage * 0.9, // 90% of max
        requestsPerSecond: 150,
        duration: 300000, // 5 minutes
        enableMemoryPressureSimulation: true
      };

      const pressureTest = await resourceProfiler.runMemoryPressureTest(pressureConfig);

      // Verify graceful handling of memory pressure
      expect(pressureTest.systemStability).toBe(true);
      expect(pressureTest.outOfMemoryErrors).toBe(0);
      expect(pressureTest.allocationFailures).toBeLessThan(10);

      // Verify adaptive behavior under pressure
      expect(pressureTest.cacheEvictions).toBeGreaterThan(0); // Should evict cache under pressure
      expect(pressureTest.requestQueueReduction).toBeGreaterThan(0); // Should reduce queue size
      expect(pressureTest.gcTriggers).toBeGreaterThan(0); // Should trigger more frequent GC

      // Verify recovery after pressure relief
      expect(pressureTest.recoveryTime).toBeLessThan(30000); // <30 seconds to recover
      expect(pressureTest.postPressurePerformance).toBeGreaterThan(0.9); // 90% performance recovery
    }, 350000); // 6 minutes timeout

    it('should detect and prevent memory leaks', async () => {
      const leakTestConfig = {
        duration: 480000, // 8 minutes
        requestsPerSecond: 100,
        enableLeakDetection: true,
        leakDetectionInterval: 60000 // Check every minute
      };

      const leakTest = await memoryAnalyzer.runLeakDetectionTest(leakTestConfig);

      // Verify no significant leaks detected
      expect(leakTest.leaksDetected).toHaveLength(0);
      expect(leakTest.suspiciousGrowthPatterns).toHaveLength(0);

      // Verify memory usage patterns are healthy
      const memoryTrend = leakTest.memoryTrend;
      expect(memoryTrend.slope).toBeLessThan(1024 * 1024); // <1MB/minute growth
      expect(memoryTrend.correlation).toBeLessThan(0.7); // Weak correlation with time

      // Verify object lifecycle management
      expect(leakTest.objectRetentionIssues).toBe(0);
      expect(leakTest.closureLeaks).toBe(0);
      expect(leakTest.eventListenerLeaks).toBe(0);

      // Verify heap snapshot analysis
      const heapAnalysis = leakTest.heapAnalysis;
      expect(heapAnalysis.dominatorIssues).toBe(0);
      expect(heapAnalysis.retainedSizeGrowth).toBeLessThan(0.1); // <10% retained size growth
    }, 520000); // 9 minutes timeout

    it('should optimize memory allocation patterns', async () => {
      const allocationConfig = {
        requestsPerSecond: 250,
        duration: 180000, // 3 minutes
        allocationPatterns: [
          { size: 'small', count: 1000, frequency: 'high' },
          { size: 'medium', count: 100, frequency: 'medium' },
          { size: 'large', count: 10, frequency: 'low' }
        ]
      };

      const allocationTest = await memoryAnalyzer.runAllocationPatternTest(allocationConfig);

      // Verify allocation efficiency
      expect(allocationTest.allocationRate).toBeGreaterThan(1000); // >1000 objects/second
      expect(allocationTest.allocationFailureRate).toBeLessThan(0.001); // <0.1% failures

      // Verify memory pool optimization
      expect(allocationTest.poolUtilization.small).toBeGreaterThan(0.8); // >80% small pool utilization
      expect(allocationTest.poolUtilization.medium).toBeGreaterThan(0.7); // >70% medium pool utilization
      expect(allocationTest.poolUtilization.large).toBeGreaterThan(0.6); // >60% large pool utilization

      // Verify allocation speed optimization
      expect(allocationTest.allocationLatency.p95).toBeLessThan(1); // <1ms p95 allocation time
      expect(allocationTest.deallocationLatency.p95).toBeLessThan(0.5); // <0.5ms p95 deallocation time

      // Verify fragmentation management
      expect(allocationTest.fragmentationReduction).toBeGreaterThan(0.2); // >20% fragmentation reduction
    }, 200000); // 3.5 minutes timeout
  });

  describe('CPU utilization', () => {
    it('should maintain CPU usage under 85% during peak load', async () => {
      const cpuTestConfig = {
        requestsPerSecond: 400,
        duration: 300000, // 5 minutes
        cpuIntensiveRequests: true,
        enableCpuProfiling: true
      };

      resourceProfiler.startCpuProfiling();

      const cpuTest = await loadGenerator.createCpuLoadTest(cpuTestConfig);
      const results = await cpuTest.execute();

      const cpuProfile = resourceProfiler.getCpuProfile();

      // Verify CPU usage stays within limits
      expect(cpuProfile.maxCpuUsage).toBeLessThan(RESOURCE_TARGETS.maxCpuUsage);
      expect(cpuProfile.averageCpuUsage).toBeLessThan(RESOURCE_TARGETS.maxCpuUsage * 0.8);

      // Verify CPU efficiency
      expect(cpuProfile.cpuEfficiency).toBeGreaterThan(0.7); // >70% CPU efficiency
      expect(cpuProfile.idleTime).toBeGreaterThan(0.1); // >10% idle time

      // Verify no CPU starvation
      expect(cpuProfile.starvationEvents).toBe(0);
      expect(cpuProfile.contextSwitchRate).toBeLessThan(10000); // <10k context switches/second

      // Verify thread utilization
      expect(cpuProfile.threadUtilization).toBeGreaterThan(0.6); // >60% thread utilization
      expect(cpuProfile.threadContention).toBeLessThan(0.1); // <10% thread contention
    }, 350000); // 6 minutes timeout

    it('should distribute CPU load effectively across cores', async () => {
      const multiCoreConfig = {
        requestsPerSecond: 300,
        duration: 240000, // 4 minutes
        enableMultiThreading: true,
        cpuCores: systemMonitor.getCpuCoreCount()
      };

      const multiCoreTest = await resourceProfiler.runMultiCoreTest(multiCoreConfig);

      // Verify load distribution
      const coreUtilization = multiCoreTest.coreUtilization;
      const avgUtilization = coreUtilization.reduce((sum, util) => sum + util, 0) / coreUtilization.length;
      const utilizationVariance = resourceProfiler.calculateVariance(coreUtilization);

      expect(avgUtilization).toBeGreaterThan(0.4); // >40% average utilization
      expect(utilizationVariance).toBeLessThan(0.2); // Low variance indicates good distribution

      // Verify scalability with core count
      const scalingEfficiency = multiCoreTest.scalingEfficiency;
      expect(scalingEfficiency).toBeGreaterThan(0.7); // >70% scaling efficiency

      // Verify no hot spots
      const maxCoreUtilization = Math.max(...coreUtilization);
      const minCoreUtilization = Math.min(...coreUtilization);
      expect(maxCoreUtilization / minCoreUtilization).toBeLessThan(2); // No core >2x another core
    }, 260000); // 4.5 minutes timeout

    it('should optimize CPU usage through request batching', async () => {
      const batchingTests = [
        { batchSize: 1, label: 'no_batching' },
        { batchSize: 10, label: 'small_batching' },
        { batchSize: 50, label: 'large_batching' }
      ];

      const batchingResults = {};

      for (const test of batchingTests) {
        adapter.configureBatching({ batchSize: test.batchSize });

        resourceProfiler.startCpuMeasurement();

        const batchTest = await loadGenerator.createBatchingTest({
          requestsPerSecond: 200,
          duration: 120000, // 2 minutes
          batchSize: test.batchSize
        });

        const result = await batchTest.execute();
        const cpuMeasurement = resourceProfiler.endCpuMeasurement();

        batchingResults[test.label] = {
          ...result,
          cpuEfficiency: cpuMeasurement.efficiency,
          cpuUsage: cpuMeasurement.averageUsage
        };
      }

      // Verify batching improves CPU efficiency
      const noBatchingCpu = batchingResults['no_batching'].cpuUsage;
      const smallBatchingCpu = batchingResults['small_batching'].cpuUsage;
      const largeBatchingCpu = batchingResults['large_batching'].cpuUsage;

      expect(smallBatchingCpu).toBeLessThan(noBatchingCpu * 0.9); // 10% improvement
      expect(largeBatchingCpu).toBeLessThan(smallBatchingCpu * 0.95); // Additional 5% improvement

      // Verify throughput improvements
      const throughputs = Object.values(batchingResults).map((r: any) => r.throughput);
      expect(throughputs[2]).toBeGreaterThan(throughputs[0] * 1.3); // >30% throughput improvement
    }, 400000); // 7 minutes timeout

    it('should handle CPU spikes and thermal throttling', async () => {
      const thermalConfig = {
        requestsPerSecond: 500,
        duration: 180000, // 3 minutes
        simulateThermalPressure: true,
        cpuSpikeIntensity: 'high'
      };

      const thermalTest = await resourceProfiler.runThermalTest(thermalConfig);

      // Verify thermal management
      expect(thermalTest.thermalThrottlingEvents).toBeGreaterThan(0); // Should detect throttling
      expect(thermalTest.maxTemperature).toBeLessThan(85); // Should stay under thermal limit

      // Verify adaptive behavior under thermal pressure
      expect(thermalTest.performanceDegradation).toBeLessThan(0.3); // <30% performance loss
      expect(thermalTest.adaptiveThrottling).toBe(true); // Should adapt to thermal constraints

      // Verify recovery after thermal event
      expect(thermalTest.thermalRecoveryTime).toBeLessThan(60000); // <1 minute recovery
      expect(thermalTest.postThermalPerformance).toBeGreaterThan(0.9); // 90% performance recovery
    });
  });

  describe('network resource utilization', () => {
    it('should manage network connections efficiently', async () => {
      const networkConfig = {
        concurrentConnections: 1000,
        requestsPerSecond: 300,
        duration: 300000, // 5 minutes
        connectionReuseRatio: 0.8
      };

      resourceProfiler.startNetworkProfiling();

      const networkTest = await loadGenerator.createNetworkTest(networkConfig);
      const results = await networkTest.execute();

      const networkProfile = resourceProfiler.getNetworkProfile();

      // Verify connection management
      expect(networkProfile.maxActiveConnections).toBeLessThan(RESOURCE_TARGETS.maxNetworkConnections);
      expect(networkProfile.connectionReuseRate).toBeGreaterThan(0.7); // >70% connection reuse

      // Verify no connection leaks
      expect(networkProfile.connectionLeaks).toBe(0);
      expect(networkProfile.timeoutConnections).toBeLessThan(10); // <10 timeout connections

      // Verify bandwidth utilization
      expect(networkProfile.bandwidthUtilization).toBeGreaterThan(0.5); // >50% bandwidth usage
      expect(networkProfile.networkLatency.p95).toBeLessThan(100); // <100ms p95 network latency

      // Verify connection pool efficiency
      expect(networkProfile.poolUtilization).toBeGreaterThan(0.8); // >80% pool utilization
      expect(networkProfile.connectionWaitTime.p95).toBeLessThan(50); // <50ms p95 wait time
    }, 350000); // 6 minutes timeout

    it('should handle network congestion gracefully', async () => {
      const congestionConfig = {
        requestsPerSecond: 400,
        duration: 240000, // 4 minutes
        simulateNetworkCongestion: true,
        packetLossRate: 0.02, // 2% packet loss
        latencyVariance: 'high'
      };

      const congestionTest = await resourceProfiler.runNetworkCongestionTest(congestionConfig);

      // Verify congestion handling
      expect(congestionTest.adaptiveBehavior).toBe(true); // Should adapt to congestion
      expect(congestionTest.retransmissionRate).toBeLessThan(0.1); // <10% retransmission rate

      // Verify performance under congestion
      expect(congestionTest.throughputDegradation).toBeLessThan(0.4); // <40% throughput loss
      expect(congestionTest.latencyIncrease).toBeLessThan(2.0); // <2x latency increase

      // Verify congestion recovery
      expect(congestionTest.congestionRecoveryTime).toBeLessThan(30000); // <30 seconds recovery
      expect(congestionTest.postCongestionPerformance).toBeGreaterThan(0.95); // 95% performance recovery
    }, 260000); // 4.5 minutes timeout

    it('should optimize network buffer usage', async () => {
      const bufferSizes = [4096, 8192, 16384, 32768]; // 4KB, 8KB, 16KB, 32KB
      const bufferResults = {};

      for (const bufferSize of bufferSizes) {
        adapter.configureNetworkBuffers({ size: bufferSize });

        const bufferTest = await resourceProfiler.runBufferOptimizationTest({
          requestsPerSecond: 250,
          duration: 90000, // 1.5 minutes
          bufferSize
        });

        bufferResults[bufferSize] = bufferTest;
      }

      // Find optimal buffer size
      const throughputs = bufferSizes.map(size => bufferResults[size].throughput);
      const latencies = bufferSizes.map(size => bufferResults[size].averageLatency);

      const optimalIndex = throughputs.indexOf(Math.max(...throughputs));
      const optimalBufferSize = bufferSizes[optimalIndex];

      // Verify optimal buffer provides best performance
      expect(bufferResults[optimalBufferSize].throughput).toBeGreaterThan(200);
      expect(bufferResults[optimalBufferSize].averageLatency).toBeLessThan(80);

      // Verify buffer efficiency
      expect(bufferResults[optimalBufferSize].bufferUtilization).toBeGreaterThan(0.7);
      expect(bufferResults[optimalBufferSize].bufferOverruns).toBe(0);
    }, 400000); // 7 minutes timeout
  });

  describe('file system resource utilization', () => {
    it('should manage file descriptors efficiently', async () => {
      const fileConfig = {
        requestsPerSecond: 200,
        duration: 300000, // 5 minutes
        fileOperationsPerRequest: 3,
        enableFileDescriptorTracking: true
      };

      resourceProfiler.startFileSystemProfiling();

      const fileTest = await loadGenerator.createFileSystemTest(fileConfig);
      const results = await fileTest.execute();

      const fsProfile = resourceProfiler.getFileSystemProfile();

      // Verify file descriptor management
      expect(fsProfile.maxFileDescriptors).toBeLessThan(RESOURCE_TARGETS.maxFileDescriptors);
      expect(fsProfile.fileDescriptorLeaks).toBe(0);

      // Verify file operation efficiency
      expect(fsProfile.fileOperationLatency.p95).toBeLessThan(10); // <10ms p95 file ops
      expect(fsProfile.fileOperationSuccessRate).toBeGreaterThan(0.999); // >99.9% success rate

      // Verify disk I/O optimization
      expect(fsProfile.diskIOUtilization).toBeGreaterThan(0.3); // >30% disk utilization
      expect(fsProfile.diskIOWaitTime).toBeLessThan(0.2); // <20% time waiting for disk
    }, 350000); // 6 minutes timeout

    it('should handle file system pressure gracefully', async () => {
      const pressureConfig = {
        requestsPerSecond: 150,
        duration: 240000, // 4 minutes
        simulateDiskPressure: true,
        diskSpaceLimit: 1024 * 1024 * 1024 // 1GB limit
      };

      const pressureTest = await resourceProfiler.runFileSystemPressureTest(pressureConfig);

      // Verify pressure handling
      expect(pressureTest.diskSpaceManagement).toBe(true); // Should manage disk space
      expect(pressureTest.fileCleanupEvents).toBeGreaterThan(0); // Should clean up files

      // Verify performance under pressure
      expect(pressureTest.performanceDegradation).toBeLessThan(0.3); // <30% performance loss
      expect(pressureTest.diskSpaceErrors).toBe(0); // No out-of-space errors

      // Verify adaptive behavior
      expect(pressureTest.cacheEvictions).toBeGreaterThan(0); // Should evict cache
      expect(pressureTest.compressionActivation).toBe(true); // Should enable compression
    }, 260000); // 4.5 minutes timeout
  });

  describe('resource optimization and tuning', () => {
    it('should demonstrate automatic resource tuning', async () => {
      const tuningConfig = {
        requestsPerSecond: 250,
        duration: 360000, // 6 minutes
        enableAutoTuning: true,
        tuningInterval: 60000 // Tune every minute
      };

      adapter.enableAutoResourceTuning();

      const tuningTest = await resourceProfiler.runAutoTuningTest(tuningConfig);

      // Verify tuning effectiveness
      expect(tuningTest.tuningEvents).toBeGreaterThan(0); // Should perform tuning
      expect(tuningTest.performanceImprovement).toBeGreaterThan(0.1); // >10% improvement

      // Verify resource optimization
      expect(tuningTest.memoryOptimization).toBeGreaterThan(0.05); // >5% memory optimization
      expect(tuningTest.cpuOptimization).toBeGreaterThan(0.05); // >5% CPU optimization

      // Verify stability during tuning
      expect(tuningTest.tuningStability).toBe(true); // Stable during tuning
      expect(tuningTest.performanceRegressions).toBe(0); // No regressions
    }, 400000); // 7 minutes timeout

    it('should validate resource monitoring accuracy', async () => {
      const monitoringConfig = {
        duration: 180000, // 3 minutes
        requestsPerSecond: 200,
        enableDetailedMonitoring: true,
        samplingInterval: 500 // 500ms sampling
      };

      const monitoringTest = await resourceProfiler.runMonitoringAccuracyTest(monitoringConfig);

      // Verify monitoring accuracy
      expect(monitoringTest.cpuMeasurementAccuracy).toBeGreaterThan(0.95); // >95% accuracy
      expect(monitoringTest.memoryMeasurementAccuracy).toBeGreaterThan(0.98); // >98% accuracy

      // Verify monitoring overhead
      expect(monitoringTest.monitoringOverhead.cpu).toBeLessThan(0.02); // <2% CPU overhead
      expect(monitoringTest.monitoringOverhead.memory).toBeLessThan(0.01); // <1% memory overhead

      // Verify data quality
      expect(monitoringTest.dataCompleteness).toBeGreaterThan(0.99); // >99% data completeness
      expect(monitoringTest.measurementConsistency).toBeGreaterThan(0.95); // >95% consistency
    }, 200000); // 3.5 minutes timeout

    it('should demonstrate resource limit enforcement', async () => {
      const limitConfig = {
        memoryLimit: 1024 * 1024 * 1024, // 1GB memory limit
        cpuLimit: 80, // 80% CPU limit
        connectionLimit: 500, // 500 connection limit
        requestsPerSecond: 300,
        duration: 240000 // 4 minutes
      };

      adapter.configureResourceLimits(limitConfig);

      const limitTest = await resourceProfiler.runResourceLimitTest(limitConfig);

      // Verify limits are enforced
      expect(limitTest.memoryLimitViolations).toBe(0); // No memory limit violations
      expect(limitTest.cpuLimitViolations).toBe(0); // No CPU limit violations
      expect(limitTest.connectionLimitViolations).toBe(0); // No connection limit violations

      // Verify graceful limit handling
      expect(limitTest.gracefulDegradation).toBe(true); // Graceful when approaching limits
      expect(limitTest.hardLimitRejections).toBeGreaterThan(0); // Should reject when at limits

      // Verify system stability under limits
      expect(limitTest.systemStability).toBe(true); // Stable under resource constraints
      expect(limitTest.limitRecoveryBehavior).toBe(true); // Recovers when resources available
    }, 260000); // 4.5 minutes timeout
  });
});