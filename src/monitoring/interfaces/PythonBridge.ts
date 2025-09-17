/**
 * Python Bridge Interface - Integration layer for existing Python monitoring infrastructure
 * Provides TypeScript/Python interoperability for AIVillage monitoring systems
 */

export interface PythonMetric {
  name: string;
  value: number;
  timestamp: number;
  tags: Record<string, string>;
  unit: string;
  type: 'counter' | 'gauge' | 'histogram';
}

export interface PythonServiceHealth {
  service: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  response_time: number;
  error_count: number;
  last_check: number;
  dependencies: string[];
  metadata: Record<string, any>;
}

export interface PythonPerformanceData {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    bytes_sent: number;
    bytes_recv: number;
    packets_sent: number;
    packets_recv: number;
  };
  process_count: number;
  uptime: number;
}

export interface PythonDistributedTracingSpan {
  trace_id: string;
  span_id: string;
  parent_span_id?: string;
  operation_name: string;
  start_time: number;
  end_time?: number;
  tags: Record<string, any>;
  logs: Array<{
    timestamp: number;
    level: string;
    message: string;
    fields?: Record<string, any>;
  }>;
  status: 'ok' | 'error' | 'timeout';
}

export interface PythonLogEntry {
  timestamp: number;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
  logger: string;
  message: string;
  module: string;
  function: string;
  line_number: number;
  thread_id: string;
  process_id: number;
  extra_fields?: Record<string, any>;
}

export interface PythonServiceDependency {
  service: string;
  endpoint: string;
  method: string;
  response_time: number;
  status_code: number;
  error_rate: number;
  dependency_type: 'database' | 'service' | 'cache' | 'queue' | 'external';
  health_score: number;
}

/**
 * Python Bridge - Communication layer with existing Python monitoring
 */
export class PythonBridge {
  private pythonProcess?: any;
  private messageQueue: Array<{id: string; data: any; resolve: Function; reject: Function}> = [];
  private isConnected = false;
  private connectionRetries = 0;
  private maxRetries = 5;

  constructor(private pythonScriptPath: string = './monitoring/python_bridge.py') {
    this.initializePythonProcess();
  }

  /**
   * Initialize Python process for communication
   */
  private async initializePythonProcess(): Promise<void> {
    try {
      const { spawn } = require('child_process');

      this.pythonProcess = spawn('python', [this.pythonScriptPath], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.pythonProcess.stdout.on('data', (data: Buffer) => {
        this.handlePythonResponse(data.toString());
      });

      this.pythonProcess.stderr.on('data', (data: Buffer) => {
        console.error(`Python bridge error: ${data.toString()}`);
      });

      this.pythonProcess.on('close', (code: number) => {
        console.log(`Python bridge process exited with code ${code}`);
        this.isConnected = false;
        if (this.connectionRetries < this.maxRetries) {
          this.connectionRetries++;
          setTimeout(() => this.initializePythonProcess(), 5000);
        }
      });

      // Wait for connection confirmation
      await this.sendCommand('ping');
      this.isConnected = true;
      this.connectionRetries = 0;
      console.log('Python bridge connected successfully');

    } catch (error) {
      console.error('Failed to initialize Python bridge:', error);
      if (this.connectionRetries < this.maxRetries) {
        this.connectionRetries++;
        setTimeout(() => this.initializePythonProcess(), 5000);
      }
    }
  }

  /**
   * Send command to Python monitoring system
   */
  private async sendCommand(command: string, params?: any): Promise<any> {
    if (!this.isConnected) {
      throw new Error('Python bridge not connected');
    }

    return new Promise((resolve, reject) => {
      const id = Date.now().toString() + Math.random().toString(36);
      const message = {
        id,
        command,
        params: params || {}
      };

      this.messageQueue.push({ id, data: message, resolve, reject });

      this.pythonProcess.stdin.write(JSON.stringify(message) + '\n');

      // Timeout after 30 seconds
      setTimeout(() => {
        const index = this.messageQueue.findIndex(msg => msg.id === id);
        if (index >= 0) {
          this.messageQueue.splice(index, 1);
          reject(new Error('Python command timeout'));
        }
      }, 30000);
    });
  }

  /**
   * Handle response from Python process
   */
  private handlePythonResponse(data: string): void {
    try {
      const lines = data.split('\n').filter(line => line.trim());

      for (const line of lines) {
        const response = JSON.parse(line);
        const pending = this.messageQueue.find(msg => msg.id === response.id);

        if (pending) {
          if (response.error) {
            pending.reject(new Error(response.error));
          } else {
            pending.resolve(response.result);
          }

          const index = this.messageQueue.indexOf(pending);
          this.messageQueue.splice(index, 1);
        }
      }
    } catch (error) {
      console.error('Failed to parse Python response:', error);
    }
  }

  /**
   * Get metrics from Python monitoring system
   */
  public async getMetrics(metricNames?: string[]): Promise<PythonMetric[]> {
    return await this.sendCommand('get_metrics', { names: metricNames });
  }

  /**
   * Get service health from Python monitoring
   */
  public async getServiceHealth(serviceName?: string): Promise<PythonServiceHealth[]> {
    return await this.sendCommand('get_service_health', { service: serviceName });
  }

  /**
   * Get performance data from Python system monitors
   */
  public async getPerformanceData(): Promise<PythonPerformanceData> {
    return await this.sendCommand('get_performance_data');
  }

  /**
   * Get distributed tracing spans from Python APM
   */
  public async getDistributedTraces(traceId?: string, limit?: number): Promise<PythonDistributedTracingSpan[]> {
    return await this.sendCommand('get_distributed_traces', { trace_id: traceId, limit });
  }

  /**
   * Get log entries from Python log aggregation
   */
  public async getLogs(
    level?: string,
    logger?: string,
    since?: number,
    limit?: number
  ): Promise<PythonLogEntry[]> {
    return await this.sendCommand('get_logs', { level, logger, since, limit });
  }

  /**
   * Get service dependencies from Python monitoring
   */
  public async getServiceDependencies(serviceName?: string): Promise<PythonServiceDependency[]> {
    return await this.sendCommand('get_service_dependencies', { service: serviceName });
  }

  /**
   * Send metric to Python monitoring system
   */
  public async sendMetric(metric: PythonMetric): Promise<void> {
    await this.sendCommand('send_metric', metric);
  }

  /**
   * Send batch of metrics to Python monitoring
   */
  public async sendMetricsBatch(metrics: PythonMetric[]): Promise<void> {
    await this.sendCommand('send_metrics_batch', { metrics });
  }

  /**
   * Register service with Python health monitoring
   */
  public async registerService(
    serviceName: string,
    healthCheckUrl: string,
    dependencies: string[] = []
  ): Promise<void> {
    await this.sendCommand('register_service', {
      name: serviceName,
      health_check_url: healthCheckUrl,
      dependencies
    });
  }

  /**
   * Start distributed trace in Python APM
   */
  public async startTrace(operationName: string, tags?: Record<string, any>): Promise<string> {
    const result = await this.sendCommand('start_trace', {
      operation_name: operationName,
      tags
    });
    return result.trace_id;
  }

  /**
   * Finish distributed trace span
   */
  public async finishTrace(traceId: string, tags?: Record<string, any>): Promise<void> {
    await this.sendCommand('finish_trace', {
      trace_id: traceId,
      tags
    });
  }

  /**
   * Send log entry to Python log aggregation
   */
  public async sendLog(logEntry: Omit<PythonLogEntry, 'timestamp'>): Promise<void> {
    await this.sendCommand('send_log', {
      ...logEntry,
      timestamp: Date.now()
    });
  }

  /**
   * Trigger Python platform validation
   */
  public async triggerPlatformValidation(
    validationType: string,
    parameters?: Record<string, any>
  ): Promise<{
    validation_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    results?: any;
  }> {
    return await this.sendCommand('trigger_platform_validation', {
      validation_type: validationType,
      parameters
    });
  }

  /**
   * Get platform validation results
   */
  public async getValidationResults(validationId: string): Promise<{
    validation_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    results: any;
    started_at: number;
    completed_at?: number;
    error?: string;
  }> {
    return await this.sendCommand('get_validation_results', {
      validation_id: validationId
    });
  }

  /**
   * Check if Python bridge is connected
   */
  public isConnectedToPython(): boolean {
    return this.isConnected;
  }

  /**
   * Disconnect from Python bridge
   */
  public disconnect(): void {
    if (this.pythonProcess) {
      this.pythonProcess.kill();
      this.pythonProcess = null;
    }
    this.isConnected = false;
  }

  /**
   * Get bridge status and health
   */
  public getBridgeStatus(): {
    connected: boolean;
    retries: number;
    queueLength: number;
    lastError?: string;
  } {
    return {
      connected: this.isConnected,
      retries: this.connectionRetries,
      queueLength: this.messageQueue.length
    };
  }
}

export default PythonBridge;