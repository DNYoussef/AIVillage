/**
 * PythonBetaNetAdapter - Real TypeScript wrapper for Python BetaNet infrastructure
 *
 * This adapter provides a production-ready bridge between TypeScript bridge components
 * and the existing Python BetaNet infrastructure, replacing mock implementations
 * with actual network communication.
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import * as net from 'net';
import { v4 as uuidv4 } from 'uuid';

// Import types from existing bridge
import {
  AIVillageRequest,
  AIVillageResponse,
  BetaNetMessage,
  HealthStatus
} from '../types';

// JSON-RPC types for Python communication
interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: string;
  method: string;
  params: any;
}

interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: string;
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
}

// Configuration for Python bridge
export interface PythonBridgeConfig {
  pythonPath?: string;
  bridgeScript: string;
  host: string;
  port: number;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
  healthCheckInterval: number;
}

// BetaNet specific configuration
export interface BetaNetConfig {
  constitutionalTier: 'Bronze' | 'Silver' | 'Gold' | 'Platinum';
  privacyMode: 'standard' | 'enhanced' | 'maximum';
  mixnodeRouting: boolean;
  zeroKnowledgeProofs: boolean;
  fogIntegration: boolean;
}

export class PythonBetaNetAdapter extends EventEmitter {
  private config: PythonBridgeConfig;
  private betaNetConfig: BetaNetConfig;
  private pythonProcess?: ChildProcess;
  private client?: net.Socket;
  private connected: boolean = false;
  private pendingRequests: Map<string, {
    resolve: (value: any) => void;
    reject: (reason: any) => void;
    timestamp: number;
  }> = new Map();
  private reconnectTimer?: NodeJS.Timeout;
  private healthCheckTimer?: NodeJS.Timeout;
  private messageBuffer: string = '';
  private metrics = {
    requestsSent: 0,
    responsesReceived: 0,
    errors: 0,
    averageLatency: 0,
    p95Latency: 0,
    latencies: [] as number[]
  };

  constructor(config: PythonBridgeConfig, betaNetConfig: BetaNetConfig) {
    super();
    this.config = {
      pythonPath: 'python3',
      ...config
    };
    this.betaNetConfig = betaNetConfig;
  }

  /**
   * Initialize the Python bridge and establish connection
   */
  public async initialize(): Promise<void> {
    try {
      // Start Python bridge process
      await this.startPythonBridge();

      // Wait for bridge to be ready
      await this.waitForBridgeReady();

      // Establish TCP connection
      await this.connectToBridge();

      // Start health checks
      this.startHealthChecks();

      // Configure BetaNet settings
      await this.configureBetaNet();

      this.emit('initialized', {
        timestamp: Date.now(),
        config: this.betaNetConfig
      });

    } catch (error) {
      console.error('Failed to initialize PythonBetaNetAdapter:', error);
      throw new Error(`Initialization failed: ${error.message}`);
    }
  }

  /**
   * Translate AIVillage request to BetaNet message using Python infrastructure
   */
  public async translateToBetaNet(request: AIVillageRequest): Promise<BetaNetMessage> {
    const startTime = Date.now();

    try {
      const response = await this.sendJsonRpcRequest('translate_to_betanet', {
        request,
        constitutional_tier: this.betaNetConfig.constitutionalTier,
        privacy_mode: this.betaNetConfig.privacyMode,
        enable_mixnode: this.betaNetConfig.mixnodeRouting,
        enable_zk_proofs: this.betaNetConfig.zeroKnowledgeProofs
      });

      this.updateMetrics(Date.now() - startTime);

      return response.result as BetaNetMessage;

    } catch (error) {
      this.metrics.errors++;
      console.error('Translation to BetaNet failed:', error);
      throw new Error(`BetaNet translation failed: ${error.message}`);
    }
  }

  /**
   * Translate BetaNet message to AIVillage response using Python infrastructure
   */
  public async translateFromBetaNet(message: BetaNetMessage): Promise<AIVillageResponse> {
    const startTime = Date.now();

    try {
      const response = await this.sendJsonRpcRequest('translate_from_betanet', {
        message,
        validate_constitutional: true,
        apply_privacy_filters: true
      });

      this.updateMetrics(Date.now() - startTime);

      return response.result as AIVillageResponse;

    } catch (error) {
      this.metrics.errors++;
      console.error('Translation from BetaNet failed:', error);
      throw new Error(`BetaNet translation failed: ${error.message}`);
    }
  }

  /**
   * Send message through BetaNet network
   */
  public async sendBetaNetMessage(message: BetaNetMessage): Promise<void> {
    const startTime = Date.now();

    try {
      await this.sendJsonRpcRequest('send_betanet_message', {
        message,
        use_fog_relay: this.betaNetConfig.fogIntegration,
        priority: message.priority || 'normal'
      });

      this.updateMetrics(Date.now() - startTime);

    } catch (error) {
      this.metrics.errors++;
      console.error('Failed to send BetaNet message:', error);
      throw error;
    }
  }

  /**
   * Receive messages from BetaNet network
   */
  public async receiveBetaNetMessages(): Promise<BetaNetMessage[]> {
    try {
      const response = await this.sendJsonRpcRequest('receive_betanet_messages', {
        timeout: 5000,
        max_messages: 100
      });

      return response.result as BetaNetMessage[];

    } catch (error) {
      console.error('Failed to receive BetaNet messages:', error);
      return [];
    }
  }

  /**
   * Get health status from Python bridge
   */
  public async getHealthStatus(): Promise<HealthStatus> {
    try {
      const response = await this.sendJsonRpcRequest('get_health_status', {});

      const pythonHealth = response.result;

      return {
        status: pythonHealth.healthy ? 'healthy' : 'unhealthy',
        components: {
          pythonBridge: pythonHealth.bridge_status,
          betaNetConnection: pythonHealth.betanet_status,
          fogIntegration: pythonHealth.fog_status,
          constitutionalValidator: pythonHealth.constitutional_status
        },
        metrics: {
          ...this.metrics,
          uptime: pythonHealth.uptime,
          activeConnections: pythonHealth.active_connections
        },
        timestamp: Date.now()
      };

    } catch (error) {
      return {
        status: 'unhealthy',
        components: {
          pythonBridge: 'error',
          betaNetConnection: 'unknown'
        },
        metrics: this.metrics,
        timestamp: Date.now()
      };
    }
  }

  /**
   * Shutdown the adapter and clean up resources
   */
  public async shutdown(): Promise<void> {
    console.log('Shutting down PythonBetaNetAdapter...');

    // Stop health checks
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
    }

    // Clear reconnect timer
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    // Notify Python bridge of shutdown
    try {
      await this.sendJsonRpcRequest('shutdown', {}, 1000);
    } catch (error) {
      // Ignore errors during shutdown
    }

    // Close TCP connection
    if (this.client) {
      this.client.destroy();
      this.client = undefined;
    }

    // Terminate Python process
    if (this.pythonProcess) {
      this.pythonProcess.kill('SIGTERM');

      // Force kill after timeout
      setTimeout(() => {
        if (this.pythonProcess) {
          this.pythonProcess.kill('SIGKILL');
        }
      }, 5000);

      this.pythonProcess = undefined;
    }

    this.connected = false;
    this.emit('shutdown', { timestamp: Date.now() });
  }

  // Private methods

  private async startPythonBridge(): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`Starting Python bridge: ${this.config.bridgeScript}`);

      const args = [
        this.config.bridgeScript,
        '--host', this.config.host,
        '--port', this.config.port.toString(),
        '--constitutional-tier', this.betaNetConfig.constitutionalTier,
        '--privacy-mode', this.betaNetConfig.privacyMode
      ];

      this.pythonProcess = spawn(this.config.pythonPath!, args, {
        cwd: 'C:/Users/17175/Desktop/AIVillage',
        env: {
          ...process.env,
          PYTHONPATH: 'C:/Users/17175/Desktop/AIVillage',
          BETANET_BRIDGE_MODE: 'production'
        }
      });

      this.pythonProcess.stdout?.on('data', (data) => {
        const output = data.toString();
        console.log(`[Python Bridge]: ${output}`);

        // Check if bridge is ready
        if (output.includes('BetaNet Bridge ready')) {
          resolve();
        }
      });

      this.pythonProcess.stderr?.on('data', (data) => {
        console.error(`[Python Bridge Error]: ${data}`);
      });

      this.pythonProcess.on('error', (error) => {
        console.error('Failed to start Python bridge:', error);
        reject(error);
      });

      this.pythonProcess.on('exit', (code, signal) => {
        console.log(`Python bridge exited with code ${code} and signal ${signal}`);
        this.handlePythonProcessExit();
      });

      // Timeout if bridge doesn't start
      setTimeout(() => {
        if (!this.connected) {
          reject(new Error('Python bridge startup timeout'));
        }
      }, 10000);
    });
  }

  private async waitForBridgeReady(): Promise<void> {
    const maxAttempts = 20;
    const delay = 500;

    for (let i = 0; i < maxAttempts; i++) {
      try {
        // Try to connect to check if port is open
        const testSocket = new net.Socket();

        await new Promise<void>((resolve, reject) => {
          testSocket.connect(this.config.port, this.config.host, () => {
            testSocket.destroy();
            resolve();
          });

          testSocket.on('error', () => {
            reject();
          });

          setTimeout(() => reject(), 100);
        });

        return; // Bridge is ready

      } catch (error) {
        // Not ready yet, wait and retry
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw new Error('Python bridge failed to become ready');
  }

  private async connectToBridge(): Promise<void> {
    return new Promise((resolve, reject) => {
      console.log(`Connecting to Python bridge at ${this.config.host}:${this.config.port}`);

      this.client = new net.Socket();

      this.client.connect(this.config.port, this.config.host, () => {
        console.log('Connected to Python bridge');
        this.connected = true;
        resolve();
      });

      this.client.on('data', (data) => {
        this.handleData(data);
      });

      this.client.on('error', (error) => {
        console.error('Python bridge connection error:', error);
        this.handleConnectionError(error);
      });

      this.client.on('close', () => {
        console.log('Python bridge connection closed');
        this.handleConnectionClose();
      });

      // Connection timeout
      setTimeout(() => {
        if (!this.connected) {
          reject(new Error('Connection to Python bridge timed out'));
        }
      }, 5000);
    });
  }

  private handleData(data: Buffer): void {
    this.messageBuffer += data.toString();

    // Process complete JSON-RPC messages
    let newlineIndex: number;
    while ((newlineIndex = this.messageBuffer.indexOf('\n')) !== -1) {
      const message = this.messageBuffer.slice(0, newlineIndex);
      this.messageBuffer = this.messageBuffer.slice(newlineIndex + 1);

      try {
        const response = JSON.parse(message) as JsonRpcResponse;
        this.handleJsonRpcResponse(response);
      } catch (error) {
        console.error('Failed to parse JSON-RPC response:', error);
      }
    }
  }

  private handleJsonRpcResponse(response: JsonRpcResponse): void {
    const pending = this.pendingRequests.get(response.id);

    if (!pending) {
      console.warn('Received response for unknown request:', response.id);
      return;
    }

    this.pendingRequests.delete(response.id);
    this.metrics.responsesReceived++;

    if (response.error) {
      pending.reject(new Error(response.error.message));
    } else {
      pending.resolve(response);
    }
  }

  private async sendJsonRpcRequest(
    method: string,
    params: any,
    timeout?: number
  ): Promise<JsonRpcResponse> {
    if (!this.connected || !this.client) {
      throw new Error('Not connected to Python bridge');
    }

    const id = uuidv4();
    const request: JsonRpcRequest = {
      jsonrpc: '2.0',
      id,
      method,
      params
    };

    return new Promise((resolve, reject) => {
      // Set up timeout
      const timeoutMs = timeout || this.config.timeout;
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request timeout: ${method}`));
      }, timeoutMs);

      // Store pending request
      this.pendingRequests.set(id, {
        resolve: (value) => {
          clearTimeout(timer);
          resolve(value);
        },
        reject: (reason) => {
          clearTimeout(timer);
          reject(reason);
        },
        timestamp: Date.now()
      });

      // Send request
      const message = JSON.stringify(request) + '\n';
      this.client!.write(message);
      this.metrics.requestsSent++;
    });
  }

  private async configureBetaNet(): Promise<void> {
    await this.sendJsonRpcRequest('configure_betanet', {
      constitutional_tier: this.betaNetConfig.constitutionalTier,
      privacy_mode: this.betaNetConfig.privacyMode,
      enable_mixnode_routing: this.betaNetConfig.mixnodeRouting,
      enable_zero_knowledge_proofs: this.betaNetConfig.zeroKnowledgeProofs,
      enable_fog_integration: this.betaNetConfig.fogIntegration
    });
  }

  private startHealthChecks(): void {
    this.healthCheckTimer = setInterval(async () => {
      try {
        const health = await this.getHealthStatus();
        this.emit('healthCheck', health);

        if (health.status === 'unhealthy') {
          console.warn('Python bridge health check failed');
          await this.reconnect();
        }
      } catch (error) {
        console.error('Health check error:', error);
      }
    }, this.config.healthCheckInterval);
  }

  private async reconnect(): Promise<void> {
    if (this.reconnectTimer) {
      return; // Already reconnecting
    }

    console.log('Attempting to reconnect to Python bridge...');

    this.reconnectTimer = setTimeout(async () => {
      try {
        await this.connectToBridge();
        this.reconnectTimer = undefined;
      } catch (error) {
        console.error('Reconnection failed:', error);
        // Try again
        this.reconnectTimer = undefined;
        await this.reconnect();
      }
    }, this.config.retryDelay);
  }

  private handleConnectionError(error: Error): void {
    console.error('Connection error:', error);
    this.connected = false;
    this.emit('error', error);
  }

  private handleConnectionClose(): void {
    this.connected = false;
    this.emit('disconnected');

    // Attempt reconnection
    this.reconnect();
  }

  private handlePythonProcessExit(): void {
    console.error('Python bridge process exited unexpectedly');
    this.connected = false;

    // Restart Python bridge
    setTimeout(() => {
      this.initialize().catch(error => {
        console.error('Failed to restart Python bridge:', error);
      });
    }, 5000);
  }

  private updateMetrics(latency: number): void {
    this.metrics.latencies.push(latency);

    // Keep only last 1000 latencies
    if (this.metrics.latencies.length > 1000) {
      this.metrics.latencies.shift();
    }

    // Calculate average
    this.metrics.averageLatency = this.metrics.latencies.reduce((a, b) => a + b, 0) /
                                  this.metrics.latencies.length;

    // Calculate P95
    const sorted = [...this.metrics.latencies].sort((a, b) => a - b);
    const p95Index = Math.floor(sorted.length * 0.95);
    this.metrics.p95Latency = sorted[p95Index] || 0;

    // Check if we're meeting <75ms target
    if (this.metrics.p95Latency > 75) {
      this.emit('performanceWarning', {
        p95Latency: this.metrics.p95Latency,
        target: 75
      });
    }
  }

  /**
   * Get current metrics
   */
  public getMetrics(): typeof this.metrics {
    return { ...this.metrics };
  }
}

export default PythonBetaNetAdapter;