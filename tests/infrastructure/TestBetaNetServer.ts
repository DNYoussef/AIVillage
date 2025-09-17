/**
 * Real Test BetaNet Server
 * Replaces mocked server with actual network communication for authentic testing
 */

import * as net from 'net';
import * as http from 'http';
import * as WebSocket from 'ws';
import { EventEmitter } from 'events';

interface BetaNetMessage {
  id: string;
  protocol: string;
  data: any;
  timestamp: number;
}

interface ServerConfig {
  port: number;
  wsPort: number;
  httpPort: number;
  latency?: number; // Simulate network latency
  errorRate?: number; // Simulate errors (0-1)
  packetLoss?: number; // Simulate packet loss (0-1)
}

export class TestBetaNetServer extends EventEmitter {
  private tcpServer: net.Server;
  private httpServer: http.Server;
  private wsServer: WebSocket.Server;
  private connections: Map<string, net.Socket | WebSocket> = new Map();
  private messageHistory: BetaNetMessage[] = [];
  private config: ServerConfig;
  private isRunning: boolean = false;

  constructor(config: ServerConfig) {
    super();
    this.config = config;
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Server already running');
    }

    await this.startTCPServer();
    await this.startHTTPServer();
    await this.startWebSocketServer();

    this.isRunning = true;
    this.emit('started');
  }

  private async startTCPServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.tcpServer = net.createServer((socket) => {
        const connectionId = `tcp-${Date.now()}-${Math.random()}`;
        this.connections.set(connectionId, socket);

        socket.on('data', async (data) => {
          await this.handleMessage(data.toString(), socket);
        });

        socket.on('error', (error) => {
          this.emit('connectionError', { connectionId, error });
        });

        socket.on('close', () => {
          this.connections.delete(connectionId);
          this.emit('connectionClosed', connectionId);
        });
      });

      this.tcpServer.listen(this.config.port, () => {
        console.log(`TCP Server listening on port ${this.config.port}`);
        resolve();
      });

      this.tcpServer.on('error', reject);
    });
  }

  private async startHTTPServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.httpServer = http.createServer(async (req, res) => {
        let body = '';

        req.on('data', chunk => {
          body += chunk.toString();
        });

        req.on('end', async () => {
          const response = await this.handleHTTPRequest(req.url!, body);

          res.writeHead(response.status, {
            'Content-Type': 'application/json',
            'X-Server-Timestamp': Date.now().toString()
          });

          res.end(JSON.stringify(response.body));
        });
      });

      this.httpServer.listen(this.config.httpPort, () => {
        console.log(`HTTP Server listening on port ${this.config.httpPort}`);
        resolve();
      });

      this.httpServer.on('error', reject);
    });
  }

  private async startWebSocketServer(): Promise<void> {
    return new Promise((resolve) => {
      this.wsServer = new WebSocket.Server({ port: this.config.wsPort });

      this.wsServer.on('connection', (ws) => {
        const connectionId = `ws-${Date.now()}-${Math.random()}`;
        this.connections.set(connectionId, ws);

        ws.on('message', async (data) => {
          await this.handleMessage(data.toString(), ws);
        });

        ws.on('error', (error) => {
          this.emit('connectionError', { connectionId, error });
        });

        ws.on('close', () => {
          this.connections.delete(connectionId);
          this.emit('connectionClosed', connectionId);
        });
      });

      console.log(`WebSocket Server listening on port ${this.config.wsPort}`);
      resolve();
    });
  }

  private async handleMessage(
    message: string,
    connection: net.Socket | WebSocket
  ): Promise<void> {
    // Simulate packet loss
    if (this.config.packetLoss && Math.random() < this.config.packetLoss) {
      return; // Drop the message
    }

    // Simulate latency
    if (this.config.latency) {
      await new Promise(resolve => setTimeout(resolve, this.config.latency));
    }

    try {
      const parsed = JSON.parse(message);
      this.messageHistory.push(parsed);

      // Simulate errors
      if (this.config.errorRate && Math.random() < this.config.errorRate) {
        const errorResponse = {
          jsonrpc: '2.0',
          id: parsed.id,
          error: {
            code: -32000,
            message: 'Simulated server error',
            data: 'Test error for validation'
          }
        };

        this.sendResponse(connection, errorResponse);
        return;
      }

      // Process the message based on method
      const response = await this.processMessage(parsed);
      this.sendResponse(connection, response);

    } catch (error) {
      const errorResponse = {
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32700,
          message: 'Parse error',
          data: error.message
        }
      };

      this.sendResponse(connection, errorResponse);
    }
  }

  private async processMessage(message: any): Promise<any> {
    const { method, params, id } = message;

    switch (method) {
      case 'ping':
        return {
          jsonrpc: '2.0',
          id,
          result: { pong: true, timestamp: Date.now() }
        };

      case 'health':
        return {
          jsonrpc: '2.0',
          id,
          result: {
            status: 'healthy',
            uptime: process.uptime(),
            connections: this.connections.size,
            messagesProcessed: this.messageHistory.length
          }
        };

      case 'processRequest':
        return this.processRequest(params, id);

      case 'validatePrivacy':
        return this.validatePrivacy(params, id);

      case 'routeProtocol':
        return this.routeProtocol(params, id);

      default:
        return {
          jsonrpc: '2.0',
          id,
          error: {
            code: -32601,
            message: 'Method not found',
            data: method
          }
        };
    }
  }

  private async processRequest(params: any, id: string): Promise<any> {
    // Simulate real processing with validation
    const { protocol, privacyTier, data } = params;

    // Validate protocol
    const validProtocols = ['betanet', 'bitchat', 'p2p', 'fog'];
    if (!validProtocols.includes(protocol)) {
      return {
        jsonrpc: '2.0',
        id,
        error: {
          code: -32602,
          message: 'Invalid protocol',
          data: protocol
        }
      };
    }

    // Simulate privacy tier processing
    const processedData = this.applyPrivacyTransformations(data, privacyTier);

    return {
      jsonrpc: '2.0',
      id,
      result: {
        success: true,
        protocol,
        privacyTier,
        data: processedData,
        timestamp: Date.now()
      }
    };
  }

  private validatePrivacy(params: any, id: string): any {
    const { privacyTier, data } = params;

    // Real privacy validation logic
    const violations = [];

    // Check for PII
    if (data.email && !data.email.includes('***')) {
      violations.push('Unmasked email detected');
    }

    if (data.ssn && privacyTier !== 'Platinum') {
      violations.push('SSN exposed in non-Platinum tier');
    }

    return {
      jsonrpc: '2.0',
      id,
      result: {
        isValid: violations.length === 0,
        violations,
        privacyTier,
        complianceScore: violations.length === 0 ? 1.0 : 0.5
      }
    };
  }

  private routeProtocol(params: any, id: string): any {
    const { protocol, data } = params;

    // Simulate protocol-specific routing
    const routingLatencies = {
      betanet: 10,
      bitchat: 15,
      p2p: 20,
      fog: 25
    };

    return {
      jsonrpc: '2.0',
      id,
      result: {
        protocol,
        routed: true,
        latency: routingLatencies[protocol] || 30,
        data
      }
    };
  }

  private applyPrivacyTransformations(data: any, tier: string): any {
    const transformed = { ...data };

    switch (tier) {
      case 'Bronze':
        // Heavy masking
        if (transformed.email) {
          transformed.email = transformed.email.replace(/@.*/, '@***');
        }
        delete transformed.userId;
        delete transformed.location;
        break;

      case 'Silver':
        // Moderate masking
        if (transformed.location) {
          transformed.location = {
            city: transformed.location.city,
            country: transformed.location.country
          };
        }
        break;

      case 'Gold':
        // Light masking
        delete transformed.internalData;
        break;

      case 'Platinum':
        // No masking
        break;
    }

    return transformed;
  }

  private async handleHTTPRequest(url: string, body: string): Promise<any> {
    if (url === '/health') {
      return {
        status: 200,
        body: {
          status: 'healthy',
          timestamp: Date.now()
        }
      };
    }

    if (url === '/bridge') {
      try {
        const message = JSON.parse(body);
        const response = await this.processMessage(message);
        return { status: 200, body: response };
      } catch (error) {
        return {
          status: 400,
          body: { error: 'Invalid request' }
        };
      }
    }

    return {
      status: 404,
      body: { error: 'Not found' }
    };
  }

  private sendResponse(
    connection: net.Socket | WebSocket,
    response: any
  ): void {
    const data = JSON.stringify(response);

    if (connection instanceof net.Socket) {
      connection.write(data + '\n');
    } else {
      connection.send(data);
    }

    this.emit('messageSent', response);
  }

  async simulateNetworkFailure(duration: number): Promise<void> {
    // Temporarily close all connections
    for (const connection of this.connections.values()) {
      if (connection instanceof net.Socket) {
        connection.destroy();
      } else {
        connection.close();
      }
    }

    await new Promise(resolve => setTimeout(resolve, duration));
  }

  async simulateHighLoad(requestsPerSecond: number, duration: number): Promise<void> {
    const endTime = Date.now() + duration;

    while (Date.now() < endTime) {
      // Generate synthetic load
      const message = {
        jsonrpc: '2.0',
        id: `load-${Date.now()}`,
        method: 'processRequest',
        params: {
          protocol: 'betanet',
          data: { load: true }
        }
      };

      // Send to random connection
      const connections = Array.from(this.connections.values());
      if (connections.length > 0) {
        const connection = connections[Math.floor(Math.random() * connections.length)];
        this.sendResponse(connection, await this.processMessage(message));
      }

      await new Promise(resolve => setTimeout(resolve, 1000 / requestsPerSecond));
    }
  }

  getMessageHistory(): BetaNetMessage[] {
    return [...this.messageHistory];
  }

  clearMessageHistory(): void {
    this.messageHistory = [];
  }

  async stop(): Promise<void> {
    // Close all connections
    for (const connection of this.connections.values()) {
      if (connection instanceof net.Socket) {
        connection.end();
      } else {
        connection.close();
      }
    }

    // Stop servers
    await new Promise<void>((resolve) => {
      this.tcpServer?.close(() => resolve());
    });

    await new Promise<void>((resolve) => {
      this.httpServer?.close(() => resolve());
    });

    this.wsServer?.close();

    this.isRunning = false;
    this.emit('stopped');
  }
}

// Export for testing
export default TestBetaNetServer;