/**
 * AIVillage API Client (TypeScript)
 *
 * Generated from OpenAPI 3.0 specification
 * Version: 1.0.0
 */

export interface Configuration {
  basePath?: string;
  accessToken?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  timeout?: number;
}

export interface RequestOptions {
  idempotencyKey?: string;
  headers?: Record<string, string>;
}

// Model interfaces matching OpenAPI schema
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  services: {
    database: 'up' | 'down';
    p2p_network: 'up' | 'down';
    agents: 'up' | 'down';
    rag_system: 'up' | 'down';
  };
  version: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  agent_preference?: 'king' | 'magi' | 'sage' | 'oracle' | 'navigator' | 'any';
  mode?: 'fast' | 'balanced' | 'comprehensive' | 'creative';
  user_context?: {
    device_type?: 'mobile' | 'desktop' | 'tablet';
    battery_level?: number;
    network_type?: 'wifi' | 'cellular' | 'offline';
  };
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  agent_used: string;
  processing_time_ms: number;
  metadata: {
    confidence: number;
    features_used: string[];
    thought_process?: string;
  };
}

export interface QueryRequest {
  query: string;
  mode?: 'fast' | 'balanced' | 'comprehensive' | 'creative' | 'analytical';
  include_sources?: boolean;
  max_results?: number;
  user_id?: string;
}

export interface QueryResponse {
  query_id: string;
  response: string;
  sources: Array<{
    title: string;
    content: string;
    confidence: number;
    url?: string;
  }>;
  metadata: {
    processing_time_ms: number;
    mode: string;
    features_enabled: Record<string, any>;
    bayesian_confidence: number;
  };
}

export interface Agent {
  id: string;
  name: string;
  category: 'governance' | 'infrastructure' | 'knowledge' | 'culture' | 'economy' | 'language' | 'health';
  capabilities: string[];
  status: 'available' | 'busy' | 'offline';
  current_load: number;
  specializations: string[];
}

export interface AgentTaskRequest {
  task_description: string;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  timeout_seconds?: number;
  context?: Record<string, any>;
}

export interface AgentTaskResponse {
  task_id: string;
  agent_id: string;
  status: 'accepted' | 'rejected' | 'completed' | 'failed';
  result: Record<string, any>;
  estimated_completion_time: string;
  metadata: Record<string, any>;
}

export interface ErrorResponse {
  detail: string;
  error_code: string;
  timestamp: string;
  request_id: string;
}

export class APIError extends Error {
  constructor(
    public status: number,
    public response: ErrorResponse,
    public headers: Record<string, string>
  ) {
    super(response.detail);
    this.name = 'APIError';
  }
}

export class RateLimitError extends APIError {
  constructor(
    public status: number,
    public response: ErrorResponse & { retry_after: number },
    public headers: Record<string, string>
  ) {
    super(status, response, headers);
    this.name = 'RateLimitError';
  }
}

export class BaseAPI {
  private config: Configuration;

  constructor(config: Configuration = {}) {
    this.config = {
      basePath: 'https://api.aivillage.io/v1',
      timeout: 30000,
      ...config
    };
  }

  protected async request<T>(
    method: string,
    path: string,
    body?: any,
    options: RequestOptions = {}
  ): Promise<T> {
    const url = `${this.config.basePath}${path}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'User-Agent': 'AIVillage-TypeScript-Client/1.0.0',
      ...this.config.headers,
      ...options.headers
    };

    // Authentication
    if (this.config.accessToken) {
      headers['Authorization'] = `Bearer ${this.config.accessToken}`;
    } else if (this.config.apiKey) {
      headers['x-api-key'] = this.config.apiKey;
    }

    // Idempotency key
    if (options.idempotencyKey) {
      headers['Idempotency-Key'] = options.idempotencyKey;
    }

    const requestConfig: RequestInit = {
      method,
      headers,
      ...(body && { body: JSON.stringify(body) })
    };

    // Add timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);
    requestConfig.signal = controller.signal;

    try {
      const response = await fetch(url, requestConfig);
      clearTimeout(timeoutId);

      const responseHeaders: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          detail: response.statusText,
          error_code: 'HTTP_ERROR',
          timestamp: new Date().toISOString(),
          request_id: responseHeaders['x-request-id'] || 'unknown'
        }));

        if (response.status === 429) {
          throw new RateLimitError(
            response.status,
            { ...errorData, retry_after: parseInt(responseHeaders['retry-after'] || '60') },
            responseHeaders
          );
        }

        throw new APIError(response.status, errorData, responseHeaders);
      }

      const data = await response.json();
      return data as T;

    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof APIError || error instanceof RateLimitError) {
        throw error;
      }

      if (error instanceof Error && error.name === 'AbortError') {
        throw new APIError(
          408,
          {
            detail: 'Request timeout',
            error_code: 'TIMEOUT',
            timestamp: new Date().toISOString(),
            request_id: 'unknown'
          },
          {}
        );
      }

      throw error;
    }
  }
}

export class HealthAPI extends BaseAPI {
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('GET', '/health');
  }
}

export class ChatAPI extends BaseAPI {
  async chat(request: ChatRequest, options: RequestOptions = {}): Promise<ChatResponse> {
    return this.request<ChatResponse>('POST', '/chat', request, options);
  }
}

export class RAGAPI extends BaseAPI {
  async processQuery(request: QueryRequest, options: RequestOptions = {}): Promise<QueryResponse> {
    return this.request<QueryResponse>('POST', '/query', request, options);
  }
}

export class AgentsAPI extends BaseAPI {
  async listAgents(params?: {
    category?: Agent['category'];
    available_only?: boolean;
  }): Promise<{ agents: Agent[]; total: number; categories: string[] }> {
    const searchParams = new URLSearchParams();
    if (params?.category) searchParams.set('category', params.category);
    if (params?.available_only !== undefined) {
      searchParams.set('available_only', params.available_only.toString());
    }

    const query = searchParams.toString();
    const path = query ? `/agents?${query}` : '/agents';

    return this.request<{ agents: Agent[]; total: number; categories: string[] }>('GET', path);
  }

  async assignAgentTask(
    agentId: string,
    request: AgentTaskRequest,
    options: RequestOptions = {}
  ): Promise<AgentTaskResponse> {
    return this.request<AgentTaskResponse>(
      'POST',
      `/agents/${encodeURIComponent(agentId)}/task`,
      request,
      options
    );
  }
}

export class P2PAPI extends BaseAPI {
  async getP2PStatus(): Promise<any> {
    return this.request<any>('GET', '/p2p/status');
  }

  async listPeers(transport?: 'bitchat' | 'betanet' | 'all'): Promise<any> {
    const query = transport ? `?transport=${transport}` : '';
    return this.request<any>('GET', `/p2p/peers${query}`);
  }
}

export class DigitalTwinAPI extends BaseAPI {
  async getDigitalTwinProfile(): Promise<any> {
    return this.request<any>('GET', '/digital-twin/profile');
  }

  async updateDigitalTwinData(
    data: any,
    options: RequestOptions = {}
  ): Promise<any> {
    return this.request<any>('POST', '/digital-twin/profile', data, options);
  }
}

/**
 * Main AIVillage API Client
 */
export class AIVillageClient {
  public health: HealthAPI;
  public chat: ChatAPI;
  public rag: RAGAPI;
  public agents: AgentsAPI;
  public p2p: P2PAPI;
  public digitalTwin: DigitalTwinAPI;

  constructor(config: Configuration = {}) {
    this.health = new HealthAPI(config);
    this.chat = new ChatAPI(config);
    this.rag = new RAGAPI(config);
    this.agents = new AgentsAPI(config);
    this.p2p = new P2PAPI(config);
    this.digitalTwin = new DigitalTwinAPI(config);
  }

  /**
   * Create client with Bearer token authentication
   */
  static withBearerToken(token: string, config: Partial<Configuration> = {}): AIVillageClient {
    return new AIVillageClient({
      ...config,
      accessToken: token
    });
  }

  /**
   * Create client with API key authentication
   */
  static withApiKey(apiKey: string, config: Partial<Configuration> = {}): AIVillageClient {
    return new AIVillageClient({
      ...config,
      apiKey: apiKey
    });
  }
}
