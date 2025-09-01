// AIVillage API Service Layer
import { APIResponse, Message, DigitalTwin, BitChatPeer, P2PMessage, MediaContent, ComputeCredit, FogNode } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

class APIService {
  private baseURL: string;
  private wsURL: string;
  private token: string | null = null;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.wsURL = WS_BASE_URL;
  }

  setAuthToken(token: string) {
    this.token = token;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<APIResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          error: data.message || 'Request failed',
        };
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Network error',
      };
    }
  }

  // Digital Twin Concierge API
  async getDigitalTwin(twinId: string): Promise<APIResponse<DigitalTwin>> {
    return this.request<DigitalTwin>(`/api/digital-twin/${twinId}`);
  }

  async sendConciergeMessage(twinId: string, message: Message): Promise<APIResponse<Message>> {
    return this.request<Message>(`/api/digital-twin/${twinId}/message`, {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }

  async getConversationHistory(twinId: string, limit?: number): Promise<APIResponse<Message[]>> {
    const params = limit ? `?limit=${limit}` : '';
    return this.request<Message[]>(`/api/digital-twin/${twinId}/history${params}`);
  }

  // BitChat P2P Messaging API
  async discoverPeers(userId: string): Promise<APIResponse<BitChatPeer[]>> {
    return this.request<BitChatPeer[]>(`/api/bitchat/${userId}/discover`);
  }

  async connectToPeer(userId: string, peerId: string): Promise<APIResponse<boolean>> {
    return this.request<boolean>(`/api/bitchat/${userId}/connect`, {
      method: 'POST',
      body: JSON.stringify({ peerId }),
    });
  }

  async sendP2PMessage(message: P2PMessage): Promise<APIResponse<boolean>> {
    return this.request<boolean>(`/api/bitchat/message`, {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }

  async getP2PConversation(userId: string, peerId: string): Promise<APIResponse<P2PMessage[]>> {
    return this.request<P2PMessage[]>(`/api/bitchat/${userId}/conversation/${peerId}`);
  }

  async getMeshNetworkStatus(userId: string): Promise<APIResponse<any>> {
    return this.request(`/api/bitchat/${userId}/mesh-status`);
  }

  // Media Display API
  async getMediaContent(contentId: string): Promise<APIResponse<MediaContent>> {
    return this.request<MediaContent>(`/api/media/${contentId}`);
  }

  async getMediaList(userId: string, type?: string): Promise<APIResponse<MediaContent[]>> {
    const params = type ? `?type=${type}` : '';
    return this.request<MediaContent[]>(`/api/media/list/${userId}${params}`);
  }

  async uploadMedia(file: File, metadata: any): Promise<APIResponse<MediaContent>> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));

    const response = await fetch(`${this.baseURL}/api/media/upload`, {
      method: 'POST',
      headers: {
        'Authorization': this.token ? `Bearer ${this.token}` : '',
      },
      body: formData,
    });

    const data = await response.json();
    return {
      success: response.ok,
      data: response.ok ? data : undefined,
      error: response.ok ? undefined : data.message,
    };
  }

  // Compute Credits Wallet API
  async getWalletBalance(userId: string): Promise<APIResponse<number>> {
    return this.request<number>(`/api/wallet/${userId}/balance`);
  }

  async getTransactionHistory(userId: string): Promise<APIResponse<ComputeCredit[]>> {
    return this.request<ComputeCredit[]>(`/api/wallet/${userId}/transactions`);
  }

  async transferCredits(fromUserId: string, toUserId: string, amount: number): Promise<APIResponse<boolean>> {
    return this.request<boolean>(`/api/wallet/transfer`, {
      method: 'POST',
      body: JSON.stringify({ fromUserId, toUserId, amount }),
    });
  }

  async getFogContributions(userId: string): Promise<APIResponse<FogNode[]>> {
    return this.request<FogNode[]>(`/api/fog/${userId}/contributions`);
  }

  async contributeToFog(userId: string, nodeId: string, resources: any): Promise<APIResponse<boolean>> {
    return this.request<boolean>(`/api/fog/contribute`, {
      method: 'POST',
      body: JSON.stringify({ userId, nodeId, resources }),
    });
  }

  // System Dashboard API
  async getSystemStatus(): Promise<APIResponse<any>> {
    return this.request(`/api/system/status`);
  }

  async getAgentStatuses(): Promise<APIResponse<any[]>> {
    return this.request(`/api/agents/status`);
  }

  async executeSystemCommand(command: string, params?: any): Promise<APIResponse<any>> {
    return this.request(`/api/system/command`, {
      method: 'POST',
      body: JSON.stringify({ command, params }),
    });
  }

  async getFogNetworkStatus(): Promise<APIResponse<any>> {
    return this.request(`/api/fog/network/status`);
  }

  // WebSocket Connections
  createWebSocket(endpoint: string, protocols?: string[]): WebSocket {
    const wsUrl = `${this.wsURL}${endpoint}`;
    const ws = new WebSocket(wsUrl, protocols);

    // Add authentication if token exists
    ws.addEventListener('open', () => {
      if (this.token) {
        ws.send(JSON.stringify({
          type: 'authenticate',
          token: this.token,
        }));
      }
    });

    return ws;
  }

  // Real-time subscriptions
  subscribeToConciergeUpdates(twinId: string, callback: (message: Message) => void): WebSocket {
    const ws = this.createWebSocket(`/ws/digital-twin/${twinId}`);

    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'message') {
          callback(data.payload);
        }
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    });

    return ws;
  }

  subscribeToP2PMessages(userId: string, callback: (message: P2PMessage) => void): WebSocket {
    const ws = this.createWebSocket(`/ws/bitchat/${userId}`);

    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'p2p_message') {
          callback(data.payload);
        }
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    });

    return ws;
  }

  subscribeToSystemUpdates(callback: (update: any) => void): WebSocket {
    const ws = this.createWebSocket('/ws/system/updates');

    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        callback(data);
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    });

    return ws;
  }

  subscribeToWalletUpdates(userId: string, callback: (update: any) => void): WebSocket {
    const ws = this.createWebSocket(`/ws/wallet/${userId}`);

    ws.addEventListener('message', (event) => {
      try {
        const data = JSON.parse(event.data);
        callback(data);
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    });

    return ws;
  }

  // Utility methods
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  async getServerInfo(): Promise<APIResponse<any>> {
    return this.request('/api/info');
  }
}

// Create singleton instance
export const apiService = new APIService();
export default apiService;
