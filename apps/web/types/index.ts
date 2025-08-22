// AIVillage TypeScript Definitions

export interface User {
  id: string;
  name: string;
  avatar?: string;
  credits: number;
  isOnline: boolean;
}

export interface DigitalTwin {
  id: string;
  userId: string;
  name: string;
  specialization: string[];
  conversationHistory: Message[];
  isActive: boolean;
}

export interface Message {
  id: string;
  sender: string;
  content: string;
  timestamp: Date;
  type: 'user' | 'ai' | 'system';
  metadata?: Record<string, any>;
}

export interface BitChatPeer {
  id: string;
  name: string;
  avatar?: string;
  status: 'online' | 'offline' | 'away';
  lastSeen: Date;
  publicKey: string;
}

export interface P2PMessage extends Message {
  encrypted: boolean;
  recipient: string;
  deliveryStatus: 'sent' | 'delivered' | 'read';
}

export interface MediaContent {
  id: string;
  type: 'image' | 'video' | 'text' | 'audio';
  url?: string;
  content?: string;
  thumbnail?: string;
  metadata: {
    size?: number;
    duration?: number;
    dimensions?: { width: number; height: number };
    format: string;
  };
}

export interface ComputeCredit {
  id: string;
  userId: string;
  amount: number;
  type: 'earned' | 'spent' | 'transferred';
  description: string;
  timestamp: Date;
  relatedTaskId?: string;
}

export interface FogNode {
  id: string;
  name: string;
  location: string;
  resources: {
    cpu: number;
    memory: number;
    storage: number;
    bandwidth: number;
  };
  status: 'active' | 'inactive' | 'maintenance';
  reputation: number;
}

export interface WebRTCConnection {
  peerId: string;
  connection: RTCPeerConnection;
  dataChannel?: RTCDataChannel;
  status: 'connecting' | 'connected' | 'disconnected' | 'failed';
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface ChatState {
  messages: Message[];
  isTyping: boolean;
  isConnected: boolean;
  activeTwin?: DigitalTwin;
}

export interface MessagingState {
  peers: BitChatPeer[];
  conversations: Record<string, P2PMessage[]>;
  activeChat?: string;
  isDiscovering: boolean;
}

export interface WalletState {
  balance: number;
  transactions: ComputeCredit[];
  fogContributions: FogNode[];
  isLoading: boolean;
}

export interface SystemDashboard {
  agents: Array<{
    id: string;
    name: string;
    status: 'active' | 'idle' | 'busy';
    performance: number;
  }>;
  fogNodes: FogNode[];
  networkHealth: {
    p2pConnections: number;
    messageLatency: number;
    nodeCount: number;
  };
  systemMetrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkTraffic: number;
  };
}
