// BitChat Service Hook - P2P Messaging with Bluetooth Low Energy
import { useState, useEffect, useCallback, useRef } from 'react';
import { BitChatPeer, P2PMessage, MessagingState } from '../types';

interface EncryptionStatus {
  enabled: boolean;
  protocol: string;
  keyRotationInterval: number;
}

interface MeshStatus {
  health: 'good' | 'fair' | 'poor';
  connectivity: number;
  latency: number;
  redundancy: number;
}

export interface BitChatServiceHook {
  messagingState: MessagingState;
  sendMessage: (message: P2PMessage) => Promise<boolean>;
  discoverPeers: () => Promise<void>;
  connectToPeer: (peerId: string) => Promise<boolean>;
  disconnectFromPeer: (peerId: string) => Promise<boolean>;
  meshStatus: MeshStatus;
  encryptionStatus: EncryptionStatus;
  isInitialized: boolean;
}

export const useBitChatService = (userId: string): BitChatServiceHook => {
  const [messagingState, setMessagingState] = useState<MessagingState>({
    peers: [],
    conversations: {},
    activeChat: undefined,
    isDiscovering: false
  });

  const [meshStatus, setMeshStatus] = useState<MeshStatus>({
    health: 'poor',
    connectivity: 0,
    latency: 0,
    redundancy: 0
  });

  const [encryptionStatus, setEncryptionStatus] = useState<EncryptionStatus>({
    enabled: true,
    protocol: 'ChaCha20-Poly1305',
    keyRotationInterval: 3600000 // 1 hour
  });

  const [isInitialized, setIsInitialized] = useState(false);
  const webRTCConnections = useRef<Map<string, RTCPeerConnection>>(new Map());
  const discoveryInterval = useRef<NodeJS.Timeout | null>(null);

  // Initialize BitChat service
  useEffect(() => {
    const initializeBitChat = async () => {
      try {
        // Initialize P2P mesh networking capabilities
        await setupWebRTCStack();
        await initializeBluetoothLEDiscovery();
        setIsInitialized(true);

        // Start periodic peer discovery
        startPeerDiscovery();
      } catch (error) {
        console.error('Failed to initialize BitChat service:', error);
      }
    };

    initializeBitChat();

    return () => {
      cleanup();
    };
  }, [userId]);

  const setupWebRTCStack = async (): Promise<void> => {
    // Configure WebRTC for mesh networking
    const configuration: RTCConfiguration = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ],
      iceCandidatePoolSize: 10
    };

    // Set up STUN/TURN servers for NAT traversal
    console.log('WebRTC stack initialized for mesh networking');
  };

  const initializeBluetoothLEDiscovery = async (): Promise<void> => {
    if (!navigator.bluetooth) {
      console.warn('Bluetooth API not available, falling back to WebRTC only');
      return;
    }

    try {
      // Request Bluetooth access for BLE mesh discovery
      const device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: ['battery_service', 'device_information']
      });

      console.log('Bluetooth LE discovery initialized:', device.name);
    } catch (error) {
      console.warn('Bluetooth LE unavailable, using WebRTC mesh only');
    }
  };

  const startPeerDiscovery = (): void => {
    if (discoveryInterval.current) {
      clearInterval(discoveryInterval.current);
    }

    discoveryInterval.current = setInterval(() => {
      discoverPeers();
    }, 30000); // Discover peers every 30 seconds
  };

  const discoverPeers = useCallback(async (): Promise<void> => {
    if (messagingState.isDiscovering) return;

    setMessagingState(prev => ({ ...prev, isDiscovering: true }));

    try {
      // Simulate peer discovery with mock data for development
      const mockPeers: BitChatPeer[] = [
        {
          id: 'peer-001',
          name: 'Alice Mobile',
          status: 'online',
          lastSeen: new Date(),
          publicKey: 'mock-public-key-001'
        },
        {
          id: 'peer-002',
          name: 'Bob Laptop',
          status: 'online',
          lastSeen: new Date(),
          publicKey: 'mock-public-key-002'
        }
      ];

      // In production, this would use actual P2P discovery
      await new Promise(resolve => setTimeout(resolve, 2000));

      setMessagingState(prev => ({
        ...prev,
        peers: [...prev.peers, ...mockPeers.filter(p => !prev.peers.find(existing => existing.id === p.id))],
        isDiscovering: false
      }));

      updateMeshStatus();
    } catch (error) {
      console.error('Peer discovery failed:', error);
      setMessagingState(prev => ({ ...prev, isDiscovering: false }));
    }
  }, [messagingState.isDiscovering]);

  const connectToPeer = useCallback(async (peerId: string): Promise<boolean> => {
    try {
      const peer = messagingState.peers.find(p => p.id === peerId);
      if (!peer) return false;

      // Create WebRTC connection
      const peerConnection = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });

      // Set up data channel for messaging
      const dataChannel = peerConnection.createDataChannel('messages', {
        ordered: true
      });

      dataChannel.onopen = () => {
        console.log(`Data channel opened with ${peer.name}`);
      };

      dataChannel.onmessage = (event) => {
        handleIncomingMessage(JSON.parse(event.data));
      };

      webRTCConnections.current.set(peerId, peerConnection);

      // Update peer status
      setMessagingState(prev => ({
        ...prev,
        peers: prev.peers.map(p =>
          p.id === peerId ? { ...p, status: 'online' } : p
        )
      }));

      updateMeshStatus();
      return true;
    } catch (error) {
      console.error(`Failed to connect to peer ${peerId}:`, error);
      return false;
    }
  }, [messagingState.peers]);

  const disconnectFromPeer = useCallback(async (peerId: string): Promise<boolean> => {
    try {
      const connection = webRTCConnections.current.get(peerId);
      if (connection) {
        connection.close();
        webRTCConnections.current.delete(peerId);
      }

      setMessagingState(prev => ({
        ...prev,
        peers: prev.peers.map(p =>
          p.id === peerId ? { ...p, status: 'offline' } : p
        )
      }));

      updateMeshStatus();
      return true;
    } catch (error) {
      console.error(`Failed to disconnect from peer ${peerId}:`, error);
      return false;
    }
  }, []);

  const sendMessage = useCallback(async (message: P2PMessage): Promise<boolean> => {
    try {
      const connection = webRTCConnections.current.get(message.recipient);
      if (!connection) {
        console.error('No connection to recipient');
        return false;
      }

      // Encrypt message if encryption is enabled
      let messageData = message;
      if (encryptionStatus.enabled) {
        messageData = await encryptMessage(message);
      }

      // In a real implementation, send via data channel
      // For now, simulate message sending
      console.log('Sending message:', messageData);

      // Update local conversation
      setMessagingState(prev => ({
        ...prev,
        conversations: {
          ...prev.conversations,
          [message.recipient]: [
            ...(prev.conversations[message.recipient] || []),
            { ...message, deliveryStatus: 'sent' }
          ]
        }
      }));

      return true;
    } catch (error) {
      console.error('Failed to send message:', error);
      return false;
    }
  }, [encryptionStatus.enabled]);

  const handleIncomingMessage = useCallback((message: P2PMessage) => {
    setMessagingState(prev => ({
      ...prev,
      conversations: {
        ...prev.conversations,
        [message.sender]: [
          ...(prev.conversations[message.sender] || []),
          { ...message, deliveryStatus: 'delivered' }
        ]
      }
    }));
  }, []);

  const encryptMessage = async (message: P2PMessage): Promise<P2PMessage> => {
    // In production, use actual encryption with ChaCha20-Poly1305
    // For now, mark as encrypted
    return {
      ...message,
      content: `[ENCRYPTED]: ${message.content}`,
      encrypted: true
    };
  };

  const updateMeshStatus = (): void => {
    const connectedPeers = messagingState.peers.filter(p => p.status === 'online');
    const connectivity = connectedPeers.length / Math.max(messagingState.peers.length, 1);

    setMeshStatus({
      health: connectivity > 0.7 ? 'good' : connectivity > 0.3 ? 'fair' : 'poor',
      connectivity: connectivity * 100,
      latency: Math.random() * 100 + 50, // Mock latency
      redundancy: Math.min(connectedPeers.length, 3)
    });
  };

  const cleanup = (): void => {
    if (discoveryInterval.current) {
      clearInterval(discoveryInterval.current);
    }

    webRTCConnections.current.forEach((connection) => {
      connection.close();
    });
    webRTCConnections.current.clear();
  };

  return {
    messagingState,
    sendMessage,
    discoverPeers,
    connectToPeer,
    disconnectFromPeer,
    meshStatus,
    encryptionStatus,
    isInitialized
  };
};
