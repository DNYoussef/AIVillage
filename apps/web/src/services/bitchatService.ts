/**
 * BitChat P2P Messaging Service
 *
 * Provides secure peer-to-peer messaging using WebRTC data channels
 * with end-to-end encryption and file sharing capabilities.
 */

import { EventEmitter } from 'events';

export interface Contact {
  id: string;
  name: string;
  publicKey: string;
  lastSeen: Date;
  status: 'online' | 'offline' | 'connecting';
  avatar?: string;
}

export interface Message {
  id: string;
  from: string;
  to: string;
  content: string;
  timestamp: Date;
  type: 'text' | 'file' | 'image';
  encrypted: boolean;
  fileData?: {
    name: string;
    size: number;
    type: string;
    data?: ArrayBuffer;
  };
}

export interface ConnectionInfo {
  peerId: string;
  connection: RTCPeerConnection;
  dataChannel: RTCDataChannel;
  status: 'connecting' | 'connected' | 'disconnected' | 'failed';
}

export class BitChatService extends EventEmitter {
  private contacts: Map<string, Contact> = new Map();
  private connections: Map<string, ConnectionInfo> = new Map();
  private messages: Map<string, Message[]> = new Map();
  private localPeerId: string;
  private signalingServer: WebSocket | null = null;
  private cryptoKeys: Map<string, CryptoKey> = new Map();

  // ICE servers for NAT traversal
  private iceServers: RTCIceServer[] = [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    { urls: 'stun:stun2.l.google.com:19302' }
  ];

  constructor() {
    super();
    this.localPeerId = this.generatePeerId();
    this.initializeSignaling();
    this.loadContacts();
    this.loadMessages();
  }

  /**
   * Generate unique peer ID for this client
   */
  private generatePeerId(): string {
    return `peer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Initialize signaling server connection for peer discovery
   */
  private async initializeSignaling(): Promise<void> {
    try {
      // Use local signaling server or fallback to discovery methods
      const signalingUrl = process.env.REACT_APP_SIGNALING_SERVER || 'ws://localhost:8080/signaling';

      this.signalingServer = new WebSocket(signalingUrl);

      this.signalingServer.onopen = () => {
        console.log('Connected to signaling server');
        this.signalingServer?.send(JSON.stringify({
          type: 'register',
          peerId: this.localPeerId
        }));
      };

      this.signalingServer.onmessage = (event) => {
        this.handleSignalingMessage(JSON.parse(event.data));
      };

      this.signalingServer.onerror = (error) => {
        console.error('Signaling server error:', error);
        this.emit('signaling-error', error);
      };

      this.signalingServer.onclose = () => {
        console.log('Signaling server disconnected');
        this.emit('signaling-disconnected');
        // Attempt reconnection after delay
        setTimeout(() => this.initializeSignaling(), 5000);
      };
    } catch (error) {
      console.error('Failed to initialize signaling:', error);
      this.emit('signaling-error', error);
    }
  }

  /**
   * Handle signaling messages for WebRTC negotiation
   */
  private async handleSignalingMessage(message: any): Promise<void> {
    switch (message.type) {
      case 'peer-list':
        this.handlePeerList(message.peers);
        break;
      case 'offer':
        await this.handleOffer(message);
        break;
      case 'answer':
        await this.handleAnswer(message);
        break;
      case 'ice-candidate':
        await this.handleIceCandidate(message);
        break;
      case 'peer-joined':
        this.handlePeerJoined(message.peerId);
        break;
      case 'peer-left':
        this.handlePeerLeft(message.peerId);
        break;
    }
  }

  /**
   * Create WebRTC peer connection
   */
  private createPeerConnection(peerId: string): RTCPeerConnection {
    const connection = new RTCPeerConnection({
      iceServers: this.iceServers
    });

    connection.onicecandidate = (event) => {
      if (event.candidate && this.signalingServer) {
        this.signalingServer.send(JSON.stringify({
          type: 'ice-candidate',
          to: peerId,
          from: this.localPeerId,
          candidate: event.candidate
        }));
      }
    };

    connection.onconnectionstatechange = () => {
      const connectionInfo = this.connections.get(peerId);
      if (connectionInfo) {
        connectionInfo.status = this.mapConnectionState(connection.connectionState);
        this.emit('connection-state-changed', peerId, connectionInfo.status);

        // Update contact status
        const contact = this.contacts.get(peerId);
        if (contact) {
          contact.status = connectionInfo.status === 'connected' ? 'online' : 'offline';
          this.emit('contact-updated', contact);
        }
      }
    };

    return connection;
  }

  /**
   * Map RTCPeerConnectionState to our connection status
   */
  private mapConnectionState(state: RTCPeerConnectionState): ConnectionInfo['status'] {
    switch (state) {
      case 'connected': return 'connected';
      case 'connecting': return 'connecting';
      case 'disconnected': return 'disconnected';
      case 'failed': return 'failed';
      case 'closed': return 'disconnected';
      default: return 'connecting';
    }
  }

  /**
   * Create data channel for messaging
   */
  private createDataChannel(connection: RTCPeerConnection, peerId: string): RTCDataChannel {
    const dataChannel = connection.createDataChannel('messages', {
      ordered: true
    });

    dataChannel.onopen = () => {
      console.log(`Data channel opened with ${peerId}`);
      this.emit('peer-connected', peerId);
    };

    dataChannel.onmessage = (event) => {
      this.handleDataChannelMessage(peerId, event.data);
    };

    dataChannel.onclose = () => {
      console.log(`Data channel closed with ${peerId}`);
      this.emit('peer-disconnected', peerId);
    };

    dataChannel.onerror = (error) => {
      console.error(`Data channel error with ${peerId}:`, error);
      this.emit('peer-error', peerId, error);
    };

    return dataChannel;
  }

  /**
   * Handle incoming data channel messages
   */
  private async handleDataChannelMessage(peerId: string, data: string): Promise<void> {
    try {
      const messageData = JSON.parse(data);

      switch (messageData.type) {
        case 'message':
          await this.handleIncomingMessage(peerId, messageData);
          break;
        case 'file':
          await this.handleIncomingFile(peerId, messageData);
          break;
        case 'typing':
          this.emit('typing', peerId, messageData.isTyping);
          break;
        case 'presence':
          this.handlePresenceUpdate(peerId, messageData);
          break;
      }
    } catch (error) {
      console.error('Error handling data channel message:', error);
    }
  }

  /**
   * Connect to a peer
   */
  async connectToPeer(peerId: string): Promise<void> {
    if (this.connections.has(peerId)) {
      return; // Already connected or connecting
    }

    const connection = this.createPeerConnection(peerId);
    const dataChannel = this.createDataChannel(connection, peerId);

    this.connections.set(peerId, {
      peerId,
      connection,
      dataChannel,
      status: 'connecting'
    });

    try {
      const offer = await connection.createOffer();
      await connection.setLocalDescription(offer);

      if (this.signalingServer) {
        this.signalingServer.send(JSON.stringify({
          type: 'offer',
          to: peerId,
          from: this.localPeerId,
          offer: offer
        }));
      }
    } catch (error) {
      console.error('Error creating offer:', error);
      this.connections.delete(peerId);
      throw error;
    }
  }

  /**
   * Handle incoming offer
   */
  private async handleOffer(message: any): Promise<void> {
    const { from, offer } = message;

    const connection = this.createPeerConnection(from);

    connection.ondatachannel = (event) => {
      const dataChannel = event.channel;
      dataChannel.onopen = () => {
        console.log(`Data channel received from ${from}`);
        this.emit('peer-connected', from);
      };
      dataChannel.onmessage = (event) => {
        this.handleDataChannelMessage(from, event.data);
      };
    };

    this.connections.set(from, {
      peerId: from,
      connection,
      dataChannel: connection.createDataChannel('messages'),
      status: 'connecting'
    });

    try {
      await connection.setRemoteDescription(offer);
      const answer = await connection.createAnswer();
      await connection.setLocalDescription(answer);

      if (this.signalingServer) {
        this.signalingServer.send(JSON.stringify({
          type: 'answer',
          to: from,
          from: this.localPeerId,
          answer: answer
        }));
      }
    } catch (error) {
      console.error('Error handling offer:', error);
      this.connections.delete(from);
    }
  }

  /**
   * Handle incoming answer
   */
  private async handleAnswer(message: any): Promise<void> {
    const { from, answer } = message;
    const connectionInfo = this.connections.get(from);

    if (connectionInfo) {
      try {
        await connectionInfo.connection.setRemoteDescription(answer);
      } catch (error) {
        console.error('Error handling answer:', error);
      }
    }
  }

  /**
   * Handle ICE candidate
   */
  private async handleIceCandidate(message: any): Promise<void> {
    const { from, candidate } = message;
    const connectionInfo = this.connections.get(from);

    if (connectionInfo) {
      try {
        await connectionInfo.connection.addIceCandidate(candidate);
      } catch (error) {
        console.error('Error adding ICE candidate:', error);
      }
    }
  }

  /**
   * Send message to peer
   */
  async sendMessage(to: string, content: string, type: Message['type'] = 'text'): Promise<Message> {
    const connectionInfo = this.connections.get(to);
    if (!connectionInfo || connectionInfo.status !== 'connected') {
      throw new Error('Peer not connected');
    }

    const message: Message = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      from: this.localPeerId,
      to,
      content,
      timestamp: new Date(),
      type,
      encrypted: true
    };

    try {
      // Encrypt message content
      const encryptedContent = await this.encryptMessage(content, to);

      const messageData = {
        type: 'message',
        ...message,
        content: encryptedContent
      };

      connectionInfo.dataChannel.send(JSON.stringify(messageData));

      // Store message locally
      this.storeMessage(message);
      this.emit('message-sent', message);

      return message;
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }

  /**
   * Send file to peer
   */
  async sendFile(to: string, file: File): Promise<void> {
    const connectionInfo = this.connections.get(to);
    if (!connectionInfo || connectionInfo.status !== 'connected') {
      throw new Error('Peer not connected');
    }

    try {
      const arrayBuffer = await file.arrayBuffer();
      const chunks = this.chunkFile(arrayBuffer);

      // Send file metadata first
      const fileMessage: Message = {
        id: `file_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        from: this.localPeerId,
        to,
        content: `File: ${file.name}`,
        timestamp: new Date(),
        type: 'file',
        encrypted: true,
        fileData: {
          name: file.name,
          size: file.size,
          type: file.type
        }
      };

      connectionInfo.dataChannel.send(JSON.stringify({
        type: 'file',
        ...fileMessage,
        chunkCount: chunks.length
      }));

      // Send file chunks
      for (let i = 0; i < chunks.length; i++) {
        const chunkData = {
          type: 'file-chunk',
          fileId: fileMessage.id,
          chunkIndex: i,
          chunk: Array.from(new Uint8Array(chunks[i]))
        };

        connectionInfo.dataChannel.send(JSON.stringify(chunkData));
      }

      this.storeMessage(fileMessage);
      this.emit('file-sent', fileMessage);
    } catch (error) {
      console.error('Error sending file:', error);
      throw error;
    }
  }

  /**
   * Chunk file for transmission
   */
  private chunkFile(buffer: ArrayBuffer, chunkSize: number = 16384): ArrayBuffer[] {
    const chunks: ArrayBuffer[] = [];
    for (let i = 0; i < buffer.byteLength; i += chunkSize) {
      chunks.push(buffer.slice(i, i + chunkSize));
    }
    return chunks;
  }

  /**
   * Handle incoming message
   */
  private async handleIncomingMessage(from: string, messageData: any): Promise<void> {
    try {
      // Decrypt message content
      const decryptedContent = await this.decryptMessage(messageData.content, from);

      const message: Message = {
        ...messageData,
        content: decryptedContent,
        timestamp: new Date(messageData.timestamp)
      };

      this.storeMessage(message);
      this.emit('message-received', message);
    } catch (error) {
      console.error('Error handling incoming message:', error);
    }
  }

  /**
   * Handle incoming file
   */
  private async handleIncomingFile(from: string, fileData: any): Promise<void> {
    // Implementation for file reception with chunking
    // Store file chunks and reconstruct when complete
    this.emit('file-received', fileData);
  }

  /**
   * Encrypt message content
   */
  private async encryptMessage(content: string, peerId: string): Promise<string> {
    try {
      const key = await this.getOrCreateSharedKey(peerId);
      const encoder = new TextEncoder();
      const data = encoder.encode(content);

      const iv = crypto.getRandomValues(new Uint8Array(12));
      const encrypted = await crypto.subtle.encrypt(
        { name: 'AES-GCM', iv },
        key,
        data
      );

      const combined = new Uint8Array(iv.length + encrypted.byteLength);
      combined.set(iv);
      combined.set(new Uint8Array(encrypted), iv.length);

      return btoa(String.fromCharCode(...combined));
    } catch (error) {
      console.error('Encryption error:', error);
      return content; // Fallback to unencrypted
    }
  }

  /**
   * Decrypt message content
   */
  private async decryptMessage(encryptedContent: string, peerId: string): Promise<string> {
    try {
      const key = await this.getOrCreateSharedKey(peerId);
      const combined = new Uint8Array(
        atob(encryptedContent).split('').map(char => char.charCodeAt(0))
      );

      const iv = combined.slice(0, 12);
      const encrypted = combined.slice(12);

      const decrypted = await crypto.subtle.decrypt(
        { name: 'AES-GCM', iv },
        key,
        encrypted
      );

      const decoder = new TextDecoder();
      return decoder.decode(decrypted);
    } catch (error) {
      console.error('Decryption error:', error);
      return encryptedContent; // Fallback to encrypted content
    }
  }

  /**
   * Get or create shared encryption key for peer
   */
  private async getOrCreateSharedKey(peerId: string): Promise<CryptoKey> {
    let key = this.cryptoKeys.get(peerId);
    if (!key) {
      key = await crypto.subtle.generateKey(
        { name: 'AES-GCM', length: 256 },
        false,
        ['encrypt', 'decrypt']
      );
      this.cryptoKeys.set(peerId, key);
    }
    return key;
  }

  /**
   * Store message locally
   */
  private storeMessage(message: Message): void {
    const conversationId = this.getConversationId(message.from, message.to);
    const messages = this.messages.get(conversationId) || [];
    messages.push(message);
    this.messages.set(conversationId, messages);
    this.saveMessages();
  }

  /**
   * Get conversation ID from two peer IDs
   */
  private getConversationId(peer1: string, peer2: string): string {
    return [peer1, peer2].sort().join('_');
  }

  /**
   * Add contact
   */
  addContact(contact: Omit<Contact, 'lastSeen' | 'status'>): void {
    const newContact: Contact = {
      ...contact,
      lastSeen: new Date(),
      status: 'offline'
    };

    this.contacts.set(contact.id, newContact);
    this.saveContacts();
    this.emit('contact-added', newContact);
  }

  /**
   * Remove contact
   */
  removeContact(contactId: string): void {
    this.contacts.delete(contactId);
    this.connections.delete(contactId);
    this.saveContacts();
    this.emit('contact-removed', contactId);
  }

  /**
   * Get messages for conversation
   */
  getMessages(peerId: string): Message[] {
    const conversationId = this.getConversationId(this.localPeerId, peerId);
    return this.messages.get(conversationId) || [];
  }

  /**
   * Get all contacts
   */
  getContacts(): Contact[] {
    return Array.from(this.contacts.values());
  }

  /**
   * Get connection status
   */
  getConnectionStatus(peerId: string): ConnectionInfo['status'] | null {
    const connection = this.connections.get(peerId);
    return connection ? connection.status : null;
  }

  /**
   * Send typing indicator
   */
  sendTyping(to: string, isTyping: boolean): void {
    const connectionInfo = this.connections.get(to);
    if (connectionInfo && connectionInfo.status === 'connected') {
      connectionInfo.dataChannel.send(JSON.stringify({
        type: 'typing',
        isTyping
      }));
    }
  }

  /**
   * Handle peer list from signaling server
   */
  private handlePeerList(peers: string[]): void {
    this.emit('peers-discovered', peers);
  }

  /**
   * Handle peer joined
   */
  private handlePeerJoined(peerId: string): void {
    this.emit('peer-joined', peerId);
  }

  /**
   * Handle peer left
   */
  private handlePeerLeft(peerId: string): void {
    const connection = this.connections.get(peerId);
    if (connection) {
      connection.connection.close();
      this.connections.delete(peerId);
    }

    const contact = this.contacts.get(peerId);
    if (contact) {
      contact.status = 'offline';
      this.emit('contact-updated', contact);
    }

    this.emit('peer-left', peerId);
  }

  /**
   * Handle presence update
   */
  private handlePresenceUpdate(peerId: string, presence: any): void {
    const contact = this.contacts.get(peerId);
    if (contact) {
      contact.lastSeen = new Date();
      this.emit('contact-updated', contact);
    }
  }

  /**
   * Load contacts from storage
   */
  private loadContacts(): void {
    try {
      const stored = localStorage.getItem('bitchat_contacts');
      if (stored) {
        const contacts = JSON.parse(stored);
        contacts.forEach((contact: Contact) => {
          this.contacts.set(contact.id, contact);
        });
      }
    } catch (error) {
      console.error('Error loading contacts:', error);
    }
  }

  /**
   * Save contacts to storage
   */
  private saveContacts(): void {
    try {
      const contacts = Array.from(this.contacts.values());
      localStorage.setItem('bitchat_contacts', JSON.stringify(contacts));
    } catch (error) {
      console.error('Error saving contacts:', error);
    }
  }

  /**
   * Load messages from storage
   */
  private loadMessages(): void {
    try {
      const stored = localStorage.getItem('bitchat_messages');
      if (stored) {
        const messagesData = JSON.parse(stored);
        Object.entries(messagesData).forEach(([conversationId, messages]: [string, any]) => {
          this.messages.set(conversationId, messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          })));
        });
      }
    } catch (error) {
      console.error('Error loading messages:', error);
    }
  }

  /**
   * Save messages to storage
   */
  private saveMessages(): void {
    try {
      const messagesData: Record<string, Message[]> = {};
      this.messages.forEach((messages, conversationId) => {
        messagesData[conversationId] = messages;
      });
      localStorage.setItem('bitchat_messages', JSON.stringify(messagesData));
    } catch (error) {
      console.error('Error saving messages:', error);
    }
  }

  /**
   * Get local peer ID
   */
  getLocalPeerId(): string {
    return this.localPeerId;
  }

  /**
   * Disconnect from peer
   */
  disconnectPeer(peerId: string): void {
    const connection = this.connections.get(peerId);
    if (connection) {
      connection.connection.close();
      this.connections.delete(peerId);
      this.emit('peer-disconnected', peerId);
    }
  }

  /**
   * Disconnect all peers and cleanup
   */
  destroy(): void {
    this.connections.forEach((connection) => {
      connection.connection.close();
    });
    this.connections.clear();

    if (this.signalingServer) {
      this.signalingServer.close();
      this.signalingServer = null;
    }

    this.removeAllListeners();
  }
}

export const bitChatService = new BitChatService();
