import React, { useState, useEffect } from 'react';
import { BitChatPeer, P2PMessage, MessagingState } from '../../types';
import { useBitChatService } from '../../hooks/useBitChatService';
import { PeerList } from './PeerList';
import { ConversationView } from './ConversationView';
import { NetworkStatus } from './NetworkStatus';
import { EncryptionBadge } from '../common/EncryptionBadge';
import './BitChatInterface.css';

interface BitChatInterfaceProps {
  userId: string;
  onPeerConnect?: (peer: BitChatPeer) => void;
  onMessageReceived?: (message: P2PMessage) => void;
}

export const BitChatInterface: React.FC<BitChatInterfaceProps> = ({
  userId,
  onPeerConnect,
  onMessageReceived
}) => {
  const [selectedPeer, setSelectedPeer] = useState<string | null>(null);
  const [showPeerDiscovery, setShowPeerDiscovery] = useState(false);

  const {
    messagingState,
    sendMessage,
    discoverPeers,
    connectToPeer,
    disconnectFromPeer,
    meshStatus,
    encryptionStatus
  } = useBitChatService(userId);

  useEffect(() => {
    // Auto-connect to nearby peers on component mount
    discoverPeers();
  }, [discoverPeers]);

  useEffect(() => {
    // Handle new peer connections
    if (onPeerConnect && messagingState.peers.length > 0) {
      const latestPeer = messagingState.peers[messagingState.peers.length - 1];
      onPeerConnect(latestPeer);
    }
  }, [messagingState.peers, onPeerConnect]);

  const handlePeerSelect = (peerId: string) => {
    setSelectedPeer(peerId);
  };

  const handleSendMessage = async (content: string, recipientId: string) => {
    const message: P2PMessage = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      sender: userId,
      content,
      timestamp: new Date(),
      type: 'user',
      encrypted: encryptionStatus.enabled,
      recipient: recipientId,
      deliveryStatus: 'sent'
    };

    const success = await sendMessage(message);
    if (success && onMessageReceived) {
      onMessageReceived(message);
    }
  };

  const selectedPeerData = messagingState.peers.find(p => p.id === selectedPeer);
  const conversation = selectedPeer ? messagingState.conversations[selectedPeer] || [] : [];

  return (
    <div className="bitchat-interface">
      <div className="bitchat-header">
        <div className="header-title">
          <h2>BitChat Mesh Network</h2>
          <EncryptionBadge
            enabled={encryptionStatus.enabled}
            protocol={encryptionStatus.protocol}
          />
        </div>
        <div className="header-controls">
          <NetworkStatus
            status={meshStatus}
            peerCount={messagingState.peers.length}
            isDiscovering={messagingState.isDiscovering}
          />
          <button
            onClick={() => setShowPeerDiscovery(!showPeerDiscovery)}
            className={`discovery-btn ${showPeerDiscovery ? 'active' : ''}`}
            aria-label="Toggle peer discovery"
          >
            📡
          </button>
        </div>
      </div>

      <div className="bitchat-content">
        <div className="peer-sidebar">
          <PeerList
            peers={messagingState.peers}
            selectedPeer={selectedPeer}
            onPeerSelect={handlePeerSelect}
            onPeerConnect={connectToPeer}
            onPeerDisconnect={disconnectFromPeer}
            conversations={messagingState.conversations}
          />

          {showPeerDiscovery && (
            <div className="peer-discovery">
              <h3>Discover Nearby Peers</h3>
              <button
                onClick={discoverPeers}
                disabled={messagingState.isDiscovering}
                className="discover-btn"
              >
                {messagingState.isDiscovering ? 'Scanning...' : 'Scan for Peers'}
              </button>
              <div className="discovery-info">
                <p>Using Bluetooth Low Energy mesh networking</p>
                <p>Range: ~100m | Encrypted: {encryptionStatus.enabled ? 'Yes' : 'No'}</p>
              </div>
            </div>
          )}
        </div>

        <div className="conversation-area">
          {selectedPeerData ? (
            <ConversationView
              peer={selectedPeerData}
              messages={conversation}
              onSendMessage={(content) => handleSendMessage(content, selectedPeer!)}
              encryptionEnabled={encryptionStatus.enabled}
            />
          ) : (
            <div className="no-conversation">
              <div className="no-conversation-icon">💬</div>
              <h3>Select a peer to start messaging</h3>
              <p>
                Connect to nearby devices using our secure P2P mesh network.
                All messages are end-to-end encrypted.
              </p>
              {messagingState.peers.length === 0 && (
                <button onClick={discoverPeers} className="discover-first-peer-btn">
                  Find Nearby Peers
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="bitchat-status">
        <div className="status-item">
          <span className="status-label">Connected Peers:</span>
          <span className="status-value">{messagingState.peers.filter(p => p.status === 'online').length}</span>
        </div>
        <div className="status-item">
          <span className="status-label">Mesh Health:</span>
          <span className={`status-value ${meshStatus.health}`}>
            {meshStatus.health === 'good' ? '🟢' : meshStatus.health === 'fair' ? '🟡' : '🔴'}
          </span>
        </div>
        <div className="status-item">
          <span className="status-label">Total Messages:</span>
          <span className="status-value">
            {Object.values(messagingState.conversations).reduce((total, conv) => total + conv.length, 0)}
          </span>
        </div>
      </div>
    </div>
  );
};
