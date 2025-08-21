// Peer List Component - Shows connected BitChat peers
import React from 'react';
import { BitChatPeer, P2PMessage } from '../../types';
import './PeerList.css';

interface PeerListProps {
  peers: BitChatPeer[];
  selectedPeer: string | null;
  onPeerSelect: (peerId: string) => void;
  onPeerConnect: (peerId: string) => Promise<boolean>;
  onPeerDisconnect: (peerId: string) => Promise<boolean>;
  conversations: Record<string, P2PMessage[]>;
  className?: string;
}

export const PeerList: React.FC<PeerListProps> = ({
  peers,
  selectedPeer,
  onPeerSelect,
  onPeerConnect,
  onPeerDisconnect,
  conversations,
  className = ''
}) => {
  const getLastMessage = (peerId: string): P2PMessage | null => {
    const peerConversation = conversations[peerId] || [];
    return peerConversation.length > 0 ? peerConversation[peerConversation.length - 1] : null;
  };

  const getUnreadCount = (peerId: string): number => {
    const peerConversation = conversations[peerId] || [];
    return peerConversation.filter(msg =>
      msg.sender !== 'current-user' && msg.deliveryStatus !== 'read'
    ).length;
  };

  const formatLastSeen = (date: Date): string => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return `${Math.floor(diff / 86400000)}d ago`;
  };

  const getStatusIcon = (status: BitChatPeer['status']) => {
    switch (status) {
      case 'online':
        return <span className="status-icon online" title="Online">🟢</span>;
      case 'away':
        return <span className="status-icon away" title="Away">🟡</span>;
      case 'offline':
        return <span className="status-icon offline" title="Offline">⚫</span>;
      default:
        return <span className="status-icon unknown" title="Unknown">❓</span>;
    }
  };

  const handlePeerAction = async (peerId: string, action: 'connect' | 'disconnect') => {
    if (action === 'connect') {
      await onPeerConnect(peerId);
    } else {
      await onPeerDisconnect(peerId);
    }
  };

  return (
    <div className={`peer-list ${className}`}>
      <div className="peer-list-header">
        <h3>Connected Peers ({peers.length})</h3>
      </div>

      <div className="peer-list-content">
        {peers.length === 0 ? (
          <div className="empty-peer-list">
            <div className="empty-icon">👥</div>
            <p>No peers discovered yet</p>
            <small>Start discovery to find nearby devices</small>
          </div>
        ) : (
          <div className="peer-items">
            {peers.map((peer) => {
              const lastMessage = getLastMessage(peer.id);
              const unreadCount = getUnreadCount(peer.id);
              const isSelected = selectedPeer === peer.id;

              return (
                <div
                  key={peer.id}
                  className={`peer-item ${isSelected ? 'selected' : ''} ${peer.status}`}
                  onClick={() => onPeerSelect(peer.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      onPeerSelect(peer.id);
                    }
                  }}
                >
                  <div className="peer-avatar">
                    {peer.avatar ? (
                      <img src={peer.avatar} alt={peer.name} />
                    ) : (
                      <div className="default-avatar">
                        {peer.name.charAt(0).toUpperCase()}
                      </div>
                    )}
                    {getStatusIcon(peer.status)}
                  </div>

                  <div className="peer-info">
                    <div className="peer-header">
                      <span className="peer-name">{peer.name}</span>
                      {unreadCount > 0 && (
                        <span className="unread-badge">{unreadCount}</span>
                      )}
                    </div>

                    <div className="peer-details">
                      {lastMessage ? (
                        <span className="last-message">
                          {lastMessage.content.length > 30
                            ? `${lastMessage.content.substring(0, 30)}...`
                            : lastMessage.content}
                        </span>
                      ) : (
                        <span className="no-messages">No messages yet</span>
                      )}
                    </div>

                    <div className="peer-meta">
                      <span className="last-seen">
                        {peer.status === 'online' ? 'Online' : formatLastSeen(peer.lastSeen)}
                      </span>
                    </div>
                  </div>

                  <div className="peer-actions">
                    {peer.status === 'online' ? (
                      <button
                        className="peer-action-btn disconnect-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePeerAction(peer.id, 'disconnect');
                        }}
                        title="Disconnect from peer"
                        aria-label={`Disconnect from ${peer.name}`}
                      >
                        ❌
                      </button>
                    ) : (
                      <button
                        className="peer-action-btn connect-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          handlePeerAction(peer.id, 'connect');
                        }}
                        title="Connect to peer"
                        aria-label={`Connect to ${peer.name}`}
                      >
                        🔗
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default PeerList;
