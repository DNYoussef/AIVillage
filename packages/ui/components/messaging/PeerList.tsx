import React from 'react';
import { BitChatPeer, P2PMessage } from '../../types';

interface PeerListProps {
  peers: BitChatPeer[];
  selectedPeer: string | null;
  onPeerSelect: (peerId: string) => void;
  onPeerConnect: (peerId: string) => Promise<void>;
  onPeerDisconnect: (peerId: string) => Promise<void>;
  conversations: Record<string, P2PMessage[]>;
}

export const PeerList: React.FC<PeerListProps> = ({
  peers,
  selectedPeer,
  onPeerSelect,
  onPeerConnect,
  onPeerDisconnect,
  conversations
}) => {
  return (
    <div className="peer-list">
      <div className="peer-list-header">
        <h3>Connected Peers</h3>
        <span className="peer-count">{peers.length} peers</span>
      </div>

      <div className="peers-container">
        {peers.length === 0 ? (
          <div className="empty-peers">
            <p>No peers connected</p>
          </div>
        ) : (
          peers.map(peer => (
            <div
              key={peer.id}
              className={`peer-item ${selectedPeer === peer.id ? 'selected' : ''} ${peer.status}`}
              onClick={() => onPeerSelect(peer.id)}
            >
              <div className="peer-avatar">
                {peer.avatar ? (
                  <img src={peer.avatar} alt={peer.name} />
                ) : (
                  <div className="default-avatar">{peer.name.charAt(0).toUpperCase()}</div>
                )}
              </div>
              <div className="peer-info">
                <div className="peer-name">{peer.name}</div>
                <div className={`peer-status ${peer.status}`}>
                  {peer.status === 'online' ? '🟢' : peer.status === 'away' ? '🟡' : '🔴'}
                  {peer.status}
                </div>
              </div>
              <div className="peer-actions">
                {conversations[peer.id] && conversations[peer.id].length > 0 && (
                  <span className="message-count">
                    {conversations[peer.id].length}
                  </span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default PeerList;
