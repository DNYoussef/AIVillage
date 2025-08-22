import React, { useState } from 'react';
import { BitChatPeer, P2PMessage } from '../../types';

interface ConversationViewProps {
  peer: BitChatPeer;
  messages: P2PMessage[];
  onSendMessage: (content: string) => void;
  encryptionEnabled: boolean;
}

export const ConversationView: React.FC<ConversationViewProps> = ({
  peer,
  messages,
  onSendMessage,
  encryptionEnabled
}) => {
  const [messageInput, setMessageInput] = useState('');

  const handleSend = () => {
    if (messageInput.trim()) {
      onSendMessage(messageInput.trim());
      setMessageInput('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="conversation-view">
      <div className="conversation-header">
        <div className="peer-info">
          <div className="peer-avatar">
            {peer.avatar ? (
              <img src={peer.avatar} alt={peer.name} />
            ) : (
              <div className="default-avatar">{peer.name.charAt(0).toUpperCase()}</div>
            )}
          </div>
          <div className="peer-details">
            <h3>{peer.name}</h3>
            <span className={`status ${peer.status}`}>{peer.status}</span>
          </div>
        </div>
        <div className="conversation-controls">
          <span className="encryption-status">
            {encryptionEnabled ? '🔒 Encrypted' : '🔓 Not Encrypted'}
          </span>
        </div>
      </div>

      <div className="messages-container">
        <div className="messages-list">
          {messages.length === 0 ? (
            <div className="empty-conversation">
              <p>No messages yet. Start the conversation!</p>
            </div>
          ) : (
            messages.map(message => (
              <div
                key={message.id}
                className={`message ${message.sender === peer.id ? 'received' : 'sent'}`}
              >
                <div className="message-content">
                  <p>{message.content}</p>
                  <div className="message-meta">
                    <span className="message-time">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                    {message.encrypted && <span className="encrypted-badge">🔒</span>}
                    <span className={`delivery-status ${message.deliveryStatus}`}>
                      {message.deliveryStatus === 'delivered' ? '✓' :
                       message.deliveryStatus === 'read' ? '✓✓' : '•'}
                    </span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="message-input-container">
        <div className="input-wrapper">
          <textarea
            value={messageInput}
            onChange={(e) => setMessageInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            rows={1}
            className="message-input"
          />
          <button
            onClick={handleSend}
            disabled={!messageInput.trim()}
            className="send-button"
            title="Send message"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConversationView;
