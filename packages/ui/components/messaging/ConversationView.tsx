// Conversation View Component - Main chat interface for P2P messaging
import React, { useState, useRef, useEffect } from 'react';
import { BitChatPeer, P2PMessage } from '../../types';
import { MessageBubble } from '../common/MessageBubble';
import { TypingIndicator } from '../common/TypingIndicator';
import { LoadingSpinner } from '../common/LoadingSpinner';
import './ConversationView.css';

interface ConversationViewProps {
  peer: BitChatPeer;
  messages: P2PMessage[];
  onSendMessage: (content: string) => Promise<void>;
  encryptionEnabled: boolean;
  className?: string;
}

export const ConversationView: React.FC<ConversationViewProps> = ({
  peer,
  messages,
  onSendMessage,
  encryptionEnabled,
  className = ''
}) => {
  const [messageInput, setMessageInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [peerTyping, setPeerTyping] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle typing indicators
  useEffect(() => {
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    if (messageInput.length > 0) {
      setIsTyping(true);
      typingTimeoutRef.current = setTimeout(() => {
        setIsTyping(false);
      }, 1000);
    } else {
      setIsTyping(false);
    }

    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, [messageInput]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async () => {
    if (!messageInput.trim() || isSending) return;

    const content = messageInput.trim();
    setMessageInput('');
    setIsSending(true);

    try {
      await onSendMessage(content);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Restore message input on failure
      setMessageInput(content);
    } finally {
      setIsSending(false);
    }

    // Focus back to input
    inputRef.current?.focus();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getMessageTime = (message: P2PMessage): string => {
    return new Date(message.timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const renderMessages = () => {
    return messages.map((message, index) => {
      const isOwn = message.sender !== peer.id;
      const showAvatar = !isOwn && (
        index === 0 ||
        messages[index - 1].sender !== message.sender ||
        new Date(message.timestamp).getTime() - new Date(messages[index - 1].timestamp).getTime() > 300000 // 5 minutes
      );

      return (
        <div key={message.id} className={`message-container ${isOwn ? 'own' : 'peer'}`}>
          {showAvatar && !isOwn && (
            <div className="message-avatar">
              {peer.avatar ? (
                <img src={peer.avatar} alt={peer.name} />
              ) : (
                <div className="default-avatar">
                  {peer.name.charAt(0).toUpperCase()}
                </div>
              )}
            </div>
          )}

          <div className="message-content">
            <MessageBubble
              message={{
                ...message,
                sender: isOwn ? 'You' : peer.name,
                type: 'user'
              }}
              showSender={!isOwn && showAvatar}
              showTime={true}
              encrypted={message.encrypted && encryptionEnabled}
              deliveryStatus={isOwn ? message.deliveryStatus : undefined}
            />
          </div>
        </div>
      );
    });
  };

  return (
    <div className={`conversation-view ${className}`}>
      <div className="conversation-header">
        <div className="peer-info">
          <div className="peer-avatar">
            {peer.avatar ? (
              <img src={peer.avatar} alt={peer.name} />
            ) : (
              <div className="default-avatar">
                {peer.name.charAt(0).toUpperCase()}
              </div>
            )}
            <div className={`status-indicator ${peer.status}`}></div>
          </div>

          <div className="peer-details">
            <h3 className="peer-name">{peer.name}</h3>
            <span className="peer-status">
              {peer.status === 'online' ? 'Active now' : `Last seen ${new Date(peer.lastSeen).toLocaleString()}`}
            </span>
          </div>
        </div>

        <div className="conversation-actions">
          <button
            className="action-btn"
            title="Voice call"
            disabled
          >
            📞
          </button>
          <button
            className="action-btn"
            title="Video call"
            disabled
          >
            🎥
          </button>
          <button
            className="action-btn"
            title="More options"
          >
            ⋮
          </button>
        </div>
      </div>

      <div className="messages-container">
        <div className="messages-list">
          {messages.length === 0 ? (
            <div className="empty-conversation">
              <div className="empty-icon">💬</div>
              <p>Start your conversation with {peer.name}</p>
              <small>Messages are {encryptionEnabled ? 'end-to-end encrypted' : 'not encrypted'}</small>
            </div>
          ) : (
            <>
              {renderMessages()}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {peerTyping && (
          <TypingIndicator isTyping={true} userName={peer.name} />
        )}
      </div>

      <div className="message-input-container">
        <div className="message-input-area">
          <button
            className="attachment-btn"
            title="Send file"
            disabled
          >
            📎
          </button>

          <textarea
            ref={inputRef}
            className="message-input"
            placeholder={`Message ${peer.name}...`}
            value={messageInput}
            onChange={(e) => setMessageInput(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={isSending}
            rows={1}
            style={{
              minHeight: '40px',
              maxHeight: '120px',
              resize: 'none'
            }}
          />

          <button
            className={`send-btn ${messageInput.trim() ? 'active' : ''}`}
            onClick={handleSendMessage}
            disabled={!messageInput.trim() || isSending}
            title="Send message"
          >
            {isSending ? (
              <LoadingSpinner size="small" variant="spinner" />
            ) : (
              '➤'
            )}
          </button>
        </div>

        <div className="input-footer">
          <div className="encryption-status">
            {encryptionEnabled ? (
              <span className="encrypted">🔒 End-to-end encrypted</span>
            ) : (
              <span className="unencrypted">⚠️ Messages not encrypted</span>
            )}
          </div>

          {isTyping && (
            <div className="typing-status">
              Typing...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConversationView;
