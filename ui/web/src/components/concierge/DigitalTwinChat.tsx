import React, { useState, useEffect, useRef } from 'react';
import { Message, DigitalTwin, ChatState } from '../../types';
import { useConciergeService } from '../../hooks/useConciergeService';
import { MessageBubble } from '../common/MessageBubble';
import { TypingIndicator } from '../common/TypingIndicator';
import { LoadingSpinner } from '../common/LoadingSpinner';
import './DigitalTwinChat.css';

interface DigitalTwinChatProps {
  twinId: string;
  userId: string;
  onClose?: () => void;
}

export const DigitalTwinChat: React.FC<DigitalTwinChatProps> = ({
  twinId,
  userId,
  onClose
}) => {
  const [input, setInput] = useState('');
  const [isExpanded, setIsExpanded] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    chatState,
    sendMessage,
    isConnected,
    error,
    reconnect
  } = useConciergeService(twinId, userId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatState.messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !isConnected) return;

    const message: Message = {
      id: Date.now().toString(),
      sender: userId,
      content: input.trim(),
      timestamp: new Date(),
      type: 'user'
    };

    await sendMessage(message);
    setInput('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  if (error && !isConnected) {
    return (
      <div className="digital-twin-chat error-state">
        <div className="error-message">
          <h3>Connection Lost</h3>
          <p>{error}</p>
          <button onClick={reconnect} className="reconnect-btn">
            Reconnect
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`digital-twin-chat ${isExpanded ? 'expanded' : 'minimized'}`}>
      <div className="chat-header">
        <div className="twin-info">
          <div className="twin-avatar">
            🤖
          </div>
          <div className="twin-details">
            <h3>{chatState.activeTwin?.name || 'Digital Twin'}</h3>
            <span className={`status ${isConnected ? 'online' : 'offline'}`}>
              {isConnected ? 'Online' : 'Connecting...'}
            </span>
          </div>
        </div>
        <div className="chat-controls">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="minimize-btn"
            aria-label={isExpanded ? 'Minimize' : 'Expand'}
          >
            {isExpanded ? '−' : '+'}
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="close-btn"
              aria-label="Close chat"
            >
              ×
            </button>
          )}
        </div>
      </div>

      {isExpanded && (
        <>
          <div className="chat-messages" role="log" aria-live="polite">
            {chatState.messages.length === 0 ? (
              <div className="welcome-message">
                <div className="welcome-avatar">🌟</div>
                <h4>Welcome to your Digital Twin!</h4>
                <p>I'm here to help with your AI Village experience. Ask me anything!</p>
              </div>
            ) : (
              chatState.messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                  isOwn={message.sender === userId}
                />
              ))
            )}

            {chatState.isTyping && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSendMessage} className="chat-input-form">
            <div className="input-container">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask your digital twin anything..."
                className="message-input"
                rows={1}
                disabled={!isConnected}
                maxLength={2000}
              />
              <button
                type="submit"
                disabled={!input.trim() || !isConnected}
                className="send-btn"
                aria-label="Send message"
              >
                <svg viewBox="0 0 24 24" className="send-icon">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              </button>
            </div>
            <div className="input-info">
              {!isConnected && <LoadingSpinner size="small" />}
              <span className="char-count">{input.length}/2000</span>
            </div>
          </form>
        </>
      )}
    </div>
  );
};
