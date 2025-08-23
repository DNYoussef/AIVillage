// Typing Indicator Component - Shows when users are typing in P2P chat
import React from 'react';
import './TypingIndicator.css';

interface TypingIndicatorProps {
  isTyping: boolean;
  userName?: string;
  className?: string;
}

export const TypingIndicator: React.FC<TypingIndicatorProps> = ({
  isTyping,
  userName,
  className = ''
}) => {
  if (!isTyping) return null;

  return (
    <div className={`typing-indicator ${className}`} role="status" aria-live="polite">
      <div className="typing-indicator-content">
        <div className="typing-dots">
          <span className="typing-dot typing-dot-1"></span>
          <span className="typing-dot typing-dot-2"></span>
          <span className="typing-dot typing-dot-3"></span>
        </div>
        <span className="typing-text">
          {userName ? `${userName} is typing...` : 'Someone is typing...'}
        </span>
      </div>
    </div>
  );
};

export default TypingIndicator;
