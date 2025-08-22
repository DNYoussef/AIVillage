import React from 'react';
import { Message } from '../../types';
import { formatTimestamp, extractUrls, parseMarkdown } from '../../utils/messageUtils';
import './MessageBubble.css';

interface MessageBubbleProps {
  message: Message;
  isOwn: boolean;
  showAvatar?: boolean;
  showTimestamp?: boolean;
  onReply?: (message: Message) => void;
  onEdit?: (messageId: string, newContent: string) => void;
  onDelete?: (messageId: string) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isOwn,
  showAvatar = true,
  showTimestamp = true,
  onReply,
  onEdit,
  onDelete,
}) => {
  const [showActions, setShowActions] = React.useState(false);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editContent, setEditContent] = React.useState(message.content);

  const handleEdit = () => {
    if (editContent.trim() !== message.content && onEdit) {
      onEdit(message.id, editContent.trim());
    }
    setIsEditing(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleEdit();
    } else if (e.key === 'Escape') {
      setEditContent(message.content);
      setIsEditing(false);
    }
  };

  const renderMessageContent = () => {
    if (isEditing) {
      return (
        <div className="message-edit-container">
          <textarea
            value={editContent}
            onChange={(e) => setEditContent(e.target.value)}
            onKeyDown={handleKeyPress}
            className="message-edit-input"
            autoFocus
            rows={Math.max(2, editContent.split('\n').length)}
          />
          <div className="edit-actions">
            <button
              onClick={handleEdit}
              className="edit-save-btn"
              disabled={!editContent.trim()}
            >
              Save
            </button>
            <button
              onClick={() => {
                setEditContent(message.content);
                setIsEditing(false);
              }}
              className="edit-cancel-btn"
            >
              Cancel
            </button>
          </div>
        </div>
      );
    }

    // Parse message content for special formatting
    const urls = extractUrls(message.content);
    const hasMarkdown = message.content.includes('**') || message.content.includes('*') || message.content.includes('`');

    return (
      <div className="message-text">
        {hasMarkdown ? (
          <div
            className="markdown-content"
            dangerouslySetInnerHTML={{ __html: parseMarkdown(message.content) }}
          />
        ) : (
          <p>{message.content}</p>
        )}

        {urls.length > 0 && (
          <div className="message-urls">
            {urls.map((url, index) => (
              <a
                key={index}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="message-link"
              >
                {url}
              </a>
            ))}
          </div>
        )}

        {message.metadata?.attachments && (
          <div className="message-attachments">
            {message.metadata.attachments.map((attachment: any, index: number) => (
              <div key={index} className="attachment">
                <span className="attachment-icon">📎</span>
                <span className="attachment-name">{attachment.name}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const getMessageIcon = () => {
    switch (message.type) {
      case 'ai':
        return '🤖';
      case 'system':
        return '⚙️';
      case 'user':
      default:
        return '👤';
    }
  };

  return (
    <div
      className={`message-bubble ${isOwn ? 'own' : 'other'} ${message.type}`}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {showAvatar && !isOwn && (
        <div className="message-avatar">
          <span className="avatar-icon">
            {getMessageIcon()}
          </span>
        </div>
      )}

      <div className="message-content">
        {!isOwn && (
          <div className="message-sender">
            {message.type === 'ai' ? 'Digital Twin' : 'User'}
          </div>
        )}

        <div className="message-body">
          {renderMessageContent()}

          {message.metadata?.confidence && message.type === 'ai' && (
            <div className="ai-confidence">
              <span className="confidence-label">Confidence:</span>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${message.metadata.confidence * 100}%` }}
                />
              </div>
              <span className="confidence-value">
                {Math.round(message.metadata.confidence * 100)}%
              </span>
            </div>
          )}
        </div>

        {showTimestamp && (
          <div className="message-timestamp">
            {formatTimestamp(message.timestamp)}
            {message.metadata?.edited && (
              <span className="edited-indicator" title="Message edited">
                ✏️
              </span>
            )}
          </div>
        )}

        {showActions && (showActions || isEditing) && (
          <div className="message-actions">
            {onReply && (
              <button
                onClick={() => onReply(message)}
                className="action-btn reply-btn"
                title="Reply"
              >
                ↩️
              </button>
            )}
            {isOwn && onEdit && (
              <button
                onClick={() => setIsEditing(true)}
                className="action-btn edit-btn"
                title="Edit"
              >
                ✏️
              </button>
            )}
            {isOwn && onDelete && (
              <button
                onClick={() => onDelete(message.id)}
                className="action-btn delete-btn"
                title="Delete"
              >
                🗑️
              </button>
            )}
            <button
              onClick={() => navigator.clipboard.writeText(message.content)}
              className="action-btn copy-btn"
              title="Copy"
            >
              📋
            </button>
          </div>
        )}
      </div>

      {isOwn && showAvatar && (
        <div className="message-avatar">
          <span className="avatar-icon">
            {getMessageIcon()}
          </span>
        </div>
      )}
    </div>
  );
};
