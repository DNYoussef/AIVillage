// Encryption Badge Component - Shows encryption status and protocol
import React from 'react';
import './EncryptionBadge.css';

interface EncryptionBadgeProps {
  enabled: boolean;
  protocol?: string;
  keyRotationInterval?: number;
  className?: string;
  size?: 'small' | 'medium' | 'large';
  showDetails?: boolean;
}

export const EncryptionBadge: React.FC<EncryptionBadgeProps> = ({
  enabled,
  protocol = 'ChaCha20-Poly1305',
  keyRotationInterval = 3600000, // 1 hour in ms
  className = '',
  size = 'medium',
  showDetails = false
}) => {
  const badgeClasses = [
    'encryption-badge',
    `encryption-badge--${size}`,
    enabled ? 'encryption-badge--enabled' : 'encryption-badge--disabled',
    className
  ].filter(Boolean).join(' ');

  const getEncryptionIcon = () => {
    if (enabled) {
      return (
        <svg className="encryption-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 1L3 5V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V5L12 1ZM12 7C13.4 7 14.8 8.6 14.8 10V11H16V18H8V11H9.2V10C9.2 8.6 10.6 7 12 7ZM12 8.2C11.2 8.2 10.4 8.7 10.4 10V11H13.6V10C13.6 8.7 12.8 8.2 12 8.2Z"/>
        </svg>
      );
    } else {
      return (
        <svg className="encryption-icon" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 1L3 5V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V5L12 1ZM18 10H17V9C17 7.3 15.7 6 14 6H10C8.3 6 7 7.3 7 9V10H6V16H18V10ZM8.2 10V9C8.2 7.9 9.1 7 10.2 7H13.8C14.9 7 15.8 7.9 15.8 9V10H8.2Z"/>
          <line x1="1" y1="1" x2="23" y2="23" stroke="currentColor" strokeWidth="2"/>
        </svg>
      );
    }
  };

  const formatRotationInterval = (ms: number): string => {
    const hours = Math.floor(ms / (1000 * 60 * 60));
    if (hours >= 24) {
      return `${Math.floor(hours / 24)}d`;
    } else if (hours > 0) {
      return `${hours}h`;
    } else {
      return `${Math.floor(ms / (1000 * 60))}m`;
    }
  };

  return (
    <div
      className={badgeClasses}
      title={enabled
        ? `End-to-end encryption enabled using ${protocol}. Keys rotate every ${formatRotationInterval(keyRotationInterval)}.`
        : 'Encryption is disabled. Messages are sent in plain text.'
      }
      role="status"
      aria-label={enabled ? 'Messages are encrypted' : 'Messages are not encrypted'}
    >
      <div className="encryption-badge-content">
        {getEncryptionIcon()}

        <div className="encryption-info">
          <span className="encryption-status">
            {enabled ? 'Encrypted' : 'Unencrypted'}
          </span>

          {showDetails && enabled && (
            <div className="encryption-details">
              <span className="encryption-protocol">{protocol}</span>
              <span className="encryption-rotation">
                Keys rotate: {formatRotationInterval(keyRotationInterval)}
              </span>
            </div>
          )}
        </div>
      </div>

      {enabled && (
        <div className="encryption-pulse" aria-hidden="true"></div>
      )}
    </div>
  );
};

export default EncryptionBadge;
