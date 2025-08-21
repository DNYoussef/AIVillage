// Network Status Component - Shows P2P mesh network health
import React from 'react';
import './NetworkStatus.css';

interface MeshStatus {
  health: 'good' | 'fair' | 'poor';
  connectivity: number;
  latency: number;
  redundancy: number;
}

interface NetworkStatusProps {
  status: MeshStatus;
  peerCount: number;
  isDiscovering: boolean;
  className?: string;
  showDetails?: boolean;
}

export const NetworkStatus: React.FC<NetworkStatusProps> = ({
  status,
  peerCount,
  isDiscovering,
  className = '',
  showDetails = false
}) => {
  const getHealthIcon = () => {
    switch (status.health) {
      case 'good':
        return '🟢';
      case 'fair':
        return '🟡';
      case 'poor':
        return '🔴';
      default:
        return '⚫';
    }
  };

  const getHealthText = () => {
    switch (status.health) {
      case 'good':
        return 'Excellent';
      case 'fair':
        return 'Good';
      case 'poor':
        return 'Poor';
      default:
        return 'Unknown';
    }
  };

  const formatLatency = (latency: number): string => {
    if (latency < 50) return 'Excellent';
    if (latency < 100) return 'Good';
    if (latency < 200) return 'Fair';
    return 'Poor';
  };

  const getSignalStrength = (connectivity: number): string => {
    if (connectivity >= 80) return '📶';
    if (connectivity >= 60) return '📶';
    if (connectivity >= 40) return '📶';
    if (connectivity >= 20) return '📶';
    return '📶';
  };

  return (
    <div className={`network-status ${className} status-${status.health}`}>
      <div className="status-main">
        <div className="status-indicator">
          {isDiscovering ? (
            <div className="discovery-animation">🔍</div>
          ) : (
            <span className="health-icon">{getHealthIcon()}</span>
          )}
        </div>

        <div className="status-info">
          <div className="status-primary">
            {isDiscovering ? (
              <span className="discovering">Discovering...</span>
            ) : (
              <>
                <span className="health-text">{getHealthText()}</span>
                <span className="peer-count">({peerCount} peers)</span>
              </>
            )}
          </div>

          {showDetails && !isDiscovering && (
            <div className="status-details">
              <div className="detail-item">
                <span className="detail-label">Signal:</span>
                <span className="detail-value">
                  {getSignalStrength(status.connectivity)} {status.connectivity.toFixed(0)}%
                </span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Latency:</span>
                <span className="detail-value">
                  {status.latency.toFixed(0)}ms ({formatLatency(status.latency)})
                </span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Redundancy:</span>
                <span className="detail-value">
                  {status.redundancy}x backup paths
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {showDetails && (
        <div className="status-graph">
          <div className="connectivity-bar">
            <div
              className="connectivity-fill"
              style={{ width: `${status.connectivity}%` }}
            ></div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NetworkStatus;
