import React from 'react';

interface MeshStatus {
  health: 'good' | 'fair' | 'poor';
  connectivity: number;
  throughput: number;
}

interface NetworkStatusProps {
  status: MeshStatus;
  peerCount: number;
  isDiscovering: boolean;
}

export const NetworkStatus: React.FC<NetworkStatusProps> = ({
  status,
  peerCount,
  isDiscovering
}) => {
  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'good': return '🟢';
      case 'fair': return '🟡';
      case 'poor': return '🔴';
      default: return '⚪';
    }
  };

  const getHealthText = (health: string) => {
    switch (health) {
      case 'good': return 'Excellent';
      case 'fair': return 'Good';
      case 'poor': return 'Poor';
      default: return 'Unknown';
    }
  };

  return (
    <div className={`network-status ${status.health}`}>
      <div className="status-header">
        <h4>Network Status</h4>
        {isDiscovering && (
          <div className="discovering-indicator">
            <span className="spinner">🔄</span>
            <span>Discovering...</span>
          </div>
        )}
      </div>

      <div className="status-grid">
        <div className="status-item">
          <span className="status-icon">{getHealthIcon(status.health)}</span>
          <div className="status-details">
            <div className="status-label">Health</div>
            <div className="status-value">{getHealthText(status.health)}</div>
          </div>
        </div>

        <div className="status-item">
          <span className="status-icon">🌐</span>
          <div className="status-details">
            <div className="status-label">Connected</div>
            <div className="status-value">{peerCount} peers</div>
          </div>
        </div>

        <div className="status-item">
          <span className="status-icon">📶</span>
          <div className="status-details">
            <div className="status-label">Signal</div>
            <div className="status-value">{status.connectivity}%</div>
          </div>
        </div>

        <div className="status-item">
          <span className="status-icon">⚡</span>
          <div className="status-details">
            <div className="status-label">Throughput</div>
            <div className="status-value">{status.throughput} KB/s</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkStatus;
