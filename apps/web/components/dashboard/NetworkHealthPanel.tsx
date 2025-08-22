// Network Health Panel - P2P mesh network status and diagnostics
import React from 'react';
import './NetworkHealthPanel.css';

interface NetworkHealth {
  p2pConnections: number;
  messageLatency: number;
  nodeCount: number;
  meshStability: number;
  bandwidthUtilization: number;
  packetLoss: number;
}

interface NetworkHealthPanelProps {
  networkHealth: NetworkHealth;
  onRunDiagnostics?: () => Promise<void>;
  className?: string;
}

export const NetworkHealthPanel: React.FC<NetworkHealthPanelProps> = ({
  networkHealth,
  onRunDiagnostics,
  className = ''
}) => {
  const getLatencyColor = (latency: number): string => {
    if (latency < 50) return '#10b981'; // green
    if (latency < 100) return '#f59e0b'; // yellow
    if (latency < 200) return '#f97316'; // orange
    return '#ef4444'; // red
  };

  const getLatencyStatus = (latency: number): string => {
    if (latency < 50) return 'Excellent';
    if (latency < 100) return 'Good';
    if (latency < 200) return 'Fair';
    return 'Poor';
  };

  const getStabilityIcon = (stability: number): string => {
    if (stability >= 95) return '🟢';
    if (stability >= 85) return '🟡';
    if (stability >= 70) return '🟠';
    return '🔴';
  };

  const getConnectionStrength = (connections: number): { strength: string; icon: string } => {
    if (connections >= 15) return { strength: 'Excellent', icon: '📶' };
    if (connections >= 10) return { strength: 'Good', icon: '📶' };
    if (connections >= 5) return { strength: 'Fair', icon: '📶' };
    return { strength: 'Poor', icon: '📶' };
  };

  const formatBandwidth = (utilization: number): string => {
    if (utilization < 1) return `${(utilization * 1000).toFixed(0)} KB/s`;
    return `${utilization.toFixed(1)} MB/s`;
  };

  const handleDiagnostics = async () => {
    if (onRunDiagnostics) {
      await onRunDiagnostics();
    }
  };

  return (
    <div className={`network-health-panel ${className}`}>
      <div className="panel-header">
        <h3>Network Health</h3>
        <div className="panel-actions">
          {onRunDiagnostics && (
            <button
              className="diagnostics-btn"
              onClick={handleDiagnostics}
              title="Run network diagnostics"
            >
              🔍 Diagnose
            </button>
          )}
        </div>
      </div>

      <div className="health-overview">
        <div className="health-card primary">
          <div className="health-icon">
            {getStabilityIcon(networkHealth.meshStability)}
          </div>
          <div className="health-info">
            <div className="health-value">
              {networkHealth.meshStability.toFixed(1)}%
            </div>
            <div className="health-label">Mesh Stability</div>
          </div>
        </div>

        <div className="health-card">
          <div className="health-icon">🔗</div>
          <div className="health-info">
            <div className="health-value">
              {networkHealth.p2pConnections}
            </div>
            <div className="health-label">
              P2P Connections
              <span className="connection-quality">
                {getConnectionStrength(networkHealth.p2pConnections).strength}
              </span>
            </div>
          </div>
        </div>

        <div className="health-card">
          <div className="health-icon">⚡</div>
          <div className="health-info">
            <div
              className="health-value"
              style={{ color: getLatencyColor(networkHealth.messageLatency) }}
            >
              {networkHealth.messageLatency.toFixed(0)}ms
            </div>
            <div className="health-label">
              Message Latency
              <span className="latency-status">
                {getLatencyStatus(networkHealth.messageLatency)}
              </span>
            </div>
          </div>
        </div>

        <div className="health-card">
          <div className="health-icon">🌐</div>
          <div className="health-info">
            <div className="health-value">
              {networkHealth.nodeCount}
            </div>
            <div className="health-label">Network Nodes</div>
          </div>
        </div>
      </div>

      <div className="detailed-metrics">
        <div className="metric-row">
          <div className="metric-item">
            <div className="metric-header">
              <span className="metric-icon">📊</span>
              <span className="metric-name">Bandwidth Utilization</span>
            </div>
            <div className="metric-content">
              <div className="bandwidth-bar">
                <div
                  className="bandwidth-fill"
                  style={{ width: `${Math.min(networkHealth.bandwidthUtilization, 100)}%` }}
                ></div>
              </div>
              <span className="bandwidth-value">
                {networkHealth.bandwidthUtilization.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        <div className="metric-row">
          <div className="metric-item">
            <div className="metric-header">
              <span className="metric-icon">📉</span>
              <span className="metric-name">Packet Loss</span>
            </div>
            <div className="metric-content">
              <div className="packet-loss-indicator">
                <div
                  className={`loss-circle ${networkHealth.packetLoss > 1 ? 'warning' : 'good'}`}
                ></div>
                <span className="loss-value">
                  {networkHealth.packetLoss.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="network-topology">
        <h4>Mesh Topology</h4>
        <div className="topology-visual">
          <div className="node center-node" title="Your Node">
            🖥️
          </div>
          {Array.from({ length: Math.min(networkHealth.p2pConnections, 8) }, (_, i) => (
            <div
              key={i}
              className="node peer-node"
              style={{
                transform: `rotate(${(360 / Math.min(networkHealth.p2pConnections, 8)) * i}deg) translateX(60px) rotate(-${(360 / Math.min(networkHealth.p2pConnections, 8)) * i}deg)`
              }}
              title={`Peer Node ${i + 1}`}
            >
              📱
            </div>
          ))}

          {/* Connection lines */}
          {Array.from({ length: Math.min(networkHealth.p2pConnections, 8) }, (_, i) => (
            <div
              key={`line-${i}`}
              className="connection-line"
              style={{
                transform: `rotate(${(360 / Math.min(networkHealth.p2pConnections, 8)) * i}deg)`
              }}
            ></div>
          ))}
        </div>
      </div>

      {networkHealth.packetLoss > 1 && (
        <div className="network-alert">
          <span className="alert-icon">⚠️</span>
          <span className="alert-message">
            High packet loss detected. Network performance may be degraded.
          </span>
        </div>
      )}
    </div>
  );
};

export default NetworkHealthPanel;
