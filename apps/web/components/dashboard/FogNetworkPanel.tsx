import React, { useState, useEffect } from 'react';
import { FogNode } from '../../types';
import './FogNetworkPanel.css';

interface NetworkHealth {
  p2pConnections: number;
  messageLatency: number;
  nodeCount: number;
  meshStability: number;
  bandwidthUtilization: number;
  packetLoss: number;
}

interface FogCommand {
  id: string;
  name: string;
  description: string;
  parameters?: Record<string, any>;
}

interface FogNetworkPanelProps {
  nodes: FogNode[];
  networkHealth: NetworkHealth;
  commands: FogCommand[];
  onExecuteCommand: (command: string, params?: any) => Promise<void>;
  onToggleNode: (nodeId: string, enabled: boolean) => Promise<void>;
  className?: string;
}

export const FogNetworkPanel: React.FC<FogNetworkPanelProps> = ({
  nodes,
  networkHealth,
  commands,
  onExecuteCommand,
  onToggleNode,
  className = ''
}) => {
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [isExecutingCommand, setIsExecutingCommand] = useState(false);
  const [networkView, setNetworkView] = useState<'grid' | 'list'>('grid');

  const getNodeStatusColor = (status: FogNode['status']) => {
    switch (status) {
      case 'active': return '#10b981';
      case 'inactive': return '#ef4444';
      case 'maintenance': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  const getNetworkHealthStatus = () => {
    const { meshStability, packetLoss, messageLatency } = networkHealth;

    if (meshStability >= 90 && packetLoss < 1 && messageLatency < 100) return 'excellent';
    if (meshStability >= 75 && packetLoss < 2 && messageLatency < 200) return 'good';
    if (meshStability >= 60 && packetLoss < 5 && messageLatency < 300) return 'fair';
    return 'poor';
  };

  const handleCommandExecution = async (command: string, params?: any) => {
    setIsExecutingCommand(true);
    try {
      await onExecuteCommand(command, params);
    } catch (error) {
      console.error('Command execution failed:', error);
    } finally {
      setIsExecutingCommand(false);
    }
  };

  const calculateTotalResources = () => {
    return nodes.reduce((total, node) => ({
      cpu: total.cpu + (node.status === 'active' ? node.resources.cpu : 0),
      memory: total.memory + (node.status === 'active' ? node.resources.memory : 0),
      storage: total.storage + (node.status === 'active' ? node.resources.storage : 0),
      bandwidth: total.bandwidth + (node.status === 'active' ? node.resources.bandwidth : 0)
    }), { cpu: 0, memory: 0, storage: 0, bandwidth: 0 });
  };

  const totalResources = calculateTotalResources();
  const activeNodes = nodes.filter(n => n.status === 'active');
  const healthStatus = getNetworkHealthStatus();

  return (
    <div className={`fog-network-panel ${className}`}>
      <div className="panel-header">
        <div className="header-title">
          <h3>Fog Network Overview</h3>
          <div className={`network-health-indicator ${healthStatus}`}>
            <span className="health-dot"></span>
            <span className="health-text">{healthStatus.toUpperCase()}</span>
          </div>
        </div>

        <div className="panel-controls">
          <div className="view-toggle">
            <button
              className={`toggle-btn ${networkView === 'grid' ? 'active' : ''}`}
              onClick={() => setNetworkView('grid')}
              title="Grid view"
            >
              ⊞
            </button>
            <button
              className={`toggle-btn ${networkView === 'list' ? 'active' : ''}`}
              onClick={() => setNetworkView('list')}
              title="List view"
            >
              ≡
            </button>
          </div>

          <button
            className="refresh-btn"
            onClick={() => handleCommandExecution('refresh_network')}
            disabled={isExecutingCommand}
          >
            {isExecutingCommand ? '⟳' : '↻'}
          </button>
        </div>
      </div>

      <div className="network-summary">
        <div className="summary-cards">
          <div className="summary-card">
            <div className="card-icon">🌐</div>
            <div className="card-content">
              <div className="card-value">{activeNodes.length}</div>
              <div className="card-label">Active Nodes</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="card-icon">🔗</div>
            <div className="card-content">
              <div className="card-value">{networkHealth.p2pConnections}</div>
              <div className="card-label">P2P Connections</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="card-icon">⚡</div>
            <div className="card-content">
              <div className="card-value">{networkHealth.messageLatency.toFixed(0)}ms</div>
              <div className="card-label">Avg Latency</div>
            </div>
          </div>

          <div className="summary-card">
            <div className="card-icon">📊</div>
            <div className="card-content">
              <div className="card-value">{networkHealth.meshStability.toFixed(1)}%</div>
              <div className="card-label">Mesh Stability</div>
            </div>
          </div>
        </div>
      </div>

      <div className="network-health">
        <h4>Network Health Metrics</h4>
        <div className="health-metrics">
          <div className="metric-item">
            <span className="metric-label">Bandwidth Utilization</span>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{
                  width: `${Math.min(networkHealth.bandwidthUtilization, 100)}%`,
                  backgroundColor: networkHealth.bandwidthUtilization > 80 ? '#ef4444' : networkHealth.bandwidthUtilization > 60 ? '#f59e0b' : '#10b981'
                }}
              ></div>
            </div>
            <span className="metric-value">{networkHealth.bandwidthUtilization.toFixed(1)}%</span>
          </div>

          <div className="metric-item">
            <span className="metric-label">Packet Loss</span>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{
                  width: `${Math.min(networkHealth.packetLoss * 20, 100)}%`,
                  backgroundColor: networkHealth.packetLoss > 2 ? '#ef4444' : networkHealth.packetLoss > 1 ? '#f59e0b' : '#10b981'
                }}
              ></div>
            </div>
            <span className="metric-value">{networkHealth.packetLoss.toFixed(2)}%</span>
          </div>
        </div>
      </div>

      <div className="resource-overview">
        <h4>Total Available Resources</h4>
        <div className="resource-stats">
          <div className="resource-item">
            <span className="resource-icon">🖥️</span>
            <span className="resource-label">CPU Cores</span>
            <span className="resource-value">{totalResources.cpu}</span>
          </div>
          <div className="resource-item">
            <span className="resource-icon">🧠</span>
            <span className="resource-label">Memory (GB)</span>
            <span className="resource-value">{(totalResources.memory / 1024).toFixed(1)}</span>
          </div>
          <div className="resource-item">
            <span className="resource-icon">💾</span>
            <span className="resource-label">Storage (TB)</span>
            <span className="resource-value">{(totalResources.storage / 1024 / 1024).toFixed(1)}</span>
          </div>
          <div className="resource-item">
            <span className="resource-icon">📡</span>
            <span className="resource-label">Bandwidth (Mbps)</span>
            <span className="resource-value">{totalResources.bandwidth}</span>
          </div>
        </div>
      </div>

      <div className={`nodes-container ${networkView}`}>
        <h4>Fog Nodes ({nodes.length})</h4>

        {networkView === 'grid' ? (
          <div className="nodes-grid">
            {nodes.map(node => (
              <div
                key={node.id}
                className={`node-card ${node.status} ${selectedNode === node.id ? 'selected' : ''}`}
                onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
              >
                <div className="node-header">
                  <div className="node-status-dot" style={{ backgroundColor: getNodeStatusColor(node.status) }}></div>
                  <div className="node-info">
                    <h5 className="node-name">{node.name}</h5>
                    <span className="node-location">{node.location}</span>
                  </div>
                  <div className="node-reputation">
                    ⭐ {node.reputation}%
                  </div>
                </div>

                <div className="node-resources">
                  <div className="resource-bar">
                    <span className="resource-label">CPU</span>
                    <div className="bar">
                      <div className="bar-fill" style={{ width: `${(node.resources.cpu / 100) * 100}%` }}></div>
                    </div>
                    <span className="resource-text">{node.resources.cpu} cores</span>
                  </div>

                  <div className="resource-bar">
                    <span className="resource-label">Memory</span>
                    <div className="bar">
                      <div className="bar-fill" style={{ width: `${(node.resources.memory / 32768) * 100}%` }}></div>
                    </div>
                    <span className="resource-text">{(node.resources.memory / 1024).toFixed(1)} GB</span>
                  </div>
                </div>

                <div className="node-actions">
                  <button
                    className={`toggle-node-btn ${node.status === 'active' ? 'active' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleNode(node.id, node.status !== 'active');
                    }}
                    title={node.status === 'active' ? 'Deactivate node' : 'Activate node'}
                  >
                    {node.status === 'active' ? '⏸️' : '▶️'}
                  </button>

                  <button
                    className="node-config-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCommandExecution('configure_node', { nodeId: node.id });
                    }}
                    title="Configure node"
                  >
                    ⚙️
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="nodes-list">
            {nodes.map(node => (
              <div
                key={node.id}
                className={`node-list-item ${node.status} ${selectedNode === node.id ? 'selected' : ''}`}
                onClick={() => setSelectedNode(selectedNode === node.id ? null : node.id)}
              >
                <div className="node-status-dot" style={{ backgroundColor: getNodeStatusColor(node.status) }}></div>
                <div className="node-main-info">
                  <h5 className="node-name">{node.name}</h5>
                  <span className="node-location">{node.location}</span>
                </div>
                <div className="node-stats">
                  <span className="stat">CPU: {node.resources.cpu}</span>
                  <span className="stat">Memory: {(node.resources.memory / 1024).toFixed(1)}GB</span>
                  <span className="stat">Rep: ⭐{node.reputation}%</span>
                </div>
                <div className="node-list-actions">
                  <button
                    className={`toggle-node-btn ${node.status === 'active' ? 'active' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleNode(node.id, node.status !== 'active');
                    }}
                  >
                    {node.status === 'active' ? '⏸️' : '▶️'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedNode && (
        <div className="node-details">
          {(() => {
            const node = nodes.find(n => n.id === selectedNode);
            if (!node) return null;

            return (
              <div className="details-panel">
                <h4>Node Details: {node.name}</h4>
                <div className="details-grid">
                  <div className="detail-item">
                    <span className="detail-label">Status:</span>
                    <span className={`detail-value status-${node.status}`}>{node.status}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Location:</span>
                    <span className="detail-value">{node.location}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Reputation:</span>
                    <span className="detail-value">⭐ {node.reputation}%</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Storage:</span>
                    <span className="detail-value">{(node.resources.storage / 1024).toFixed(0)} GB</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Bandwidth:</span>
                    <span className="detail-value">{node.resources.bandwidth} Mbps</span>
                  </div>
                </div>

                <div className="node-commands">
                  <h5>Quick Actions</h5>
                  <div className="command-buttons">
                    <button
                      onClick={() => handleCommandExecution('restart_node', { nodeId: node.id })}
                      disabled={isExecutingCommand}
                    >
                      🔄 Restart
                    </button>
                    <button
                      onClick={() => handleCommandExecution('optimize_node', { nodeId: node.id })}
                      disabled={isExecutingCommand}
                    >
                      ⚡ Optimize
                    </button>
                    <button
                      onClick={() => handleCommandExecution('maintenance_mode', { nodeId: node.id })}
                      disabled={isExecutingCommand}
                    >
                      🔧 Maintenance
                    </button>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {nodes.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">🌐</div>
          <h4>No Fog Nodes Available</h4>
          <p>Connect to the AIVillage network to discover fog computing nodes</p>
          <button
            onClick={() => handleCommandExecution('discover_nodes')}
            className="discover-nodes-btn"
            disabled={isExecutingCommand}
          >
            {isExecutingCommand ? 'Discovering...' : 'Discover Nodes'}
          </button>
        </div>
      )}
    </div>
  );
};

export default FogNetworkPanel;
