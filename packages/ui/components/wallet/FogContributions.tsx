import React, { useState } from 'react';
import { FogNode } from '../../types';
import './FogContributions.css';

interface FogContributionsProps {
  contributions: FogNode[];
  onContribute: (nodeId: string, resourceAmount: number) => void;
  onWithdraw: (nodeId: string) => void;
  earningRate: {
    current: number;
    projected: number;
  };
  networkStats: {
    totalNodes: number;
    networkLoad: number;
    averageLatency: number;
  };
}

export const FogContributions: React.FC<FogContributionsProps> = ({
  contributions,
  onContribute,
  onWithdraw,
  earningRate,
  networkStats,
}) => {
  const [showAddNode, setShowAddNode] = useState(false);
  const [newNodeName, setNewNodeName] = useState('');
  const [newNodeLocation, setNewNodeLocation] = useState('');
  const [resourceAllocation, setResourceAllocation] = useState({
    cpu: 50,
    memory: 50,
    storage: 50,
    bandwidth: 50,
  });

  const handleAddNode = () => {
    if (!newNodeName.trim() || !newNodeLocation.trim()) return;

    // In a real implementation, this would create a new fog node
    console.log('Adding new fog node:', {
      name: newNodeName,
      location: newNodeLocation,
      resources: resourceAllocation,
    });

    setShowAddNode(false);
    setNewNodeName('');
    setNewNodeLocation('');
    setResourceAllocation({ cpu: 50, memory: 50, storage: 50, bandwidth: 50 });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return '#10b981';
      case 'inactive': return '#ef4444';
      case 'maintenance': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  const getReputationStars = (reputation: number) => {
    const stars = Math.round(reputation);
    return '★'.repeat(stars) + '☆'.repeat(5 - stars);
  };

  const calculateEarnings = (node: FogNode) => {
    const resourceUtilization =
      (node.resources.cpu + node.resources.memory + node.resources.storage + node.resources.bandwidth) / 4;
    return (resourceUtilization / 100) * earningRate.current * node.reputation;
  };

  return (
    <div className="fog-contributions">
      <div className="contributions-header">
        <div className="header-stats">
          <div className="stat-card">
            <div className="stat-value">{contributions.length}</div>
            <div className="stat-label">Active Nodes</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{earningRate.current}</div>
            <div className="stat-label">Credits/Hour</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{earningRate.projected.toFixed(1)}</div>
            <div className="stat-label">Projected Rate</div>
          </div>
        </div>

        <button
          onClick={() => setShowAddNode(!showAddNode)}
          className="add-node-btn"
        >
          + Add Fog Node
        </button>
      </div>

      {showAddNode && (
        <div className="add-node-form">
          <h3>Add New Fog Node</h3>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="node-name">Node Name</label>
              <input
                id="node-name"
                type="text"
                value={newNodeName}
                onChange={(e) => setNewNodeName(e.target.value)}
                placeholder="My Home Node"
              />
            </div>

            <div className="form-group">
              <label htmlFor="node-location">Location</label>
              <input
                id="node-location"
                type="text"
                value={newNodeLocation}
                onChange={(e) => setNewNodeLocation(e.target.value)}
                placeholder="Living Room"
              />
            </div>
          </div>

          <div className="resource-allocation">
            <h4>Resource Allocation</h4>
            {Object.entries(resourceAllocation).map(([resource, value]) => (
              <div key={resource} className="resource-slider">
                <label>{resource.toUpperCase()}: {value}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={value}
                  onChange={(e) => setResourceAllocation(prev => ({
                    ...prev,
                    [resource]: parseInt(e.target.value),
                  }))}
                />
              </div>
            ))}
          </div>

          <div className="form-actions">
            <button onClick={() => setShowAddNode(false)} className="cancel-btn">
              Cancel
            </button>
            <button onClick={handleAddNode} className="create-btn">
              Create Node
            </button>
          </div>
        </div>
      )}

      <div className="network-overview">
        <h3>Network Overview</h3>
        <div className="network-metrics">
          <div className="metric">
            <span className="metric-icon">🌐</span>
            <div className="metric-info">
              <div className="metric-label">Total Network Nodes</div>
              <div className="metric-value">{networkStats.totalNodes}</div>
            </div>
          </div>
          <div className="metric">
            <span className="metric-icon">📊</span>
            <div className="metric-info">
              <div className="metric-label">Network Load</div>
              <div className="metric-value">{networkStats.networkLoad}%</div>
            </div>
          </div>
          <div className="metric">
            <span className="metric-icon">⚡</span>
            <div className="metric-info">
              <div className="metric-label">Avg. Latency</div>
              <div className="metric-value">{networkStats.averageLatency}ms</div>
            </div>
          </div>
        </div>
      </div>

      {contributions.length === 0 ? (
        <div className="no-contributions">
          <div className="no-contributions-icon">☁️</div>
          <h3>No Fog Contributions</h3>
          <p>Start contributing your device resources to the fog network and earn credits!</p>
          <button onClick={() => setShowAddNode(true)} className="get-started-btn">
            Get Started
          </button>
        </div>
      ) : (
        <div className="contributions-list">
          {contributions.map((node) => (
            <div key={node.id} className={`contribution-item ${node.status}`}>
              <div className="node-header">
                <div className="node-info">
                  <h4>{node.name}</h4>
                  <p className="node-location">📍 {node.location}</p>
                  <div className="node-status">
                    <span
                      className="status-indicator"
                      style={{ backgroundColor: getStatusColor(node.status) }}
                    ></span>
                    <span className="status-text">{node.status}</span>
                  </div>
                </div>

                <div className="node-earnings">
                  <div className="earnings-amount">
                    +{calculateEarnings(node).toFixed(2)} credits/hour
                  </div>
                  <div className="reputation">
                    <span className="reputation-stars">
                      {getReputationStars(node.reputation)}
                    </span>
                    <span className="reputation-score">({node.reputation})</span>
                  </div>
                </div>
              </div>

              <div className="resource-usage">
                {Object.entries(node.resources).map(([resource, usage]) => (
                  <div key={resource} className="resource-bar">
                    <div className="resource-label">
                      {resource.toUpperCase()}
                    </div>
                    <div className="resource-progress">
                      <div
                        className={`resource-fill ${resource}`}
                        style={{ width: `${usage}%` }}
                      ></div>
                      <span className="resource-percentage">{usage}%</span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="node-actions">
                <button
                  onClick={() => onContribute(node.id, 10)}
                  className="contribute-btn"
                  disabled={node.status !== 'active'}
                  title="Increase contribution"
                >
                  ⬆️ Increase
                </button>

                <button
                  onClick={() => onWithdraw(node.id)}
                  className="withdraw-btn"
                  title="Withdraw from fog network"
                >
                  💰 Withdraw
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
