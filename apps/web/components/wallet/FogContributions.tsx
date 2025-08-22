import React from 'react';
import { FogNode } from '../../types';

interface EarningRate {
  current: number;
  potential: number;
  trend: 'up' | 'down' | 'stable';
}

interface FogStats {
  totalNodes: number;
  networkLoad: number;
  averageLatency: number;
}

interface FogContributionsProps {
  contributions: FogNode[];
  onContribute: (nodeId: string, resourceAmount: number) => Promise<void>;
  onWithdraw: (nodeId: string) => Promise<void>;
  earningRate: EarningRate;
  networkStats: FogStats;
}

export const FogContributions: React.FC<FogContributionsProps> = ({
  contributions,
  onContribute,
  onWithdraw,
  earningRate,
  networkStats
}) => {
  return (
    <div className="fog-contributions">
      <div className="contributions-header">
        <h3>Fog Network Contributions</h3>
        <div className="earning-summary">
          <span className="earning-rate">{earningRate.current} credits/hour</span>
        </div>
      </div>

      <div className="network-overview">
        <div className="network-stat">
          <span className="stat-label">Total Nodes</span>
          <span className="stat-value">{networkStats.totalNodes}</span>
        </div>
      </div>

      <div className="contributions-list">
        {contributions.length === 0 ? (
          <div className="empty-contributions">
            <p>No active contributions</p>
            <button
              className="contribute-btn"
              onClick={() => onContribute('new-node', 1)}
            >
              Start Contributing
            </button>
          </div>
        ) : (
          contributions.map(node => (
            <div key={node.id} className={`contribution-item ${node.status}`}>
              <div className="node-info">
                <h4>{node.name}</h4>
                <p>{node.location}</p>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default FogContributions;
