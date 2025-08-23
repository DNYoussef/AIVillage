import React from 'react';
import { FogNode } from '../../types';

interface Agent {
  id: string;
  name: string;
  status: 'active' | 'idle' | 'busy';
  performance: number;
}

interface NetworkTopologyViewProps {
  agents: Agent[];
  fogNodes: FogNode[];
  networkHealth: {
    p2pConnections: number;
    messageLatency: number;
    nodeCount: number;
  };
  onNodeClick: (nodeId: string) => void;
}

export const NetworkTopologyView: React.FC<NetworkTopologyViewProps> = ({
  agents,
  fogNodes,
  networkHealth,
  onNodeClick
}) => {
  return (
    <div className="network-topology-view">
      <div className="panel-header">
        <h3>Network Topology</h3>
        <div className="topology-stats">
          <span>Agents: {agents.length}</span>
          <span>Fog Nodes: {fogNodes.length}</span>
          <span>Connections: {networkHealth.p2pConnections}</span>
        </div>
      </div>

      <div className="topology-visualization">
        <svg width="800" height="400" className="topology-svg">
          <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7"
             refX="0" refY="3.5" orient="auto">
              <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
            </marker>
          </defs>

          {/* Render fog nodes */}
          {fogNodes.map((node, index) => (
            <g key={node.id}>
              <circle
                cx={150 + (index % 3) * 200}
                cy={100 + Math.floor(index / 3) * 150}
                r="20"
                fill={node.status === 'active' ? '#10b981' : '#ef4444'}
                stroke="#374151"
                strokeWidth="2"
                style={{ cursor: 'pointer' }}
                onClick={() => onNodeClick(node.id)}
              />
              <text
                x={150 + (index % 3) * 200}
                y={100 + Math.floor(index / 3) * 150}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize="10"
                fill="white"
              >
                {node.name.slice(0, 4)}
              </text>
            </g>
          ))}

          {/* Render agents */}
          {agents.map((agent, index) => (
            <g key={agent.id}>
              <rect
                x={100 + (index % 4) * 150}
                y={250 + Math.floor(index / 4) * 100}
                width="30"
                height="20"
                fill={agent.status === 'active' ? '#3b82f6' : '#6b7280'}
                stroke="#374151"
                strokeWidth="1"
                rx="5"
                style={{ cursor: 'pointer' }}
                onClick={() => onNodeClick(agent.id)}
              />
              <text
                x={115 + (index % 4) * 150}
                y={260 + Math.floor(index / 4) * 100}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize="8"
                fill="white"
              >
                AI
              </text>
            </g>
          ))}

          {/* Render connections */}
          {fogNodes.slice(0, 2).map((_, index) => (
            <line
              key={`connection-${index}`}
              x1={150 + index * 200}
              y1={120}
              x2={150 + (index + 1) * 200}
              y2={120}
              stroke="#6b7280"
              strokeWidth="2"
              markerEnd="url(#arrowhead)"
            />
          ))}
        </svg>
      </div>

      <div className="topology-legend">
        <div className="legend-item">
          <div className="legend-color fog-node"></div>
          <span>Fog Nodes</span>
        </div>
        <div className="legend-item">
          <div className="legend-color agent"></div>
          <span>AI Agents</span>
        </div>
        <div className="legend-item">
          <div className="legend-line"></div>
          <span>P2P Connections</span>
        </div>
      </div>
    </div>
  );
};

export default NetworkTopologyView;
