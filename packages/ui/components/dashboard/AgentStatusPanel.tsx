// Agent Status Panel - Shows AIVillage agent health and performance
import React from 'react';
import { LoadingSpinner } from '../common/LoadingSpinner';
import './AgentStatusPanel.css';

interface Agent {
  id: string;
  name: string;
  type: 'specialist' | 'coordinator' | 'worker';
  status: 'active' | 'idle' | 'busy' | 'error';
  performance: number;
  tasksCompleted: number;
  uptime: number;
  lastActivity: Date;
}

interface AgentStatusPanelProps {
  agents: Agent[];
  onRestartAgent: (agentId: string) => Promise<boolean>;
  onScaleAgents: (agentType: string, count: number) => Promise<boolean>;
  isLoading?: boolean;
  className?: string;
}

export const AgentStatusPanel: React.FC<AgentStatusPanelProps> = ({
  agents,
  onRestartAgent,
  onScaleAgents,
  isLoading = false,
  className = ''
}) => {
  const getStatusIcon = (status: Agent['status']) => {
    switch (status) {
      case 'active': return '🟢';
      case 'idle': return '🟡';
      case 'busy': return '🔵';
      case 'error': return '🔴';
      default: return '⚫';
    }
  };

  const getTypeIcon = (type: Agent['type']) => {
    switch (type) {
      case 'coordinator': return '👑';
      case 'specialist': return '🎯';
      case 'worker': return '⚡';
      default: return '🤖';
    }
  };

  const formatUptime = (ms: number): string => {
    const hours = Math.floor(ms / (1000 * 60 * 60));
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h`;
    return `${Math.floor(ms / (1000 * 60))}m`;
  };

  const getPerformanceColor = (performance: number): string => {
    if (performance >= 90) return 'excellent';
    if (performance >= 75) return 'good';
    if (performance >= 50) return 'fair';
    return 'poor';
  };

  const handleRestartAgent = async (agentId: string) => {
    try {
      await onRestartAgent(agentId);
    } catch (error) {
      console.error(`Failed to restart agent ${agentId}:`, error);
    }
  };

  if (isLoading) {
    return (
      <div className={`agent-status-panel loading ${className}`}>
        <LoadingSpinner message="Loading agents..." />
      </div>
    );
  }

  const agentsByType = agents.reduce((acc, agent) => {
    if (!acc[agent.type]) acc[agent.type] = [];
    acc[agent.type].push(agent);
    return acc;
  }, {} as Record<string, Agent[]>);

  return (
    <div className={`agent-status-panel ${className}`}>
      <div className="panel-header">
        <h3>Agent Fleet Status</h3>
        <div className="panel-summary">
          <span className="total-agents">{agents.length} Total</span>
          <span className="active-agents">
            {agents.filter(a => a.status === 'active').length} Active
          </span>
          <span className="error-agents">
            {agents.filter(a => a.status === 'error').length} Errors
          </span>
        </div>
      </div>

      <div className="agent-groups">
        {Object.entries(agentsByType).map(([type, typeAgents]) => (
          <div key={type} className="agent-group">
            <div className="group-header">
              <div className="group-title">
                {getTypeIcon(type as Agent['type'])}
                <span className="type-name">
                  {type.charAt(0).toUpperCase() + type.slice(1)} Agents
                </span>
                <span className="type-count">({typeAgents.length})</span>
              </div>
              <button
                className="scale-btn"
                onClick={() => onScaleAgents(type, 1)}
                title={`Add ${type} agent`}
              >
                +
              </button>
            </div>

            <div className="agent-list">
              {typeAgents.map(agent => (
                <div key={agent.id} className={`agent-card status-${agent.status}`}>
                  <div className="agent-info">
                    <div className="agent-header">
                      <div className="agent-identity">
                        <span className="agent-icon">{getTypeIcon(agent.type)}</span>
                        <span className="agent-name">{agent.name}</span>
                        <span className="status-indicator" title={agent.status}>
                          {getStatusIcon(agent.status)}
                        </span>
                      </div>

                      {agent.status === 'error' && (
                        <button
                          className="restart-btn"
                          onClick={() => handleRestartAgent(agent.id)}
                          title="Restart agent"
                        >
                          🔄
                        </button>
                      )}
                    </div>

                    <div className="agent-metrics">
                      <div className="metric">
                        <span className="metric-label">Performance</span>
                        <div className="performance-bar">
                          <div
                            className={`performance-fill ${getPerformanceColor(agent.performance)}`}
                            style={{ width: `${agent.performance}%` }}
                          ></div>
                        </div>
                        <span className="metric-value">{agent.performance.toFixed(1)}%</span>
                      </div>

                      <div className="metric">
                        <span className="metric-label">Tasks</span>
                        <span className="metric-value">{agent.tasksCompleted}</span>
                      </div>

                      <div className="metric">
                        <span className="metric-label">Uptime</span>
                        <span className="metric-value">{formatUptime(agent.uptime)}</span>
                      </div>

                      <div className="metric">
                        <span className="metric-label">Last Activity</span>
                        <span className="metric-value">
                          {new Date(agent.lastActivity).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {agents.length === 0 && (
        <div className="empty-agents">
          <div className="empty-icon">🤖</div>
          <p>No agents detected</p>
          <small>Start the agent fleet to see status</small>
        </div>
      )}
    </div>
  );
};

export default AgentStatusPanel;
