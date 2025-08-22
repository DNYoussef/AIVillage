import React, { useState, useEffect } from 'react';
import { SystemDashboard, FogNode } from '../../types';
import { useSystemService } from '../../hooks/useSystemService';
import { AgentStatusPanel } from './AgentStatusPanel';
import { FogNetworkPanel } from './FogNetworkPanel';
import { SystemMetricsPanel } from './SystemMetricsPanel';
import { AlertsPanel } from './AlertsPanel';
import { QuickActionsPanel } from './QuickActionsPanel';
import { NetworkTopologyView } from './NetworkTopologyView';
import './SystemControlDashboard.css';

interface SystemControlDashboardProps {
  userId: string;
  onComponentToggle?: (component: string, enabled: boolean) => void;
  onEmergencyStop?: () => void;
}

export const SystemControlDashboard: React.FC<SystemControlDashboardProps> = ({
  userId,
  onComponentToggle,
  onEmergencyStop
}) => {
  const [activePanel, setActivePanel] = useState<'overview' | 'agents' | 'fog' | 'metrics' | 'topology'>('overview');
  const [refreshRate, setRefreshRate] = useState<5 | 10 | 30>(10); // seconds
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [emergencyMode, setEmergencyMode] = useState(false);

  const {
    systemState,
    agentCommands,
    fogCommands,
    systemAlerts,
    isConnected,
    lastUpdate,
    error,
    executeCommand,
    refreshSystem
  } = useSystemService(userId);

  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (autoRefresh && isConnected) {
      interval = setInterval(() => {
        refreshSystem();
      }, refreshRate * 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, refreshRate, isConnected, refreshSystem]);

  const handleEmergencyStop = async () => {
    setEmergencyMode(true);
    try {
      await executeCommand('emergency_stop_all');
      if (onEmergencyStop) {
        onEmergencyStop();
      }
    } catch (err) {
      console.error('Emergency stop failed:', err);
    } finally {
      setEmergencyMode(false);
    }
  };

  const handleComponentToggle = async (component: string, enabled: boolean) => {
    try {
      await executeCommand(enabled ? `enable_${component}` : `disable_${component}`);
      if (onComponentToggle) {
        onComponentToggle(component, enabled);
      }
    } catch (err) {
      console.error(`Failed to toggle ${component}:`, err);
    }
  };

  const renderOverviewPanel = () => (
    <div className="overview-panel">
      <div className="system-status-cards">
        <div className="status-card agents">
          <div className="card-header">
            <h3>AI Agents</h3>
            <span className="status-indicator">
              {systemState.agents.filter(a => a.status === 'active').length}/
              {systemState.agents.length}
            </span>
          </div>
          <div className="card-content">
            <div className="agent-distribution">
              <div className="agent-type active">
                Active: {systemState.agents.filter(a => a.status === 'active').length}
              </div>
              <div className="agent-type idle">
                Idle: {systemState.agents.filter(a => a.status === 'idle').length}
              </div>
              <div className="agent-type busy">
                Busy: {systemState.agents.filter(a => a.status === 'busy').length}
              </div>
            </div>
          </div>
        </div>

        <div className="status-card fog-network">
          <div className="card-header">
            <h3>Fog Network</h3>
            <span className={`network-health ${systemState.networkHealth.nodeCount > 5 ? 'good' : 'fair'}`}>
              {systemState.fogNodes.filter(n => n.status === 'active').length} nodes
            </span>
          </div>
          <div className="card-content">
            <div className="network-stats">
              <div className="stat">
                <span className="stat-label">P2P Connections</span>
                <span className="stat-value">{systemState.networkHealth.p2pConnections}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Avg. Latency</span>
                <span className="stat-value">{systemState.networkHealth.messageLatency}ms</span>
              </div>
            </div>
          </div>
        </div>

        <div className="status-card system-metrics">
          <div className="card-header">
            <h3>System Load</h3>
            <span className={`load-indicator ${systemState.systemMetrics.cpuUsage > 80 ? 'high' : 'normal'}`}>
              {systemState.systemMetrics.cpuUsage.toFixed(1)}%
            </span>
          </div>
          <div className="card-content">
            <div className="metrics-bars">
              <div className="metric-bar">
                <span className="metric-label">CPU</span>
                <div className="bar-container">
                  <div
                    className="bar-fill cpu"
                    style={{ width: `${systemState.systemMetrics.cpuUsage}%` }}
                  />
                </div>
                <span className="metric-value">{systemState.systemMetrics.cpuUsage.toFixed(1)}%</span>
              </div>
              <div className="metric-bar">
                <span className="metric-label">RAM</span>
                <div className="bar-container">
                  <div
                    className="bar-fill memory"
                    style={{ width: `${systemState.systemMetrics.memoryUsage}%` }}
                  />
                </div>
                <span className="metric-value">{systemState.systemMetrics.memoryUsage.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="status-card alerts">
          <div className="card-header">
            <h3>System Alerts</h3>
            <span className={`alert-count ${systemAlerts.critical.length > 0 ? 'critical' : 'normal'}`}>
              {systemAlerts.critical.length + systemAlerts.warnings.length}
            </span>
          </div>
          <div className="card-content">
            <div className="alert-summary">
              {systemAlerts.critical.length > 0 && (
                <div className="alert-item critical">
                  🔴 {systemAlerts.critical.length} Critical
                </div>
              )}
              {systemAlerts.warnings.length > 0 && (
                <div className="alert-item warning">
                  🟡 {systemAlerts.warnings.length} Warnings
                </div>
              )}
              {systemAlerts.critical.length === 0 && systemAlerts.warnings.length === 0 && (
                <div className="alert-item normal">
                  ✅ All systems normal
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="real-time-charts">
        <SystemMetricsPanel
          metrics={systemState.systemMetrics}
          historicalData={[]} // TODO: Add historical data
          timeRange="1h"
          autoRefresh={autoRefresh}
        />
      </div>
    </div>
  );

  if (error) {
    return (
      <div className="dashboard-error">
        <div className="error-icon">⚠️</div>
        <h3>Dashboard Connection Error</h3>
        <p>{error}</p>
        <button onClick={refreshSystem} className="retry-btn">
          Reconnect
        </button>
      </div>
    );
  }

  return (
    <div className="system-control-dashboard">
      <div className="dashboard-header">
        <div className="dashboard-title">
          <h1>AIVillage System Control</h1>
          <div className="connection-status">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
            <span className="status-text">
              {isConnected ? `Connected • Last update: ${lastUpdate?.toLocaleTimeString()}` : 'Disconnected'}
            </span>
          </div>
        </div>

        <div className="dashboard-controls">
          <div className="refresh-controls">
            <label htmlFor="auto-refresh">
              <input
                id="auto-refresh"
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
              Auto-refresh
            </label>
            <select
              value={refreshRate}
              onChange={(e) => setRefreshRate(parseInt(e.target.value) as 5 | 10 | 30)}
              disabled={!autoRefresh}
            >
              <option value={5}>5s</option>
              <option value={10}>10s</option>
              <option value={30}>30s</option>
            </select>
          </div>

          <button
            onClick={handleEmergencyStop}
            className="emergency-stop-btn"
            disabled={emergencyMode}
            title="Emergency stop all systems"
          >
            {emergencyMode ? '🔄 Stopping...' : '🛑 Emergency Stop'}
          </button>
        </div>
      </div>

      <div className="dashboard-navigation">
        <nav className="panel-tabs">
          <button
            onClick={() => setActivePanel('overview')}
            className={`tab-btn ${activePanel === 'overview' ? 'active' : ''}`}
          >
            📊 Overview
          </button>
          <button
            onClick={() => setActivePanel('agents')}
            className={`tab-btn ${activePanel === 'agents' ? 'active' : ''}`}
          >
            🤖 Agents ({systemState.agents.filter(a => a.status === 'active').length})
          </button>
          <button
            onClick={() => setActivePanel('fog')}
            className={`tab-btn ${activePanel === 'fog' ? 'active' : ''}`}
          >
            ☁️ Fog Network ({systemState.fogNodes.filter(n => n.status === 'active').length})
          </button>
          <button
            onClick={() => setActivePanel('metrics')}
            className={`tab-btn ${activePanel === 'metrics' ? 'active' : ''}`}
          >
            📈 Metrics
          </button>
          <button
            onClick={() => setActivePanel('topology')}
            className={`tab-btn ${activePanel === 'topology' ? 'active' : ''}`}
          >
            🌐 Network Map
          </button>
        </nav>
      </div>

      <div className="dashboard-content">
        {activePanel === 'overview' && renderOverviewPanel()}

        {activePanel === 'agents' && (
          <AgentStatusPanel
            agents={systemState.agents}
            commands={agentCommands}
            onExecuteCommand={executeCommand}
            onToggleAgent={handleComponentToggle}
          />
        )}

        {activePanel === 'fog' && (
          <FogNetworkPanel
            nodes={systemState.fogNodes}
            networkHealth={systemState.networkHealth}
            commands={fogCommands}
            onExecuteCommand={executeCommand}
            onToggleNode={handleComponentToggle}
          />
        )}

        {activePanel === 'metrics' && (
          <SystemMetricsPanel
            metrics={systemState.systemMetrics}
            historicalData={[]}
            timeRange="24h"
            autoRefresh={autoRefresh}
          />
        )}

        {activePanel === 'topology' && (
          <NetworkTopologyView
            agents={systemState.agents}
            fogNodes={systemState.fogNodes}
            networkHealth={systemState.networkHealth}
            onNodeClick={(nodeId) => console.log('Node clicked:', nodeId)}
          />
        )}
      </div>

      <div className="dashboard-footer">
        <QuickActionsPanel
          onRefresh={refreshSystem}
          onToggleAutoRefresh={() => setAutoRefresh(!autoRefresh)}
          onExportData={() => console.log('Export data')}
          onImportConfig={() => console.log('Import config')}
          isLoading={!isConnected}
        />

        <AlertsPanel
          alerts={systemAlerts}
          onDismissAlert={(alertId) => console.log('Dismiss alert:', alertId)}
          onViewDetails={(alertId) => console.log('View alert details:', alertId)}
        />
      </div>
    </div>
  );
};
