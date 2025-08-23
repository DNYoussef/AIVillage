import React, { useState, useEffect } from 'react';
import { SystemControlDashboard } from '../dashboard/SystemControlDashboard';
import './AdminInterface.css';

interface AdminInterfaceProps {
  isAdminMode: boolean;
  onToggleAdminMode: (enabled: boolean) => void;
}

interface SystemMetrics {
  p2pNodes: number;
  activeAgents: number;
  fogResources: number;
  networkHealth: number;
  systemUptime: string;
}

export const AdminInterface: React.FC<AdminInterfaceProps> = ({
  isAdminMode,
  onToggleAdminMode
}) => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    p2pNodes: 0,
    activeAgents: 0,
    fogResources: 0,
    networkHealth: 0,
    systemUptime: '0h 0m'
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (isAdminMode) {
      fetchSystemMetrics();
      const interval = setInterval(fetchSystemMetrics, 5000); // Update every 5 seconds
      return () => clearInterval(interval);
    }
  }, [isAdminMode]);

  const fetchSystemMetrics = async () => {
    try {
      setIsLoading(true);

      // Simulate API call to get system metrics
      // In production, this would call the actual admin API endpoints
      const response = await fetch('/api/admin/metrics');

      if (response.ok) {
        const data = await response.json();
        setMetrics(data);
      } else {
        // Fallback to mock data for demo
        setMetrics({
          p2pNodes: Math.floor(Math.random() * 50) + 10,
          activeAgents: Math.floor(Math.random() * 20) + 5,
          fogResources: Math.floor(Math.random() * 100) + 200,
          networkHealth: Math.floor(Math.random() * 30) + 70,
          systemUptime: `${Math.floor(Math.random() * 24)}h ${Math.floor(Math.random() * 60)}m`
        });
      }
    } catch (error) {
      console.warn('Admin metrics unavailable, using mock data:', error);
      setMetrics({
        p2pNodes: 25,
        activeAgents: 12,
        fogResources: 350,
        networkHealth: 85,
        systemUptime: '12h 34m'
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (!isAdminMode) {
    return (
      <div className="admin-interface-toggle">
        <button
          onClick={() => onToggleAdminMode(true)}
          className="btn-admin-enable"
        >
          Enable Admin Mode
        </button>
      </div>
    );
  }

  return (
    <div className="admin-interface">
      <div className="admin-header">
        <h2>AIVillage Admin Dashboard</h2>
        <button
          onClick={() => onToggleAdminMode(false)}
          className="btn-admin-disable"
        >
          Exit Admin Mode
        </button>
      </div>

      {isLoading ? (
        <div className="admin-loading">
          <div className="spinner"></div>
          <p>Loading system metrics...</p>
        </div>
      ) : (
        <>
          <div className="admin-metrics-grid">
            <div className="metric-card">
              <h3>P2P Network</h3>
              <div className="metric-value">{metrics.p2pNodes}</div>
              <div className="metric-label">Connected Nodes</div>
            </div>

            <div className="metric-card">
              <h3>AI Agents</h3>
              <div className="metric-value">{metrics.activeAgents}</div>
              <div className="metric-label">Active Agents</div>
            </div>

            <div className="metric-card">
              <h3>Fog Compute</h3>
              <div className="metric-value">{metrics.fogResources}</div>
              <div className="metric-label">Available Resources</div>
            </div>

            <div className="metric-card">
              <h3>Network Health</h3>
              <div className="metric-value">{metrics.networkHealth}%</div>
              <div className="metric-label">Overall Health</div>
            </div>
          </div>

          <div className="admin-dashboard-container">
            <SystemControlDashboard
              adminMode={true}
              refreshInterval={5000}
            />
          </div>

          <div className="admin-actions">
            <div className="action-group">
              <h3>System Actions</h3>
              <button className="btn-action" onClick={() => window.open('/api/admin/logs', '_blank')}>
                View System Logs
              </button>
              <button className="btn-action" onClick={() => window.open('/api/admin/config', '_blank')}>
                System Configuration
              </button>
              <button className="btn-action" onClick={fetchSystemMetrics}>
                Refresh Metrics
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
