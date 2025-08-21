import React, { useState, useEffect } from 'react';
import './AlertsPanel.css';

interface SystemAlert {
  id: string;
  type: 'critical' | 'warning' | 'info';
  message: string;
  timestamp: Date;
  category?: 'system' | 'network' | 'security' | 'performance';
  source?: string;
  resolved?: boolean;
}

interface AlertsGroup {
  critical: SystemAlert[];
  warnings: SystemAlert[];
  info: SystemAlert[];
}

interface AlertsPanelProps {
  alerts: AlertsGroup;
  onDismissAlert: (alertId: string) => void;
  onViewDetails: (alertId: string) => void;
  onResolveAlert?: (alertId: string) => void;
  maxDisplayed?: number;
  className?: string;
}

export const AlertsPanel: React.FC<AlertsPanelProps> = ({
  alerts,
  onDismissAlert,
  onViewDetails,
  onResolveAlert,
  maxDisplayed = 10,
  className = ''
}) => {
  const [filterType, setFilterType] = useState<'all' | 'critical' | 'warning' | 'info'>('all');
  const [isExpanded, setIsExpanded] = useState(false);
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());

  // Combine all alerts and sort by timestamp
  const allAlerts = [
    ...alerts.critical,
    ...alerts.warnings,
    ...alerts.info
  ].filter(alert => !dismissedAlerts.has(alert.id))
   .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

  // Filter alerts based on selected type
  const filteredAlerts = filterType === 'all' 
    ? allAlerts 
    : allAlerts.filter(alert => alert.type === filterType);

  // Limit displayed alerts
  const displayedAlerts = isExpanded 
    ? filteredAlerts 
    : filteredAlerts.slice(0, maxDisplayed);

  const getAlertIcon = (type: SystemAlert['type']) => {
    switch (type) {
      case 'critical': return '🔴';
      case 'warning': return '🟡';
      case 'info': return '🔵';
      default: return '⚪';
    }
  };

  const getAlertPriority = (type: SystemAlert['type']) => {
    switch (type) {
      case 'critical': return 3;
      case 'warning': return 2;
      case 'info': return 1;
      default: return 0;
    }
  };

  const getCategoryIcon = (category?: string) => {
    switch (category) {
      case 'system': return '⚙️';
      case 'network': return '🌐';
      case 'security': return '🔒';
      case 'performance': return '📊';
      default: return '📋';
    }
  };

  const formatTimestamp = (timestamp: Date) => {
    const now = new Date();
    const diff = now.getTime() - timestamp.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    if (minutes > 0) return `${minutes}m ago`;
    return 'Just now';
  };

  const handleDismissAlert = (alertId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    setDismissedAlerts(prev => new Set([...prev, alertId]));
    onDismissAlert(alertId);
  };

  const handleResolveAlert = (alertId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    if (onResolveAlert) {
      onResolveAlert(alertId);
    }
    setDismissedAlerts(prev => new Set([...prev, alertId]));
  };

  const criticalCount = alerts.critical.filter(a => !dismissedAlerts.has(a.id)).length;
  const warningCount = alerts.warnings.filter(a => !dismissedAlerts.has(a.id)).length;
  const infoCount = alerts.info.filter(a => !dismissedAlerts.has(a.id)).length;
  const totalCount = criticalCount + warningCount + infoCount;

  return (
    <div className={`alerts-panel ${className}`}>
      <div className="panel-header">
        <div className="header-title">
          <h3>System Alerts</h3>
          {totalCount > 0 && (
            <div className="alert-counter">
              <span className="total-count">{totalCount}</span>
              {criticalCount > 0 && <span className="critical-badge">{criticalCount}</span>}
            </div>
          )}
        </div>
        
        <div className="panel-controls">
          <div className="filter-buttons">
            <button
              className={`filter-btn ${filterType === 'all' ? 'active' : ''}`}
              onClick={() => setFilterType('all')}
            >
              All ({totalCount})
            </button>
            <button
              className={`filter-btn critical ${filterType === 'critical' ? 'active' : ''} ${criticalCount === 0 ? 'disabled' : ''}`}
              onClick={() => setFilterType('critical')}
              disabled={criticalCount === 0}
            >
              Critical ({criticalCount})
            </button>
            <button
              className={`filter-btn warning ${filterType === 'warning' ? 'active' : ''} ${warningCount === 0 ? 'disabled' : ''}`}
              onClick={() => setFilterType('warning')}
              disabled={warningCount === 0}
            >
              Warnings ({warningCount})
            </button>
            <button
              className={`filter-btn info ${filterType === 'info' ? 'active' : ''} ${infoCount === 0 ? 'disabled' : ''}`}
              onClick={() => setFilterType('info')}
              disabled={infoCount === 0}
            >
              Info ({infoCount})
            </button>
          </div>
          
          {filteredAlerts.length > maxDisplayed && (
            <button
              className="expand-btn"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? '▲ Collapse' : `▼ Show All (${filteredAlerts.length})`}
            </button>
          )}
        </div>
      </div>

      {totalCount === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">✅</div>
          <h4>No Active Alerts</h4>
          <p>All systems are operating normally</p>
        </div>
      ) : (
        <div className="alerts-list">
          {displayedAlerts.map(alert => (
            <div
              key={alert.id}
              className={`alert-item ${alert.type} ${alert.resolved ? 'resolved' : ''}`}
              onClick={() => onViewDetails(alert.id)}
            >
              <div className="alert-icon">
                {getAlertIcon(alert.type)}
              </div>
              
              <div className="alert-content">
                <div className="alert-main">
                  <div className="alert-message">{alert.message}</div>
                  <div className="alert-meta">
                    <span className="alert-time">{formatTimestamp(alert.timestamp)}</span>
                    {alert.category && (
                      <span className="alert-category">
                        {getCategoryIcon(alert.category)} {alert.category}
                      </span>
                    )}
                    {alert.source && (
                      <span className="alert-source">from {alert.source}</span>
                    )}
                  </div>
                </div>
                
                <div className="alert-type-badge">
                  {alert.type.toUpperCase()}
                </div>
              </div>
              
              <div className="alert-actions">
                {onResolveAlert && !alert.resolved && (
                  <button
                    className="resolve-btn"
                    onClick={(e) => handleResolveAlert(alert.id, e)}
                    title="Mark as resolved"
                  >
                    ✓
                  </button>
                )}
                
                <button
                  className="dismiss-btn"
                  onClick={(e) => handleDismissAlert(alert.id, e)}
                  title="Dismiss alert"
                >
                  ✕
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {criticalCount > 0 && (
        <div className="critical-summary">
          <div className="summary-header">
            <span className="critical-icon">🚨</span>
            <span className="summary-text">
              {criticalCount} critical {criticalCount === 1 ? 'alert' : 'alerts'} requiring immediate attention
            </span>
          </div>
          
          <div className="quick-actions">
            <button
              className="acknowledge-all-btn"
              onClick={() => {
                alerts.critical.forEach(alert => {
                  if (!dismissedAlerts.has(alert.id)) {
                    handleDismissAlert(alert.id, {} as React.MouseEvent);
                  }
                });
              }}
            >
              Acknowledge All Critical
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertsPanel;