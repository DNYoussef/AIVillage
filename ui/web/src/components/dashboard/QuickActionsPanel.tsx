import React, { useState } from 'react';
import './QuickActionsPanel.css';

interface QuickActionsPanelProps {
  onRefresh: () => void;
  onToggleAutoRefresh: () => void;
  onExportData: () => void;
  onImportConfig: () => void;
  onEmergencyStop?: () => void;
  onSystemOptimize?: () => void;
  onBackupSystem?: () => void;
  onRestoreSystem?: () => void;
  isLoading?: boolean;
  autoRefreshEnabled?: boolean;
  className?: string;
}

interface ActionButton {
  id: string;
  label: string;
  icon: string;
  action: () => void;
  type?: 'primary' | 'secondary' | 'warning' | 'danger';
  disabled?: boolean;
  tooltip?: string;
  confirmRequired?: boolean;
}

export const QuickActionsPanel: React.FC<QuickActionsPanelProps> = ({
  onRefresh,
  onToggleAutoRefresh,
  onExportData,
  onImportConfig,
  onEmergencyStop,
  onSystemOptimize,
  onBackupSystem,
  onRestoreSystem,
  isLoading = false,
  autoRefreshEnabled = false,
  className = ''
}) => {
  const [showConfirmDialog, setShowConfirmDialog] = useState<string | null>(null);
  const [executingAction, setExecutingAction] = useState<string | null>(null);

  const handleActionClick = async (actionId: string, action: () => void, confirmRequired = false) => {
    if (confirmRequired && showConfirmDialog !== actionId) {
      setShowConfirmDialog(actionId);
      return;
    }

    setExecutingAction(actionId);
    setShowConfirmDialog(null);

    try {
      await action();
    } catch (error) {
      console.error(`Action ${actionId} failed:`, error);
    } finally {
      setExecutingAction(null);
    }
  };

  const primaryActions: ActionButton[] = [
    {
      id: 'refresh',
      label: 'Refresh',
      icon: '↻',
      action: onRefresh,
      type: 'primary',
      disabled: isLoading,
      tooltip: 'Refresh all dashboard data'
    },
    {
      id: 'auto-refresh',
      label: autoRefreshEnabled ? 'Auto ON' : 'Auto OFF',
      icon: autoRefreshEnabled ? '⏸️' : '▶️',
      action: onToggleAutoRefresh,
      type: 'secondary',
      tooltip: `${autoRefreshEnabled ? 'Disable' : 'Enable'} automatic refresh`
    },
    {
      id: 'optimize',
      label: 'Optimize',
      icon: '⚡',
      action: onSystemOptimize || (() => console.log('System optimize')),
      type: 'primary',
      disabled: !onSystemOptimize || isLoading,
      tooltip: 'Optimize system performance'
    }
  ];

  const dataActions: ActionButton[] = [
    {
      id: 'export',
      label: 'Export',
      icon: '📤',
      action: onExportData,
      type: 'secondary',
      tooltip: 'Export system data and configuration'
    },
    {
      id: 'import',
      label: 'Import',
      icon: '📥',
      action: onImportConfig,
      type: 'secondary',
      tooltip: 'Import configuration file'
    },
    {
      id: 'backup',
      label: 'Backup',
      icon: '💾',
      action: onBackupSystem || (() => console.log('System backup')),
      type: 'secondary',
      disabled: !onBackupSystem,
      tooltip: 'Create system backup'
    }
  ];

  const criticalActions: ActionButton[] = [
    {
      id: 'emergency-stop',
      label: 'Emergency Stop',
      icon: '🛑',
      action: onEmergencyStop || (() => console.log('Emergency stop')),
      type: 'danger',
      disabled: !onEmergencyStop || isLoading,
      confirmRequired: true,
      tooltip: 'Emergency stop all systems'
    },
    {
      id: 'restore',
      label: 'Restore',
      icon: '🔄',
      action: onRestoreSystem || (() => console.log('System restore')),
      type: 'warning',
      disabled: !onRestoreSystem,
      confirmRequired: true,
      tooltip: 'Restore system from backup'
    }
  ];

  const renderActionButton = (buttonAction: ActionButton) => (
    <button
      key={buttonAction.id}
      className={`quick-action-btn ${buttonAction.type || 'secondary'} ${executingAction === buttonAction.id ? 'executing' : ''}`}
      onClick={() => handleActionClick(buttonAction.id, buttonAction.action, buttonAction.confirmRequired)}
      disabled={buttonAction.disabled || executingAction === buttonAction.id}
      title={buttonAction.tooltip}
    >
      <span className="action-icon">
        {executingAction === buttonAction.id ? '⟳' : buttonAction.icon}
      </span>
      <span className="action-label">{buttonAction.label}</span>
      {showConfirmDialog === buttonAction.id && (
        <span className="confirm-indicator">❗</span>
      )}
    </button>
  );

  const renderConfirmDialog = () => {
    if (!showConfirmDialog) return null;

    const action = [...primaryActions, ...dataActions, ...criticalActions]
      .find(a => a.id === showConfirmDialog);

    if (!action) return null;

    return (
      <div className="confirm-overlay">
        <div className="confirm-dialog">
          <div className="confirm-header">
            <span className="confirm-icon">{action.icon}</span>
            <h4>Confirm Action</h4>
          </div>

          <div className="confirm-content">
            <p>Are you sure you want to <strong>{action.label.toLowerCase()}</strong>?</p>
            {action.id === 'emergency-stop' && (
              <p className="warning-text">
                ⚠️ This will stop all running systems and agents immediately.
              </p>
            )}
            {action.id === 'restore' && (
              <p className="warning-text">
                ⚠️ This will overwrite current system configuration.
              </p>
            )}
          </div>

          <div className="confirm-actions">
            <button
              className="cancel-btn"
              onClick={() => setShowConfirmDialog(null)}
            >
              Cancel
            </button>
            <button
              className={`confirm-btn ${action.type}`}
              onClick={() => handleActionClick(action.id, action.action, false)}
            >
              {action.label}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={`quick-actions-panel ${className}`}>
      <div className="panel-header">
        <h3>Quick Actions</h3>
        <div className="panel-status">
          {isLoading && (
            <div className="loading-indicator">
              <span className="loading-spinner">⟳</span>
              <span>Loading...</span>
            </div>
          )}
        </div>
      </div>

      <div className="actions-container">
        <div className="action-group primary-actions">
          <h4 className="group-title">System Control</h4>
          <div className="action-buttons">
            {primaryActions.map(renderActionButton)}
          </div>
        </div>

        <div className="action-group data-actions">
          <h4 className="group-title">Data Management</h4>
          <div className="action-buttons">
            {dataActions.map(renderActionButton)}
          </div>
        </div>

        <div className="action-group critical-actions">
          <h4 className="group-title">Emergency Operations</h4>
          <div className="action-buttons">
            {criticalActions.map(renderActionButton)}
          </div>
        </div>
      </div>

      <div className="panel-footer">
        <div className="status-indicators">
          <div className="status-item">
            <span className="status-dot auto-refresh" data-active={autoRefreshEnabled}></span>
            <span className="status-text">Auto-refresh {autoRefreshEnabled ? 'ON' : 'OFF'}</span>
          </div>
          <div className="status-item">
            <span className="status-dot system-status" data-status="online"></span>
            <span className="status-text">System Online</span>
          </div>
        </div>

        <div className="last-action">
          {executingAction && (
            <span className="executing-text">
              Executing {primaryActions.concat(dataActions, criticalActions)
                .find(a => a.id === executingAction)?.label.toLowerCase()}...
            </span>
          )}
        </div>
      </div>

      {renderConfirmDialog()}
    </div>
  );
};

export default QuickActionsPanel;
