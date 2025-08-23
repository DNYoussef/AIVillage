import React, { useState, useEffect } from 'react';
import { DigitalTwinChat } from './components/concierge/DigitalTwinChat';
import { BitChatInterface } from './components/messaging/BitChatInterface';
import { MediaDisplayEngine } from './components/media/MediaDisplayEngine';
import { ComputeCreditsWallet } from './components/wallet/ComputeCreditsWallet';
import { SystemControlDashboard } from './components/dashboard/SystemControlDashboard';
import { apiService } from './services/apiService';
import './App.css';

interface AppProps {
  userId?: string;
}

const App: React.FC<AppProps> = ({ userId = 'demo-user' }) => {
  const [activeComponent, setActiveComponent] = useState<'concierge' | 'messaging' | 'media' | 'wallet' | 'dashboard'>('concierge');
  const [isSystemHealthy, setIsSystemHealthy] = useState(true);
  const [notifications, setNotifications] = useState<Array<{ id: string; message: string; type: 'info' | 'warning' | 'error' }>>([]);

  useEffect(() => {
    // Check system health on startup
    const checkHealth = async () => {
      const healthy = await apiService.healthCheck();
      setIsSystemHealthy(healthy);

      if (!healthy) {
        setNotifications(prev => [...prev, {
          id: Date.now().toString(),
          message: 'Backend services are unavailable. Some features may not work.',
          type: 'warning'
        }]);
      }
    };

    checkHealth();

    // Set up authentication if needed
    const token = localStorage.getItem('aivillage_token');
    if (token) {
      apiService.setAuthToken(token);
    }
  }, []);

  const dismissNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const renderActiveComponent = () => {
    switch (activeComponent) {
      case 'concierge':
        return (
          <DigitalTwinChat
            twinId={`twin-${userId}`}
            userId={userId}
            onClose={() => console.log('Concierge closed')}
          />
        );
      case 'messaging':
        return (
          <BitChatInterface
            userId={userId}
            onPeerConnect={(peer) => {
              setNotifications(prev => [...prev, {
                id: Date.now().toString(),
                message: `Connected to ${peer.name}`,
                type: 'info'
              }]);
            }}
            onMessageReceived={(message) => {
              console.log('Message received:', message);
            }}
          />
        );
      case 'media':
        return (
          <MediaDisplayEngine
            fogEnabled={true}
            onContentLoad={(content) => {
              console.log('Media content loaded:', content);
            }}
            onError={(error) => {
              setNotifications(prev => [...prev, {
                id: Date.now().toString(),
                message: `Media error: ${error}`,
                type: 'error'
              }]);
            }}
          />
        );
      case 'wallet':
        return (
          <ComputeCreditsWallet
            userId={userId}
            onTransactionComplete={(tx) => {
              setNotifications(prev => [...prev, {
                id: Date.now().toString(),
                message: `Transaction completed: ${tx.type} ${tx.amount} credits`,
                type: 'info'
              }]);
            }}
            onFogNodeUpdate={(node) => {
              console.log('Fog node updated:', node);
            }}
          />
        );
      case 'dashboard':
        return (
          <SystemControlDashboard
            userId={userId}
            onComponentToggle={(component, enabled) => {
              setNotifications(prev => [...prev, {
                id: Date.now().toString(),
                message: `${component} ${enabled ? 'enabled' : 'disabled'}`,
                type: 'info'
              }]);
            }}
            onEmergencyStop={() => {
              setNotifications(prev => [...prev, {
                id: Date.now().toString(),
                message: 'Emergency stop activated',
                type: 'warning'
              }]);
            }}
          />
        );
      default:
        return <div>Component not found</div>;
    }
  };

  return (
    <div className="aivillage-app">
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-logo">🏘️</div>
          <div className="brand-info">
            <h1>AIVillage</h1>
            <p>Distributed AI Platform</p>
          </div>
        </div>

        <nav className="app-navigation">
          <button
            onClick={() => setActiveComponent('concierge')}
            className={`nav-btn ${activeComponent === 'concierge' ? 'active' : ''}`}
            title="Digital Twin Concierge"
          >
            🤖 Concierge
          </button>
          <button
            onClick={() => setActiveComponent('messaging')}
            className={`nav-btn ${activeComponent === 'messaging' ? 'active' : ''}`}
            title="P2P Messaging"
          >
            💬 Messaging
          </button>
          <button
            onClick={() => setActiveComponent('media')}
            className={`nav-btn ${activeComponent === 'media' ? 'active' : ''}`}
            title="Media Display"
          >
            🖼️ Media
          </button>
          <button
            onClick={() => setActiveComponent('wallet')}
            className={`nav-btn ${activeComponent === 'wallet' ? 'active' : ''}`}
            title="Compute Credits"
          >
            💰 Wallet
          </button>
          <button
            onClick={() => setActiveComponent('dashboard')}
            className={`nav-btn ${activeComponent === 'dashboard' ? 'active' : ''}`}
            title="System Control"
          >
            📊 Dashboard
          </button>
        </nav>

        <div className="system-status">
          <div className={`status-indicator ${isSystemHealthy ? 'healthy' : 'unhealthy'}`}>
            {isSystemHealthy ? '✅' : '⚠️'}
          </div>
          <span className="status-text">
            {isSystemHealthy ? 'System Online' : 'System Issues'}
          </span>
        </div>
      </header>

      <main className="app-main">
        <div className="component-container">
          {renderActiveComponent()}
        </div>
      </main>

      {notifications.length > 0 && (
        <div className="notifications-container">
          {notifications.map((notification) => (
            <div
              key={notification.id}
              className={`notification ${notification.type}`}
            >
              <div className="notification-content">
                <span className="notification-icon">
                  {notification.type === 'info' ? 'ℹ️' : notification.type === 'warning' ? '⚠️' : '🚨'}
                </span>
                <span className="notification-message">{notification.message}</span>
              </div>
              <button
                onClick={() => dismissNotification(notification.id)}
                className="notification-dismiss"
                aria-label="Dismiss notification"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <footer className="app-footer">
        <div className="footer-info">
          <span>AIVillage v2.0.0</span>
          <span>•</span>
          <span>Connected to {isSystemHealthy ? 'Fog Network' : 'Local Mode'}</span>
          <span>•</span>
          <span>User: {userId}</span>
        </div>
        <div className="footer-links">
          <a href="https://github.com/DNYoussef/AIVillage" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          <a href="/docs" target="_blank" rel="noopener noreferrer">
            Documentation
          </a>
          <a href="/support" target="_blank" rel="noopener noreferrer">
            Support
          </a>
        </div>
      </footer>
    </div>
  );
};

export default App;
