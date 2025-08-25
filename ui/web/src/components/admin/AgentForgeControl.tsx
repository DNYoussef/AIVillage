import React, { useState, useEffect } from 'react';
import './AgentForgeControl.css';

interface PhaseStatus {
  phase_name: string;
  status: 'ready' | 'running' | 'completed' | 'error';
  progress: number;
  message: string;
  start_time?: string;
  duration_seconds?: number;
  current_step?: string;
  total_steps?: number;
  estimated_time_remaining?: number;
  models_completed?: number;
  total_models?: number;
}

interface ModelInfo {
  model_id: string;
  model_name: string;
  phase_name: string;
  parameter_count: number;
  created_at: string;
  training_status?: 'training' | 'completed' | 'failed' | 'pending';
  focus?: string;
  artifacts?: any;
}

interface SystemMetrics {
  cpu: { usage_percent: number; count: number };
  memory: { usage_percent: number; available_gb: number; total_gb: number };
  gpu?: { gpu_memory_used?: number; gpu_memory_total?: number; gpu_name?: string };
}

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

interface AgentFleetMetrics {
  totalAgents: number;
  activeAgents: number;
  errorAgents: number;
  averagePerformance: number;
  totalTasksCompleted: number;
}

interface ConnectionHealth {
  status: 'connected' | 'disconnected' | 'error' | 'reconnecting';
  lastConnected: Date | null;
  reconnectAttempts: number;
  latency?: number;
}

export const AgentForgeControl: React.FC = () => {
  const [phases, setPhases] = useState<PhaseStatus[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [agentFleetMetrics, setAgentFleetMetrics] = useState<AgentFleetMetrics | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [chatMessage, setChatMessage] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [connectionHealth, setConnectionHealth] = useState<ConnectionHealth>({
    status: 'disconnected',
    lastConnected: null,
    reconnectAttempts: 0
  });
  const [lastPingTime, setLastPingTime] = useState<number>(0);

  const API_BASE = 'http://localhost:8083';
  const CHAT_API = 'http://localhost:8084';
  const WS_URL = 'ws://localhost:8085/ws';
  const AGENT_API = 'http://localhost:8086';

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    connectWebSocket();
    // Load initial data
    loadPhases();
    loadModels();
    loadSystemMetrics();
    loadAgents();
    loadAgentFleetMetrics();

    // Set up polling for fallback
    const interval = setInterval(() => {
      loadPhases();
      loadSystemMetrics();
      loadAgents();
      loadAgentFleetMetrics();
    }, 5000);

    return () => {
      clearInterval(interval);
      if (ws) {
        ws.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      const websocket = new WebSocket(WS_URL);

      websocket.onopen = () => {
        setConnectionStatus('connected');
        // Subscribe to relevant channels
        websocket.send(JSON.stringify({
          type: 'subscribe',
          channel: 'agent_forge_phases'
        }));
        websocket.send(JSON.stringify({
          type: 'subscribe',
          channel: 'system_metrics'
        }));
        websocket.send(JSON.stringify({
          type: 'subscribe',
          channel: 'model_updates'
        }));
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      };

      websocket.onclose = () => {
        setConnectionStatus('disconnected');
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      websocket.onerror = () => {
        setConnectionStatus('error');
      };

      setWs(websocket);
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      setConnectionStatus('error');
    }
  };

  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'phase_update':
        setPhases(prev => prev.map(phase =>
          phase.phase_name === data.phase_name
            ? { ...phase, status: data.status, progress: data.progress, message: data.message }
            : phase
        ));
        break;
      case 'system_metrics':
        setSystemMetrics(data.metrics);
        break;
      case 'model_update':
        if (data.event_type === 'created') {
          loadModels(); // Reload models when new ones are created
        }
        break;
    }
  };

  const loadPhases = async () => {
    try {
      const response = await fetch(`${API_BASE}/phases/status`);
      const data = await response.json();
      setPhases(data.phases || []);
    } catch (error) {
      console.error('Failed to load phases:', error);
    }
  };

  const loadModels = async () => {
    try {
      const response = await fetch(`${CHAT_API}/models`);
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  };

  const loadSystemMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/system/metrics`);
      const data = await response.json();
      setSystemMetrics(data);
    } catch (error) {
      console.error('Failed to load system metrics:', error);
    }
  };

  const loadAgents = async () => {
    try {
      const response = await fetch(`${AGENT_API}/agents/status`, {
        headers: { 'Accept': 'application/json' }
      });
      if (response.ok) {
        const data = await response.json();
        setAgents((data.agents || []).map((agent: any) => ({
          ...agent,
          lastActivity: new Date(agent.lastActivity)
        })));
      } else {
        console.warn('Agent service unavailable, using fallback');
        setAgents(generateMockAgents());
      }
    } catch (error) {
      console.error('Failed to load agents:', error);
      setAgents(generateMockAgents());
    }
  };

  const loadAgentFleetMetrics = async () => {
    try {
      const response = await fetch(`${AGENT_API}/agents/metrics`);
      if (response.ok) {
        const data = await response.json();
        setAgentFleetMetrics(data);
      } else {
        if (agents.length > 0) {
          const metrics = {
            totalAgents: agents.length,
            activeAgents: agents.filter(a => a.status === 'active').length,
            errorAgents: agents.filter(a => a.status === 'error').length,
            averagePerformance: agents.reduce((sum, a) => sum + a.performance, 0) / agents.length,
            totalTasksCompleted: agents.reduce((sum, a) => sum + a.tasksCompleted, 0)
          };
          setAgentFleetMetrics(metrics);
        }
      }
    } catch (error) {
      console.error('Failed to load agent fleet metrics:', error);
    }
  };

  const restartAgent = async (agentId: string): Promise<boolean> => {
    try {
      const response = await fetch(`${AGENT_API}/agents/${agentId}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        setAgents(prev => prev.map(agent =>
          agent.id === agentId
            ? { ...agent, status: 'idle' as const, lastActivity: new Date() }
            : agent
        ));
        setTimeout(loadAgents, 2000);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to restart agent:', error);
      return false;
    }
  };

  const scaleAgents = async (agentType: string, count: number): Promise<boolean> => {
    try {
      const response = await fetch(`${AGENT_API}/agents/scale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_type: agentType, count })
      });
      if (response.ok) {
        setTimeout(loadAgents, 1000);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Failed to scale agents:', error);
      return false;
    }
  };

  const startPhase = async (phaseName: string) => {
    try {
      let endpoint = '';
      switch (phaseName) {
        case 'Cognate':
          endpoint = '/phases/cognate/start';
          break;
        case 'EvoMerge':
          endpoint = '/phases/evomerge/start';
          break;
        default:
          alert(`Phase ${phaseName} not yet implemented`);
          return;
      }

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });

      if (response.ok) {
        const data = await response.json();
        console.log(`Started ${phaseName} phase:`, data);
        loadPhases(); // Refresh phase status
      } else {
        alert(`Failed to start ${phaseName} phase`);
      }
    } catch (error) {
      console.error(`Error starting ${phaseName} phase:`, error);
      alert(`Error starting ${phaseName} phase`);
    }
  };

  const sendChatMessage = async () => {
    if (!selectedModel || !chatMessage.trim()) return;

    try {
      const response = await fetch(`${CHAT_API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: selectedModel,
          message: chatMessage
        })
      });

      if (response.ok) {
        const data = await response.json();
        setChatHistory(prev => [...prev,
          { role: 'user', content: chatMessage },
          { role: 'assistant', content: data.response, model: data.model_name, time: data.response_time_ms }
        ]);
        setChatMessage('');
      } else {
        alert('Failed to send message');
      }
    } catch (error) {
      console.error('Chat error:', error);
      alert('Chat error occurred');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return '#4ade80';
      case 'running': return '#3b82f6';
      case 'completed': return '#10b981';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'ready': return '○';
      case 'running': return '⟳';
      case 'completed': return '✓';
      case 'error': return '✗';
      default: return '○';
    }
  };

  const getAgentStatusIcon = (status: Agent['status']) => {
    switch (status) {
      case 'active': return '🟢';
      case 'idle': return '🟡';
      case 'busy': return '🔵';
      case 'error': return '🔴';
      default: return '⚫';
    }
  };

  const getAgentTypeIcon = (type: Agent['type']) => {
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

  // Mock data generator for fallback
  const generateMockAgents = (): Agent[] => {
    const types: Agent['type'][] = ['specialist', 'coordinator', 'worker'];
    const statuses: Agent['status'][] = ['active', 'idle', 'busy', 'error'];

    return Array.from({ length: 6 }, (_, i) => ({
      id: `agent-${i + 1}`,
      name: `Agent ${i + 1}`,
      type: types[i % types.length],
      status: statuses[Math.floor(Math.random() * statuses.length)],
      performance: Math.round(60 + Math.random() * 35),
      tasksCompleted: Math.floor(Math.random() * 100),
      uptime: Date.now() - Math.random() * 86400000,
      lastActivity: new Date(Date.now() - Math.random() * 3600000)
    }));
  };

  return (
    <div className="agent-forge-control">
      <div className="control-header">
        <h2>Agent Forge Training Control</h2>
        <div className="connection-status">
          <span className={`status-indicator ${connectionStatus}`}>
            {connectionStatus === 'connected' ? '●' : '○'}
          </span>
          <span>WebSocket: {connectionStatus}</span>
        </div>
      </div>

      {/* Phase Control Section */}
      <div className="phase-control-section">
        <h3>Phase Control</h3>
        <div className="phases-grid">
          {phases.map((phase) => (
            <div key={phase.phase_name} className="phase-card">
              <div className="phase-header">
                <h4>{phase.phase_name}</h4>
                <span
                  className="status-badge"
                  style={{ backgroundColor: getStatusColor(phase.status) }}
                >
                  {getStatusIcon(phase.status)} {phase.status}
                </span>
              </div>

              <div className="progress-section">
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${phase.progress * 100}%` }}
                  ></div>
                </div>
                <span className="progress-text">{Math.round(phase.progress * 100)}%</span>
              </div>

              {phase.models_completed !== undefined && phase.total_models && (
                <div className="model-progress">
                  <small>Models: {phase.models_completed}/{phase.total_models}</small>
                </div>
              )}

              {phase.current_step && (
                <div className="step-progress">
                  <small>{phase.current_step}</small>
                  {phase.estimated_time_remaining && (
                    <small className="time-remaining">
                      ~{Math.round(phase.estimated_time_remaining / 60)}m remaining
                    </small>
                  )}
                </div>
              )}

              <div className="phase-message">{phase.message}</div>

              <button
                className={`phase-button ${phase.status}`}
                onClick={() => startPhase(phase.phase_name)}
                disabled={phase.status === 'running'}
              >
                {phase.status === 'running' ? 'RUNNING...' : `START ${phase.phase_name.toUpperCase()}`}
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* System Metrics Dashboard */}
      {systemMetrics && (
        <div className="metrics-section">
          <h3>System Resources</h3>
          <div className="metrics-grid">
            <div className="metric-card">
              <h4>CPU Usage</h4>
              <div className="metric-value">{systemMetrics.cpu.usage_percent.toFixed(1)}%</div>
              <div className="metric-bar">
                <div
                  className="metric-fill"
                  style={{ width: `${systemMetrics.cpu.usage_percent}%` }}
                ></div>
              </div>
              <div className="metric-detail">{systemMetrics.cpu.count} cores</div>
            </div>

            <div className="metric-card">
              <h4>Memory Usage</h4>
              <div className="metric-value">{systemMetrics.memory.usage_percent.toFixed(1)}%</div>
              <div className="metric-bar">
                <div
                  className="metric-fill"
                  style={{ width: `${systemMetrics.memory.usage_percent}%` }}
                ></div>
              </div>
              <div className="metric-detail">
                {(systemMetrics.memory.total_gb - systemMetrics.memory.available_gb).toFixed(1)} /
                {systemMetrics.memory.total_gb.toFixed(1)} GB
              </div>
            </div>

            {systemMetrics.gpu && (
              <div className="metric-card">
                <h4>GPU Memory</h4>
                <div className="metric-value">
                  {systemMetrics.gpu.gpu_memory_used?.toFixed(1) || 0} /
                  {systemMetrics.gpu.gpu_memory_total?.toFixed(1) || 0} GB
                </div>
                <div className="metric-bar">
                  <div
                    className="metric-fill"
                    style={{
                      width: `${((systemMetrics.gpu.gpu_memory_used || 0) / (systemMetrics.gpu.gpu_memory_total || 1)) * 100}%`
                    }}
                  ></div>
                </div>
                <div className="metric-detail">{systemMetrics.gpu.gpu_name || 'GPU'}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Agent Fleet Status Section */}
      {agents.length > 0 && (
        <div className="agent-fleet-section">
          <div className="section-header">
            <h3>Agent Fleet Status</h3>
            {agentFleetMetrics && (
              <div className="fleet-summary">
                <span className="summary-stat">
                  <span className="stat-value">{agentFleetMetrics.totalAgents}</span>
                  <span className="stat-label">Total</span>
                </span>
                <span className="summary-stat">
                  <span className="stat-value active">{agentFleetMetrics.activeAgents}</span>
                  <span className="stat-label">Active</span>
                </span>
                <span className="summary-stat">
                  <span className="stat-value error">{agentFleetMetrics.errorAgents}</span>
                  <span className="stat-label">Errors</span>
                </span>
                <span className="summary-stat">
                  <span className="stat-value performance">{agentFleetMetrics.averagePerformance.toFixed(1)}%</span>
                  <span className="stat-label">Avg Performance</span>
                </span>
              </div>
            )}
          </div>

          <div className="agent-groups">
            {Object.entries(
              agents.reduce((acc, agent) => {
                if (!acc[agent.type]) acc[agent.type] = [];
                acc[agent.type].push(agent);
                return acc;
              }, {} as Record<string, Agent[]>)
            ).map(([type, typeAgents]) => (
              <div key={type} className="agent-group">
                <div className="group-header">
                  <div className="group-title">
                    {getAgentTypeIcon(type as Agent['type'])}
                    <span className="type-name">
                      {type.charAt(0).toUpperCase() + type.slice(1)} Agents
                    </span>
                    <span className="type-count">({typeAgents.length})</span>
                  </div>
                  <button
                    className="scale-btn"
                    onClick={() => scaleAgents(type, 1)}
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
                            <span className="agent-icon">{getAgentTypeIcon(agent.type)}</span>
                            <span className="agent-name">{agent.name}</span>
                            <span className="status-indicator" title={agent.status}>
                              {getAgentStatusIcon(agent.status)}
                            </span>
                          </div>

                          {agent.status === 'error' && (
                            <button
                              className="restart-btn"
                              onClick={() => restartAgent(agent.id)}
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

          {/* Agent Fleet Empty State */}
          {agents.length === 0 && (
            <div className="empty-agents">
              <div className="empty-icon">🤖</div>
              <p>No agents detected</p>
              <small>Start the agent fleet to see status</small>
            </div>
          )}
        </div>
      )}

      {/* Model Management Section */}
      <div className="models-section">
        <h3>Trained Models</h3>

        {models.length > 0 ? (
          <div className="models-grid">
            {models.map((model) => (
              <div key={model.model_id} className="model-card">
                <div className="model-header">
                  <h4>{model.model_name}</h4>
                  <div className="model-badges">
                    <span className="phase-badge">{model.phase_name}</span>
                    <span className="params-badge">
                      {(model.parameter_count / 1000000).toFixed(1)}M params
                    </span>
                  </div>
                </div>

                <div className="model-details">
                  <div className="detail-row">
                    <span className="detail-label">Status:</span>
                    <span className={`detail-value status-${model.training_status || 'unknown'}`}>
                      {model.training_status || 'unknown'}
                    </span>
                  </div>
                  {model.focus && (
                    <div className="detail-row">
                      <span className="detail-label">Focus:</span>
                      <span className="detail-value">{model.focus}</span>
                    </div>
                  )}
                  <div className="detail-row">
                    <span className="detail-label">Parameters:</span>
                    <span className="detail-value">
                      {model.parameter_count.toLocaleString()}
                    </span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Created:</span>
                    <span className="detail-value">
                      {new Date(model.created_at).toLocaleString()}
                    </span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Model ID:</span>
                    <span className="detail-value model-id">{model.model_id}</span>
                  </div>
                </div>

                <div className="model-actions">
                  <button
                    className="model-action-button test"
                    onClick={() => setSelectedModel(model.model_id)}
                  >
                    {selectedModel === model.model_id ? 'Selected' : 'Test Model'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-models">
            <p>No trained models available yet. Complete a training phase to see models here.</p>
          </div>
        )}
      </div>

      {/* Model Chat Interface */}
      <div className="chat-section">
        <h3>Model Testing</h3>

        {models.length > 0 && (
          <div className="model-selection">
            <label>Select Model:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="">Choose a model...</option>
              {models.map(model => (
                <option key={model.model_id} value={model.model_id}>
                  {model.model_name} ({model.phase_name}) - {(model.parameter_count / 1000000).toFixed(1)}M params
                </option>
              ))}
            </select>
          </div>
        )}

        {selectedModel && (
          <div className="chat-interface">
            <div className="chat-history">
              {chatHistory.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-content">{msg.content}</div>
                  {msg.model && (
                    <div className="message-meta">
                      {msg.model} - {msg.time}ms
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="chat-input">
              <input
                type="text"
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                placeholder="Chat with the model..."
                onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
              />
              <button onClick={sendChatMessage} disabled={!chatMessage.trim()}>
                Send
              </button>
            </div>
          </div>
        )}

        {models.length === 0 && (
          <div className="no-models">
            <p>No trained models available yet. Complete a training phase to chat with models.</p>
          </div>
        )}
      </div>
    </div>
  );
};
