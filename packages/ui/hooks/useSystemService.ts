// System Service Hook - Monitor and control AIVillage system components
import { useState, useEffect, useCallback, useRef } from 'react';
import { SystemDashboard, FogNode } from '../types';

interface SystemMetrics {
  cpuUsage: number;
  memoryUsage: number;
  networkTraffic: number;
  activeConnections: number;
  messagesThroughput: number;
  errorRate: number;
}

interface AgentStatus {
  id: string;
  name: string;
  type: 'specialist' | 'coordinator' | 'worker';
  status: 'active' | 'idle' | 'busy' | 'error';
  performance: number;
  tasksCompleted: number;
  uptime: number;
  lastActivity: Date;
}

interface NetworkHealth {
  p2pConnections: number;
  messageLatency: number;
  nodeCount: number;
  meshStability: number;
  bandwidthUtilization: number;
  packetLoss: number;
}

export interface SystemServiceHook {
  systemDashboard: SystemDashboard;
  systemMetrics: SystemMetrics;
  networkHealth: NetworkHealth;
  agents: AgentStatus[];
  isMonitoring: boolean;
  startMonitoring: () => void;
  stopMonitoring: () => void;
  restartAgent: (agentId: string) => Promise<boolean>;
  scaleAgents: (agentType: string, count: number) => Promise<boolean>;
  optimizeSystem: () => Promise<boolean>;
  exportSystemReport: () => string;
  getSystemAlerts: () => Array<{ id: string; type: 'warning' | 'error' | 'info'; message: string; timestamp: Date }>;
}

export const useSystemService = (userId: string): SystemServiceHook => {
  const [systemDashboard, setSystemDashboard] = useState<SystemDashboard>({
    agents: [],
    fogNodes: [],
    networkHealth: {
      p2pConnections: 0,
      messageLatency: 0,
      nodeCount: 0
    },
    systemMetrics: {
      cpuUsage: 0,
      memoryUsage: 0,
      networkTraffic: 0
    }
  });

  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    cpuUsage: 0,
    memoryUsage: 0,
    networkTraffic: 0,
    activeConnections: 0,
    messagesThroughput: 0,
    errorRate: 0
  });

  const [networkHealth, setNetworkHealth] = useState<NetworkHealth>({
    p2pConnections: 0,
    messageLatency: 0,
    nodeCount: 0,
    meshStability: 0,
    bandwidthUtilization: 0,
    packetLoss: 0
  });

  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [systemAlerts, setSystemAlerts] = useState<Array<{ id: string; type: 'warning' | 'error' | 'info'; message: string; timestamp: Date }>>([]);

  const monitoringInterval = useRef<NodeJS.Timeout | null>(null);
  const performanceHistory = useRef<SystemMetrics[]>([]);

  // Initialize system monitoring
  useEffect(() => {
    initializeSystemMonitoring();
    return () => {
      if (monitoringInterval.current) {
        clearInterval(monitoringInterval.current);
      }
    };
  }, []);

  const initializeSystemMonitoring = (): void => {
    // Initialize demo agents
    const demoAgents: AgentStatus[] = [
      {
        id: 'agent-king',
        name: 'King Agent',
        type: 'coordinator',
        status: 'active',
        performance: 95,
        tasksCompleted: 147,
        uptime: 86400000, // 24 hours
        lastActivity: new Date()
      },
      {
        id: 'agent-magi-001',
        name: 'Magi Infrastructure',
        type: 'specialist',
        status: 'active',
        performance: 88,
        tasksCompleted: 203,
        uptime: 172800000, // 48 hours
        lastActivity: new Date(Date.now() - 30000)
      },
      {
        id: 'agent-oracle',
        name: 'Oracle Knowledge',
        type: 'specialist',
        status: 'busy',
        performance: 92,
        tasksCompleted: 89,
        uptime: 43200000, // 12 hours
        lastActivity: new Date()
      },
      {
        id: 'agent-worker-001',
        name: 'Compute Worker #1',
        type: 'worker',
        status: 'idle',
        performance: 76,
        tasksCompleted: 45,
        uptime: 21600000, // 6 hours
        lastActivity: new Date(Date.now() - 120000)
      }
    ];

    setAgents(demoAgents);
    setSystemDashboard(prev => ({
      ...prev,
      agents: demoAgents.map(a => ({
        id: a.id,
        name: a.name,
        status: a.status,
        performance: a.performance
      }))
    }));

    // Initialize fog nodes
    const demoFogNodes: FogNode[] = [
      {
        id: 'fog-primary',
        name: 'Primary Compute Node',
        location: 'Local Network',
        resources: { cpu: 85, memory: 16384, storage: 512000, bandwidth: 1000 },
        status: 'active',
        reputation: 98
      },
      {
        id: 'fog-secondary',
        name: 'Secondary Node',
        location: 'Edge Network',
        resources: { cpu: 60, memory: 8192, storage: 256000, bandwidth: 500 },
        status: 'active',
        reputation: 94
      }
    ];

    setSystemDashboard(prev => ({ ...prev, fogNodes: demoFogNodes }));
  };

  const startMonitoring = useCallback((): void => {
    if (isMonitoring) return;

    setIsMonitoring(true);
    monitoringInterval.current = setInterval(() => {
      updateSystemMetrics();
      updateNetworkHealth();
      updateAgentStatuses();
      checkSystemAlerts();
    }, 5000); // Update every 5 seconds

    console.log('System monitoring started');
  }, [isMonitoring]);

  const stopMonitoring = useCallback((): void => {
    if (!isMonitoring) return;

    setIsMonitoring(false);
    if (monitoringInterval.current) {
      clearInterval(monitoringInterval.current);
      monitoringInterval.current = null;
    }

    console.log('System monitoring stopped');
  }, [isMonitoring]);

  const updateSystemMetrics = (): void => {
    // Simulate realistic system metrics
    const newMetrics: SystemMetrics = {
      cpuUsage: Math.max(0, Math.min(100, 30 + Math.random() * 40 + Math.sin(Date.now() / 60000) * 15)),
      memoryUsage: Math.max(0, Math.min(100, 50 + Math.random() * 30 + Math.sin(Date.now() / 120000) * 10)),
      networkTraffic: Math.max(0, 10 + Math.random() * 80 + Math.sin(Date.now() / 30000) * 20),
      activeConnections: Math.floor(5 + Math.random() * 15),
      messagesThroughput: Math.floor(100 + Math.random() * 500),
      errorRate: Math.max(0, Math.min(5, Math.random() * 2))
    };

    setSystemMetrics(newMetrics);

    // Keep performance history for analysis
    performanceHistory.current.push(newMetrics);
    if (performanceHistory.current.length > 100) {
      performanceHistory.current.shift();
    }

    // Update dashboard
    setSystemDashboard(prev => ({
      ...prev,
      systemMetrics: {
        cpuUsage: newMetrics.cpuUsage,
        memoryUsage: newMetrics.memoryUsage,
        networkTraffic: newMetrics.networkTraffic
      }
    }));
  };

  const updateNetworkHealth = (): void => {
    const newHealth: NetworkHealth = {
      p2pConnections: Math.floor(8 + Math.random() * 12),
      messageLatency: Math.max(10, 50 + Math.random() * 100),
      nodeCount: Math.floor(15 + Math.random() * 10),
      meshStability: Math.max(70, 85 + Math.random() * 15),
      bandwidthUtilization: Math.max(0, 30 + Math.random() * 50),
      packetLoss: Math.max(0, Math.random() * 2)
    };

    setNetworkHealth(newHealth);
    setSystemDashboard(prev => ({
      ...prev,
      networkHealth: {
        p2pConnections: newHealth.p2pConnections,
        messageLatency: newHealth.messageLatency,
        nodeCount: newHealth.nodeCount
      }
    }));
  };

  const updateAgentStatuses = (): void => {
    setAgents(prev => prev.map(agent => {
      // Simulate agent activity
      const performanceChange = (Math.random() - 0.5) * 5;
      const newPerformance = Math.max(0, Math.min(100, agent.performance + performanceChange));

      // Randomly change status occasionally
      let newStatus = agent.status;
      if (Math.random() < 0.1) { // 10% chance to change status
        const statuses: AgentStatus['status'][] = ['active', 'idle', 'busy'];
        newStatus = statuses[Math.floor(Math.random() * statuses.length)];
      }

      return {
        ...agent,
        performance: newPerformance,
        status: newStatus,
        tasksCompleted: agent.status === 'active' ? agent.tasksCompleted + Math.floor(Math.random() * 3) : agent.tasksCompleted,
        lastActivity: agent.status === 'active' ? new Date() : agent.lastActivity
      };
    }));
  };

  const checkSystemAlerts = (): void => {
    const alerts = [];

    // Check system metrics for alerts
    if (systemMetrics.cpuUsage > 90) {
      alerts.push({
        id: `alert-cpu-${Date.now()}`,
        type: 'warning' as const,
        message: `High CPU usage detected: ${systemMetrics.cpuUsage.toFixed(1)}%`,
        timestamp: new Date()
      });
    }

    if (systemMetrics.errorRate > 3) {
      alerts.push({
        id: `alert-errors-${Date.now()}`,
        type: 'error' as const,
        message: `Error rate elevated: ${systemMetrics.errorRate.toFixed(2)}%`,
        timestamp: new Date()
      });
    }

    if (networkHealth.packetLoss > 1) {
      alerts.push({
        id: `alert-network-${Date.now()}`,
        type: 'warning' as const,
        message: `Network packet loss detected: ${networkHealth.packetLoss.toFixed(2)}%`,
        timestamp: new Date()
      });
    }

    // Check agent health
    const errorAgents = agents.filter(a => a.status === 'error');
    if (errorAgents.length > 0) {
      alerts.push({
        id: `alert-agents-${Date.now()}`,
        type: 'error' as const,
        message: `${errorAgents.length} agent(s) in error state`,
        timestamp: new Date()
      });
    }

    if (alerts.length > 0) {
      setSystemAlerts(prev => [...alerts, ...prev.slice(0, 20)]); // Keep last 20 alerts
    }
  };

  const restartAgent = useCallback(async (agentId: string): Promise<boolean> => {
    try {
      setAgents(prev => prev.map(agent =>
        agent.id === agentId
          ? { ...agent, status: 'active', performance: 100, lastActivity: new Date() }
          : agent
      ));

      setSystemAlerts(prev => [{
        id: `restart-${Date.now()}`,
        type: 'info',
        message: `Agent ${agentId} restarted successfully`,
        timestamp: new Date()
      }, ...prev]);

      return true;
    } catch (error) {
      console.error(`Failed to restart agent ${agentId}:`, error);
      return false;
    }
  }, []);

  const scaleAgents = useCallback(async (agentType: string, count: number): Promise<boolean> => {
    try {
      const newAgents: AgentStatus[] = [];
      for (let i = 0; i < count; i++) {
        newAgents.push({
          id: `agent-${agentType}-${Date.now()}-${i}`,
          name: `${agentType} Agent #${i + 1}`,
          type: agentType as AgentStatus['type'],
          status: 'active',
          performance: 85 + Math.random() * 15,
          tasksCompleted: 0,
          uptime: 0,
          lastActivity: new Date()
        });
      }

      setAgents(prev => [...prev, ...newAgents]);

      setSystemAlerts(prev => [{
        id: `scale-${Date.now()}`,
        type: 'info',
        message: `Scaled ${agentType} agents by ${count}`,
        timestamp: new Date()
      }, ...prev]);

      return true;
    } catch (error) {
      console.error(`Failed to scale ${agentType} agents:`, error);
      return false;
    }
  }, []);

  const optimizeSystem = useCallback(async (): Promise<boolean> => {
    try {
      // Simulate system optimization
      setSystemMetrics(prev => ({
        ...prev,
        cpuUsage: Math.max(10, prev.cpuUsage * 0.8),
        memoryUsage: Math.max(20, prev.memoryUsage * 0.9),
        errorRate: Math.max(0, prev.errorRate * 0.5)
      }));

      setAgents(prev => prev.map(agent => ({
        ...agent,
        performance: Math.min(100, agent.performance * 1.1)
      })));

      setSystemAlerts(prev => [{
        id: `optimize-${Date.now()}`,
        type: 'info',
        message: 'System optimization completed successfully',
        timestamp: new Date()
      }, ...prev]);

      return true;
    } catch (error) {
      console.error('System optimization failed:', error);
      return false;
    }
  }, []);

  const exportSystemReport = useCallback((): string => {
    const report = {
      timestamp: new Date().toISOString(),
      systemMetrics,
      networkHealth,
      agents: agents.map(a => ({
        id: a.id,
        name: a.name,
        status: a.status,
        performance: a.performance,
        uptime: a.uptime
      })),
      performanceHistory: performanceHistory.current,
      alerts: systemAlerts.slice(0, 10)
    };

    return JSON.stringify(report, null, 2);
  }, [systemMetrics, networkHealth, agents, systemAlerts]);

  const getSystemAlerts = useCallback(() => {
    return systemAlerts;
  }, [systemAlerts]);

  return {
    systemDashboard,
    systemMetrics,
    networkHealth,
    agents,
    isMonitoring,
    startMonitoring,
    stopMonitoring,
    restartAgent,
    scaleAgents,
    optimizeSystem,
    exportSystemReport,
    getSystemAlerts
  };
};
