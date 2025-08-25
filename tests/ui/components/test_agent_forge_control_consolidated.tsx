/**
 * Comprehensive Consolidated Test Suite for Agent Forge Control
 * Integrates all UI testing patterns from multiple implementations
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { jest } from '@jest/globals';
import '@testing-library/jest-dom';
import WS from 'jest-websocket-mock';

import { AgentForgeControl } from '../../../ui/web/src/components/admin/AgentForgeControl';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Test data fixtures
const mockPhases = [
  {
    phase_name: 'Cognate',
    status: 'ready' as const,
    progress: 0,
    message: 'Ready to start',
    current_step: 'initialization',
    total_steps: 5
  },
  {
    phase_name: 'EvoMerge',
    status: 'running' as const,
    progress: 0.65,
    message: 'Processing model evolution',
    current_step: 'merging models',
    total_steps: 3,
    models_completed: 2,
    total_models: 3
  }
];

const mockModels = [
  {
    model_id: 'model-1',
    model_name: 'Cognate Model Alpha',
    phase_name: 'Cognate',
    parameter_count: 25000000,
    created_at: '2024-01-15T10:30:00Z',
    training_status: 'completed' as const,
    focus: 'reasoning'
  },
  {
    model_id: 'model-2',
    model_name: 'EvoMerge Beta',
    phase_name: 'EvoMerge',
    parameter_count: 50000000,
    created_at: '2024-01-15T14:45:00Z',
    training_status: 'training' as const
  }
];

const mockAgents = [
  {
    id: 'agent-1',
    name: 'Coordinator Alpha',
    type: 'coordinator' as const,
    status: 'active' as const,
    performance: 95.5,
    tasksCompleted: 234,
    uptime: 86400000,
    lastActivity: new Date('2024-01-15T15:30:00Z')
  },
  {
    id: 'agent-2',
    name: 'Specialist Beta',
    type: 'specialist' as const,
    status: 'error' as const,
    performance: 72.3,
    tasksCompleted: 156,
    uptime: 43200000,
    lastActivity: new Date('2024-01-15T14:15:00Z')
  },
  {
    id: 'agent-3',
    name: 'Worker Gamma',
    type: 'worker' as const,
    status: 'busy' as const,
    performance: 88.7,
    tasksCompleted: 89,
    uptime: 21600000,
    lastActivity: new Date('2024-01-15T15:29:00Z')
  }
];

const mockSystemMetrics = {
  cpu: { usage_percent: 45.6, count: 8 },
  memory: { usage_percent: 67.2, available_gb: 8.5, total_gb: 32 },
  gpu: { gpu_memory_used: 12.4, gpu_memory_total: 24, gpu_name: 'RTX 4090' }
};

const mockAgentFleetMetrics = {
  totalAgents: 3,
  activeAgents: 1,
  errorAgents: 1,
  averagePerformance: 85.5,
  totalTasksCompleted: 479
};

describe('AgentForgeControl - Consolidated Test Suite', () => {
  let server: WS;

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();

    // Setup WebSocket mock
    server = new WS('ws://localhost:8085/ws', { jsonProtocol: true });

    // Setup default successful API responses
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/phases/status')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ phases: mockPhases })
        });
      }
      if (url.includes('/models')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ models: mockModels })
        });
      }
      if (url.includes('/system/metrics')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockSystemMetrics)
        });
      }
      if (url.includes('/agents/status')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ agents: mockAgents })
        });
      }
      if (url.includes('/agents/metrics')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockAgentFleetMetrics)
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  afterEach(() => {
    WS.clean();
    jest.restoreAllMocks();
  });

  describe('Component Rendering', () => {
    test('renders main header with gradient title', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByText('Agent Forge Training Control')).toBeInTheDocument();
      });
    });

    test('displays WebSocket connection status', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByText(/WebSocket:/)).toBeInTheDocument();
      });
    });

    test('renders phase control section when phases loaded', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByText('Phase Control')).toBeInTheDocument();
        expect(screen.getByText('Cognate')).toBeInTheDocument();
        expect(screen.getByText('EvoMerge')).toBeInTheDocument();
      });
    });

    test('renders system metrics when data available', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByText('System Resources')).toBeInTheDocument();
        expect(screen.getByText('CPU Usage')).toBeInTheDocument();
        expect(screen.getByText('45.6%')).toBeInTheDocument();
      });
    });
  });

  describe('Agent Fleet Monitoring', () => {
    test('displays agent fleet status section with summary metrics', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByText('Agent Fleet Status')).toBeInTheDocument();
        expect(screen.getByText('3')).toBeInTheDocument(); // Total agents
        expect(screen.getByText('1')).toBeInTheDocument(); // Active agents
        expect(screen.getByText('85.5%')).toBeInTheDocument(); // Avg performance
      });
    });

    test('groups agents by type correctly', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByText('Coordinator Agents')).toBeInTheDocument();
        expect(screen.getByText('Specialist Agents')).toBeInTheDocument();
        expect(screen.getByText('Worker Agents')).toBeInTheDocument();
      });
    });

    test('displays agent status icons correctly', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        // Check for emoji icons (these appear as text in tests)
        expect(screen.getByTitle('active')).toBeInTheDocument();
        expect(screen.getByTitle('error')).toBeInTheDocument();
        expect(screen.getByTitle('busy')).toBeInTheDocument();
      });
    });

    test('shows restart button only for error status agents', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        const restartButtons = screen.getAllByTitle('Restart agent');
        expect(restartButtons).toHaveLength(1); // Only error agent should have restart button
      });
    });

    test('handles agent restart action', async () => {
      mockFetch.mockImplementationOnce((url: string) => {
        if (url.includes('/agents/agent-2/restart')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
        }
        return mockFetch(url);
      });

      render(<AgentForgeControl />);

      await waitFor(() => {
        const restartButton = screen.getByTitle('Restart agent');
        fireEvent.click(restartButton);
      });

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/agents/agent-2/restart'),
          expect.objectContaining({ method: 'POST' })
        );
      });
    });

    test('handles agent scaling action', async () => {
      mockFetch.mockImplementationOnce((url: string) => {
        if (url.includes('/agents/scale')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
        }
        return mockFetch(url);
      });

      render(<AgentForgeControl />);

      await waitFor(() => {
        const scaleButton = screen.getByTitle('Add coordinator agent');
        fireEvent.click(scaleButton);
      });

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          expect.stringContaining('/agents/scale'),
          expect.objectContaining({
            method: 'POST',
            body: JSON.stringify({ agent_type: 'coordinator', count: 1 })
          })
        );
      });
    });
  });

  describe('Real-time WebSocket Updates', () => {
    test('establishes WebSocket connection on mount', async () => {
      render(<AgentForgeControl />);

      await server.connected;
      expect(server).toHaveReceivedMessages([
        { type: 'subscribe', channel: 'agent_forge_phases' },
        { type: 'subscribe', channel: 'system_metrics' },
        { type: 'subscribe', channel: 'model_updates' },
        { type: 'subscribe', channel: 'agent_updates' },
        { type: 'subscribe', channel: 'agent_health' },
        { type: 'subscribe', channel: 'performance_metrics' }
      ]);
    });

    test('handles phase update messages', async () => {
      render(<AgentForgeControl />);

      await server.connected;

      // Send phase update
      act(() => {
        server.send({
          type: 'phase_update',
          phase_name: 'Cognate',
          status: 'running',
          progress: 0.25,
          message: 'Processing data'
        });
      });

      await waitFor(() => {
        expect(screen.getByText('Processing data')).toBeInTheDocument();
      });
    });

    test('handles agent health updates', async () => {
      render(<AgentForgeControl />);

      await server.connected;

      // Send agent health update
      act(() => {
        server.send({
          type: 'agent_health',
          agents: mockAgents.map(agent => ({
            ...agent,
            performance: agent.performance + 5,
            lastActivity: new Date().toISOString()
          }))
        });
      });

      await waitFor(() => {
        expect(screen.getByText('100.5%')).toBeInTheDocument(); // Updated performance
      });
    });

    test('handles connection loss and reconnection', async () => {
      render(<AgentForgeControl />);

      await server.connected;

      // Simulate connection loss
      act(() => {
        server.close();
      });

      await waitFor(() => {
        expect(screen.getByText(/disconnected/)).toBeInTheDocument();
      });

      // Reconnection should be attempted after delay
      await waitFor(() => {
        expect(screen.getByText(/reconnecting/)).toBeInTheDocument();
      }, { timeout: 10000 });
    });
  });

  describe('Performance and Error Handling', () => {
    test('handles API failures gracefully with fallback data', async () => {
      // Mock API failure for agents
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('/agents/status')) {
          return Promise.reject(new Error('Service unavailable'));
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      });

      render(<AgentForgeControl />);

      // Should still render with mock data
      await waitFor(() => {
        expect(screen.getByText('Agent Fleet Status')).toBeInTheDocument();
      });
    });

    test('displays loading states appropriately', async () => {
      // Delay the API response
      mockFetch.mockImplementation(() =>
        new Promise(resolve => setTimeout(() => resolve({
          ok: true,
          json: () => Promise.resolve({ phases: [] })
        }), 100))
      );

      render(<AgentForgeControl />);

      // Component should render even during loading
      expect(screen.getByText('Agent Forge Training Control')).toBeInTheDocument();
    });

    test('measures component rendering performance', async () => {
      const startTime = performance.now();

      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByText('Agent Forge Training Control')).toBeInTheDocument();
      });

      const renderTime = performance.now() - startTime;
      expect(renderTime).toBeLessThan(1000); // Should render within 1 second
    });
  });

  describe('Integration Testing', () => {
    test('phase start triggers API call and updates state', async () => {
      mockFetch.mockImplementationOnce((url: string) => {
        if (url.includes('/phases/cognate/start')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
        }
        return mockFetch(url);
      });

      render(<AgentForgeControl />);

      await waitFor(() => {
        const startButton = screen.getByText('START COGNATE');
        fireEvent.click(startButton);
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/phases/cognate/start'),
        expect.objectContaining({ method: 'POST' })
      );
    });

    test('model selection enables chat interface', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        const modelSelect = screen.getByRole('combobox');
        fireEvent.change(modelSelect, { target: { value: 'model-1' } });
      });

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Chat with the model...')).toBeInTheDocument();
      });
    });

    test('chat message sending works end-to-end', async () => {
      mockFetch.mockImplementationOnce((url: string) => {
        if (url.includes('/chat')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              response: 'Hello! I am the model.',
              model_name: 'Cognate Model Alpha',
              response_time_ms: 245
            })
          });
        }
        return mockFetch(url);
      });

      render(<AgentForgeControl />);

      // Select model first
      await waitFor(() => {
        const modelSelect = screen.getByRole('combobox');
        fireEvent.change(modelSelect, { target: { value: 'model-1' } });
      });

      // Type and send message
      await waitFor(() => {
        const chatInput = screen.getByPlaceholderText('Chat with the model...');
        fireEvent.change(chatInput, { target: { value: 'Hello model!' } });

        const sendButton = screen.getByText('Send');
        fireEvent.click(sendButton);
      });

      // Check for response
      await waitFor(() => {
        expect(screen.getByText('Hello! I am the model.')).toBeInTheDocument();
        expect(screen.getByText('245ms')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility and UX', () => {
    test('has proper ARIA labels and roles', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toHaveAccessibleName(/select model/i);
        expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
      });
    });

    test('supports keyboard navigation', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        const chatInput = screen.getByPlaceholderText('Chat with the model...');
        chatInput.focus();

        fireEvent.keyPress(chatInput, { key: 'Enter', code: 'Enter' });
        // Should not crash or cause errors
      });
    });

    test('displays proper tooltips for interactive elements', async () => {
      render(<AgentForgeControl />);

      await waitFor(() => {
        expect(screen.getByTitle('Restart agent')).toBeInTheDocument();
        expect(screen.getByTitle('Add coordinator agent')).toBeInTheDocument();
      });
    });
  });
});

// Performance benchmark test
describe('AgentForgeControl - Performance Benchmarks', () => {
  test('initial render performance under load', async () => {
    const startTime = performance.now();
    const { unmount } = render(<AgentForgeControl />);
    const mountTime = performance.now() - startTime;

    expect(mountTime).toBeLessThan(100); // Should mount in under 100ms

    const unmountStart = performance.now();
    unmount();
    const unmountTime = performance.now() - unmountStart;

    expect(unmountTime).toBeLessThan(50); // Should unmount quickly
  });

  test('WebSocket message processing performance', async () => {
    const server = new WS('ws://localhost:8085/ws', { jsonProtocol: true });
    render(<AgentForgeControl />);

    await server.connected;

    const startTime = performance.now();

    // Send 100 rapid updates
    for (let i = 0; i < 100; i++) {
      act(() => {
        server.send({
          type: 'phase_update',
          phase_name: 'Cognate',
          status: 'running',
          progress: i / 100,
          message: `Processing step ${i}`
        });
      });
    }

    const processingTime = performance.now() - startTime;
    expect(processingTime).toBeLessThan(1000); // Should process 100 messages in under 1s

    WS.clean();
  });
});
