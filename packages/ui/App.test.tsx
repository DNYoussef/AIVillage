import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

// Mock the services
jest.mock('./hooks/useSystemService', () => ({
  useSystemService: () => ({
    systemState: {
      agents: [
        { id: '1', name: 'Agent 1', status: 'active', performance: 95 },
        { id: '2', name: 'Agent 2', status: 'idle', performance: 80 }
      ],
      fogNodes: [
        { id: 'node1', name: 'Node 1', location: 'US-East', status: 'active', resources: { cpu: 50, memory: 60, storage: 70, bandwidth: 80 }, reputation: 95 }
      ],
      networkHealth: {
        p2pConnections: 25,
        messageLatency: 120,
        nodeCount: 8
      },
      systemMetrics: {
        cpuUsage: 45,
        memoryUsage: 67,
        networkTraffic: 35
      }
    },
    agentCommands: [],
    fogCommands: [],
    systemAlerts: {
      critical: [],
      warnings: []
    },
    isConnected: true,
    lastUpdate: new Date(),
    error: null,
    executeCommand: jest.fn(),
    refreshSystem: jest.fn()
  })
}));

jest.mock('./hooks/useConciergeService', () => ({
  useConciergeService: () => ({
    chatState: {
      messages: [],
      isTyping: false,
      isConnected: true,
      activeTwin: {
        id: 'twin-1',
        userId: 'demo-user',
        name: 'Assistant',
        specialization: ['general'],
        conversationHistory: [],
        isActive: true
      }
    },
    sendMessage: jest.fn(),
    clearMessages: jest.fn(),
    setTwinPersonality: jest.fn(),
    error: null,
    isLoading: false
  })
}));

jest.mock('./hooks/useBitChatService', () => ({
  useBitChatService: () => ({
    isConnected: true,
    peers: [],
    messages: [],
    sendMessage: jest.fn(),
    connectToPeer: jest.fn()
  })
}));

jest.mock('./hooks/useWalletService', () => ({
  useWalletService: () => ({
    balance: 1250,
    transactions: [],
    isLoading: false,
    earnCredits: jest.fn(),
    spendCredits: jest.fn()
  })
}));

describe('App Component', () => {
  test('renders without crashing', () => {
    render(<App />);
    expect(screen.getByText('AIVillage')).toBeInTheDocument();
  });

  test('displays navigation tabs', () => {
    render(<App />);

    expect(screen.getByText('🤖 Concierge')).toBeInTheDocument();
    expect(screen.getByText('💬 Messaging')).toBeInTheDocument();
    expect(screen.getByText('💰 Wallet')).toBeInTheDocument();
    expect(screen.getByText('📊 Dashboard')).toBeInTheDocument();
  });

  test('switches between tabs', () => {
    render(<App />);

    // Concierge should be active by default based on App.tsx
    expect(screen.getByText('Digital Twin Chat')).toBeInTheDocument();

    // Click on Messaging tab
    fireEvent.click(screen.getByText('💬 Messaging'));
    expect(screen.getByText('BitChat Mesh Network')).toBeInTheDocument();

    // Click on Wallet tab
    fireEvent.click(screen.getByText('💰 Wallet'));
    expect(screen.getByText('Compute Credits Wallet')).toBeInTheDocument();

    // Click on Dashboard tab
    fireEvent.click(screen.getByText('📊 Dashboard'));
    expect(screen.getByText('AIVillage System Control')).toBeInTheDocument();
  });

  test('dashboard tab shows system metrics', () => {
    render(<App />);

    // Switch to dashboard first
    fireEvent.click(screen.getByText('📊 Dashboard'));

    // Should show system control dashboard
    expect(screen.getByText('AIVillage System Control')).toBeInTheDocument();
  });

  test('concierge tab shows chat interface', () => {
    render(<App />);

    // Concierge is default, or click it
    fireEvent.click(screen.getByText('🤖 Concierge'));

    // Should show chat interface
    expect(screen.getByText('Digital Twin Chat')).toBeInTheDocument();
  });

  test('messaging tab shows peer-to-peer interface', () => {
    render(<App />);

    fireEvent.click(screen.getByText('💬 Messaging'));

    // Should show P2P messaging interface
    expect(screen.getByText('BitChat Mesh Network')).toBeInTheDocument();
    expect(screen.getByText('Network Status')).toBeInTheDocument();
    expect(screen.getByText('Connected Peers')).toBeInTheDocument();
  });

  test('wallet tab shows credit information', () => {
    render(<App />);

    fireEvent.click(screen.getByText('💰 Wallet'));

    // Should show wallet interface
    expect(screen.getByText('Compute Credits Wallet')).toBeInTheDocument();
  });
});
