import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

// Mock the services
jest.mock('./hooks/useSystemService', () => ({
  useSystemService: () => ({
    systemMetrics: {
      cpuUsage: 45,
      memoryUsage: 67,
      networkLatency: 120,
      activeAgents: 8
    },
    isConnected: true,
    lastUpdate: new Date().toISOString()
  })
}));

jest.mock('./hooks/useConciergeService', () => ({
  useConciergeService: () => ({
    isConnected: true,
    messages: [],
    sendMessage: jest.fn(),
    clearMessages: jest.fn()
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

    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Concierge')).toBeInTheDocument();
    expect(screen.getByText('Messaging')).toBeInTheDocument();
    expect(screen.getByText('Wallet')).toBeInTheDocument();
  });

  test('switches between tabs', () => {
    render(<App />);

    // Dashboard should be active by default
    expect(screen.getByText('System Control Dashboard')).toBeInTheDocument();

    // Click on Concierge tab
    fireEvent.click(screen.getByText('Concierge'));
    expect(screen.getByText('Digital Twin Concierge')).toBeInTheDocument();

    // Click on Messaging tab
    fireEvent.click(screen.getByText('Messaging'));
    expect(screen.getByText('BitChat P2P Messaging')).toBeInTheDocument();

    // Click on Wallet tab
    fireEvent.click(screen.getByText('Wallet'));
    expect(screen.getByText('Compute Credits Wallet')).toBeInTheDocument();
  });

  test('dashboard tab shows system metrics', () => {
    render(<App />);

    // Should show system metrics
    expect(screen.getByText('System Metrics')).toBeInTheDocument();
    expect(screen.getByText('Agent Status')).toBeInTheDocument();
    expect(screen.getByText('Network Health')).toBeInTheDocument();
  });

  test('concierge tab shows chat interface', () => {
    render(<App />);

    fireEvent.click(screen.getByText('Concierge'));

    // Should show chat interface
    expect(screen.getByText('Digital Twin Concierge')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Ask your digital twin anything...')).toBeInTheDocument();
  });

  test('messaging tab shows peer-to-peer interface', () => {
    render(<App />);

    fireEvent.click(screen.getByText('Messaging'));

    // Should show P2P messaging interface
    expect(screen.getByText('BitChat P2P Messaging')).toBeInTheDocument();
    expect(screen.getByText('Network Status')).toBeInTheDocument();
    expect(screen.getByText('Connected Peers')).toBeInTheDocument();
  });

  test('wallet tab shows credit information', () => {
    render(<App />);

    fireEvent.click(screen.getByText('Wallet'));

    // Should show wallet interface
    expect(screen.getByText('Compute Credits Wallet')).toBeInTheDocument();
    expect(screen.getByText('Credit Earning Tips')).toBeInTheDocument();
    expect(screen.getByText('Transaction History')).toBeInTheDocument();
  });
});
