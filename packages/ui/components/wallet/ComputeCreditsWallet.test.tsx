import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ComputeCreditsWallet } from './ComputeCreditsWallet';

// Mock the wallet service
const mockWalletService = {
  walletState: {
    balance: 1250,
    transactions: [
      { id: 'tx1', userId: 'test', type: 'earned', amount: 50, description: 'Fog compute contribution', timestamp: new Date() },
      { id: 'tx2', userId: 'test', type: 'spent', amount: -25, description: 'AI model inference', timestamp: new Date() }
    ],
    fogContributions: [],
    isLoading: false
  },
  transferCredits: jest.fn(),
  contributeToFog: jest.fn(),
  withdrawFromFog: jest.fn(),
  refreshBalance: jest.fn(),
  earningRate: {
    current: 5.2,
    potential: 15.8,
    trend: 'up'
  },
  fogStats: {
    totalNodes: 142,
    networkLoad: 76,
    averageLatency: 45
  },
  marketRates: {
    creditToUSD: 0.0085,
    demandMultiplier: 1.3
  },
  error: null,
  isLoading: false
};

jest.mock('../../hooks/useWalletService', () => ({
  useWalletService: () => mockWalletService
}));

describe('ComputeCreditsWallet', () => {
  const defaultProps = {
    userId: 'test-user',
    onTransactionComplete: jest.fn(),
    onFogNodeUpdate: jest.fn()
  };

  test('renders wallet interface', () => {
    render(<ComputeCreditsWallet {...defaultProps} />);

    expect(screen.getByText('Compute Credits Wallet')).toBeInTheDocument();
    expect(screen.getByText('1,250')).toBeInTheDocument();
  });

  test('displays transaction history when transactions tab clicked', () => {
    render(<ComputeCreditsWallet {...defaultProps} />);

    // Click transactions tab first
    fireEvent.click(screen.getByText('Transactions'));
    
    expect(screen.getByText('Fog compute contribution')).toBeInTheDocument();
    expect(screen.getByText('AI model inference')).toBeInTheDocument();
  });

  test('shows current balance', () => {
    render(<ComputeCreditsWallet {...defaultProps} />);

    expect(screen.getByText('Total Balance')).toBeInTheDocument();
    expect(screen.getByText('1,250')).toBeInTheDocument();
  });

  test('displays loading state', () => {
    const loadingService = { 
      ...mockWalletService, 
      isLoading: true,
      walletState: {
        ...mockWalletService.walletState,
        isLoading: true
      }
    };
    
    // Temporarily override the mock
    const originalMock = require('../../hooks/useWalletService').useWalletService;
    require('../../hooks/useWalletService').useWalletService = jest.fn(() => loadingService);

    render(<ComputeCreditsWallet {...defaultProps} />);

    expect(screen.getByText('Syncing...')).toBeInTheDocument();
    
    // Restore original mock
    require('../../hooks/useWalletService').useWalletService = originalMock;
  });
});
