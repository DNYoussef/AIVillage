import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import ComputeCreditsWallet from './ComputeCreditsWallet';

// Mock the wallet service
const mockWalletService = {
  balance: 1250,
  transactions: [
    { id: 'tx1', type: 'earn', amount: 50, description: 'Fog compute contribution', timestamp: new Date().toISOString() },
    { id: 'tx2', type: 'spend', amount: -25, description: 'AI model inference', timestamp: new Date().toISOString() }
  ],
  isLoading: false,
  earnCredits: jest.fn(),
  spendCredits: jest.fn()
};

jest.mock('../../hooks/useWalletService', () => ({
  useWalletService: () => mockWalletService
}));

describe('ComputeCreditsWallet', () => {
  test('renders wallet interface', () => {
    render(<ComputeCreditsWallet />);

    expect(screen.getByText('Compute Credits Wallet')).toBeInTheDocument();
    expect(screen.getByText('1,250 Credits')).toBeInTheDocument();
  });

  test('displays transaction history', () => {
    render(<ComputeCreditsWallet />);

    expect(screen.getByText('Fog compute contribution')).toBeInTheDocument();
    expect(screen.getByText('AI model inference')).toBeInTheDocument();
    expect(screen.getByText('+50')).toBeInTheDocument();
    expect(screen.getByText('-25')).toBeInTheDocument();
  });

  test('shows current balance', () => {
    render(<ComputeCreditsWallet />);

    expect(screen.getByText('Current Balance')).toBeInTheDocument();
    expect(screen.getByText('1,250 Credits')).toBeInTheDocument();
  });

  test('displays loading state', () => {
    const loadingService = { ...mockWalletService, isLoading: true };
    jest.doMock('../../hooks/useWalletService', () => ({
      useWalletService: () => loadingService
    }));

    render(<ComputeCreditsWallet />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});
