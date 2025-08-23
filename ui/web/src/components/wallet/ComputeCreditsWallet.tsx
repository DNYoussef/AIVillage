import React, { useState, useEffect } from 'react';
import { ComputeCredit, FogNode, WalletState } from '../../types';
import { useWalletService } from '../../hooks/useWalletService';
import { TransactionHistory } from './TransactionHistory';
import { FogContributions } from './FogContributions';
import { CreditEarningTips } from './CreditEarningTips';
import { WalletChart } from './WalletChart';
import './ComputeCreditsWallet.css';

interface ComputeCreditsWalletProps {
  userId: string;
  onTransactionComplete?: (transaction: ComputeCredit) => void;
  onFogNodeUpdate?: (node: FogNode) => void;
}

export const ComputeCreditsWallet: React.FC<ComputeCreditsWalletProps> = ({
  userId,
  onTransactionComplete,
  onFogNodeUpdate
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'transactions' | 'contributions' | 'earning'>('overview');
  const [showTransferModal, setShowTransferModal] = useState(false);
  const [transferAmount, setTransferAmount] = useState('');
  const [transferRecipient, setTransferRecipient] = useState('');

  const {
    walletState,
    transferCredits,
    contributeToFog,
    withdrawFromFog,
    refreshBalance,
    earningRate,
    fogStats,
    marketRates,
    error,
    isLoading
  } = useWalletService(userId);

  useEffect(() => {
    // Refresh balance every 30 seconds
    const interval = setInterval(() => {
      refreshBalance();
    }, 30000);

    return () => clearInterval(interval);
  }, [refreshBalance]);

  const handleTransferCredits = async () => {
    if (!transferAmount || !transferRecipient) return;

    const amount = parseFloat(transferAmount);
    if (isNaN(amount) || amount <= 0 || amount > walletState.balance) return;

    const success = await transferCredits(transferRecipient, amount);
    if (success) {
      setShowTransferModal(false);
      setTransferAmount('');
      setTransferRecipient('');

      if (onTransactionComplete) {
        const transaction: ComputeCredit = {
          id: Date.now().toString(),
          userId,
          amount: -amount,
          type: 'transferred',
          description: `Transfer to ${transferRecipient}`,
          timestamp: new Date()
        };
        onTransactionComplete(transaction);
      }
    }
  };

  const handleFogContribution = async (nodeId: string, resourceAmount: number) => {
    const success = await contributeToFog(nodeId, resourceAmount);
    if (success && onFogNodeUpdate) {
      const updatedNode = walletState.fogContributions.find(n => n.id === nodeId);
      if (updatedNode) {
        onFogNodeUpdate(updatedNode);
      }
    }
  };

  const renderOverviewTab = () => (
    <div className="wallet-overview">
      <div className="balance-card">
        <div className="balance-header">
          <h3>Total Balance</h3>
          <button
            onClick={refreshBalance}
            className="refresh-btn"
            disabled={isLoading}
            aria-label="Refresh balance"
          >
            🔄
          </button>
        </div>
        <div className="balance-amount">
          <span className="balance-number">{walletState.balance.toLocaleString()}</span>
          <span className="balance-unit">Credits</span>
        </div>
        <div className="balance-usd">
          ≈ ${(walletState.balance * marketRates.creditToUSD).toFixed(2)} USD
        </div>
      </div>

      <div className="earning-stats">
        <div className="stat-item">
          <span className="stat-label">Earning Rate</span>
          <span className="stat-value">{earningRate.current} credits/hour</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Active Contributions</span>
          <span className="stat-value">{walletState.fogContributions.length} nodes</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total Earned</span>
          <span className="stat-value">
            {walletState.transactions
              .filter(t => t.type === 'earned')
              .reduce((sum, t) => sum + t.amount, 0)
              .toLocaleString()}
          </span>
        </div>
      </div>

      <WalletChart
        transactions={walletState.transactions}
        timeframe="30d"
        showEarnings={true}
        showSpending={true}
      />

      <div className="quick-actions">
        <button
          onClick={() => setShowTransferModal(true)}
          className="action-btn primary"
          disabled={walletState.balance <= 0}
        >
          💸 Transfer Credits
        </button>
        <button
          onClick={() => setActiveTab('contributions')}
          className="action-btn secondary"
        >
          ☁️ Contribute to Fog
        </button>
        <button
          onClick={() => setActiveTab('earning')}
          className="action-btn secondary"
        >
          📈 Earning Tips
        </button>
      </div>

      <div className="fog-network-status">
        <h4>Fog Network Status</h4>
        <div className="network-stats">
          <div className="network-stat">
            <span className="stat-icon">🌐</span>
            <div>
              <div className="stat-title">Total Nodes</div>
              <div className="stat-number">{fogStats.totalNodes}</div>
            </div>
          </div>
          <div className="network-stat">
            <span className="stat-icon">⚡</span>
            <div>
              <div className="stat-title">Network Load</div>
              <div className="stat-number">{fogStats.networkLoad}%</div>
            </div>
          </div>
          <div className="network-stat">
            <span className="stat-icon">🎯</span>
            <div>
              <div className="stat-title">Avg. Latency</div>
              <div className="stat-number">{fogStats.averageLatency}ms</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  if (error) {
    return (
      <div className="wallet-error">
        <div className="error-icon">⚠️</div>
        <h3>Wallet Error</h3>
        <p>{error}</p>
        <button onClick={refreshBalance} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="compute-credits-wallet">
      <div className="wallet-header">
        <div className="wallet-title">
          <h2>Compute Credits Wallet</h2>
          <div className="wallet-status">
            <span className={`status-indicator ${isLoading ? 'loading' : 'active'}`}>
              {isLoading ? '🔄' : '✅'}
            </span>
            <span>
              {isLoading ? 'Syncing...' : 'Up to date'}
            </span>
          </div>
        </div>

        <div className="wallet-tabs">
          <button
            onClick={() => setActiveTab('overview')}
            className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          >
            Overview
          </button>
          <button
            onClick={() => setActiveTab('transactions')}
            className={`tab-btn ${activeTab === 'transactions' ? 'active' : ''}`}
          >
            Transactions
          </button>
          <button
            onClick={() => setActiveTab('contributions')}
            className={`tab-btn ${activeTab === 'contributions' ? 'active' : ''}`}
          >
            Fog Contributions
          </button>
          <button
            onClick={() => setActiveTab('earning')}
            className={`tab-btn ${activeTab === 'earning' ? 'active' : ''}`}
          >
            Earning Guide
          </button>
        </div>
      </div>

      <div className="wallet-content">
        {activeTab === 'overview' && renderOverviewTab()}

        {activeTab === 'transactions' && (
          <TransactionHistory
            transactions={walletState.transactions}
            onTransactionClick={(tx) => console.log('Transaction details:', tx)}
            isLoading={isLoading}
          />
        )}

        {activeTab === 'contributions' && (
          <FogContributions
            contributions={walletState.fogContributions}
            onContribute={handleFogContribution}
            onWithdraw={withdrawFromFog}
            earningRate={earningRate}
            networkStats={fogStats}
          />
        )}

        {activeTab === 'earning' && (
          <CreditEarningTips
            currentBalance={walletState.balance}
            earningRate={earningRate}
            marketRates={marketRates}
            onOptimizeEarnings={(strategy) => {
              console.log('Optimize earnings:', strategy);
            }}
          />
        )}
      </div>

      {showTransferModal && (
        <div className="transfer-modal-overlay">
          <div className="transfer-modal">
            <div className="modal-header">
              <h3>Transfer Credits</h3>
              <button
                onClick={() => setShowTransferModal(false)}
                className="close-btn"
                aria-label="Close"
              >
                ×
              </button>
            </div>

            <div className="modal-content">
              <div className="transfer-form">
                <div className="form-group">
                  <label htmlFor="recipient">Recipient ID</label>
                  <input
                    id="recipient"
                    type="text"
                    value={transferRecipient}
                    onChange={(e) => setTransferRecipient(e.target.value)}
                    placeholder="Enter user ID or wallet address"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="amount">Amount</label>
                  <input
                    id="amount"
                    type="number"
                    value={transferAmount}
                    onChange={(e) => setTransferAmount(e.target.value)}
                    placeholder="0.00"
                    min="0"
                    max={walletState.balance}
                    step="0.01"
                  />
                  <div className="amount-helper">
                    Available: {walletState.balance.toLocaleString()} credits
                  </div>
                </div>
              </div>
            </div>

            <div className="modal-actions">
              <button
                onClick={() => setShowTransferModal(false)}
                className="cancel-btn"
              >
                Cancel
              </button>
              <button
                onClick={handleTransferCredits}
                className="transfer-btn"
                disabled={!transferAmount || !transferRecipient || isLoading}
              >
                {isLoading ? 'Processing...' : 'Transfer'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ComputeCreditsWallet;
