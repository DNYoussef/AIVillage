import React, { useState } from 'react';
import { ComputeCredit } from '../../types';
import './TransactionHistory.css';

interface TransactionHistoryProps {
  transactions: ComputeCredit[];
  onTransactionClick: (transaction: ComputeCredit) => void;
  isLoading: boolean;
}

export const TransactionHistory: React.FC<TransactionHistoryProps> = ({
  transactions,
  onTransactionClick,
  isLoading,
}) => {
  const [filter, setFilter] = useState<'all' | 'earned' | 'spent' | 'transferred'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'amount'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const filteredTransactions = transactions.filter(tx =>
    filter === 'all' || tx.type === filter
  );

  const sortedTransactions = [...filteredTransactions].sort((a, b) => {
    let comparison = 0;

    if (sortBy === 'date') {
      comparison = a.timestamp.getTime() - b.timestamp.getTime();
    } else if (sortBy === 'amount') {
      comparison = Math.abs(a.amount) - Math.abs(b.amount);
    }

    return sortOrder === 'asc' ? comparison : -comparison;
  });

  const getTransactionIcon = (type: string) => {
    switch (type) {
      case 'earned': return '🟢';
      case 'spent': return '🔴';
      case 'transferred': return '🟡';
      default: return '⚪';
    }
  };

  const getTransactionColor = (type: string) => {
    switch (type) {
      case 'earned': return '#10b981';
      case 'spent': return '#ef4444';
      case 'transferred': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  if (isLoading) {
    return (
      <div className="transaction-history loading">
        <div className="loading-spinner"></div>
        <p>Loading transactions...</p>
      </div>
    );
  }

  return (
    <div className="transaction-history">
      <div className="history-controls">
        <div className="filter-controls">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="filter-select"
          >
            <option value="all">All Transactions</option>
            <option value="earned">Earned</option>
            <option value="spent">Spent</option>
            <option value="transferred">Transferred</option>
          </select>
        </div>

        <div className="sort-controls">
          <select
            value={`${sortBy}-${sortOrder}`}
            onChange={(e) => {
              const [field, order] = e.target.value.split('-');
              setSortBy(field as 'date' | 'amount');
              setSortOrder(order as 'asc' | 'desc');
            }}
            className="sort-select"
          >
            <option value="date-desc">Newest First</option>
            <option value="date-asc">Oldest First</option>
            <option value="amount-desc">Highest Amount</option>
            <option value="amount-asc">Lowest Amount</option>
          </select>
        </div>
      </div>

      {sortedTransactions.length === 0 ? (
        <div className="no-transactions">
          <div className="no-transactions-icon">💳</div>
          <h3>No transactions found</h3>
          <p>No transactions match the selected filter.</p>
        </div>
      ) : (
        <div className="transactions-list">
          {sortedTransactions.map((transaction) => (
            <div
              key={transaction.id}
              className={`transaction-item ${transaction.type}`}
              onClick={() => onTransactionClick(transaction)}
            >
              <div className="transaction-icon">
                {getTransactionIcon(transaction.type)}
              </div>

              <div className="transaction-details">
                <div className="transaction-description">
                  {transaction.description}
                </div>
                <div className="transaction-meta">
                  <span className="transaction-date">
                    {transaction.timestamp.toLocaleDateString()}
                  </span>
                  <span className="transaction-time">
                    {transaction.timestamp.toLocaleTimeString()}
                  </span>
                  {transaction.relatedTaskId && (
                    <span className="transaction-task">
                      Task: {transaction.relatedTaskId}
                    </span>
                  )}
                </div>
              </div>

              <div className="transaction-amount">
                <span
                  className={`amount ${transaction.amount >= 0 ? 'positive' : 'negative'}`}
                  style={{ color: getTransactionColor(transaction.type) }}
                >
                  {transaction.amount >= 0 ? '+' : ''}{transaction.amount.toLocaleString()}
                </span>
                <span className="amount-unit">credits</span>
              </div>

              <div className="transaction-actions">
                <button
                  className="details-btn"
                  title="View details"
                >
                  🔍
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="history-summary">
        <div className="summary-item">
          <span className="summary-label">Total Earned:</span>
          <span className="summary-value positive">
            +{transactions
              .filter(t => t.type === 'earned')
              .reduce((sum, t) => sum + t.amount, 0)
              .toLocaleString()} credits
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Total Spent:</span>
          <span className="summary-value negative">
            {transactions
              .filter(t => t.type === 'spent')
              .reduce((sum, t) => sum + t.amount, 0)
              .toLocaleString()} credits
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Net Transfer:</span>
          <span className="summary-value">
            {transactions
              .filter(t => t.type === 'transferred')
              .reduce((sum, t) => sum + t.amount, 0)
              .toLocaleString()} credits
          </span>
        </div>
      </div>
    </div>
  );
};
