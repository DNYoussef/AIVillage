import React from 'react';
import { ComputeCredit } from '../../types';

interface TransactionHistoryProps {
  transactions: ComputeCredit[];
  onTransactionClick: (transaction: ComputeCredit) => void;
  isLoading: boolean;
}

export const TransactionHistory: React.FC<TransactionHistoryProps> = ({
  transactions,
  onTransactionClick,
  isLoading
}) => {
  if (isLoading) {
    return (
      <div className="transaction-history loading">
        <div className="loading-spinner">Loading transactions...</div>
      </div>
    );
  }

  if (transactions.length === 0) {
    return (
      <div className="transaction-history empty">
        <div className="empty-state">
          <span className="empty-icon">📋</span>
          <p>No transactions yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="transaction-history">
      <div className="history-header">
        <h3>Transaction History</h3>
        <span className="transaction-count">{transactions.length} transactions</span>
      </div>

      <div className="transactions-list">
        {transactions.map(transaction => (
          <div
            key={transaction.id}
            className={`transaction-item ${transaction.type}`}
            onClick={() => onTransactionClick(transaction)}
          >
            <div className="transaction-icon">
              {transaction.type === 'earned' ? '💰' :
               transaction.type === 'spent' ? '💸' : '🔄'}
            </div>
            <div className="transaction-details">
              <div className="transaction-description">
                {transaction.description}
              </div>
              <div className="transaction-timestamp">
                {transaction.timestamp.toLocaleDateString()} {transaction.timestamp.toLocaleTimeString()}
              </div>
            </div>
            <div className={`transaction-amount ${transaction.type}`}>
              {transaction.type === 'spent' ? '-' : '+'}
              {Math.abs(transaction.amount).toLocaleString()} credits
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TransactionHistory;
