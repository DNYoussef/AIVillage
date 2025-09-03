import React from 'react';
import { ComputeCredit } from '../../types';

interface WalletChartProps {
  transactions: ComputeCredit[];
  timeframe: '7d' | '30d' | '90d';
  showEarnings: boolean;
  showSpending: boolean;
}

export const WalletChart: React.FC<WalletChartProps> = ({
  transactions,
  timeframe,
  showEarnings,
  showSpending
}) => {
  // Simple chart implementation
  const chartWidth = 400;
  const chartHeight = 200;

  return (
    <div className="wallet-chart">
      <div className="chart-header">
        <h4>Credit Flow Chart</h4>
      </div>

      <div className="chart-container">
        <svg width={chartWidth} height={chartHeight} className="credit-chart">
          <rect width="100%" height="100%" fill="#f3f4f6" />
          <text x={chartWidth/2} y={chartHeight/2} textAnchor="middle" dominantBaseline="central">
            No chart data available
          </text>
        </svg>
      </div>

      <div className="chart-stats">
        <div className="stat-item">
          <span className="stat-label">Total Transactions:</span>
          <span className="stat-value">{transactions.length}</span>
        </div>
      </div>
    </div>
  );
};

export default WalletChart;
