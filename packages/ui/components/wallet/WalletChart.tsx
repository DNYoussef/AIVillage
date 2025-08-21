import React, { useMemo } from 'react';
import { ComputeCredit } from '../../types';
import './WalletChart.css';

interface WalletChartProps {
  transactions: ComputeCredit[];
  timeframe: '7d' | '30d' | '90d' | '1y';
  showEarnings: boolean;
  showSpending: boolean;
}

export const WalletChart: React.FC<WalletChartProps> = ({
  transactions,
  timeframe,
  showEarnings,
  showSpending,
}) => {
  const chartData = useMemo(() => {
    const now = new Date();
    const timeframes = {
      '7d': 7,
      '30d': 30,
      '90d': 90,
      '1y': 365,
    };

    const daysBack = timeframes[timeframe];
    const startDate = new Date(now.getTime() - (daysBack * 24 * 60 * 60 * 1000));

    // Filter transactions within timeframe
    const filteredTransactions = transactions.filter(
      tx => tx.timestamp >= startDate
    );

    // Group by date
    const dailyData: Record<string, { earned: number; spent: number; date: Date }> = {};

    filteredTransactions.forEach(tx => {
      const dateKey = tx.timestamp.toDateString();
      if (!dailyData[dateKey]) {
        dailyData[dateKey] = { earned: 0, spent: 0, date: tx.timestamp };
      }

      if (tx.type === 'earned') {
        dailyData[dateKey].earned += tx.amount;
      } else if (tx.type === 'spent' || tx.type === 'transferred') {
        dailyData[dateKey].spent += Math.abs(tx.amount);
      }
    });

    // Convert to array and sort by date
    const dataPoints = Object.values(dailyData)
      .sort((a, b) => a.date.getTime() - b.date.getTime());

    // Calculate running balance
    let runningBalance = 0;
    const balanceData = dataPoints.map(point => {
      runningBalance += point.earned - point.spent;
      return {
        ...point,
        balance: runningBalance,
      };
    });

    return balanceData;
  }, [transactions, timeframe]);

  const maxValue = useMemo(() => {
    if (chartData.length === 0) return 100;

    const values = [];
    if (showEarnings) {
      values.push(...chartData.map(d => d.earned));
    }
    if (showSpending) {
      values.push(...chartData.map(d => d.spent));
    }

    return Math.max(...values, 1);
  }, [chartData, showEarnings, showSpending]);

  const formatDate = (date: Date) => {
    if (timeframe === '7d') {
      return date.toLocaleDateString([], { weekday: 'short' });
    } else if (timeframe === '30d') {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } else {
      return date.toLocaleDateString([], { month: 'short' });
    }
  };

  const getBarHeight = (value: number) => {
    return (value / maxValue) * 200; // 200px max height
  };

  if (chartData.length === 0) {
    return (
      <div className="wallet-chart no-data">
        <div className="no-data-message">
          <div className="no-data-icon">📈</div>
          <h4>No transaction data</h4>
          <p>Start earning or spending credits to see your activity chart.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="wallet-chart">
      <div className="chart-header">
        <h4>Transaction Activity ({timeframe})</h4>

        <div className="chart-legend">
          {showEarnings && (
            <div className="legend-item earnings">
              <div className="legend-color"></div>
              <span>Earnings</span>
            </div>
          )}
          {showSpending && (
            <div className="legend-item spending">
              <div className="legend-color"></div>
              <span>Spending</span>
            </div>
          )}
        </div>
      </div>

      <div className="chart-container">
        <div className="chart-y-axis">
          <span className="y-axis-label top">{maxValue.toLocaleString()}</span>
          <span className="y-axis-label middle">{(maxValue / 2).toLocaleString()}</span>
          <span className="y-axis-label bottom">0</span>
        </div>

        <div className="chart-bars">
          {chartData.map((data, index) => (
            <div key={index} className="bar-group">
              <div className="bars">
                {showEarnings && (
                  <div
                    className="bar earnings"
                    style={{ height: `${getBarHeight(data.earned)}px` }}
                    title={`Earned: ${data.earned.toLocaleString()} credits`}
                  />
                )}
                {showSpending && (
                  <div
                    className="bar spending"
                    style={{ height: `${getBarHeight(data.spent)}px` }}
                    title={`Spent: ${data.spent.toLocaleString()} credits`}
                  />
                )}
              </div>

              <div className="bar-label">
                {formatDate(data.date)}
              </div>

              <div className="bar-tooltip">
                <div className="tooltip-content">
                  <div className="tooltip-date">
                    {data.date.toLocaleDateString()}
                  </div>
                  {showEarnings && (
                    <div className="tooltip-item earnings">
                      <span>Earned:</span>
                      <span>+{data.earned.toLocaleString()}</span>
                    </div>
                  )}
                  {showSpending && (
                    <div className="tooltip-item spending">
                      <span>Spent:</span>
                      <span>-{data.spent.toLocaleString()}</span>
                    </div>
                  )}
                  <div className="tooltip-item net">
                    <span>Net:</span>
                    <span className={data.earned - data.spent >= 0 ? 'positive' : 'negative'}>
                      {data.earned - data.spent >= 0 ? '+' : ''}
                      {(data.earned - data.spent).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="chart-summary">
        <div className="summary-item">
          <span className="summary-label">Total Earned:</span>
          <span className="summary-value positive">
            +{chartData.reduce((sum, d) => sum + d.earned, 0).toLocaleString()}
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Total Spent:</span>
          <span className="summary-value negative">
            -{chartData.reduce((sum, d) => sum + d.spent, 0).toLocaleString()}
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Net Change:</span>
          <span className={`summary-value ${
            chartData.reduce((sum, d) => sum + d.earned - d.spent, 0) >= 0 ? 'positive' : 'negative'
          }`}>
            {chartData.reduce((sum, d) => sum + d.earned - d.spent, 0) >= 0 ? '+' : ''}
            {chartData.reduce((sum, d) => sum + d.earned - d.spent, 0).toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  );
};
