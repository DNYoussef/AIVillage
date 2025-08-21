import React, { useState } from 'react';
import './CreditEarningTips.css';

interface CreditEarningTipsProps {
  currentBalance: number;
  earningRate: {
    current: number;
    projected: number;
  };
  marketRates: {
    creditToUSD: number;
    usdToCredit: number;
  };
  onOptimizeEarnings: (strategy: string) => void;
}

export const CreditEarningTips: React.FC<CreditEarningTipsProps> = ({
  currentBalance,
  earningRate,
  marketRates,
  onOptimizeEarnings,
}) => {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);

  const strategies = [
    {
      id: 'fog_contribution',
      title: 'Contribute to Fog Network',
      description: 'Share your device resources (CPU, storage, bandwidth) with the fog network',
      potential: '+25-50 credits/hour',
      difficulty: 'Easy',
      icon: '☁️',
      requirements: ['Stable internet', 'Available device resources'],
      tips: [
        'Keep your device online for maximum earnings',
        'Optimize resource allocation based on demand',
        'Maintain high reputation score for better rates',
      ],
    },
    {
      id: 'p2p_relay',
      title: 'P2P Message Relay',
      description: 'Help relay messages in the peer-to-peer network',
      potential: '+10-20 credits/hour',
      difficulty: 'Easy',
      icon: '🔄',
      requirements: ['Good network connection', 'Active participation'],
      tips: [
        'Enable message relay in your settings',
        'Maintain good connectivity to peers',
        'Process messages efficiently',
      ],
    },
    {
      id: 'ai_training',
      title: 'AI Model Training',
      description: 'Contribute computational power for training AI models',
      potential: '+50-100 credits/hour',
      difficulty: 'Medium',
      icon: '🤖',
      requirements: ['High-performance GPU', 'Significant power consumption'],
      tips: [
        'Use during off-peak hours for better rates',
        'Ensure adequate cooling and power',
        'Monitor performance metrics',
      ],
    },
    {
      id: 'data_validation',
      title: 'Data Validation',
      description: 'Validate and verify data integrity in the network',
      potential: '+15-30 credits/hour',
      difficulty: 'Easy',
      icon: '✅',
      requirements: ['Good reputation', 'Attention to detail'],
      tips: [
        'Maintain high accuracy in validations',
        'Participate in consensus mechanisms',
        'Build trust with other validators',
      ],
    },
    {
      id: 'storage_hosting',
      title: 'Distributed Storage',
      description: 'Provide storage space for the distributed file system',
      potential: '+20-40 credits/hour',
      difficulty: 'Medium',
      icon: '🗎',
      requirements: ['Available storage space', 'Reliable uptime'],
      tips: [
        'Ensure data redundancy and backup',
        'Optimize storage efficiency',
        'Monitor disk health regularly',
      ],
    },
  ];

  const calculatePotentialEarnings = (strategy: any) => {
    const baseRate = parseFloat(strategy.potential.match(/\d+/)?.[0] || '0');
    const hours24 = baseRate * 24;
    const monthly = hours24 * 30;
    return { daily: hours24, monthly };
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'Easy': return '#10b981';
      case 'Medium': return '#f59e0b';
      case 'Hard': return '#ef4444';
      default: return '#6b7280';
    }
  };

  return (
    <div className="credit-earning-tips">
      <div className="earnings-overview">
        <h3>Earning Optimization</h3>

        <div className="current-stats">
          <div className="stat-card">
            <div className="stat-value">{currentBalance.toLocaleString()}</div>
            <div className="stat-label">Current Balance</div>
            <div className="stat-usd">
              ≈ ${(currentBalance * marketRates.creditToUSD).toFixed(2)} USD
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-value">{earningRate.current}</div>
            <div className="stat-label">Current Rate/Hour</div>
            <div className="stat-projection">
              Projected: {earningRate.projected.toFixed(1)}/hour
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-value">
              {(earningRate.current * 24 * 30).toLocaleString()}
            </div>
            <div className="stat-label">Monthly Potential</div>
            <div className="stat-usd">
              ≈ ${(earningRate.current * 24 * 30 * marketRates.creditToUSD).toFixed(2)} USD
            </div>
          </div>
        </div>
      </div>

      <div className="earning-strategies">
        <h3>Earning Strategies</h3>

        <div className="strategies-grid">
          {strategies.map((strategy) => {
            const earnings = calculatePotentialEarnings(strategy);
            const isSelected = selectedStrategy === strategy.id;

            return (
              <div
                key={strategy.id}
                className={`strategy-card ${isSelected ? 'selected' : ''}`}
                onClick={() => setSelectedStrategy(isSelected ? null : strategy.id)}
              >
                <div className="strategy-header">
                  <div className="strategy-icon">{strategy.icon}</div>
                  <div className="strategy-info">
                    <h4>{strategy.title}</h4>
                    <span
                      className="difficulty-badge"
                      style={{ backgroundColor: getDifficultyColor(strategy.difficulty) }}
                    >
                      {strategy.difficulty}
                    </span>
                  </div>
                  <div className="potential-earnings">
                    <div className="earnings-amount">{strategy.potential}</div>
                    <div className="earnings-period">per hour</div>
                  </div>
                </div>

                <p className="strategy-description">{strategy.description}</p>

                <div className="earnings-projection">
                  <div className="projection-item">
                    <span className="projection-label">Daily:</span>
                    <span className="projection-value">~{earnings.daily} credits</span>
                  </div>
                  <div className="projection-item">
                    <span className="projection-label">Monthly:</span>
                    <span className="projection-value">~{earnings.monthly.toLocaleString()} credits</span>
                  </div>
                </div>

                {isSelected && (
                  <div className="strategy-details">
                    <div className="requirements">
                      <h5>Requirements:</h5>
                      <ul>
                        {strategy.requirements.map((req, index) => (
                          <li key={index}>{req}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="tips">
                      <h5>Tips for Success:</h5>
                      <ul>
                        {strategy.tips.map((tip, index) => (
                          <li key={index}>{tip}</li>
                        ))}
                      </ul>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onOptimizeEarnings(strategy.id);
                      }}
                      className="optimize-btn"
                    >
                      Start Earning
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="market-insights">
        <h3>Market Insights</h3>

        <div className="insights-grid">
          <div className="insight-card">
            <div className="insight-icon">📈</div>
            <div className="insight-content">
              <h4>Credit Value</h4>
              <p>1 Credit = ${marketRates.creditToUSD.toFixed(4)} USD</p>
              <p>Current rate is stable with slight upward trend</p>
            </div>
          </div>

          <div className="insight-card">
            <div className="insight-icon">⬆️</div>
            <div className="insight-content">
              <h4>Demand Peak Hours</h4>
              <p>6PM - 11PM (Local Time)</p>
              <p>Earnings increase by 20-40% during peak demand</p>
            </div>
          </div>

          <div className="insight-card">
            <div className="insight-icon">🌐</div>
            <div className="insight-content">
              <h4>Network Growth</h4>
              <p>+15% new nodes this month</p>
              <p>Higher network participation = more earning opportunities</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
