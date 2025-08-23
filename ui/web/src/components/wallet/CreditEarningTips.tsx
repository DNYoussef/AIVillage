import React from 'react';

interface EarningRate {
  current: number;
  potential: number;
  trend: 'up' | 'down' | 'stable';
}

interface MarketRates {
  creditToUSD: number;
  demandMultiplier: number;
}

interface CreditEarningTipsProps {
  currentBalance: number;
  earningRate: EarningRate;
  marketRates: MarketRates;
  onOptimizeEarnings: (strategy: string) => void;
}

export const CreditEarningTips: React.FC<CreditEarningTipsProps> = ({
  currentBalance,
  earningRate,
  marketRates,
  onOptimizeEarnings
}) => {
  return (
    <div className="credit-earning-tips">
      <div className="tips-header">
        <h3>Credit Earning Guide</h3>
        <div className="current-stats">
          <div className="stat">
            <span className="stat-label">Current Rate</span>
            <span className="stat-value">{earningRate.current} credits/hour</span>
          </div>
        </div>
      </div>

      <div className="earning-strategies">
        <h4>Earning Strategies</h4>
        <p>Multiple ways to earn credits coming soon!</p>
      </div>

      <div className="market-insights">
        <h4>Market Insights</h4>
        <div className="insights-grid">
          <div className="insight-card">
            <h5>Credit Value</h5>
            <p>${marketRates.creditToUSD.toFixed(4)} USD per credit</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreditEarningTips;
