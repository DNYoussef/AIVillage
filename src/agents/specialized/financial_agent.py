"""Financial Agent - Economic Modeling and Financial Analysis Specialist"""

from dataclasses import dataclass
import logging
import math
from typing import Any

from src.production.rag.rag_system.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


@dataclass
class FinancialAnalysisRequest:
    """Request for financial analysis"""

    analysis_type: str  # 'portfolio', 'risk', 'valuation', 'forecast'
    data_source: str
    parameters: dict[str, Any]
    time_horizon: str = "1Y"


class FinancialAgent(AgentInterface):
    """Specialized agent for financial analysis including:
    - Portfolio optimization and risk analysis
    - Financial modeling and forecasting
    - Market sentiment analysis
    - Economic indicator analysis
    - Trading strategy backtesting
    """

    def __init__(self, agent_id: str = "financial_agent"):
        self.agent_id = agent_id
        self.agent_type = "Financial"
        self.capabilities = [
            "portfolio_optimization",
            "risk_analysis",
            "financial_modeling",
            "market_forecasting",
            "sentiment_analysis",
            "trading_strategies",
            "economic_indicators",
            "regulatory_compliance",
        ]
        self.portfolios = {}
        self.market_data = {}
        self.analysis_history = []
        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate financial analysis responses"""
        if "portfolio" in prompt.lower():
            return "I can optimize portfolios using modern portfolio theory and risk metrics like VaR and Sharpe ratio."
        if "risk" in prompt.lower():
            return "I analyze financial risk using VaR, CVaR, beta analysis, and stress testing scenarios."
        if "forecast" in prompt.lower():
            return "I provide financial forecasts using time series analysis, Monte Carlo simulations, and economic models."
        if "valuation" in prompt.lower():
            return "I perform asset valuation using DCF, comparables, and options pricing models."
        return "I'm a Financial Agent specialized in quantitative finance, risk management, and economic analysis."

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for financial text"""
        import hashlib

        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(hash_value % 1000) / 1000.0] * 384

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank results based on financial relevance"""
        keywords = [
            "portfolio",
            "risk",
            "valuation",
            "forecast",
            "trading",
            "market",
            "finance",
            "economic",
        ]

        for result in results:
            score = 0
            text = str(result.get("content", ""))
            for keyword in keywords:
                score += text.lower().count(keyword)
            result["financial_relevance_score"] = score

        return sorted(results, key=lambda x: x.get("financial_relevance_score", 0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return agent capabilities and status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "portfolios_managed": len(self.portfolios),
            "market_data_points": len(self.market_data),
            "analyses_performed": len(self.analysis_history),
            "initialized": self.initialized,
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate with other agents"""
        response = await recipient.generate(f"Financial Agent says: {message}")
        return f"Received response: {response}"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate latent space for financial analysis"""
        analysis_type = "risk" if "risk" in query.lower() else "portfolio"
        latent_representation = f"FINANCE[{analysis_type}:{query[:50]}]"
        return analysis_type, latent_representation

    async def optimize_portfolio(self, assets: list[str], constraints: dict[str, Any]) -> dict[str, Any]:
        """Optimize portfolio allocation using modern portfolio theory"""
        try:
            # Simulate expected returns and covariance matrix
            n_assets = len(assets)
            expected_returns = [0.08 + 0.05 * (i / n_assets) for i in range(n_assets)]

            # Simple covariance matrix simulation
            covariance_matrix = []
            for i in range(n_assets):
                row = []
                for j in range(n_assets):
                    if i == j:
                        row.append(0.04)  # Variance
                    else:
                        row.append(0.01)  # Covariance
                covariance_matrix.append(row)

            # Simplified equal-weight optimization (in practice, use scipy.optimize)
            weights = [1.0 / n_assets] * n_assets

            # Calculate portfolio metrics
            portfolio_return = sum(w * r for w, r in zip(weights, expected_returns, strict=False))
            portfolio_variance = sum(
                weights[i] * weights[j] * covariance_matrix[i][j] for i in range(n_assets) for j in range(n_assets)
            )
            portfolio_volatility = math.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

            result = {
                "assets": assets,
                "weights": dict(zip(assets, weights, strict=False)),
                "expected_return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "var_95": -1.645 * portfolio_volatility,  # Value at Risk
                "max_drawdown": 0.15,
            }

            return result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {"error": str(e)}

    async def calculate_risk_metrics(self, portfolio_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        try:
            # Simulate portfolio returns
            returns = [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.04, -0.01, 0.02]
            portfolio_value = portfolio_data.get("value", 1000000)

            # Calculate VaR (Value at Risk)
            returns_sorted = sorted(returns)
            var_95 = returns_sorted[int(0.05 * len(returns))] * portfolio_value
            var_99 = returns_sorted[int(0.01 * len(returns))] * portfolio_value

            # Calculate CVaR (Conditional Value at Risk)
            cvar_95 = (
                sum(r for r in returns_sorted if r <= returns_sorted[int(0.05 * len(returns))])
                / int(0.05 * len(returns))
                * portfolio_value
            )

            # Calculate other metrics
            mean_return = sum(returns) / len(returns)
            volatility = math.sqrt(sum((r - mean_return) ** 2 for r in returns) / len(returns))

            # Beta calculation (simplified)
            market_returns = [
                0.015,
                -0.015,
                0.025,
                -0.005,
                0.018,
                0.012,
                -0.025,
                0.035,
                -0.008,
                0.015,
            ]
            beta = self._calculate_beta(returns, market_returns)

            result = {
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "volatility": volatility,
                "beta": beta,
                "max_drawdown": max(0, max(returns) - min(returns)),
                "sharpe_ratio": mean_return / volatility if volatility > 0 else 0,
                "risk_score": min(10, max(1, volatility * 100)),
                "stress_test_scenarios": {
                    "market_crash_2008": -0.37,
                    "covid_crash_2020": -0.34,
                    "dot_com_bubble_2000": -0.49,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Risk calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_beta(self, asset_returns: list[float], market_returns: list[float]) -> float:
        """Calculate beta coefficient"""
        if len(asset_returns) != len(market_returns):
            return 1.0

        asset_mean = sum(asset_returns) / len(asset_returns)
        market_mean = sum(market_returns) / len(market_returns)

        covariance = sum(
            (a - asset_mean) * (m - market_mean) for a, m in zip(asset_returns, market_returns, strict=False)
        ) / len(asset_returns)
        market_variance = sum((m - market_mean) ** 2 for m in market_returns) / len(market_returns)

        return covariance / market_variance if market_variance > 0 else 1.0

    async def perform_dcf_valuation(self, company_data: dict[str, Any]) -> dict[str, Any]:
        """Perform Discounted Cash Flow valuation"""
        try:
            # Extract key financial data
            revenue = company_data.get("revenue", 1000000000)
            growth_rate = company_data.get("growth_rate", 0.05)
            discount_rate = company_data.get("discount_rate", 0.10)
            terminal_growth = company_data.get("terminal_growth", 0.03)
            projection_years = 5

            # Project cash flows
            cash_flows = []
            current_cf = revenue * 0.15  # Assume 15% FCF margin

            for year in range(projection_years):
                current_cf *= 1 + growth_rate
                pv_factor = (1 + discount_rate) ** (year + 1)
                present_value = current_cf / pv_factor
                cash_flows.append(
                    {
                        "year": year + 1,
                        "cash_flow": current_cf,
                        "present_value": present_value,
                    }
                )

            # Calculate terminal value
            terminal_cf = current_cf * (1 + terminal_growth)
            terminal_value = terminal_cf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / ((1 + discount_rate) ** projection_years)

            # Calculate enterprise value
            sum_pv_cf = sum(cf["present_value"] for cf in cash_flows)
            enterprise_value = sum_pv_cf + terminal_pv

            # Equity value calculation
            debt = company_data.get("debt", 0)
            cash = company_data.get("cash", 0)
            equity_value = enterprise_value - debt + cash

            result = {
                "enterprise_value": enterprise_value,
                "equity_value": equity_value,
                "terminal_value": terminal_value,
                "terminal_pv": terminal_pv,
                "projected_cash_flows": cash_flows,
                "assumptions": {
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    "growth_rate": growth_rate,
                },
                "sensitivity_analysis": {
                    "discount_rate_+1%": equity_value * 0.9,
                    "discount_rate_-1%": equity_value * 1.1,
                    "growth_rate_+1%": equity_value * 1.15,
                    "growth_rate_-1%": equity_value * 0.85,
                },
            }

            return result

        except Exception as e:
            logger.error(f"DCF valuation failed: {e}")
            return {"error": str(e)}

    async def forecast_market_trends(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Forecast market trends using time series analysis"""
        try:
            # Simulate historical data
            historical_prices = [100 + i + 5 * math.sin(i / 10) for i in range(100)]

            # Simple trend analysis
            recent_prices = historical_prices[-20:]
            trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)

            # Generate forecasts
            forecast_periods = 12
            forecasts = []
            last_price = historical_prices[-1]

            for i in range(forecast_periods):
                # Simple trend + noise forecast
                forecast_price = last_price + trend * (i + 1) + 2 * math.sin((i + 1) / 3)
                confidence_interval = 0.1 * forecast_price  # 10% CI

                forecasts.append(
                    {
                        "period": i + 1,
                        "price_forecast": forecast_price,
                        "upper_bound": forecast_price + confidence_interval,
                        "lower_bound": forecast_price - confidence_interval,
                        "probability": max(0.5, 1 - i * 0.05),  # Decreasing confidence
                    }
                )

            # Market regime analysis
            volatility = sum(
                abs(historical_prices[i] - historical_prices[i - 1]) for i in range(1, len(historical_prices))
            ) / len(historical_prices)

            if volatility < 2:
                market_regime = "low_volatility"
            elif volatility > 5:
                market_regime = "high_volatility"
            else:
                market_regime = "normal_volatility"

            result = {
                "forecasts": forecasts,
                "trend": "bullish" if trend > 0 else "bearish",
                "trend_strength": abs(trend),
                "market_regime": market_regime,
                "volatility": volatility,
                "support_levels": [min(recent_prices), min(historical_prices[-50:])],
                "resistance_levels": [max(recent_prices), max(historical_prices[-50:])],
                "technical_indicators": {
                    "moving_average_20": sum(recent_prices) / len(recent_prices),
                    "rsi": 45.0,  # Simplified RSI
                    "macd": trend * 10,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Market forecast failed: {e}")
            return {"error": str(e)}

    async def analyze_credit_risk(self, borrower_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze credit risk using financial ratios and scoring"""
        try:
            # Extract financial metrics
            debt_to_income = borrower_data.get("debt_to_income", 0.3)
            credit_score = borrower_data.get("credit_score", 700)
            income = borrower_data.get("annual_income", 50000)
            debt = borrower_data.get("total_debt", 15000)

            # Calculate credit risk score
            base_score = 100

            # Adjust for debt-to-income ratio
            if debt_to_income > 0.4:
                base_score -= 20
            elif debt_to_income > 0.3:
                base_score -= 10

            # Adjust for credit score
            if credit_score < 600:
                base_score -= 30
            elif credit_score < 700:
                base_score -= 15

            # Income stability factor
            if income < 30000:
                base_score -= 10
            elif income > 100000:
                base_score += 10

            # Determine risk category
            if base_score >= 80:
                risk_category = "low_risk"
                default_probability = 0.02
            elif base_score >= 60:
                risk_category = "medium_risk"
                default_probability = 0.08
            else:
                risk_category = "high_risk"
                default_probability = 0.20

            result = {
                "credit_risk_score": base_score,
                "risk_category": risk_category,
                "default_probability": default_probability,
                "recommended_rate": 0.05 + default_probability * 3,  # Base rate + risk premium
                "key_factors": {
                    "debt_to_income_ratio": debt_to_income,
                    "credit_score": credit_score,
                    "annual_income": income,
                    "total_debt": debt,
                },
                "recommendations": [
                    "Verify income documentation",
                    "Review credit history details",
                    "Consider collateral requirements" if base_score < 70 else "Standard terms acceptable",
                ],
            }

            return result

        except Exception as e:
            logger.error(f"Credit risk analysis failed: {e}")
            return {"error": str(e)}

    async def initialize(self):
        """Initialize the Financial agent"""
        try:
            logger.info("Initializing Financial Agent...")

            # Initialize market data simulation
            self.market_data = {
                "SPX": {"price": 4500, "change": 0.015},
                "NASDAQ": {"price": 15000, "change": 0.022},
                "VIX": {"price": 18.5, "change": -0.05},
            }

            self.initialized = True
            logger.info(f"Financial Agent {self.agent_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Financial Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Cleanup resources"""
        try:
            # Save analysis results if needed
            for portfolio_id in self.portfolios:
                logger.info(f"Saving portfolio data: {portfolio_id}")

            self.initialized = False
            logger.info(f"Financial Agent {self.agent_id} shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
