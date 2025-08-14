"""Data Science Agent - Statistical Analysis and ML Model Training Specialist"""

import logging
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class DataAnalysisRequest:
    """Request for data analysis task"""

    task_type: str  # 'statistical', 'ml_training', 'visualization', 'preprocessing'
    data_source: str
    parameters: dict[str, Any]
    output_format: str = "json"
    priority: int = 5


class DataScienceAgent(BaseAgent):
    """Specialized agent for data science tasks including:
    - Statistical analysis and hypothesis testing
    - Machine learning model training and evaluation
    - Data preprocessing and feature engineering
    - Visualization generation
    - A/B testing and experimentation
    """

    def __init__(self, agent_id: str = "data_science_agent"):
        capabilities = [
            "statistical_analysis",
            "ml_model_training",
            "data_preprocessing",
            "feature_engineering",
            "visualization",
            "ab_testing",
            "time_series_analysis",
            "anomaly_detection",
        ]
        super().__init__(agent_id, "DataScience", capabilities)
        self.models_cache = {}
        self.analysis_history = []

    async def generate(self, prompt: str) -> str:
        """Generate data analysis insights from prompt"""
        if "statistical analysis" in prompt.lower():
            return "I can perform statistical analysis. Please provide data source and analysis parameters."
        if "machine learning" in prompt.lower():
            return "I can train ML models. Specify the target variable and model type."
        if "anomaly detection" in prompt.lower():
            return "I can detect anomalies in your data using isolation forest or other methods."
        return "I'm a Data Science Agent specialized in statistical analysis, ML training, and data insights."

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank results based on data science relevance"""
        # Simple relevance scoring based on data science keywords
        keywords = [
            "analysis",
            "model",
            "data",
            "statistics",
            "prediction",
            "regression",
            "classification",
        ]

        for result in results:
            score = 0
            text = str(result.get("content", ""))
            for keyword in keywords:
                score += text.lower().count(keyword)
            result["ds_relevance_score"] = score

        return sorted(results, key=lambda x: x.get("ds_relevance_score", 0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return agent capabilities and status"""
        info = await super().introspect()
        info.update(
            {
                "models_cached": len(self.models_cache),
                "analyses_performed": len(self.analysis_history),
                "status": "active" if self.initialized else "initializing",
            }
        )
        return info

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate latent space for data analysis"""
        analysis_type = "statistical" if "stat" in query.lower() else "ml"
        latent_representation = f"LATENT[{analysis_type}:{query[:50]}]"
        return analysis_type, latent_representation

    async def initialize(self):
        """Initialize the data science agent"""
        try:
            await self._load_pretrained_models()
            await self._setup_analysis_pipeline()
            self.initialized = True
            logger.info(f"Data Science Agent {self.agent_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Data Science Agent: {e}")
            self.initialized = False

    async def _load_pretrained_models(self):
        """Load commonly used pretrained models"""
        try:
            # Load standard models for common tasks
            self.models_cache["sentiment"] = {
                "tokenizer": AutoTokenizer.from_pretrained("distilbert-base-uncased"),
                "model": AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased"),
            }
        except Exception as e:
            logger.warning(f"Could not load pretrained models: {e}")

    async def _setup_analysis_pipeline(self):
        """Setup data analysis pipeline"""
        self.pipeline_config = {
            "max_data_size": 10_000_000,  # 10MB limit
            "supported_formats": ["csv", "json", "parquet", "excel"],
            "visualization_backend": "matplotlib",
            "parallel_processing": True,
        }

    async def perform_statistical_analysis(self, data: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        results = {
            "descriptive_stats": {},
            "correlations": {},
            "distributions": {},
            "hypothesis_tests": {},
        }

        try:
            if not SKLEARN_AVAILABLE:
                return {"error": "Required data science libraries not available. Install pandas, numpy, scikit-learn."}

            # Descriptive statistics
            results["descriptive_stats"] = {
                "summary": data.describe().to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "data_types": data.dtypes.astype(str).to_dict(),
            }

            # Correlation analysis for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                results["correlations"] = data[numeric_cols].corr().to_dict()

            # Distribution analysis
            for col in numeric_cols[:10]:  # Limit to first 10 columns
                results["distributions"][col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "skewness": float(data[col].skew()),
                    "kurtosis": float(data[col].kurt()),
                }

            # Hypothesis testing if specified
            if "hypothesis_test" in config:
                test_type = config["hypothesis_test"].get("type", "t_test")
                if test_type == "t_test" and "groups" in config["hypothesis_test"]:
                    from scipy import stats

                    group_col = config["hypothesis_test"]["groups"]
                    value_col = config["hypothesis_test"]["value"]
                    groups = data.groupby(group_col)[value_col].apply(list)
                    if len(groups) == 2:
                        t_stat, p_value = stats.ttest_ind(groups.iloc[0], groups.iloc[1])
                        results["hypothesis_tests"]["t_test"] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": p_value < 0.05,
                        }

        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            results["error"] = str(e)

        return results

    async def train_ml_model(self, data: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Train machine learning model based on configuration"""
        results = {
            "model_type": config.get("model_type", "auto"),
            "metrics": {},
            "feature_importance": {},
            "model_id": None,
        }

        try:
            if not SKLEARN_AVAILABLE:
                return {"error": "Required data science libraries not available. Install pandas, numpy, scikit-learn."}

            # Prepare data
            target_col = config.get("target_column")
            feature_cols = [col for col in data.columns if col != target_col]

            X = data[feature_cols]
            y = data[target_col]

            # Handle categorical variables
            X = pd.get_dummies(X, drop_first=True)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Select and train model
            model_type = config.get("model_type", "auto")

            if model_type == "auto":
                # Auto-select based on target type
                if y.dtype == "object" or len(y.unique()) < 10:
                    from sklearn.ensemble import RandomForestClassifier

                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    task_type = "classification"
                else:
                    from sklearn.ensemble import RandomForestRegressor

                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    task_type = "regression"
            else:
                # Use specified model
                model = self._get_model_by_type(model_type)
                task_type = "classification" if "Classifier" in model_type else "regression"

            # Train model
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            if task_type == "classification":
                from sklearn.metrics import (
                    accuracy_score,
                    f1_score,
                    precision_score,
                    recall_score,
                )

                y_pred = model.predict(X_test_scaled)
                results["metrics"] = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, average="weighted")),
                    "recall": float(recall_score(y_test, y_pred, average="weighted")),
                    "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
                }
            else:
                from sklearn.metrics import (
                    mean_absolute_error,
                    mean_squared_error,
                    r2_score,
                )

                y_pred = model.predict(X_test_scaled)
                results["metrics"] = {
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "mae": float(mean_absolute_error(y_test, y_pred)),
                    "r2_score": float(r2_score(y_test, y_pred)),
                }

            # Feature importance for tree-based models
            if hasattr(model, "feature_importances_"):
                importance_dict = dict(zip(X.columns, model.feature_importances_, strict=False))
                results["feature_importance"] = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
                )  # Top 20 features

            # Store model
            model_id = f"model_{len(self.models_cache)}_{task_type}"
            self.models_cache[model_id] = {
                "model": model,
                "scaler": scaler,
                "feature_cols": list(X.columns),
                "target_col": target_col,
                "task_type": task_type,
            }
            results["model_id"] = model_id

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            results["error"] = str(e)

        return results

    async def perform_anomaly_detection(self, data: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Detect anomalies in the dataset"""
        results = {
            "anomalies": [],
            "anomaly_scores": {},
            "method": config.get("method", "isolation_forest"),
        }

        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler

            # Prepare data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            X = data[numeric_cols]

            # Scale data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train anomaly detector
            contamination = config.get("contamination", 0.1)
            detector = IsolationForest(contamination=contamination, random_state=42)
            predictions = detector.fit_predict(X_scaled)

            # Get anomaly scores
            scores = detector.score_samples(X_scaled)

            # Identify anomalies
            anomaly_indices = np.where(predictions == -1)[0]
            results["anomalies"] = anomaly_indices.tolist()
            results["anomaly_scores"] = {
                "mean_score": float(scores.mean()),
                "std_score": float(scores.std()),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
            }
            results["total_anomalies"] = len(anomaly_indices)
            results["anomaly_rate"] = len(anomaly_indices) / len(data)

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            results["error"] = str(e)

        return results

    async def perform_time_series_analysis(self, data: Any, config: dict[str, Any]) -> dict[str, Any]:
        """Perform time series analysis and forecasting"""
        results = {"trend": {}, "seasonality": {}, "forecast": {}, "metrics": {}}

        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Ensure datetime index
            date_col = config.get("date_column", data.columns[0])
            value_col = config.get("value_column", data.columns[1])

            ts_data = data.set_index(pd.to_datetime(data[date_col]))[value_col]
            ts_data = ts_data.sort_index()

            # Decomposition
            decomposition = seasonal_decompose(ts_data, model="additive", period=config.get("period", 12))

            results["trend"] = {
                "direction": "increasing"
                if decomposition.trend.iloc[-1] > decomposition.trend.iloc[0]
                else "decreasing",
                "strength": float(
                    np.corrcoef(
                        range(len(decomposition.trend.dropna())),
                        decomposition.trend.dropna(),
                    )[0, 1]
                ),
            }

            results["seasonality"] = {
                "present": decomposition.seasonal.std() > 0.01,
                "strength": float(decomposition.seasonal.std()),
            }

            # Forecasting with ARIMA
            if config.get("forecast", False):
                model = ARIMA(ts_data, order=(1, 1, 1))
                model_fit = model.fit()

                forecast_periods = config.get("forecast_periods", 10)
                forecast = model_fit.forecast(steps=forecast_periods)

                results["forecast"] = {
                    "values": forecast.tolist(),
                    "periods": forecast_periods,
                    "model": "ARIMA(1,1,1)",
                }

                results["metrics"] = {
                    "aic": float(model_fit.aic),
                    "bic": float(model_fit.bic),
                }

        except Exception as e:
            logger.error(f"Time series analysis failed: {e}")
            results["error"] = str(e)

        return results

    def _get_model_by_type(self, model_type: str):
        """Get sklearn model by type name"""
        models = {
            "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
            "RandomForestRegressor": "sklearn.ensemble.RandomForestRegressor",
            "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier",
            "GradientBoostingRegressor": "sklearn.ensemble.GradientBoostingRegressor",
            "LogisticRegression": "sklearn.linear_model.LogisticRegression",
            "LinearRegression": "sklearn.linear_model.LinearRegression",
            "SVM": "sklearn.svm.SVC",
            "SVR": "sklearn.svm.SVR",
        }

        if model_type in models:
            module_name, class_name = models[model_type].rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)()
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor()

    async def process_request(self, request: DataAnalysisRequest) -> dict[str, Any]:
        """Process incoming data analysis request"""
        try:
            # Load data
            data = await self._load_data(request.data_source)

            # Route to appropriate handler
            if request.task_type == "statistical":
                result = await self.perform_statistical_analysis(data, request.parameters)
            elif request.task_type == "ml_training":
                result = await self.train_ml_model(data, request.parameters)
            elif request.task_type == "anomaly_detection":
                result = await self.perform_anomaly_detection(data, request.parameters)
            elif request.task_type == "time_series":
                result = await self.perform_time_series_analysis(data, request.parameters)
            else:
                result = {"error": f"Unknown task type: {request.task_type}"}

            # Store in history
            self.analysis_history.append(
                {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "task_type": request.task_type,
                    "result_summary": self._summarize_result(result),
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e)}

    async def _load_data(self, data_source: str) -> Any:
        """Load data from various sources"""
        if data_source.endswith(".csv"):
            return pd.read_csv(data_source)
        if data_source.endswith(".json"):
            return pd.read_json(data_source)
        if data_source.endswith(".parquet"):
            return pd.read_parquet(data_source)
        if data_source.endswith(".xlsx"):
            return pd.read_excel(data_source)
        raise ValueError(f"Unsupported data format: {data_source}")

    def _summarize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Create summary of analysis result"""
        summary = {}
        if "metrics" in result:
            summary["metrics"] = result["metrics"]
        if "anomalies" in result:
            summary["anomaly_count"] = len(result["anomalies"])
        if "descriptive_stats" in result:
            summary["analyzed_columns"] = len(result["descriptive_stats"].get("summary", {}))
        return summary

    async def collaborate_with_agent(self, agent_id: str, task: dict[str, Any]) -> dict[str, Any]:
        """Collaborate with other agents for complex tasks"""
        logger.info(f"Collaborating with {agent_id} on task: {task}")
        # Simplified collaboration without message bus dependency
        return {"status": "collaboration_initiated", "agent_id": agent_id, "task": task}

    async def shutdown(self):
        """Cleanup resources"""
        try:
            # Save models if needed
            for model_id, _model_data in self.models_cache.items():
                logger.info(f"Saving model {model_id}")
                # Implement model persistence if needed

            self.initialized = False
            logger.info(f"Data Science Agent {self.agent_id} shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
