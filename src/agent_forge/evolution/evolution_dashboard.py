"""Evolution Monitoring Dashboard - Real-time visualization of agent evolution

Provides comprehensive monitoring and visualization of the self-evolving agent ecosystem.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from flask import Flask, jsonify, render_template, request
from plotly.utils import PlotlyJSONEncoder

from .agent_evolution_engine import AgentEvolutionEngine

matplotlib.use("Agg")  # Non-interactive backend


logger = logging.getLogger(__name__)


class EvolutionDashboard:
    """Real-time dashboard for monitoring agent evolution"""

    def __init__(self, evolution_engine: AgentEvolutionEngine, port: int = 5000):
        self.evolution_engine = evolution_engine
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()

        # Dashboard data cache
        self.cached_data = {}
        self.last_update = None
        self.cache_duration = timedelta(minutes=1)  # Cache for 1 minute

    def setup_routes(self):
        """Setup Flask routes for the dashboard"""

        @self.app.route("/")
        def dashboard():
            return render_template("evolution_dashboard.html")

        @self.app.route("/api/evolution_status")
        async def evolution_status():
            try:
                data = await self.get_evolution_status()
                return jsonify(data)
            except Exception as e:
                logger.error(f"Failed to get evolution status: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/population_fitness")
        async def population_fitness():
            try:
                data = await self.get_population_fitness_data()
                return jsonify(data)
            except Exception as e:
                logger.error(f"Failed to get population fitness: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/specialization_distribution")
        async def specialization_distribution():
            try:
                data = await self.get_specialization_distribution()
                return jsonify(data)
            except Exception as e:
                logger.error(f"Failed to get specialization distribution: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/performance_trends")
        async def performance_trends():
            try:
                agent_id = request.args.get("agent_id")
                data = await self.get_performance_trends(agent_id)
                return jsonify(data)
            except Exception as e:
                logger.error(f"Failed to get performance trends: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/diversity_analysis")
        async def diversity_analysis():
            try:
                data = await self.get_diversity_analysis()
                return jsonify(data)
            except Exception as e:
                logger.error(f"Failed to get diversity analysis: {e}")
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/trigger_evolution", methods=["POST"])
        async def trigger_evolution():
            try:
                # Validate request data
                if not request.json:
                    return jsonify({"error": "Missing JSON payload"}), 400
                    
                generations = request.json.get("generations", 1)
                
                # Validate generations parameter
                if not isinstance(generations, int) or generations < 1 or generations > 100:
                    return jsonify({
                        "error": "Invalid generations parameter. Must be integer between 1 and 100"
                    }), 400
                
                # Check if evolution engine is available
                if not self.evolution_engine:
                    return jsonify({
                        "error": "Evolution engine not initialized"
                    }), 500
                    
                # Check if evolution is already running
                if hasattr(self.evolution_engine, 'is_running') and self.evolution_engine.is_running:
                    return jsonify({
                        "error": "Evolution is already running. Please wait for current cycle to complete."
                    }), 409
                
                # Verify there are agents to evolve
                try:
                    status = await self.evolution_engine.get_evolution_dashboard_data()
                    total_agents = status["population_stats"]["total_agents"]
                    
                    if total_agents == 0:
                        return jsonify({
                            "error": "No agents available for evolution. Initialize population first."
                        }), 400
                        
                except Exception as e:
                    logger.warning(f"Could not verify agent population: {e}")
                    # Continue anyway - the engine should handle this
                
                # Run evolution with timeout and validation
                logger.info(f"Starting evolution cycle with {generations} generations")
                start_time = time.time()
                
                try:
                    results = await asyncio.wait_for(
                        self.evolution_engine.run_evolution_cycle(generations=generations),
                        timeout=300.0  # 5 minute timeout
                    )
                except asyncio.TimeoutError:
                    return jsonify({
                        "error": "Evolution cycle timed out after 5 minutes"
                    }), 408
                
                duration = time.time() - start_time

                # Validate results structure
                if not isinstance(results, dict):
                    return jsonify({
                        "error": "Evolution cycle returned invalid results"
                    }), 400

                expected_fields = {
                    "initial_population": int,
                    "generations_run": int,
                    "best_fitness_history": list,
                    "diversity_history": list,
                    "specialization_distribution": list,
                }

                missing = [f for f in expected_fields if f not in results]
                if missing:
                    return jsonify({
                        "error": f"Evolution results missing fields: {', '.join(missing)}"
                    }), 400

                for field, field_type in expected_fields.items():
                    if not isinstance(results[field], field_type):
                        return jsonify({
                            "error": f"Invalid type for field '{field}'"
                        }), 400

                generations_run = results["generations_run"]
                histories = [
                    results["best_fitness_history"],
                    results["diversity_history"],
                    results["specialization_distribution"],
                ]
                if generations_run < 1 or any(len(hist) != generations_run for hist in histories):
                    return jsonify({
                        "error": "Evolution results inconsistent with generations run"
                    }), 400

                if results.get("status") == "failed":
                    return jsonify({
                        "error": f"Evolution failed: {results.get('error', 'Unknown error')}"
                    }), 500

                logger.info(f"Evolution cycle completed successfully in {duration:.2f}s")
                
                return jsonify({
                    "success": True,
                    "results": results,
                    "duration_seconds": duration,
                    "generations_requested": generations,
                    "completed_at": datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to trigger evolution: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return jsonify({
                    "error": f"Evolution failed: {str(e)}",
                    "type": type(e).__name__
                }), 500

        @self.app.route("/api/emergency_rollback", methods=["POST"])
        async def emergency_rollback():
            try:
                generations_back = request.json.get("generations_back", 1)
                success = await self.evolution_engine.emergency_rollback(
                    generations_back
                )
                return jsonify({"success": success})
            except Exception as e:
                logger.error(f"Failed to rollback: {e}")
                return jsonify({"error": str(e)}), 500

    async def get_evolution_status(self) -> dict[str, Any]:
        """Get current evolution status"""
        if self._is_cache_valid("evolution_status"):
            return self.cached_data["evolution_status"]

        dashboard_data = await self.evolution_engine.get_evolution_dashboard_data()

        status = {
            "current_generation": dashboard_data["population_stats"][
                "current_generation"
            ],
            "total_agents": dashboard_data["population_stats"]["total_agents"],
            "avg_fitness": dashboard_data["population_stats"]["avg_fitness"],
            "max_fitness": dashboard_data["population_stats"]["max_fitness"],
            "diversity": dashboard_data["population_stats"]["diversity"],
            "timestamp": dashboard_data["timestamp"],
            "is_evolving": False,  # Would be set during active evolution
            "last_evolution": self._get_last_evolution_time(),
        }

        self.cached_data["evolution_status"] = status
        self.last_update = datetime.now()

        return status

    async def get_population_fitness_data(self) -> dict[str, Any]:
        """Get population fitness data for visualization"""
        if self._is_cache_valid("population_fitness"):
            return self.cached_data["population_fitness"]

        dashboard_data = await self.evolution_engine.get_evolution_dashboard_data()
        fitness_scores = dashboard_data["fitness_scores"]

        # Sort agents by fitness
        sorted_agents = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)

        # Create visualization data
        agent_names = [
            agent_id.split("_")[-1] for agent_id, _ in sorted_agents
        ]  # Extract specialization
        fitness_values = [fitness for _, fitness in sorted_agents]

        # Create Plotly bar chart
        fig = px.bar(
            x=agent_names,
            y=fitness_values,
            title="Agent Population Fitness Scores",
            labels={"x": "Agent Specialization", "y": "Fitness Score"},
            color=fitness_values,
            color_continuous_scale="viridis",
        )

        fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)

        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        data = {
            "chart": chart_json,
            "summary": {
                "best_agent": sorted_agents[0][0] if sorted_agents else None,
                "best_fitness": sorted_agents[0][1] if sorted_agents else 0,
                "worst_fitness": sorted_agents[-1][1] if sorted_agents else 0,
                "fitness_std": np.std(fitness_values) if fitness_values else 0,
            },
        }

        self.cached_data["population_fitness"] = data
        return data

    async def get_specialization_distribution(self) -> dict[str, Any]:
        """Get specialization distribution visualization"""
        if self._is_cache_valid("specialization_distribution"):
            return self.cached_data["specialization_distribution"]

        dashboard_data = await self.evolution_engine.get_evolution_dashboard_data()
        spec_dist = dashboard_data["population_stats"]["specialization_distribution"]

        # Create pie chart
        fig = px.pie(
            values=list(spec_dist.values()),
            names=list(spec_dist.keys()),
            title="Agent Specialization Distribution",
        )

        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=500)

        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        data = {
            "chart": chart_json,
            "distribution": spec_dist,
            "total_specializations": len(spec_dist),
            "most_common": max(spec_dist.items(), key=lambda x: x[1])
            if spec_dist
            else None,
        }

        self.cached_data["specialization_distribution"] = data
        return data

    async def get_performance_trends(
        self, agent_id: str | None = None
    ) -> dict[str, Any]:
        """Get performance trends for agents"""
        cache_key = f"performance_trends_{agent_id or 'all'}"
        if self._is_cache_valid(cache_key):
            return self.cached_data[cache_key]

        if agent_id:
            # Single agent trends
            trends = self.evolution_engine.kpi_tracker.get_performance_trends(agent_id)

            if not trends:
                return {"error": f"No trend data for agent {agent_id}"}

            # Create line chart
            fig = go.Figure()

            for metric, values in trends.items():
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        mode="lines+markers",
                        name=metric.replace("_", " ").title(),
                        line=dict(width=2),
                    )
                )

            fig.update_layout(
                title=f"Performance Trends - {agent_id}",
                xaxis_title="Time Steps",
                yaxis_title="Performance Score",
                height=400,
            )

            chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

            data = {
                "chart": chart_json,
                "agent_id": agent_id,
                "latest_scores": {
                    metric: values[-1] if values else 0
                    for metric, values in trends.items()
                },
            }

        else:
            # All agents overview
            dashboard_data = await self.evolution_engine.get_evolution_dashboard_data()
            all_trends = dashboard_data["performance_trends"]

            # Create multi-agent comparison
            fig = go.Figure()

            for agent_id, trends in all_trends.items():
                if "task_success_rate" in trends:
                    fig.add_trace(
                        go.Scatter(
                            y=trends["task_success_rate"],
                            mode="lines",
                            name=agent_id.split("_")[-1],
                            opacity=0.7,
                        )
                    )

            fig.update_layout(
                title="Task Success Rate Trends - All Agents",
                xaxis_title="Time Steps",
                yaxis_title="Success Rate",
                height=400,
            )

            chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

            data = {
                "chart": chart_json,
                "agent_count": len(all_trends),
                "available_agents": list(all_trends.keys()),
            }

        self.cached_data[cache_key] = data
        return data

    async def get_diversity_analysis(self) -> dict[str, Any]:
        """Get population diversity analysis"""
        if self._is_cache_valid("diversity_analysis"):
            return self.cached_data["diversity_analysis"]

        # Load historical diversity data
        diversity_history = []
        generation_files = list(
            self.evolution_engine.evolution_data_path.glob(
                "evolution_results_gen_*.json"
            )
        )

        for file_path in sorted(generation_files):
            try:
                with open(file_path) as f:
                    results = json.load(f)
                    if "diversity_history" in results:
                        diversity_history.extend(results["diversity_history"])
            except Exception as e:
                logger.warning(f"Failed to load diversity data from {file_path}: {e}")

        if not diversity_history:
            diversity_history = [0.5]  # Default value

        # Create diversity trend chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=diversity_history,
                mode="lines+markers",
                name="Population Diversity",
                line=dict(color="purple", width=3),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            title="Population Diversity Over Time",
            xaxis_title="Generation",
            yaxis_title="Diversity Score",
            height=400,
            yaxis=dict(range=[0, 1]),
        )

        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        current_diversity = await self.evolution_engine.get_evolution_dashboard_data()
        current_div_score = current_diversity["population_stats"]["diversity"]

        data = {
            "chart": chart_json,
            "current_diversity": current_div_score,
            "diversity_trend": "increasing"
            if len(diversity_history) > 1
            and diversity_history[-1] > diversity_history[-2]
            else "decreasing",
            "avg_diversity": np.mean(diversity_history),
            "generations_tracked": len(diversity_history),
        }

        self.cached_data["diversity_analysis"] = data
        return data

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cached_data or self.last_update is None:
            return False

        return datetime.now() - self.last_update < self.cache_duration

    def _get_last_evolution_time(self) -> str | None:
        """Get timestamp of last evolution run"""
        try:
            evolution_files = list(
                self.evolution_engine.evolution_data_path.glob(
                    "evolution_results_gen_*.json"
                )
            )
            if not evolution_files:
                return None

            latest_file = max(evolution_files, key=lambda x: x.stat().st_mtime)
            return datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        except Exception:
            return None

    def run(self, debug: bool = False):
        """Run the dashboard server"""
        logger.info(f"Starting Evolution Dashboard on port {self.port}")
        self.app.run(host="0.0.0.0", port=self.port, debug=debug)


class PerformanceAnalyzer:
    """Advanced performance analysis for agent evolution"""

    def __init__(self, evolution_engine: AgentEvolutionEngine):
        self.evolution_engine = evolution_engine

    async def generate_evolution_report(self) -> dict[str, Any]:
        """Generate comprehensive evolution analysis report"""
        dashboard_data = await self.evolution_engine.get_evolution_dashboard_data()

        # Performance analysis
        fitness_scores = dashboard_data["fitness_scores"]
        performance_analysis = {
            "top_performers": self._get_top_performers(fitness_scores),
            "performance_distribution": self._analyze_performance_distribution(
                fitness_scores
            ),
            "improvement_opportunities": self._identify_improvement_opportunities(
                fitness_scores
            ),
        }

        # Specialization analysis
        spec_dist = dashboard_data["population_stats"]["specialization_distribution"]
        specialization_analysis = {
            "distribution_balance": self._analyze_specialization_balance(spec_dist),
            "specialization_effectiveness": await self._analyze_specialization_effectiveness(),
            "niche_opportunities": self._identify_niche_opportunities(spec_dist),
        }

        # Evolution trends
        evolution_trends = await self._analyze_evolution_trends()

        report = {
            "timestamp": datetime.now().isoformat(),
            "generation": dashboard_data["population_stats"]["current_generation"],
            "performance_analysis": performance_analysis,
            "specialization_analysis": specialization_analysis,
            "evolution_trends": evolution_trends,
            "recommendations": self._generate_recommendations(
                performance_analysis, specialization_analysis
            ),
        }

        return report

    def _get_top_performers(
        self, fitness_scores: dict[str, float], top_n: int = 5
    ) -> list[dict[str, Any]]:
        """Get top performing agents"""
        sorted_agents = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)

        top_performers = []
        for i, (agent_id, fitness) in enumerate(sorted_agents[:top_n]):
            top_performers.append(
                {
                    "rank": i + 1,
                    "agent_id": agent_id,
                    "fitness": fitness,
                    "specialization": agent_id.split("_")[-1]
                    if "_" in agent_id
                    else "unknown",
                }
            )

        return top_performers

    def _analyze_performance_distribution(
        self, fitness_scores: dict[str, float]
    ) -> dict[str, float]:
        """Analyze distribution of performance scores"""
        scores = list(fitness_scores.values())

        if not scores:
            return {}

        return {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "q25": np.percentile(scores, 25),
            "q75": np.percentile(scores, 75),
        }

    def _identify_improvement_opportunities(
        self, fitness_scores: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Identify agents with improvement opportunities"""
        sorted_agents = sorted(fitness_scores.items(), key=lambda x: x[1])
        bottom_quartile = len(sorted_agents) // 4

        opportunities = []
        for agent_id, fitness in sorted_agents[: max(1, bottom_quartile)]:
            opportunities.append(
                {
                    "agent_id": agent_id,
                    "current_fitness": fitness,
                    "improvement_potential": "high" if fitness < 0.3 else "medium",
                    "suggested_actions": [
                        "hyperparameter_tuning",
                        "specialization_refinement",
                    ],
                }
            )

        return opportunities

    def _analyze_specialization_balance(
        self, spec_dist: dict[str, int]
    ) -> dict[str, Any]:
        """Analyze balance of specializations"""
        total_agents = sum(spec_dist.values())

        if total_agents == 0:
            return {}

        # Calculate ideal distribution (equal distribution)
        ideal_per_spec = total_agents / len(spec_dist)

        # Calculate imbalance
        imbalances = {}
        for spec, count in spec_dist.items():
            imbalances[spec] = abs(count - ideal_per_spec) / ideal_per_spec

        return {
            "total_specializations": len(spec_dist),
            "ideal_per_specialization": ideal_per_spec,
            "most_overrepresented": max(imbalances, key=imbalances.get)
            if imbalances
            else None,
            "most_underrepresented": min(spec_dist, key=spec_dist.get)
            if spec_dist
            else None,
            "balance_score": 1.0
            - (np.std(list(spec_dist.values())) / np.mean(list(spec_dist.values())))
            if spec_dist
            else 0.0,
        }

    async def _analyze_specialization_effectiveness(self) -> dict[str, float]:
        """Analyze effectiveness of each specialization"""
        dashboard_data = await self.evolution_engine.get_evolution_dashboard_data()
        fitness_scores = dashboard_data["fitness_scores"]

        # Group agents by specialization
        spec_performance = {}
        for agent_id, fitness in fitness_scores.items():
            spec = agent_id.split("_")[-1] if "_" in agent_id else "unknown"
            if spec not in spec_performance:
                spec_performance[spec] = []
            spec_performance[spec].append(fitness)

        # Calculate average performance per specialization
        spec_effectiveness = {}
        for spec, performances in spec_performance.items():
            spec_effectiveness[spec] = np.mean(performances) if performances else 0.0

        return spec_effectiveness

    def _identify_niche_opportunities(self, spec_dist: dict[str, int]) -> list[str]:
        """Identify underserved specialization niches"""
        total_agents = sum(spec_dist.values())
        avg_per_spec = total_agents / len(spec_dist) if spec_dist else 0

        underserved = []
        for spec, count in spec_dist.items():
            if count < avg_per_spec * 0.5:  # Less than 50% of average
                underserved.append(spec)

        return underserved

    async def _analyze_evolution_trends(self) -> dict[str, Any]:
        """Analyze evolution trends over generations"""
        # Load historical data
        fitness_history = []
        diversity_history = []

        generation_files = list(
            self.evolution_engine.evolution_data_path.glob(
                "evolution_results_gen_*.json"
            )
        )

        for file_path in sorted(generation_files):
            try:
                with open(file_path) as f:
                    results = json.load(f)

                    if "best_fitness_history" in results:
                        fitness_history.extend(results["best_fitness_history"])

                    if "diversity_history" in results:
                        diversity_history.extend(results["diversity_history"])

            except Exception as e:
                logger.warning(f"Failed to load evolution data from {file_path}: {e}")

        trends = {}

        if fitness_history:
            trends["fitness_trend"] = (
                "improving"
                if len(fitness_history) > 1 and fitness_history[-1] > fitness_history[0]
                else "declining"
            )
            trends["fitness_growth_rate"] = (
                (fitness_history[-1] - fitness_history[0]) / len(fitness_history)
                if len(fitness_history) > 1
                else 0
            )
            trends["best_fitness_ever"] = max(fitness_history)

        if diversity_history:
            trends["diversity_trend"] = (
                "increasing"
                if len(diversity_history) > 1
                and diversity_history[-1] > diversity_history[0]
                else "decreasing"
            )
            trends["diversity_stability"] = (
                1.0 - (np.std(diversity_history) / np.mean(diversity_history))
                if diversity_history
                else 0
            )

        trends["generations_analyzed"] = len(generation_files)

        return trends

    def _generate_recommendations(
        self,
        performance_analysis: dict[str, Any],
        specialization_analysis: dict[str, Any],
    ) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Performance-based recommendations
        if (
            performance_analysis.get("performance_distribution", {}).get("mean", 0)
            < 0.5
        ):
            recommendations.append(
                "Overall population fitness is low. Consider increasing mutation rate or improving evaluation tasks."
            )

        if len(performance_analysis.get("improvement_opportunities", [])) > 5:
            recommendations.append(
                "Many agents underperforming. Consider targeted optimization or population restart."
            )

        # Specialization-based recommendations
        balance_score = specialization_analysis.get("distribution_balance", {}).get(
            "balance_score", 0
        )
        if balance_score < 0.7:
            recommendations.append(
                "Specialization distribution is imbalanced. Consider adjusting selection pressure."
            )

        niche_opportunities = specialization_analysis.get("niche_opportunities", [])
        if len(niche_opportunities) > 3:
            recommendations.append(
                f"Underserved specializations detected: {', '.join(niche_opportunities[:3])}. Consider targeted agent creation."
            )

        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Population appears healthy. Continue current evolution strategy."
            )

        return recommendations


def create_dashboard_html_template() -> str:
    """Create HTML template for evolution dashboard"""
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Evolution Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-evolving { background-color: #28a745; color: white; }
        .status-idle { background-color: #6c757d; color: white; }
    </style>
</head>
<body class="bg-light">
    <div class="container-fluid">
        <div class="row">
            <div class="col-12 py-3">
                <h1 class="text-center mb-4">Agent Evolution Dashboard</h1>

                <!-- Status Row -->
                <div class="row" id="status-row">
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3 id="current-generation">-</h3>
                            <p>Current Generation</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3 id="total-agents">-</h3>
                            <p>Active Agents</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3 id="avg-fitness">-</h3>
                            <p>Average Fitness</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card text-center">
                            <h3 id="diversity-score">-</h3>
                            <p>Population Diversity</p>
                        </div>
                    </div>
                </div>

                <!-- Control Panel -->
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5>Evolution Control Panel</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <button class="btn btn-primary" onclick="triggerEvolution()">Run Evolution Cycle</button>
                                        <button class="btn btn-warning ml-2" onclick="emergencyRollback()">Emergency Rollback</button>
                                    </div>
                                    <div class="col-md-6 text-right">
                                        <span class="status-badge" id="evolution-status">Idle</span>
                                        <small class="text-muted ml-2" id="last-update">Never</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Charts Row -->
                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h5>Population Fitness</h5>
                            <div id="fitness-chart"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h5>Specialization Distribution</h5>
                            <div id="specialization-chart"></div>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h5>Performance Trends</h5>
                            <div id="performance-trends-chart"></div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h5>Diversity Analysis</h5>
                            <div id="diversity-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Dashboard JavaScript
        let refreshInterval;

        function updateDashboard() {
            // Update status
            fetch('/api/evolution_status')
                .then(response => response.json())
                .then(data => {
                    $('#current-generation').text(data.current_generation || '-');
                    $('#total-agents').text(data.total_agents || '-');
                    $('#avg-fitness').text((data.avg_fitness || 0).toFixed(3));
                    $('#diversity-score').text((data.diversity || 0).toFixed(3));

                    const status = data.is_evolving ? 'Evolving' : 'Idle';
                    $('#evolution-status').text(status).attr('class',
                        `status-badge ${data.is_evolving ? 'status-evolving' : 'status-idle'}`);

                    if (data.timestamp) {
                        $('#last-update').text(`Last update: ${new Date(data.timestamp).toLocaleTimeString()}`);
                    }
                })
                .catch(error => console.error('Failed to update status:', error));

            // Update fitness chart
            fetch('/api/population_fitness')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        const chartData = JSON.parse(data.chart);
                        Plotly.newPlot('fitness-chart', chartData.data, chartData.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Failed to update fitness chart:', error));

            // Update specialization chart
            fetch('/api/specialization_distribution')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        const chartData = JSON.parse(data.chart);
                        Plotly.newPlot('specialization-chart', chartData.data, chartData.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Failed to update specialization chart:', error));

            // Update performance trends
            fetch('/api/performance_trends')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        const chartData = JSON.parse(data.chart);
                        Plotly.newPlot('performance-trends-chart', chartData.data, chartData.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Failed to update performance trends:', error));

            // Update diversity analysis
            fetch('/api/diversity_analysis')
                .then(response => response.json())
                .then(data => {
                    if (data.chart) {
                        const chartData = JSON.parse(data.chart);
                        Plotly.newPlot('diversity-chart', chartData.data, chartData.layout, {responsive: true});
                    }
                })
                .catch(error => console.error('Failed to update diversity analysis:', error));
        }

        function triggerEvolution() {
            const generations = prompt('Number of generations to run:', '1');
            if (generations && !isNaN(generations)) {
                fetch('/api/trigger_evolution', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({generations: parseInt(generations)})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Evolution triggered successfully!');
                        updateDashboard();
                    } else {
                        alert('Failed to trigger evolution: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(error => {
                    console.error('Evolution trigger failed:', error);
                    alert('Failed to trigger evolution');
                });
            }
        }

        function emergencyRollback() {
            if (confirm('Are you sure you want to perform an emergency rollback?')) {
                const generations = prompt('Generations to roll back:', '1');
                if (generations && !isNaN(generations)) {
                    fetch('/api/emergency_rollback', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({generations_back: parseInt(generations)})
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Rollback completed successfully!');
                            updateDashboard();
                        } else {
                            alert('Rollback failed');
                        }
                    })
                    .catch(error => {
                        console.error('Rollback failed:', error);
                        alert('Rollback failed');
                    });
                }
            }
        }

        // Initialize dashboard
        $(document).ready(function() {
            updateDashboard();

            // Auto-refresh every 30 seconds
            refreshInterval = setInterval(updateDashboard, 30000);
        });

        // Clean up on page unload
        $(window).on('beforeunload', function() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        });
    </script>
</body>
</html>
    """

    return template


# Create templates directory and save HTML template
def setup_dashboard_templates(base_path: str = "agent_forge/evolution"):
    """Setup dashboard templates directory"""
    templates_dir = Path(base_path) / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    template_file = templates_dir / "evolution_dashboard.html"
    with open(template_file, "w") as f:
        f.write(create_dashboard_html_template())

    logger.info(f"Dashboard template created at {template_file}")


if __name__ == "__main__":
    # Example usage
    async def run_dashboard():
        from .agent_evolution_engine import AgentEvolutionEngine

        # Setup templates
        setup_dashboard_templates()

        # Initialize evolution engine
        evolution_engine = AgentEvolutionEngine()
        await evolution_engine.initialize_population()

        # Create and run dashboard
        dashboard = EvolutionDashboard(evolution_engine, port=5000)
        dashboard.run(debug=True)

    asyncio.run(run_dashboard())
