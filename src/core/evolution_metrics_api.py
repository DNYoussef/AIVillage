"""Evolution Metrics API Server

Provides HTTP API endpoints for evolution metrics system on port 8081.
"""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sqlite3
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import the integrated metrics module
try:
    from evolution_metrics_integrated import (
        IntegratedEvolutionMetrics,
        get_health_status,
        get_metrics_instance,
    )
    INTEGRATED_AVAILABLE = True
except ImportError:
    INTEGRATED_AVAILABLE = False


class EvolutionMetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for evolution metrics API."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health/evolution":
            self.handle_health_check()
        elif self.path == "/metrics/current":
            self.handle_current_metrics()
        elif self.path == "/metrics/leaderboard":
            self.handle_leaderboard()
        elif self.path.startswith("/metrics/agent/"):
            self.handle_agent_metrics()
        else:
            self.send_error(404, "Not Found")

    def handle_health_check(self):
        """Handle health check endpoint."""
        health_data = self.get_health_status()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(health_data, indent=2).encode())

    def handle_current_metrics(self):
        """Handle current metrics endpoint."""
        metrics_data = self.get_current_metrics()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(metrics_data, indent=2).encode())

    def handle_leaderboard(self):
        """Handle leaderboard endpoint."""
        leaderboard = self.get_leaderboard()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(leaderboard, indent=2).encode())

    def handle_agent_metrics(self):
        """Handle agent-specific metrics endpoint."""
        agent_id = self.path.split("/")[-1]
        agent_data = self.get_agent_metrics(agent_id)

        if agent_data:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(agent_data, indent=2).encode())
        else:
            self.send_error(404, f"Agent {agent_id} not found")

    def get_health_status(self):
        """Get health status data."""
        if INTEGRATED_AVAILABLE:
            return get_health_status()

        # Fallback implementation
        db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")

        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": {
                "connected": os.path.exists(db_path),
                "path": db_path,
                "wal_mode": False
            },
            "redis": {
                "available": False,
                "connected": False
            },
            "metrics": {
                "flush_threshold": int(os.getenv("AIVILLAGE_METRICS_FLUSH_THRESHOLD", "50"))
            },
            "api": {
                "port": 8081,
                "endpoints": [
                    "/health/evolution",
                    "/metrics/current",
                    "/metrics/leaderboard",
                    "/metrics/agent/{agent_id}"
                ]
            }
        }

        # Check database details
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Check WAL mode
                cursor.execute("PRAGMA journal_mode")
                mode = cursor.fetchone()[0]
                health["database"]["wal_mode"] = mode.lower() == "wal"

                # Get metrics count
                cursor.execute("SELECT COUNT(*) FROM fitness_metrics")
                health["metrics"]["total_collected"] = cursor.fetchone()[0]

                # Get current round
                cursor.execute("SELECT id, status FROM evolution_rounds ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    health["database"]["current_round"] = row[0]
                    health["database"]["round_status"] = row[1]

                conn.close()
            except Exception as e:
                health["status"] = "degraded"
                health["error"] = str(e)

        return health

    def get_current_metrics(self):
        """Get current metrics summary."""
        db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")

        if not os.path.exists(db_path):
            return {"error": "Database not found"}

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get latest round metrics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT agent_id) as agent_count,
                    AVG(fitness_score) as avg_fitness,
                    MAX(fitness_score) as max_fitness,
                    MIN(fitness_score) as min_fitness
                FROM fitness_metrics
                WHERE round_id = (SELECT MAX(id) FROM evolution_rounds)
            """)

            row = cursor.fetchone()

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "current_round": {
                    "agent_count": row[0] or 0,
                    "avg_fitness": row[1] or 0.0,
                    "max_fitness": row[2] or 0.0,
                    "min_fitness": row[3] or 0.0
                }
            }

            # Get resource usage
            cursor.execute("""
                SELECT 
                    AVG(cpu_usage) as avg_cpu,
                    AVG(memory_usage_mb) as avg_memory
                FROM resource_metrics
                WHERE round_id = (SELECT MAX(id) FROM evolution_rounds)
            """)

            row = cursor.fetchone()
            metrics["resources"] = {
                "avg_cpu_percent": row[0] or 0.0,
                "avg_memory_mb": row[1] or 0.0
            }

            conn.close()
            return metrics

        except Exception as e:
            return {"error": str(e)}

    def get_leaderboard(self):
        """Get agent leaderboard."""
        db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")

        if not os.path.exists(db_path):
            return {"error": "Database not found"}

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    agent_id,
                    AVG(fitness_score) as avg_fitness,
                    COUNT(*) as evaluation_count
                FROM fitness_metrics
                GROUP BY agent_id
                ORDER BY avg_fitness DESC
                LIMIT 10
            """)

            leaderboard = {
                "timestamp": datetime.now().isoformat(),
                "top_agents": []
            }

            for i, row in enumerate(cursor.fetchall(), 1):
                leaderboard["top_agents"].append({
                    "rank": i,
                    "agent_id": row[0],
                    "avg_fitness": row[1],
                    "evaluations": row[2]
                })

            conn.close()
            return leaderboard

        except Exception as e:
            return {"error": str(e)}

    def get_agent_metrics(self, agent_id):
        """Get metrics for specific agent."""
        db_path = os.getenv("AIVILLAGE_DB_PATH", "./data/evolution_metrics.db")

        if not os.path.exists(db_path):
            return None

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    fitness_score,
                    performance_metrics,
                    timestamp
                FROM fitness_metrics
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (agent_id,))

            rows = cursor.fetchall()
            if not rows:
                conn.close()
                return None

            agent_data = {
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "recent_metrics": []
            }

            for row in rows:
                metric = {
                    "fitness_score": row[0],
                    "timestamp": row[2]
                }
                if row[1]:
                    try:
                        metric["kpis"] = json.loads(row[1])
                    except:
                        pass
                agent_data["recent_metrics"].append(metric)

            conn.close()
            return agent_data

        except Exception as e:
            return {"error": str(e)}

    def log_message(self, format, *args):
        """Suppress default logging."""


def run_api_server(port=8081):
    """Run the evolution metrics API server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, EvolutionMetricsHandler)

    print(f"Evolution Metrics API Server starting on port {port}")
    print("Available endpoints:")
    print("  GET /health/evolution - Health check")
    print("  GET /metrics/current - Current metrics summary")
    print("  GET /metrics/leaderboard - Top agents leaderboard")
    print("  GET /metrics/agent/{agent_id} - Agent-specific metrics")
    print("\nPress Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    # Set working directory
    os.chdir(Path(__file__).parent.parent.parent)

    # Start API server
    port = int(os.getenv("EVOLUTION_METRICS_PORT", "8081"))
    run_api_server(port)
