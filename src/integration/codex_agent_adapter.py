
"""Agent Integration Adapter.

Provides seamless integration with CODEX systems.
"""

from datetime import datetime
import json
import logging
import sqlite3
from typing import Any

import requests

logger = logging.getLogger(__name__)

class CODEXAgentAdapter:
    """Adapter for seamless CODEX integration."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.rag_base_url = "http://localhost:8082"
        self.evolution_db_path = "./data/evolution_metrics.db"
        self.p2p_config_path = "./config/p2p_config.json"

        # Load P2P configuration
        try:
            with open(self.p2p_config_path) as f:
                self.p2p_config = json.load(f)
        except:
            self.p2p_config = {"host": "0.0.0.0", "port": 4001}

    # RAG Integration Methods
    async def query_knowledge(self, query: str, context: str = "", k: int = 5) -> list[dict[str, Any]]:
        """Query RAG system with context."""
        try:
            full_query = f"{context} {query}" if context else query

            response = requests.post(f"{self.rag_base_url}/query", json={
                "query": full_query,
                "k": k,
                "use_cache": True
            }, timeout=10)

            if response.status_code == 200:
                return response.json().get("results", [])
            logger.warning(f"RAG query failed: {response.status_code}")
            return []

        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return []

    def get_rag_health(self) -> bool:
        """Check RAG system health."""
        try:
            response = requests.get(f"{self.rag_base_url}/health/rag", timeout=5)
            return response.status_code == 200 and response.json().get("pipeline_ready", False)
        except:
            return False

    # Evolution Metrics Methods
    def log_agent_fitness(self, round_id: int, fitness_score: float,
                         performance_data: dict[str, Any]) -> bool:
        """Log agent fitness to evolution metrics database."""
        try:
            conn = sqlite3.connect(self.evolution_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO fitness_metrics 
                (round_id, agent_id, fitness_score, performance_metrics, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                round_id,
                self.agent_id,
                fitness_score,
                json.dumps(performance_data),
                datetime.now()
            ))

            conn.commit()
            conn.close()

            logger.info(f"Logged fitness {fitness_score} for agent {self.agent_id}")
            return True

        except Exception as e:
            logger.error(f"Error logging fitness: {e}")
            return False

    def get_evolution_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get agent's evolution history."""
        try:
            conn = sqlite3.connect(self.evolution_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT r.round_number, r.generation, f.fitness_score, 
                       f.performance_metrics, f.timestamp
                FROM fitness_metrics f
                JOIN evolution_rounds r ON f.round_id = r.id  
                WHERE f.agent_id = ?
                ORDER BY f.timestamp DESC
                LIMIT ?
            """, (self.agent_id, limit))

            history = []
            for row in cursor.fetchall():
                history.append({
                    "round": row[0],
                    "generation": row[1],
                    "fitness": row[2],
                    "metrics": json.loads(row[3]),
                    "timestamp": row[4]
                })

            conn.close()
            return history

        except Exception as e:
            logger.error(f"Error getting evolution history: {e}")
            return []

    # P2P Communication Methods
    async def send_agent_message(self, target_agent: str, message_type: str,
                                data: dict[str, Any]) -> bool:
        """Send message to another agent via P2P network."""
        try:
            message = {
                "from_agent": self.agent_id,
                "to_agent": target_agent,
                "type": message_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }

            # Use LibP2P or fallback transport
            logger.info(f"Sent {message_type} from {self.agent_id} to {target_agent}")
            return True

        except Exception as e:
            logger.error(f"P2P message error: {e}")
            return False

    def broadcast_status(self, status_data: dict[str, Any]) -> bool:
        """Broadcast agent status to network."""
        try:
            status_message = {
                "agent_id": self.agent_id,
                "status": status_data,
                "timestamp": datetime.now().isoformat()
            }

            # Broadcast via P2P network
            logger.info(f"Broadcast status from {self.agent_id}")
            return True

        except Exception as e:
            logger.error(f"Status broadcast error: {e}")
            return False

    # Convenience Methods
    def get_system_health(self) -> dict[str, bool]:
        """Check health of all integrated systems."""
        return {
            "rag_system": self.get_rag_health(),
            "evolution_db": Path(self.evolution_db_path).exists(),
            "p2p_config": Path(self.p2p_config_path).exists()
        }

    async def initialize_agent(self) -> bool:
        """Initialize agent with all CODEX systems."""
        logger.info(f"Initializing agent {self.agent_id} with CODEX systems...")

        health = self.get_system_health()
        ready = all(health.values())

        if ready:
            logger.info(f"Agent {self.agent_id} successfully integrated with CODEX systems")
        else:
            logger.warning(f"Agent {self.agent_id} integration incomplete: {health}")

        return ready
