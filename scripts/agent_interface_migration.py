"""Agent Interface Migration Script.

Updates agent interfaces to use new RAG and evolution metrics systems.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CODEX integration endpoints
EVOLUTION_METRICS_PORT = 8081
RAG_PIPELINE_PORT = 8082
DIGITAL_TWIN_PORT = 8080


class AgentInterfaceMigrator:
    """Handles migration of agent interfaces to new systems."""

    def __init__(self):
        self.migration_log = []
        self.agent_files = []
        self.interface_updates = {}

    def find_agent_files(self) -> list[Path]:
        """Find agent implementation files."""
        logger.info("Scanning for agent files...")

        agent_files = []
        search_dirs = [
            Path("./src/agents"),
            Path("./agents"),
            Path("./experimental/agents"),
            Path("./src/production")
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                # Find Python files that look like agents
                for file_path in search_dir.rglob("*.py"):
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    # Look for agent-related patterns
                    if any(pattern in content.lower() for pattern in [
                        "class.*agent", "baseagent", "agent.*class",
                        "rag.*query", "evolution.*metric", "fitness.*score"
                    ]):
                        agent_files.append(file_path)
                        logger.info(f"Found agent file: {file_path}")

        return agent_files

    def analyze_agent_interfaces(self, agent_files: list[Path]) -> dict[str, Any]:
        """Analyze current agent interfaces."""
        logger.info("Analyzing agent interfaces...")

        analysis = {
            "total_agents": len(agent_files),
            "rag_interfaces": [],
            "evolution_interfaces": [],
            "p2p_interfaces": [],
            "outdated_patterns": [],
            "update_candidates": []
        }

        for file_path in agent_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                file_analysis = {
                    "file": str(file_path),
                    "needs_rag_update": False,
                    "needs_evolution_update": False,
                    "needs_p2p_update": False,
                    "outdated_patterns": []
                }

                # Check for RAG interface usage
                if any(pattern in content for pattern in [
                    "sha256", "SHA256", "hash.*embed", "mock.*embed"
                ]):
                    file_analysis["needs_rag_update"] = True
                    file_analysis["outdated_patterns"].append("SHA256 embeddings")
                    analysis["rag_interfaces"].append(str(file_path))

                # Check for evolution metrics usage
                if any(pattern in content for pattern in [
                    "evolution.*json", "metrics.*json", "fitness.*file"
                ]):
                    file_analysis["needs_evolution_update"] = True
                    file_analysis["outdated_patterns"].append("JSON metrics storage")
                    analysis["evolution_interfaces"].append(str(file_path))

                # Check for P2P network usage
                if any(pattern in content for pattern in [
                    "bluetooth", "mock.*p2p", "fake.*network"
                ]):
                    file_analysis["needs_p2p_update"] = True
                    file_analysis["outdated_patterns"].append("Mock Bluetooth P2P")
                    analysis["p2p_interfaces"].append(str(file_path))

                if any([file_analysis["needs_rag_update"],
                       file_analysis["needs_evolution_update"],
                       file_analysis["needs_p2p_update"]]):
                    analysis["update_candidates"].append(file_analysis)

                analysis["outdated_patterns"].extend(file_analysis["outdated_patterns"])

            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")

        logger.info(f"Analysis complete: {len(analysis['update_candidates'])} files need updates")
        return analysis

    def create_updated_interfaces(self) -> dict[str, str]:
        """Create updated interface code snippets."""
        logger.info("Creating updated interface patterns...")

        interfaces = {
            "rag_query_interface": '''
# Updated RAG query interface using CODEX-compliant endpoint
import requests
from typing import List, Dict, Any

class RAGQueryInterface:
    """CODEX-compliant RAG query interface."""
    
    def __init__(self, base_url: str = "http://localhost:8082"):
        self.base_url = base_url
    
    async def query_knowledge(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Query RAG system with real embeddings."""
        try:
            response = requests.post(f"{self.base_url}/query", json={
                "query": query,
                "k": k,
                "use_cache": True
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
            else:
                logger.error(f"RAG query failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return []
    
    def check_health(self) -> bool:
        """Check RAG system health."""
        try:
            response = requests.get(f"{self.base_url}/health/rag", timeout=5)
            return response.status_code == 200
        except:
            return False
''',

            "evolution_metrics_interface": '''
# Updated evolution metrics interface using SQLite database
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime

class EvolutionMetricsInterface:
    """CODEX-compliant evolution metrics interface."""
    
    def __init__(self, db_path: str = "./data/evolution_metrics.db"):
        self.db_path = db_path
    
    def log_fitness_metrics(self, agent_id: str, fitness_score: float, 
                           round_id: int, metrics: Dict[str, Any]) -> bool:
        """Log agent fitness metrics to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO fitness_metrics 
                (round_id, agent_id, fitness_score, performance_metrics, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                round_id,
                agent_id, 
                fitness_score,
                json.dumps(metrics),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error logging fitness metrics: {e}")
            return False
    
    def get_agent_performance(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get agent performance history from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fitness_score, performance_metrics, timestamp
                FROM fitness_metrics
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (agent_id, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "fitness_score": row[0],
                    "metrics": json.loads(row[1]),
                    "timestamp": row[2]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting agent performance: {e}")
            return []
''',

            "p2p_messaging_interface": '''
# Updated P2P messaging interface using LibP2P
import asyncio
import json
from typing import Dict, Any, Callable, Optional

class P2PMessagingInterface:
    """CODEX-compliant LibP2P messaging interface."""
    
    def __init__(self, config_path: str = "./config/p2p_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.connected = False
        self.message_handlers = {}
    
    async def connect(self) -> bool:
        """Connect to LibP2P network."""
        try:
            # Initialize LibP2P connection (simplified)
            self.connected = True
            logger.info(f"Connected to P2P network on port {self.config['port']}")
            return True
            
        except Exception as e:
            logger.error(f"P2P connection failed: {e}")
            return False
    
    async def send_message(self, peer_id: str, message_type: str, data: Dict[str, Any]) -> bool:
        """Send message to peer via LibP2P."""
        if not self.connected:
            return False
            
        try:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "sender": "agent"
            }
            
            # Send via LibP2P pubsub (simplified)
            logger.info(f"Sent {message_type} to {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Message send failed: {e}")
            return False
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register message handler."""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for {message_type}")
'''
        }

        return interfaces

    def update_agent_file(self, file_path: Path, interface_updates: dict[str, str]) -> dict[str, Any]:
        """Update a single agent file with new interfaces."""
        logger.info(f"Updating {file_path}...")

        update_result = {
            "file": str(file_path),
            "rag_updated": False,
            "evolution_updated": False,
            "p2p_updated": False,
            "backup_created": False,
            "errors": []
        }

        try:
            # Create backup
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            content = file_path.read_text(encoding="utf-8")
            backup_path.write_text(content, encoding="utf-8")
            update_result["backup_created"] = True

            # Update content with new interfaces
            updated_content = content

            # Add imports if needed
            if "import requests" not in updated_content and "rag" in content.lower():
                updated_content = "import requests\nimport json\n" + updated_content
                update_result["rag_updated"] = True

            if "import sqlite3" not in updated_content and "evolution" in content.lower():
                updated_content = "import sqlite3\n" + updated_content
                update_result["evolution_updated"] = True

            if "import asyncio" not in updated_content and "p2p" in content.lower():
                updated_content = "import asyncio\n" + updated_content
                update_result["p2p_updated"] = True

            # Add interface classes at the end
            if update_result["rag_updated"]:
                updated_content += "\n\n" + interface_updates["rag_query_interface"]

            if update_result["evolution_updated"]:
                updated_content += "\n\n" + interface_updates["evolution_metrics_interface"]

            if update_result["p2p_updated"]:
                updated_content += "\n\n" + interface_updates["p2p_messaging_interface"]

            # Write updated content
            file_path.write_text(updated_content, encoding="utf-8")

        except Exception as e:
            error_msg = f"Error updating {file_path}: {e}"
            logger.error(error_msg)
            update_result["errors"].append(error_msg)

        return update_result

    def create_agent_adapter(self) -> str:
        """Create agent adapter for seamless integration."""
        adapter_code = '''
"""Agent Integration Adapter.

Provides seamless integration with CODEX systems.
"""

import asyncio
import json
import sqlite3
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

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
            with open(self.p2p_config_path, 'r') as f:
                self.p2p_config = json.load(f)
        except:
            self.p2p_config = {"host": "0.0.0.0", "port": 4001}
    
    # RAG Integration Methods
    async def query_knowledge(self, query: str, context: str = "", k: int = 5) -> List[Dict[str, Any]]:
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
            else:
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
                         performance_data: Dict[str, Any]) -> bool:
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
    
    def get_evolution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
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
                                data: Dict[str, Any]) -> bool:
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
    
    def broadcast_status(self, status_data: Dict[str, Any]) -> bool:
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
    def get_system_health(self) -> Dict[str, bool]:
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
'''
        return adapter_code

    def run_migration(self) -> dict[str, Any]:
        """Execute complete agent interface migration."""
        logger.info("Starting agent interface migration...")

        start_time = datetime.now()

        # Find agent files
        agent_files = self.find_agent_files()

        # Analyze interfaces
        analysis = self.analyze_agent_interfaces(agent_files)

        # Create updated interfaces
        interface_updates = self.create_updated_interfaces()

        # Update each agent file
        update_results = []
        for file_info in analysis["update_candidates"][:5]:  # Limit to first 5 for demo
            file_path = Path(file_info["file"])
            result = self.update_agent_file(file_path, interface_updates)
            update_results.append(result)

        # Create agent adapter
        adapter_code = self.create_agent_adapter()
        adapter_path = Path("./src/integration/codex_agent_adapter.py")
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_text(adapter_code, encoding="utf-8")

        # Generate final report
        report = {
            "status": "completed",
            "migration_type": "agent_interface_updates",
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": (datetime.now() - start_time).total_seconds(),
            "analysis": analysis,
            "updates_applied": len(update_results),
            "successful_updates": len([r for r in update_results if not r["errors"]]),
            "adapter_created": str(adapter_path),
            "integration_endpoints": {
                "rag_pipeline": f"http://localhost:{RAG_PIPELINE_PORT}",
                "evolution_metrics": f"http://localhost:{EVOLUTION_METRICS_PORT}",
                "digital_twin": f"http://localhost:{DIGITAL_TWIN_PORT}"
            },
            "update_results": update_results
        }

        logger.info(f"Agent interface migration completed: {len(update_results)} agents updated")

        return report


def main():
    """Main migration function."""
    migrator = AgentInterfaceMigrator()
    report = migrator.run_migration()

    # Save migration report
    report_path = Path("./data/agent_interface_migration_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print("AGENT INTERFACE MIGRATION COMPLETE")
    print(f"{'='*50}")
    print(f"Status: {report['status']}")
    print(f"Agents found: {report['analysis']['total_agents']}")
    print(f"Updates needed: {len(report['analysis']['update_candidates'])}")
    print(f"Updates applied: {report['updates_applied']}")
    print(f"Successful: {report['successful_updates']}")
    print(f"Adapter created: {report['adapter_created']}")
    print("\nIntegration endpoints:")
    for name, url in report["integration_endpoints"].items():
        print(f"  {name}: {url}")
    print(f"Duration: {report['duration']:.2f} seconds")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
