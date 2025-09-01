"""
MCP Server Coordinator - Memory Coordination Specialist Agent
Manages distributed memory system and facilitates information sharing between agents.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import logging

from .mcp_server_knowledge_base import MCPKnowledgeBase

@dataclass
class MemoryEntry:
    """Represents a memory entry in the distributed system"""
    key: str
    value: Any
    namespace: str
    ttl: Optional[datetime] = None
    encrypted: bool = False
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

class MCPServerCoordinator:
    """
    Memory Coordination Specialist Agent
    Manages distributed memory system and enables knowledge persistence across sessions
    """
    
    def __init__(self, storage_dir: str = "C:/Users/17175/Desktop/AIVillage/.mcp"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_db_path = self.storage_dir / "memory.db"
        self.knowledge_base = MCPKnowledgeBase()
        self.logger = self._setup_logging()
        
        self._initialize_memory_db()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the coordinator"""
        logger = logging.getLogger("mcp_coordinator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler(self.storage_dir / "coordinator.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_memory_db(self):
        """Initialize the SQLite memory database"""
        with sqlite3.connect(self.memory_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    namespace TEXT,
                    ttl TEXT,
                    encrypted INTEGER DEFAULT 0,
                    tags TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_namespace ON memory_entries(namespace)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags ON memory_entries(tags)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_coordination (
                    session_id TEXT,
                    agent_id TEXT,
                    task_id TEXT,
                    status TEXT,
                    data TEXT,
                    timestamp TEXT,
                    PRIMARY KEY (session_id, agent_id, task_id)
                )
            """)
    
    # Core Memory Operations
    async def store_memory(self, key: str, value: Any, namespace: str = "default", 
                          ttl: Optional[int] = None, encrypted: bool = False, 
                          tags: List[str] = None) -> bool:
        """Store data in the distributed memory system"""
        try:
            entry = MemoryEntry(
                key=key,
                value=json.dumps(value) if not isinstance(value, str) else value,
                namespace=namespace,
                ttl=datetime.now() + timedelta(seconds=ttl) if ttl else None,
                encrypted=encrypted,
                tags=tags or [],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_entries 
                    (key, value, namespace, ttl, encrypted, tags, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.value,
                    entry.namespace,
                    entry.ttl.isoformat() if entry.ttl else None,
                    1 if entry.encrypted else 0,
                    json.dumps(entry.tags),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat()
                ))
            
            self.logger.info(f"Stored memory: {key} in namespace: {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store memory {key}: {e}")
            return False
    
    async def retrieve_memory(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Retrieve data from the distributed memory system"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute("""
                    SELECT value, ttl, encrypted FROM memory_entries 
                    WHERE key = ? AND namespace = ?
                """, (key, namespace))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                value, ttl_str, encrypted = result
                
                # Check if entry has expired
                if ttl_str:
                    ttl = datetime.fromisoformat(ttl_str)
                    if datetime.now() > ttl:
                        await self.delete_memory(key, namespace)
                        return None
                
                # Handle decryption if needed
                if encrypted:
                    # TODO: Implement encryption/decryption
                    pass
                
                # Try to parse JSON, return as string if it fails
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
                    
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {key}: {e}")
            return None
    
    async def search_memory(self, pattern: str, namespace: str = None) -> List[Dict[str, Any]]:
        """Search for memories using patterns"""
        try:
            query = "SELECT key, value, namespace, tags FROM memory_entries WHERE key LIKE ?"
            params = [f"%{pattern}%"]
            
            if namespace:
                query += " AND namespace = ?"
                params.append(namespace)
                
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute(query, params)
                results = []
                
                for key, value, ns, tags_str in cursor.fetchall():
                    try:
                        parsed_value = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        parsed_value = value
                    
                    results.append({
                        "key": key,
                        "value": parsed_value,
                        "namespace": ns,
                        "tags": json.loads(tags_str) if tags_str else []
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to search memory with pattern {pattern}: {e}")
            return []
    
    async def delete_memory(self, key: str, namespace: str = "default") -> bool:
        """Delete memory entry"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("DELETE FROM memory_entries WHERE key = ? AND namespace = ?", (key, namespace))
            
            self.logger.info(f"Deleted memory: {key} from namespace: {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {key}: {e}")
            return False
    
    # Namespace Management
    async def create_namespace(self, namespace: str, description: str = "") -> bool:
        """Create a new namespace for memory organization"""
        metadata = {
            "description": description,
            "created_at": datetime.now().isoformat(),
            "entry_count": 0
        }
        
        return await self.store_memory(
            f"_namespace_metadata_{namespace}",
            metadata,
            namespace="system"
        )
    
    async def list_namespaces(self) -> List[str]:
        """List all available namespaces"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT namespace FROM memory_entries")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Failed to list namespaces: {e}")
            return []
    
    # Agent Coordination
    async def coordinate_agents(self, session_id: str, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate multiple agents for a task"""
        coordination_data = {
            "session_id": session_id,
            "agents": agents,
            "started_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Store coordination metadata
        await self.store_memory(
            f"session_{session_id}_coordination",
            coordination_data,
            namespace="coordination"
        )
        
        # Initialize each agent's workspace
        for agent in agents:
            agent_workspace = {
                "agent_id": agent["id"],
                "agent_type": agent["type"],
                "assigned_tasks": agent.get("tasks", []),
                "status": "initialized",
                "started_at": datetime.now().isoformat()
            }
            
            await self.store_memory(
                f"agent_{agent['id']}_workspace",
                agent_workspace,
                namespace=f"session_{session_id}"
            )
        
        return coordination_data
    
    async def update_agent_status(self, session_id: str, agent_id: str, 
                                 status: str, data: Dict[str, Any] = None) -> bool:
        """Update agent status and progress"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO agent_coordination 
                    (session_id, agent_id, task_id, status, data, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    agent_id,
                    f"status_update_{datetime.now().timestamp()}",
                    status,
                    json.dumps(data) if data else None,
                    datetime.now().isoformat()
                ))
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update agent status: {e}")
            return False
    
    # MCP Server Integration
    async def recommend_mcp_servers(self, task_type: str, requirements: List[str] = None) -> Dict[str, Any]:
        """Recommend optimal MCP servers for a specific task"""
        recommended_servers = self.knowledge_base.recommend_servers_for_task(task_type)
        
        recommendations = {
            "primary_servers": recommended_servers,
            "integration_pattern": self.knowledge_base.get_integration_pattern(f"{task_type}_stack"),
            "best_practices": self.knowledge_base.get_best_practices("multi_agent_coordination"),
            "performance_considerations": []
        }
        
        # Add performance considerations based on requirements
        if requirements:
            for req in requirements:
                if "speed" in req.lower():
                    recommendations["performance_considerations"].append("high_speed_required")
                elif "concurrent" in req.lower():
                    recommendations["performance_considerations"].append("high_concurrency")
        
        # Store recommendation for future learning
        await self.store_memory(
            f"recommendation_{task_type}_{datetime.now().timestamp()}",
            recommendations,
            namespace="recommendations"
        )
        
        return recommendations
    
    # Performance Analytics
    async def get_memory_analytics(self) -> Dict[str, Any]:
        """Get memory usage analytics"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                # Get namespace statistics
                cursor = conn.execute("""
                    SELECT namespace, COUNT(*) as count, 
                           MIN(created_at) as oldest, 
                           MAX(updated_at) as newest
                    FROM memory_entries 
                    GROUP BY namespace
                """)
                
                namespace_stats = []
                total_entries = 0
                
                for namespace, count, oldest, newest in cursor.fetchall():
                    namespace_stats.append({
                        "namespace": namespace,
                        "entries": count,
                        "oldest_entry": oldest,
                        "newest_entry": newest
                    })
                    total_entries += count
                
                # Get expired entries count
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM memory_entries 
                    WHERE ttl IS NOT NULL AND ttl < ?
                """, (datetime.now().isoformat(),))
                
                expired_count = cursor.fetchone()[0]
                
                return {
                    "total_entries": total_entries,
                    "expired_entries": expired_count,
                    "namespace_statistics": namespace_stats,
                    "database_size_mb": self.memory_db_path.stat().st_size / (1024 * 1024),
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get memory analytics: {e}")
            return {}
    
    # Cleanup and Maintenance
    async def cleanup_expired_memories(self) -> int:
        """Clean up expired memory entries"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM memory_entries 
                    WHERE ttl IS NOT NULL AND ttl < ?
                """, (datetime.now().isoformat(),))
                
                cleaned_count = cursor.rowcount
                self.logger.info(f"Cleaned up {cleaned_count} expired memory entries")
                return cleaned_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired memories: {e}")
            return 0
    
    async def optimize_memory_db(self) -> bool:
        """Optimize the memory database"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
            
            self.logger.info("Memory database optimized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to optimize memory database: {e}")
            return False

# Usage example and initialization
async def main():
    """Example usage of the MCP Server Coordinator"""
    coordinator = MCPServerCoordinator()
    
    # Store some sample data
    await coordinator.store_memory("project_config", {"name": "AIVillage", "type": "multi_agent"}, "project")
    await coordinator.store_memory("agent_patterns", ["researcher", "coder", "tester"], "patterns")
    
    # Coordinate agents for a development task
    agents = [
        {"id": "researcher_01", "type": "researcher", "tasks": ["analyze_requirements"]},
        {"id": "coder_01", "type": "coder", "tasks": ["implement_features"]},
        {"id": "tester_01", "type": "tester", "tasks": ["create_tests"]}
    ]
    
    coordination = await coordinator.coordinate_agents("dev_session_001", agents)
    print(f"Coordination initialized: {coordination}")
    
    # Get MCP server recommendations
    recommendations = await coordinator.recommend_mcp_servers("code_development", ["speed", "concurrent"])
    print(f"Recommended servers: {recommendations}")
    
    # Get analytics
    analytics = await coordinator.get_memory_analytics()
    print(f"Memory analytics: {analytics}")

if __name__ == "__main__":
    asyncio.run(main())