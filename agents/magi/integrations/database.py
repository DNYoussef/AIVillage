"""Database integration for MAGI."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import logging
import json
import sqlite3
import aiosqlite
from pathlib import Path

from ..core.exceptions import DatabaseError
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    """
    Manages database interactions for MAGI.
    
    Responsibilities:
    - Tool persistence
    - Execution history
    - Performance metrics
    - System state
    - Analytics data
    """
    
    def __init__(self, db_path: Union[str, Path] = "magi.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._ensure_tables()
    
    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tools table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    code TEXT,
                    parameters TEXT,
                    version TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0
                )
            """)
            
            # Tool executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tool_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT,
                    parameters TEXT,
                    result TEXT,
                    success BOOLEAN,
                    execution_time REAL,
                    error TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (tool_name) REFERENCES tools (name)
                )
            """)
            
            # Techniques table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS techniques (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    parameters TEXT,
                    version TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0
                )
            """)
            
            # Technique executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS technique_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    technique_name TEXT,
                    thought TEXT,
                    result TEXT,
                    confidence REAL,
                    execution_time REAL,
                    error TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (technique_name) REFERENCES techniques (name)
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT,
                    target_type TEXT,
                    target_name TEXT,
                    value REAL,
                    timestamp TIMESTAMP
                )
            """)
            
            # System state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT,
                    state TEXT,
                    timestamp TIMESTAMP
                )
            """)
            
            # Analytics data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT,
                    data TEXT,
                    timestamp TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def store_tool(self, tool_data: Dict[str, Any]) -> None:
        """
        Store tool data in database.
        
        Args:
            tool_data: Tool data to store
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO tools (
                    name, description, code, parameters, version,
                    created_at, updated_at, usage_count, success_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_data['name'],
                    tool_data['description'],
                    tool_data['code'],
                    json.dumps(tool_data['parameters']),
                    tool_data['version'],
                    tool_data.get('created_at', datetime.now()),
                    datetime.now(),
                    tool_data.get('usage_count', 0),
                    tool_data.get('success_rate', 0.0)
                )
            )
            await db.commit()
    
    async def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get tool data from database.
        
        Args:
            name: Tool name
            
        Returns:
            Tool data if found, None otherwise
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM tools WHERE name = ?",
                (name,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    return {
                        'name': row[0],
                        'description': row[1],
                        'code': row[2],
                        'parameters': json.loads(row[3]),
                        'version': row[4],
                        'created_at': row[5],
                        'updated_at': row[6],
                        'usage_count': row[7],
                        'success_rate': row[8]
                    }
                return None
    
    async def store_tool_execution(
        self,
        execution_data: Dict[str, Any]
    ) -> None:
        """
        Store tool execution data.
        
        Args:
            execution_data: Execution data to store
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO tool_executions (
                    tool_name, parameters, result, success,
                    execution_time, error, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_data['tool_name'],
                    json.dumps(execution_data['parameters']),
                    json.dumps(execution_data['result']),
                    execution_data['success'],
                    execution_data['execution_time'],
                    execution_data.get('error'),
                    execution_data.get('timestamp', datetime.now())
                )
            )
            await db.commit()
    
    async def store_technique(self, technique_data: Dict[str, Any]) -> None:
        """
        Store technique data in database.
        
        Args:
            technique_data: Technique data to store
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO techniques (
                    name, description, parameters, version,
                    created_at, updated_at, usage_count, success_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    technique_data['name'],
                    technique_data['description'],
                    json.dumps(technique_data['parameters']),
                    technique_data['version'],
                    technique_data.get('created_at', datetime.now()),
                    datetime.now(),
                    technique_data.get('usage_count', 0),
                    technique_data.get('success_rate', 0.0)
                )
            )
            await db.commit()
    
    async def store_technique_execution(
        self,
        execution_data: Dict[str, Any]
    ) -> None:
        """
        Store technique execution data.
        
        Args:
            execution_data: Execution data to store
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO technique_executions (
                    technique_name, thought, result, confidence,
                    execution_time, error, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_data['technique_name'],
                    execution_data['thought'],
                    json.dumps(execution_data['result']),
                    execution_data['confidence'],
                    execution_data['execution_time'],
                    execution_data.get('error'),
                    execution_data.get('timestamp', datetime.now())
                )
            )
            await db.commit()
    
    async def store_metric(
        self,
        metric_type: str,
        target_type: str,
        target_name: str,
        value: float
    ) -> None:
        """
        Store performance metric.
        
        Args:
            metric_type: Type of metric
            target_type: Type of target (tool, technique, system)
            target_name: Name of target
            value: Metric value
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO performance_metrics (
                    metric_type, target_type, target_name, value, timestamp
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    metric_type,
                    target_type,
                    target_name,
                    value,
                    datetime.now()
                )
            )
            await db.commit()
    
    async def store_system_state(
        self,
        component: str,
        state: Dict[str, Any]
    ) -> None:
        """
        Store system state.
        
        Args:
            component: System component
            state: Component state
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO system_state (component, state, timestamp)
                VALUES (?, ?, ?)
                """,
                (
                    component,
                    json.dumps(state),
                    datetime.now()
                )
            )
            await db.commit()
    
    async def store_analytics(
        self,
        category: str,
        data: Dict[str, Any]
    ) -> None:
        """
        Store analytics data.
        
        Args:
            category: Analytics category
            data: Analytics data
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO analytics_data (category, data, timestamp)
                VALUES (?, ?, ?)
                """,
                (
                    category,
                    json.dumps(data),
                    datetime.now()
                )
            )
            await db.commit()
    
    async def get_tool_history(
        self,
        tool_name: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution history for a tool.
        
        Args:
            tool_name: Tool name
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT * FROM tool_executions WHERE tool_name = ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with db.execute(query, (tool_name,)) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    {
                        'tool_name': row[1],
                        'parameters': json.loads(row[2]),
                        'result': json.loads(row[3]),
                        'success': row[4],
                        'execution_time': row[5],
                        'error': row[6],
                        'timestamp': row[7]
                    }
                    for row in rows
                ]
    
    async def get_technique_history(
        self,
        technique_name: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get execution history for a technique.
        
        Args:
            technique_name: Technique name
            limit: Maximum number of records to return
            
        Returns:
            List of execution records
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT * FROM technique_executions WHERE technique_name = ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with db.execute(query, (technique_name,)) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    {
                        'technique_name': row[1],
                        'thought': row[2],
                        'result': json.loads(row[3]),
                        'confidence': row[4],
                        'execution_time': row[5],
                        'error': row[6],
                        'timestamp': row[7]
                    }
                    for row in rows
                ]
    
    async def get_metrics(
        self,
        metric_type: Optional[str] = None,
        target_type: Optional[str] = None,
        target_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get performance metrics.
        
        Args:
            metric_type: Filter by metric type (optional)
            target_type: Filter by target type (optional)
            target_name: Filter by target name (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of metric records
        """
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT * FROM performance_metrics WHERE 1=1"
            params = []
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)
            if target_type:
                query += " AND target_type = ?"
                params.append(target_type)
            if target_name:
                query += " AND target_name = ?"
                params.append(target_name)
            
            query += " ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                
                return [
                    {
                        'metric_type': row[1],
                        'target_type': row[2],
                        'target_name': row[3],
                        'value': row[4],
                        'timestamp': row[5]
                    }
                    for row in rows
                ]

# Example usage
if __name__ == "__main__":
    async def main():
        # Create database manager
        db = DatabaseManager("magi.db")
        
        # Store a tool
        await db.store_tool({
            'name': 'example_tool',
            'description': 'An example tool',
            'code': 'def example(): pass',
            'parameters': {'param1': {'type': 'str'}},
            'version': '1.0.0'
        })
        
        # Store a tool execution
        await db.store_tool_execution({
            'tool_name': 'example_tool',
            'parameters': {'param1': 'value1'},
            'result': {'output': 'success'},
            'success': True,
            'execution_time': 0.1
        })
        
        # Get tool history
        history = await db.get_tool_history('example_tool', limit=10)
        print(f"Tool history: {history}")
    
    asyncio.run(main())
