"""Enhanced data collection and storage system for AI Village."""

import logging
import json
import sqlite3
import shutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import aiosqlite
from config.unified_config import UnifiedConfig, DatabaseConfig

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize DatabaseManager.
        
        Args:
            db_config: Database configuration
        """
        self.db_path = Path(db_config.path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_interval = db_config.backup_interval
        self.max_backup_count = db_config.max_backup_count
        self.vacuum_threshold = db_config.vacuum_threshold
        self.maintenance_task = None
        
        logger.info(f"Initialized DatabaseManager with database at {self.db_path}")
    
    async def initialize(self):
        """Initialize database schema and start maintenance."""
        try:
            await self._init_database()
            await self._start_maintenance()
            logger.info("Successfully initialized DatabaseManager")
        except Exception as e:
            logger.error(f"Error initializing DatabaseManager: {str(e)}")
            raise
    
    async def _init_database(self):
        """Initialize database schema."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            # Create interactions table with additional fields
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    was_complex BOOLEAN NOT NULL,
                    performance_metrics TEXT,
                    metadata TEXT,
                    token_usage INTEGER,
                    response_time REAL,
                    error_count INTEGER DEFAULT 0,
                    feedback_score REAL
                )
            """)
            
            # Create training_data table with enhanced fields
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    frontier_model TEXT NOT NULL,
                    local_model TEXT NOT NULL,
                    input TEXT NOT NULL,
                    output TEXT NOT NULL,
                    quality_score REAL,
                    timestamp REAL NOT NULL,
                    metadata TEXT,
                    training_status TEXT DEFAULT 'pending',
                    validation_score REAL,
                    iteration_number INTEGER DEFAULT 0
                )
            """)
            
            # Create performance_metrics table with detailed tracking
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    context TEXT,
                    aggregation_period TEXT,
                    confidence_score REAL
                )
            """)
            
            # Create backup_history table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS backup_history (
                    id INTEGER PRIMARY KEY,
                    backup_path TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            
            # Create indexes for better query performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_agent_type ON interactions(agent_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_training_status ON training_data(training_status)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_composite ON performance_metrics(agent_type, model_type, metric_name)")
            
            await conn.commit()
    
    async def _start_maintenance(self):
        """Start the maintenance loop."""
        async def maintenance_loop():
            while True:
                try:
                    await self.create_backup()
                    await self.run_maintenance()
                    await asyncio.sleep(self.backup_interval * 3600)
                except Exception as e:
                    logger.error(f"Error in maintenance loop: {str(e)}")
                    await asyncio.sleep(300)
        
        self.maintenance_task = asyncio.create_task(maintenance_loop())
    
    async def shutdown(self):
        """Shutdown the database manager."""
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
    
    async def store_interaction(self, agent_type: str, interaction: Dict[str, Any], was_complex: bool = False):
        """Store an interaction in the database."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute("""
                INSERT INTO interactions 
                (agent_type, model_used, timestamp, prompt, response, was_complex,
                 metadata, token_usage, response_time, error_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_type,
                interaction.get("model", "unknown"),
                interaction.get("timestamp", datetime.now().timestamp()),
                interaction.get("prompt", ""),
                interaction.get("response", ""),
                was_complex,
                json.dumps(interaction.get("metadata", {})),
                interaction.get("metadata", {}).get("usage", {}).get("total_tokens", 0),
                interaction.get("metadata", {}).get("response_time", 0),
                interaction.get("metadata", {}).get("error_count", 0)
            ))
            await conn.commit()
    
    async def store_training_example(self, agent_type: str, frontier_model: str, local_model: str, example: Dict[str, Any], quality_score: float):
        """Store a training example in the database."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            await conn.execute("""
                INSERT INTO training_data
                (agent_type, frontier_model, local_model, input, output,
                 quality_score, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_type,
                frontier_model,
                local_model,
                example["input"],
                example["output"],
                quality_score,
                datetime.now().timestamp(),
                json.dumps(example.get("metadata", {}))
            ))
            await conn.commit()
    
    async def store_performance_metrics(self, agent_type: str, model_type: str, metrics: Dict[str, float], context: Optional[Dict[str, Any]] = None):
        """Store performance metrics in the database."""
        async with aiosqlite.connect(str(self.db_path)) as conn:
            for metric_name, metric_value in metrics.items():
                await conn.execute("""
                    INSERT INTO performance_metrics
                    (agent_type, model_type, metric_name, metric_value, timestamp,
                     context, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_type,
                    model_type,
                    metric_name,
                    metric_value,
                    datetime.now().timestamp(),
                    json.dumps(context) if context else None,
                    0.8  # Default confidence score
                ))
            await conn.commit()
    
    async def get_interactions(self, agent_type: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get stored interactions."""
        query = "SELECT * FROM interactions"
        params = []
        
        if agent_type:
            query += " WHERE agent_type = ?"
            params.append(agent_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_training_data(self, agent_type: Optional[str] = None, min_quality: Optional[float] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get training data."""
        query = "SELECT * FROM training_data WHERE 1=1"
        params = []
        
        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type)
            
        if min_quality is not None:
            query += " AND quality_score >= ?"
            params.append(min_quality)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_performance_metrics(self, agent_type: str, model_type: str, metric_names: List[str]) -> Dict[str, float]:
        """Get performance metrics."""
        placeholders = ','.join('?' * len(metric_names))
        query = f"""
            SELECT metric_name, metric_value
            FROM performance_metrics
            WHERE agent_type = ?
            AND model_type = ?
            AND metric_name IN ({placeholders})
            AND timestamp = (
                SELECT MAX(timestamp)
                FROM performance_metrics
                WHERE agent_type = ?
                AND model_type = ?
                AND metric_name = performance_metrics.metric_name
            )
        """
        params = [agent_type, model_type] + metric_names + [agent_type, model_type]
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return {row[0]: row[1] for row in rows}
    
    async def create_backup(self) -> str:
        """Create a database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"backups/agent_data_{timestamp}.db"
        backup_path.parent.mkdir(exist_ok=True)
        
        shutil.copy2(str(self.db_path), str(backup_path))
        return str(backup_path)
    
    async def run_maintenance(self) -> int:
        """Run database maintenance."""
        cutoff_time = datetime.now() - timedelta(days=30)
        cutoff_timestamp = cutoff_time.timestamp()
        
        async with aiosqlite.connect(str(self.db_path)) as conn:
            async with conn.execute(
                "DELETE FROM interactions WHERE timestamp < ?",
                (cutoff_timestamp,)
            ) as cursor:
                cleaned_count = cursor.rowcount
            
            await conn.commit()
            return cleaned_count

class DataCollector:
    """Enhanced system for collecting and storing agent data."""
    
    def __init__(self, config: UnifiedConfig):
        """Initialize DataCollector."""
        self.config = config
        self.db_manager = DatabaseManager(config.get_db_config())
        self.performance_config = config.get_performance_config()
        
        logger.info("Initialized DataCollector")
    
    async def initialize(self):
        """Initialize the data collector and its components."""
        try:
            logger.info("Initializing DataCollector...")
            await self.db_manager.initialize()
            logger.info("Successfully initialized DataCollector")
        except Exception as e:
            logger.error(f"Error initializing DataCollector: {str(e)}")
            raise
    
    async def shutdown(self):
        """Shutdown the data collector."""
        try:
            await self.db_manager.shutdown()
            logger.info("Successfully shut down DataCollector")
        except Exception as e:
            logger.error(f"Error shutting down DataCollector: {str(e)}")
            raise
    
    async def store_interaction(self, agent_type: str, interaction: Dict[str, Any], was_complex: bool = False):
        """Store an agent interaction."""
        try:
            await self.db_manager.store_interaction(agent_type, interaction, was_complex)
        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")
            raise
    
    async def store_training_example(self, agent_type: str, frontier_model: str, local_model: str, example: Dict[str, Any], quality_score: float):
        """Store a training example."""
        try:
            await self.db_manager.store_training_example(agent_type, frontier_model, local_model, example, quality_score)
        except Exception as e:
            logger.error(f"Error storing training example: {str(e)}")
            raise
    
    async def store_performance_metrics(self, agent_type: str, model_type: str, metrics: Dict[str, float], context: Optional[Dict[str, Any]] = None):
        """Store performance metrics."""
        try:
            await self.db_manager.store_performance_metrics(agent_type, model_type, metrics, context)
        except Exception as e:
            logger.error(f"Error storing performance metrics: {str(e)}")
            raise
    
    async def get_interactions(self, agent_type: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get stored interactions."""
        try:
            return await self.db_manager.get_interactions(agent_type, limit)
        except Exception as e:
            logger.error(f"Error retrieving interactions: {str(e)}")
            raise
    
    async def get_training_data(self, agent_type: Optional[str] = None, min_quality: Optional[float] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get training data."""
        try:
            return await self.db_manager.get_training_data(agent_type, min_quality, limit)
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            raise
    
    async def get_performance_metrics(self, agent_type: str, model_type: str, metric_names: List[str]) -> Dict[str, float]:
        """Get performance metrics."""
        try:
            return await self.db_manager.get_performance_metrics(agent_type, model_type, metric_names)
        except Exception as e:
            logger.error(f"Error retrieving performance metrics: {str(e)}")
            raise
    
    async def run_maintenance(self) -> int:
        """Run database maintenance."""
        try:
            return await self.db_manager.run_maintenance()
        except Exception as e:
            logger.error(f"Error running maintenance: {str(e)}")
            raise
