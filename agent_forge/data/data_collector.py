"""Enhanced data collection and storage system for AI Village."""

import logging
import json
import sqlite3
import shutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
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
        
        self._init_database()
        self._schedule_maintenance()
        
        logger.info(f"Initialized DatabaseManager with database at {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Create interactions table with additional fields
            conn.execute("""
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    frontier_model TEXT NOT NULL,
                    local_model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    quality_score REAL,
                    timestamp REAL NOT NULL,
                    performance_metrics TEXT,
                    training_status TEXT DEFAULT 'pending',
                    validation_score REAL,
                    iteration_number INTEGER DEFAULT 0
                )
            """)
            
            # Create performance_metrics table with detailed tracking
            conn.execute("""
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
            conn.execute("""
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_agent_type ON interactions(agent_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_status ON training_data(training_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_composite ON performance_metrics(agent_type, model_type, metric_name)")
            
            conn.commit()
    
    def _schedule_maintenance(self):
        """Schedule database maintenance tasks."""
        async def maintenance_loop():
            while True:
                try:
                    # Create backup
                    await self.create_backup()
                    
                    # Clean old backups
                    await self.clean_old_backups()
                    
                    # Vacuum database if needed
                    await self.vacuum_if_needed()
                    
                    # Wait for next maintenance interval
                    await asyncio.sleep(self.backup_interval * 3600)  # Convert hours to seconds
                    
                except Exception as e:
                    logger.error(f"Error in maintenance loop: {str(e)}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
        
        # Start maintenance loop
        asyncio.create_task(maintenance_loop())
    
    async def create_backup(self):
        """Create a database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.parent / f"backups/agent_data_{timestamp}.db"
        backup_path.parent.mkdir(exist_ok=True)
        
        try:
            # Create backup
            shutil.copy2(str(self.db_path), str(backup_path))
            
            # Record backup in history
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO backup_history 
                    (backup_path, timestamp, size_bytes, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    str(backup_path),
                    datetime.now().timestamp(),
                    backup_path.stat().st_size,
                    "success",
                    None
                ))
                conn.commit()
            
            logger.info(f"Created database backup at {backup_path}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Backup creation failed: {error_msg}")
            
            # Record failed backup attempt
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO backup_history 
                    (backup_path, timestamp, size_bytes, status, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    str(backup_path),
                    datetime.now().timestamp(),
                    0,
                    "failed",
                    error_msg
                ))
                conn.commit()
    
    async def clean_old_backups(self):
        """Remove old backups exceeding max_backup_count."""
        backup_dir = self.db_path.parent / "backups"
        if not backup_dir.exists():
            return
            
        backups = sorted(backup_dir.glob("agent_data_*.db"), key=lambda x: x.stat().st_mtime)
        
        # Remove old backups
        while len(backups) > self.max_backup_count:
            backup_to_remove = backups.pop(0)
            try:
                backup_to_remove.unlink()
                logger.info(f"Removed old backup: {backup_to_remove}")
            except Exception as e:
                logger.error(f"Error removing backup {backup_to_remove}: {str(e)}")
    
    async def vacuum_if_needed(self):
        """Vacuum database if it exceeds the threshold."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Check row counts
                cursor = conn.execute("SELECT COUNT(*) FROM interactions")
                interaction_count = cursor.fetchone()[0]
                
                if interaction_count > self.vacuum_threshold:
                    logger.info("Running database vacuum...")
                    conn.execute("VACUUM")
                    logger.info("Database vacuum completed")
        
        except Exception as e:
            logger.error(f"Error during database vacuum: {str(e)}")

class DataCollector:
    """Enhanced system for collecting and storing agent data."""
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize DataCollector.
        
        Args:
            config: Unified configuration instance
        """
        self.db_manager = DatabaseManager(config.get_db_config())
        self.performance_config = config.get_performance_config()
        
        logger.info("Initialized DataCollector")
    
    async def store_interaction(self, 
                              agent_type: str,
                              interaction: Dict[str, Any],
                              was_complex: bool,
                              performance_metrics: Optional[Dict[str, float]] = None):
        """
        Store an agent interaction with enhanced metadata.
        
        Args:
            agent_type: Type of agent
            interaction: Interaction data
            was_complex: Whether the task was complex
            performance_metrics: Optional performance metrics
        """
        try:
            with sqlite3.connect(str(self.db_manager.db_path)) as conn:
                conn.execute("""
                    INSERT INTO interactions 
                    (agent_type, model_used, timestamp, prompt, response, was_complex,
                     performance_metrics, metadata, token_usage, response_time, error_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_type,
                    interaction["model"],
                    interaction["timestamp"],
                    interaction["prompt"],
                    interaction["response"],
                    was_complex,
                    json.dumps(performance_metrics) if performance_metrics else None,
                    json.dumps(interaction["metadata"]),
                    interaction["metadata"].get("usage", {}).get("total_tokens", 0),
                    interaction["metadata"].get("response_time", 0),
                    interaction["metadata"].get("error_count", 0)
                ))
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")
            raise
    
    async def store_training_example(self,
                                   agent_type: str,
                                   frontier_model: str,
                                   local_model: str,
                                   prompt: str,
                                   response: str,
                                   quality_score: Optional[float] = None,
                                   performance_metrics: Optional[Dict[str, float]] = None):
        """
        Store a training example with enhanced tracking.
        
        Args:
            agent_type: Type of agent
            frontier_model: Name of frontier model
            local_model: Name of local model
            prompt: Input prompt
            response: Model response
            quality_score: Optional quality score
            performance_metrics: Optional performance metrics
        """
        try:
            with sqlite3.connect(str(self.db_manager.db_path)) as conn:
                conn.execute("""
                    INSERT INTO training_data
                    (agent_type, frontier_model, local_model, prompt, response,
                     quality_score, timestamp, performance_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_type,
                    frontier_model,
                    local_model,
                    prompt,
                    response,
                    quality_score,
                    datetime.now().timestamp(),
                    json.dumps(performance_metrics) if performance_metrics else None
                ))
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error storing training example: {str(e)}")
            raise
    
    async def store_performance_metrics(self,
                                      agent_type: str,
                                      model_type: str,
                                      metrics: Dict[str, float],
                                      context: Optional[Dict[str, Any]] = None,
                                      aggregation_period: str = "hour"):
        """
        Store performance metrics with enhanced context.
        
        Args:
            agent_type: Type of agent
            model_type: Type of model
            metrics: Performance metrics
            context: Optional context information
            aggregation_period: Time period for aggregation
        """
        try:
            current_time = datetime.now().timestamp()
            
            with sqlite3.connect(str(self.db_manager.db_path)) as conn:
                for metric_name, metric_value in metrics.items():
                    confidence = self._calculate_confidence(metric_value, context)
                    
                    conn.execute("""
                        INSERT INTO performance_metrics
                        (agent_type, model_type, metric_name, metric_value, timestamp,
                         context, aggregation_period, confidence_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        agent_type,
                        model_type,
                        metric_name,
                        metric_value,
                        current_time,
                        json.dumps(context) if context else None,
                        aggregation_period,
                        confidence
                    ))
                conn.commit()
        
        except Exception as e:
            logger.error(f"Error storing performance metrics: {str(e)}")
            raise
    
    def _calculate_confidence(self, 
                            metric_value: float, 
                            context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for a metric value."""
        # Start with base confidence
        confidence = 0.8
        
        # Adjust based on context if available
        if context:
            sample_size = context.get('sample_size', 0)
            if sample_size > 100:
                confidence += 0.1
            elif sample_size < 10:
                confidence -= 0.2
                
            if context.get('high_variance', False):
                confidence -= 0.1
                
            if context.get('outliers_removed', False):
                confidence += 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    async def get_training_data(self,
                              agent_type: Optional[str] = None,
                              local_model: Optional[str] = None,
                              min_quality: Optional[float] = None,
                              limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get training data with enhanced filtering.
        
        Args:
            agent_type: Optional agent type filter
            local_model: Optional local model filter
            min_quality: Optional minimum quality score
            limit: Maximum number of examples
            
        Returns:
            List of training examples
        """
        query = "SELECT * FROM training_data WHERE 1=1"
        params = []
        
        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type)
            
        if local_model:
            query += " AND local_model = ?"
            params.append(local_model)
            
        if min_quality is not None:
            query += " AND quality_score >= ?"
            params.append(min_quality)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with sqlite3.connect(str(self.db_manager.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
        
        except Exception as e:
            logger.error(f"Error retrieving training data: {str(e)}")
            raise
    
    async def get_performance_history(self,
                                    agent_type: str,
                                    model_type: str,
                                    metric_name: str,
                                    days: int = 7,
                                    aggregation: str = "hour") -> List[Dict[str, Any]]:
        """
        Get performance history with aggregation.
        
        Args:
            agent_type: Type of agent
            model_type: Type of model
            metric_name: Name of the metric
            days: Number of days of history
            aggregation: Aggregation period ("hour", "day", "week")
            
        Returns:
            List of aggregated metric values
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        # Define aggregation SQL based on period
        if aggregation == "hour":
            time_group = "strftime('%Y-%m-%d %H', datetime(timestamp, 'unixepoch'))"
        elif aggregation == "day":
            time_group = "strftime('%Y-%m-%d', datetime(timestamp, 'unixepoch'))"
        elif aggregation == "week":
            time_group = "strftime('%Y-%W', datetime(timestamp, 'unixepoch'))"
        else:
            raise ValueError(f"Invalid aggregation period: {aggregation}")
        
        query = f"""
            SELECT 
                {time_group} as period,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                COUNT(*) as sample_size,
                AVG(confidence_score) as avg_confidence
            FROM performance_metrics
            WHERE agent_type = ? 
            AND model_type = ? 
            AND metric_name = ?
            AND timestamp >= ?
            GROUP BY period
            ORDER BY period ASC
        """
        
        try:
            with sqlite3.connect(str(self.db_manager.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, (agent_type, model_type, metric_name, cutoff_time))
                return [dict(row) for row in cursor.fetchall()]
        
        except Exception as e:
            logger.error(f"Error retrieving performance history: {str(e)}")
            raise
    
    async def export_data(self, 
                         export_dir: Optional[str] = None,
                         format: str = "json") -> Dict[str, str]:
        """
        Export all data with format options.
        
        Args:
            export_dir: Optional directory for export files
            format: Export format ("json" or "csv")
            
        Returns:
            Dictionary mapping data type to export file path
        """
        export_dir = Path(export_dir) if export_dir else self.db_manager.db_path.parent / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        export_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            with sqlite3.connect(str(self.db_manager.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                
                # Export each table
                for table in ["interactions", "training_data", "performance_metrics"]:
                    cursor = conn.execute(f"SELECT * FROM {table}")
                    rows = [dict(row) for row in cursor.fetchall()]
                    
                    if format == "json":
                        file_path = export_dir / f"{table}_{timestamp}.json"
                        with open(file_path, 'w') as f:
                            json.dump(rows, f, indent=2)
                    else:  # csv format
                        import csv
                        file_path = export_dir / f"{table}_{timestamp}.csv"
                        with open(file_path, 'w', newline='') as f:
                            if rows:
                                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                                writer.writeheader()
                                writer.writerows(rows)
                    
                    export_files[table] = str(file_path)
            
            return export_files
        
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise
