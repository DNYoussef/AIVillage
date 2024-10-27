import logging
import json
import sqlite3
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DataCollector:
    """
    System for collecting and storing API outputs, performance metrics,
    and training data from agent interactions.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataCollector.
        
        Args:
            data_dir: Directory to store collected data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = self.data_dir / "agent_data.db"
        self._init_database()
        
        logger.info(f"Initialized DataCollector with data directory: {data_dir}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(str(self.db_path)) as conn:
            # Create interactions table
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
                    metadata TEXT
                )
            """)
            
            # Create training_data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INTEGER PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    frontier_model TEXT NOT NULL,
                    local_model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    quality_score REAL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create performance_metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            conn.commit()
    
    def store_interaction(self, 
                         agent_type: str,
                         interaction: Dict[str, Any],
                         was_complex: bool,
                         performance_metrics: Optional[Dict[str, float]] = None):
        """
        Store an agent interaction.
        
        Args:
            agent_type: Type of agent ("king", "sage", or "magi")
            interaction: Interaction data
            was_complex: Whether the task was complex
            performance_metrics: Optional performance metrics
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO interactions 
                (agent_type, model_used, timestamp, prompt, response, was_complex, 
                 performance_metrics, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_type,
                interaction["model"],
                interaction["timestamp"],
                interaction["prompt"],
                interaction["response"],
                was_complex,
                json.dumps(performance_metrics) if performance_metrics else None,
                json.dumps(interaction["metadata"])
            ))
            conn.commit()
    
    def store_training_example(self,
                             agent_type: str,
                             frontier_model: str,
                             local_model: str,
                             prompt: str,
                             response: str,
                             quality_score: Optional[float] = None):
        """
        Store a training example for local model improvement.
        
        Args:
            agent_type: Type of agent
            frontier_model: Name of frontier model
            local_model: Name of local model to train
            prompt: Input prompt
            response: Model response
            quality_score: Optional quality score
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                INSERT INTO training_data
                (agent_type, frontier_model, local_model, prompt, response, 
                 quality_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_type,
                frontier_model,
                local_model,
                prompt,
                response,
                quality_score,
                datetime.now().timestamp()
            ))
            conn.commit()
    
    def store_performance_metrics(self,
                                agent_type: str,
                                model_type: str,
                                metrics: Dict[str, float]):
        """
        Store performance metrics.
        
        Args:
            agent_type: Type of agent
            model_type: Type of model ("frontier" or "local")
            metrics: Performance metrics
        """
        current_time = datetime.now().timestamp()
        
        with sqlite3.connect(str(self.db_path)) as conn:
            for metric_name, metric_value in metrics.items():
                conn.execute("""
                    INSERT INTO performance_metrics
                    (agent_type, model_type, metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    agent_type,
                    model_type,
                    metric_name,
                    metric_value,
                    current_time
                ))
            conn.commit()
    
    def get_training_data(self,
                         agent_type: Optional[str] = None,
                         local_model: Optional[str] = None,
                         min_quality: Optional[float] = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get training data for local model training.
        
        Args:
            agent_type: Optional agent type to filter by
            local_model: Optional local model name to filter by
            min_quality: Optional minimum quality score
            limit: Maximum number of examples to return
            
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
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_performance_history(self,
                              agent_type: str,
                              model_type: str,
                              metric_name: str,
                              days: int = 7) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific metric.
        
        Args:
            agent_type: Type of agent
            model_type: Type of model
            metric_name: Name of the metric
            days: Number of days of history
            
        Returns:
            List of metric values with timestamps
        """
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT metric_value, timestamp
                FROM performance_metrics
                WHERE agent_type = ? AND model_type = ? AND metric_name = ?
                AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (agent_type, model_type, metric_name, cutoff_time))
            return [dict(row) for row in cursor.fetchall()]
    
    def export_data(self, export_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export all data to JSON files.
        
        Args:
            export_dir: Optional directory for export files
            
        Returns:
            Dictionary mapping data type to export file path
        """
        export_dir = Path(export_dir) if export_dir else self.data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        export_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Export interactions
            cursor = conn.execute("SELECT * FROM interactions")
            interactions = [dict(row) for row in cursor.fetchall()]
            interactions_file = export_dir / f"interactions_{timestamp}.json"
            with open(interactions_file, 'w') as f:
                json.dump(interactions, f, indent=2)
            export_files['interactions'] = str(interactions_file)
            
            # Export training data
            cursor = conn.execute("SELECT * FROM training_data")
            training_data = [dict(row) for row in cursor.fetchall()]
            training_file = export_dir / f"training_data_{timestamp}.json"
            with open(training_file, 'w') as f:
                json.dump(training_data, f, indent=2)
            export_files['training_data'] = str(training_file)
            
            # Export performance metrics
            cursor = conn.execute("SELECT * FROM performance_metrics")
            metrics = [dict(row) for row in cursor.fetchall()]
            metrics_file = export_dir / f"performance_metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            export_files['performance_metrics'] = str(metrics_file)
        
        return export_files
