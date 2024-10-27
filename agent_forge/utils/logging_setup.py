import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime
import gzip
import shutil

class LogAnalyzer:
    """Utility class for analyzing log files and extracting insights."""
    
    def __init__(self, log_dir: str):
        """
        Initialize LogAnalyzer.
        
        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = Path(log_dir)
    
    def analyze_errors(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze error patterns in recent logs.
        
        Args:
            days: Number of days of logs to analyze
            
        Returns:
            Dictionary containing error analysis
        """
        error_counts: Dict[str, int] = {}
        error_examples: Dict[str, str] = {}
        
        for log_file in self._get_recent_logs(days):
            for line in self._read_log_file(log_file):
                if "ERROR" in line:
                    error_type = self._extract_error_type(line)
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    if error_type not in error_examples:
                        error_examples[error_type] = line
        
        return {
            "total_errors": sum(error_counts.values()),
            "error_types": error_counts,
            "examples": error_examples
        }
    
    def analyze_performance(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze performance metrics from logs.
        
        Args:
            days: Number of days of logs to analyze
            
        Returns:
            Dictionary containing performance analysis
        """
        metrics: Dict[str, list] = {
            "response_times": [],
            "token_usage": [],
            "model_usage": {}
        }
        
        for log_file in self._get_recent_logs(days):
            for line in self._read_log_file(log_file):
                if "response_time" in line:
                    metrics["response_times"].append(
                        self._extract_metric(line, "response_time")
                    )
                if "tokens_used" in line:
                    metrics["token_usage"].append(
                        self._extract_metric(line, "tokens_used")
                    )
                if "model_used" in line:
                    model = self._extract_model(line)
                    metrics["model_usage"][model] = metrics["model_usage"].get(model, 0) + 1
        
        return {
            "average_response_time": self._calculate_average(metrics["response_times"]),
            "total_tokens": sum(metrics["token_usage"]),
            "model_distribution": metrics["model_usage"]
        }
    
    def _get_recent_logs(self, days: int) -> list:
        """Get list of recent log files."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        log_files = []
        
        for file in self.log_dir.glob("*.log*"):
            if file.stat().st_mtime >= cutoff:
                log_files.append(file)
        
        return sorted(log_files, key=lambda f: f.stat().st_mtime, reverse=True)
    
    def _read_log_file(self, file_path: Path) -> list:
        """Read lines from a log file, handling compression."""
        if str(file_path).endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                return f.readlines()
        else:
            with open(file_path, 'r') as f:
                return f.readlines()
    
    @staticmethod
    def _extract_error_type(line: str) -> str:
        """Extract error type from log line."""
        try:
            return line.split("ERROR")[1].split(":")[0].strip()
        except:
            return "Unknown"
    
    @staticmethod
    def _extract_metric(line: str, metric: str) -> float:
        """Extract numeric metric from log line."""
        try:
            return float(line.split(f"{metric}=")[1].split()[0])
        except:
            return 0.0
    
    @staticmethod
    def _extract_model(line: str) -> str:
        """Extract model name from log line."""
        try:
            return line.split("model_used=")[1].split()[0]
        except:
            return "unknown"
    
    @staticmethod
    def _calculate_average(values: list) -> float:
        """Calculate average of values."""
        return sum(values) / len(values) if values else 0.0

def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        max_size: Maximum size of each log file
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("ai_village")
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "ai_village.log",
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Create performance log
    perf_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "performance.log",
        maxBytes=max_size,
        backupCount=backup_count
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(file_formatter)
    
    perf_logger = logging.getLogger("ai_village.performance")
    perf_logger.addHandler(perf_handler)
    
    # Create error log
    error_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / "error.log",
        maxBytes=max_size,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    error_logger = logging.getLogger("ai_village.error")
    error_logger.addHandler(error_handler)
    
    return logger

def log_performance_metrics(metrics: Dict[str, Any]):
    """
    Log performance metrics.
    
    Args:
        metrics: Performance metrics to log
    """
    logger = logging.getLogger("ai_village.performance")
    logger.info(f"Performance metrics: {json.dumps(metrics)}")

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an error with context.
    
    Args:
        error: The exception to log
        context: Optional context information
    """
    logger = logging.getLogger("ai_village.error")
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    logger.error(f"Error occurred: {json.dumps(error_info)}")

def compress_old_logs(log_dir: str, days_old: int = 30):
    """
    Compress logs older than specified days.
    
    Args:
        log_dir: Directory containing log files
        days_old: Age in days after which to compress logs
    """
    log_dir = Path(log_dir)
    cutoff = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    
    for log_file in log_dir.glob("*.log"):
        if log_file.stat().st_mtime < cutoff and not log_file.name.endswith('.gz'):
            with open(log_file, 'rb') as f_in:
                with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log_file.unlink()  # Remove original file
