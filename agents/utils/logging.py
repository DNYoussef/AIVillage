"""Logging utilities for the AI Village system."""

import logging
from typing import Optional
from pathlib import Path

def setup_logger(
    name: str,
    level: Optional[int] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        level: Logging level (defaults to INFO if not specified)
        log_file: Optional file path for logging to file
        format_string: Optional custom format string for log messages
    
    Returns:
        Configured logger instance
    """
    # Set default level if not specified
    if level is None:
        level = logging.INFO
    
    # Set default format if not specified
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_task_logger(task_name: str) -> logging.Logger:
    """Get a logger specifically for task execution."""
    return setup_logger(
        f"task.{task_name}",
        log_file=f"logs/tasks/{task_name}.log"
    )

def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get a logger specifically for agent operations."""
    return setup_logger(
        f"agent.{agent_name}",
        log_file=f"logs/agents/{agent_name}.log"
    )

def get_system_logger() -> logging.Logger:
    """Get the main system logger."""
    return setup_logger(
        "system",
        log_file="logs/system.log"
    )

# Create logs directory
Path("logs").mkdir(exist_ok=True)
Path("logs/tasks").mkdir(exist_ok=True)
Path("logs/agents").mkdir(exist_ok=True)

# Initialize system logger
system_logger = get_system_logger()

# Alias for backward compatibility
get_logger = setup_logger
