"""Logging utilities for core techniques."""

import logging
from typing import Optional
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create handler
        handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add formatter to handler
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Set level
        logger.setLevel(logging.INFO)
    
    return logger

def setup_technique_logging(technique_name: str) -> logging.Logger:
    """Set up logging for a specific technique."""
    logger = get_logger(f"technique.{technique_name}")
    
    # Add file handler for technique-specific logging
    file_handler = logging.FileHandler(
        f"logs/techniques/{technique_name}.log"
    )
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    )
    logger.addHandler(file_handler)
    
    return logger

def log_execution_time(logger: logging.Logger, start_time: datetime):
    """Log execution time of a technique."""
    duration = datetime.now() - start_time
    logger.info(f"Execution time: {duration.total_seconds():.2f} seconds")

def log_technique_result(
    logger: logging.Logger,
    technique_name: str,
    success: bool,
    details: Optional[str] = None
):
    """Log the result of a technique execution."""
    status = "succeeded" if success else "failed"
    message = f"Technique {technique_name} {status}"
    if details:
        message += f": {details}"
    
    if success:
        logger.info(message)
    else:
        logger.error(message)
