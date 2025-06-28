# exceptions.py

from typing import Any, Dict
from langroid.utils.logging import setup_logger

logger = setup_logger()

class AIVillageException(Exception):
    """Base exception class for AI Village project."""
    def __init__(self, message: str, context: Dict[str, Any] = None):
        super().__init__(message)
        self.context = context or {}
        logger.error(f"AIVillageException: {message}", extra={"context": self.context})

class PlanningException(AIVillageException):
    """Exception raised for errors in the planning process."""
    pass

class ModelInteractionException(AIVillageException):
    """Exception raised for errors in interacting with the AI model."""
    pass

class DataProcessingException(AIVillageException):
    """Exception raised for errors in data processing or manipulation."""
    pass

# Add more specific exception classes as needed
