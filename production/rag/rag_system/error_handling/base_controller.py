from abc import ABC, abstractmethod
from typing import Any


class BaseErrorController(ABC):
    @abstractmethod
    def handle_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """Handle an error and return a dictionary with error information and suggested action.

        :param error: The exception that was raised
        :param context: Additional context about where the error occurred
        :return: A dictionary containing error information and suggested action
        """

    @abstractmethod
    def reset(self):
        """Reset the error controller to its initial state."""
