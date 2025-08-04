# rag_system/error_handling/error_control.py

from typing import Any

from rag_system.utils.logging import setup_logger


class ErrorController:
    def __init__(self) -> None:
        self.logger = setup_logger(__name__)
        # Track how many times each error type has been seen
        self.error_counts: dict[str, int] = {}
        self.total_errors: int = 0

    def handle_error(
        self,
        error_message: str,
        exception: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Handle errors in a centralized manner.

        :param error_message: A descriptive message about the error
        :param exception: The exception that was raised
        :param context: Optional dictionary containing contextual information about the error
        """
        self.logger.error(f"{error_message}: {exception!s}")

        if context:
            self.logger.error(f"Error context: {context}")

        # Log the full stack trace
        self.logger.exception("Full stack trace:")

        # Implement additional error handling logic here, such as:
        # - Sending error notifications
        # - Updating error statistics
        # - Triggering recovery mechanisms

        self._attempt_recovery(exception, context)

    def _attempt_recovery(self, exception: Exception, context: dict[str, Any] | None = None):
        """Attempt to recover from the error and update error statistics."""
        error_type = type(exception).__name__
        self.total_errors += 1
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        if isinstance(exception, ValueError):
            self.logger.info("Attempting basic recovery from ValueError...")
            # Use a fallback value if provided in the context
            if context and "fallback" in context:
                self.logger.info("Using fallback value provided in context.")
                return context["fallback"]
        elif isinstance(exception, IOError):
            self.logger.info("Attempting basic recovery from IOError by retrying callback if available...")
            retry_cb = context.get("retry_callback") if context else None
            if callable(retry_cb):
                try:
                    return retry_cb()
                except Exception as retry_err:
                    self.logger.exception(f"Retry failed: {retry_err}")
        else:
            self.logger.warning("No specific recovery mechanism for this error type.")
        return None

    def log_warning(self, warning_message: str, context: dict[str, Any] | None = None) -> None:
        """Log a warning message with optional context.

        :param warning_message: The warning message to log
        :param context: Optional dictionary containing contextual information about the warning
        """
        self.logger.warning(warning_message)

        if context:
            self.logger.warning(f"Warning context: {context}")

    def get_error_statistics(self) -> dict[str, Any]:
        """Return aggregated statistics about recorded errors."""
        if self.error_counts:
            most_common = max(self.error_counts, key=self.error_counts.get)
        else:
            most_common = None

        return {
            "total_errors": self.total_errors,
            "errors_by_type": dict(self.error_counts),
            "most_common_error": most_common,
        }
