# PURPOSE: Resilient chat engine with graceful degradation and offline fallback.
#          Wraps Twin inference and applies optional conformal calibration.
#
# ARCHITECTURE: Circuit breaker pattern with local fallback mode
# • Remote mode: Twin service at TWIN_URL (HTTP JSON API)
# • Local mode: Basic chat processing when Twin service unavailable
# • Hybrid mode: Automatic failover with health monitoring
# • Configuration: Environment-driven mode selection
# • Fallback: Meaningful responses in offline scenarios
#
# IMPLEMENTATION NOTES
# • Feature-flag via env var CALIBRATION_ENABLED=1
# • Falls back gracefully if calibrator fails >3× in a row
# • Twin client discovered via TWIN_URL with health checks
# • Circuit breaker prevents cascade failures
# • Local chat processor for offline functionality
# • Exposes ChatEngine.process_chat(message, conversation_id)
#   ↳ returns dict {"response": str,
#                   "conversation_id": str,
#                   "raw_prob": float,
#                   "calibrated_prob": float|None,
#                   "timestamp": datetime,
#                   "mode": str,  # 'remote', 'local', 'fallback'
#                   "service_status": str}  # 'healthy', 'degraded', 'offline'

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
import logging
import os
import time
from typing import Any

import requests

from .resilience.error_handling import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    RetryConfig,
    get_resilience_manager,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configuration with environment overrides
TWIN_URL = os.getenv("TWIN_URL", "https://twin:8001/v1/chat")
CALIB_ENABLED = os.getenv("CALIBRATION_ENABLED", "0") == "1"
CHAT_MODE = os.getenv("CHAT_MODE", "hybrid").lower()  # remote, local, hybrid
OFFLINE_RESPONSES_ENABLED = os.getenv("OFFLINE_RESPONSES_ENABLED", "1") == "1"
SERVICE_HEALTH_CHECK_INTERVAL = int(os.getenv("SERVICE_HEALTH_CHECK_INTERVAL", "30"))
CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
CIRCUIT_BREAKER_TIMEOUT_MS = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT_MS", "60000"))


class ChatMode(Enum):
    """Chat engine operation modes."""

    REMOTE = "remote"  # Always use twin service
    LOCAL = "local"  # Always use local processing
    HYBRID = "hybrid"  # Auto-failover with health monitoring


class ServiceStatus(Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OFFLINE = "offline"


if CALIB_ENABLED:
    try:
        from calibration.conformal import ConformalCalibrator

        _calibrator = ConformalCalibrator.load_default()
        logger.info("Calibration enabled (loaded default model)")
    except Exception:  # pragma: no cover - fallback is tested separately
        logger.exception("Failed to init calibrator – continuing without")
        CALIB_ENABLED = False
        _calibrator = None
else:  # pragma: no cover - branch for disabled feature
    _calibrator = None


class LocalChatProcessor:
    """Local chat processing for offline mode and fallback scenarios."""

    def __init__(self):
        self._conversation_history = {}
        self._response_templates = {
            "greeting": [
                "Hello! I'm running in local mode. How can I assist you?",
                "Hi there! I'm operating offline but ready to help.",
                "Greetings! Local chat mode is active. What can I do for you?",
            ],
            "question": [
                "That's an interesting question. In local mode, I have limited capabilities but I'm here to help.",
                "I understand you're asking about that topic. While I'm offline, I can still provide basic assistance.",
                "Good question! I'm in local mode with reduced functionality, but I'll do my best to help.",
            ],
            "help": [
                "I'm operating in local mode with basic functionality. I can echo messages, provide status updates, and maintain conversation context.",
                "Local mode offers limited but functional chat capabilities. I can respond to messages and track our conversation.",
                "In offline mode, I provide basic chat responses and maintain conversation continuity.",
            ],
            "status": [
                "Status: Local chat mode active. Twin service unavailable. Basic functionality operational.",
                "System Status: Running in offline mode. Core chat functions available with limited capabilities.",
                "Service Status: Local fallback mode. Essential chat operations functioning.",
            ],
            "default": [
                "I received your message. I'm currently in local mode with limited capabilities.",
                "Thanks for your message. I'm operating offline but can still provide basic responses.",
                "Message received. I'm in local mode - functionality is reduced but I'm here to help.",
            ],
        }

    def process_local_chat(self, message: str, conversation_id: str) -> dict[str, Any]:
        """Process chat message using local fallback logic."""
        message_lower = message.lower().strip()

        # Initialize conversation history
        if conversation_id not in self._conversation_history:
            self._conversation_history[conversation_id] = []

        # Add user message to history
        self._conversation_history[conversation_id].append(
            {"role": "user", "message": message, "timestamp": time.time()}
        )

        # Determine response type based on message content
        response_type = self._categorize_message(message_lower)

        # Handle special commands
        if message_lower.startswith("/status"):
            response_text = f"Local Chat Mode: Active | Twin Service: Offline | Conversation ID: {conversation_id}"
        elif message_lower.startswith("/help"):
            response_text = self._get_help_response()
        elif message_lower.startswith("/echo"):
            echo_text = message[5:].strip() if len(message) > 5 else "echo"
            response_text = f"Echo: {echo_text}"
        else:
            # Generate contextual response
            response_text = self._generate_response(response_type, message, conversation_id)

        # Add assistant response to history
        self._conversation_history[conversation_id].append(
            {"role": "assistant", "message": response_text, "timestamp": time.time()}
        )

        # Generate mock confidence scores
        raw_prob = 0.7 + (hash(message) % 30) / 100  # Deterministic but varied

        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "raw_prob": raw_prob,
            "calibrated_prob": None,  # No calibration in local mode
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": "local",
            "service_status": ServiceStatus.OFFLINE.value,
            "processing_time_ms": 50,  # Simulate processing time
            "conversation_length": len(self._conversation_history[conversation_id]),
        }

    def _categorize_message(self, message_lower: str) -> str:
        """Categorize message to select appropriate response template."""
        if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "greeting"
        elif any(word in message_lower for word in ["help", "assistance", "support"]):
            return "help"
        elif any(word in message_lower for word in ["status", "health", "system"]):
            return "status"
        elif "?" in message_lower or any(word in message_lower for word in ["what", "how", "why", "when", "where"]):
            return "question"
        else:
            return "default"

    def _generate_response(self, response_type: str, original_message: str, conversation_id: str) -> str:
        """Generate contextual response based on message type and history."""
        templates = self._response_templates.get(response_type, self._response_templates["default"])
        base_response = templates[hash(original_message) % len(templates)]

        # Add conversation context for engagement
        history_length = len(self._conversation_history.get(conversation_id, []))
        if history_length > 2:
            base_response += f" (This is our {history_length//2 + 1}th exchange in this conversation.)"

        return base_response

    def _get_help_response(self) -> str:
        """Generate help response with available commands."""
        return (
            "Local Chat Mode Help:\n"
            "• /status - Show system status\n"
            "• /help - Show this help message\n"
            "• /echo [text] - Echo back your text\n"
            "• Regular messages - Basic conversational responses\n"
            "Note: Advanced features require Twin service connectivity."
        )

    def get_conversation_history(self, conversation_id: str) -> list[dict]:
        """Get conversation history for a given conversation ID."""
        return self._conversation_history.get(conversation_id, [])

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history for a given conversation ID."""
        if conversation_id in self._conversation_history:
            del self._conversation_history[conversation_id]
            return True
        return False


class ChatEngine:
    """Resilient chat engine with circuit breaker pattern and graceful degradation."""

    def __init__(self) -> None:
        self._calib_enabled = CALIB_ENABLED
        self._consecutive_calib_errors = 0
        self._mode = ChatMode(CHAT_MODE)
        self._current_status = ServiceStatus.HEALTHY
        self._last_health_check = 0

        # Initialize local processor
        self._local_processor = LocalChatProcessor()

        # Initialize resilience components
        self._resilience_manager = get_resilience_manager()

        # Configure circuit breaker for twin service
        cb_config = CircuitBreakerConfig(
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            timeout_ms=CIRCUIT_BREAKER_TIMEOUT_MS,
            success_threshold=3,
            half_open_max_calls=5,
        )
        self._circuit_breaker = self._resilience_manager.get_circuit_breaker("twin_service", cb_config)

        # Configure retry handler
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay_ms=500,
            max_delay_ms=5000,
            retryable_exceptions=[requests.exceptions.ConnectionError, requests.exceptions.Timeout],
        )
        self._retry_handler = self._resilience_manager.get_retry_handler("twin_service", retry_config)

        logger.info(f"ChatEngine initialized in {self._mode.value} mode")

    def health_check_twin_service(self) -> bool:
        """Check if Twin service is available and healthy."""
        try:
            # Use a lightweight health check endpoint if available
            health_url = TWIN_URL.replace("/v1/chat", "/health")
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            # Fallback: try the main endpoint with a simple request
            try:
                test_payload = {"prompt": "health_check", "conversation_id": "health_check"}
                response = requests.post(TWIN_URL, json=test_payload, timeout=5)
                return response.status_code in (200, 400)  # 400 might be validation error but service is up
            except Exception:
                return False

    def _update_service_status(self) -> ServiceStatus:
        """Update and return current service status."""
        current_time = time.time()

        # Check if we need to perform health check
        if current_time - self._last_health_check > SERVICE_HEALTH_CHECK_INTERVAL:
            self._last_health_check = current_time

            if self._mode == ChatMode.LOCAL:
                self._current_status = ServiceStatus.OFFLINE
            elif self._mode == ChatMode.REMOTE:
                if self.health_check_twin_service():
                    self._current_status = ServiceStatus.HEALTHY
                else:
                    self._current_status = ServiceStatus.OFFLINE
            else:  # HYBRID mode
                if self._circuit_breaker.state.value == "closed":
                    if self.health_check_twin_service():
                        self._current_status = ServiceStatus.HEALTHY
                    else:
                        self._current_status = ServiceStatus.DEGRADED
                elif self._circuit_breaker.state.value == "half_open":
                    self._current_status = ServiceStatus.DEGRADED
                else:  # open
                    self._current_status = ServiceStatus.OFFLINE

        return self._current_status

    def _call_twin_service(self, message: str, conversation_id: str) -> dict[str, Any]:
        """Make resilient call to twin service with circuit breaker protection."""

        def twin_request():
            twin_payload = {"prompt": message, "conversation_id": conversation_id}
            response = requests.post(TWIN_URL, json=twin_payload, timeout=15)
            response.raise_for_status()
            return response.json()

        # Use circuit breaker and retry logic
        try:
            return self._circuit_breaker.call(self._retry_handler.execute_with_retry, twin_request)
        except CircuitBreakerOpenError:
            logger.warning("Circuit breaker is open, falling back to local mode")
            raise
        except Exception as e:
            logger.error(f"Twin service call failed: {e}")
            raise

    # ------------------------------------------------------------------
    def process_chat(self, message: str, conversation_id: str) -> dict[str, Any]:
        """Process chat message with resilient architecture and graceful degradation."""
        started = time.time()

        # Update service status
        current_status = self._update_service_status()

        # Determine processing mode based on configuration and service status
        if self._mode == ChatMode.LOCAL:
            return self._process_local_chat(message, conversation_id, started)
        elif self._mode == ChatMode.REMOTE:
            return self._process_remote_chat(message, conversation_id, started)
        else:  # HYBRID mode
            return self._process_hybrid_chat(message, conversation_id, started, current_status)

    def _process_local_chat(self, message: str, conversation_id: str, started: float) -> dict[str, Any]:
        """Process chat in local-only mode."""
        logger.info("Processing chat in local mode")
        result = self._local_processor.process_local_chat(message, conversation_id)
        result["processing_time_ms"] = int((time.time() - started) * 1000)
        return result

    def _process_remote_chat(self, message: str, conversation_id: str, started: float) -> dict[str, Any]:
        """Process chat in remote-only mode."""
        logger.info("Processing chat in remote mode")
        try:
            data = self._call_twin_service(message, conversation_id)
            return self._format_remote_response(data, conversation_id, started, "remote", ServiceStatus.HEALTHY)
        except Exception as e:
            logger.error(f"Remote processing failed: {e}")
            # In remote-only mode, we still fail but with better error info
            if OFFLINE_RESPONSES_ENABLED:
                logger.info("Falling back to local mode due to remote failure")
                return self._process_fallback_chat(message, conversation_id, started, str(e))
            else:
                raise

    def _process_hybrid_chat(
        self, message: str, conversation_id: str, started: float, status: ServiceStatus
    ) -> dict[str, Any]:
        """Process chat in hybrid mode with intelligent failover."""
        # Try remote first if service appears healthy
        if status == ServiceStatus.HEALTHY:
            try:
                logger.info("Processing chat in hybrid mode (attempting remote)")
                data = self._call_twin_service(message, conversation_id)
                return self._format_remote_response(data, conversation_id, started, "remote", status)
            except (CircuitBreakerOpenError, Exception) as e:
                logger.warning(f"Remote processing failed in hybrid mode, falling back: {e}")
                return self._process_fallback_chat(message, conversation_id, started, str(e))
        else:
            # Service is degraded or offline, use local processing
            logger.info(f"Processing chat in hybrid mode (local fallback, status: {status.value})")
            return self._process_fallback_chat(message, conversation_id, started, "Service unavailable")

    def _process_fallback_chat(
        self, message: str, conversation_id: str, started: float, error_reason: str
    ) -> dict[str, Any]:
        """Process chat using local fallback with error context."""
        result = self._local_processor.process_local_chat(message, conversation_id)
        result.update(
            {
                "mode": "fallback",
                "service_status": self._current_status.value,
                "processing_time_ms": int((time.time() - started) * 1000),
                "fallback_reason": error_reason,
                "notice": "This response was generated in offline mode with limited capabilities.",
            }
        )
        return result

    def _format_remote_response(
        self, data: dict, conversation_id: str, started: float, mode: str, status: ServiceStatus
    ) -> dict[str, Any]:
        """Format response from remote twin service."""
        resp_text: str = data.get("response", "")
        raw_prob: float = float(data.get("raw_prob", 0.5))
        calibrated: float | None = data.get("calibrated_prob")

        # Apply calibration if enabled and available
        if calibrated is None and self._calib_enabled:
            try:
                calibrated = _calibrator.calibrate(raw_prob)  # type: ignore[attr-defined]
                self._consecutive_calib_errors = 0
            except Exception as exc:
                logger.warning("Calibration error: %s", exc)
                self._consecutive_calib_errors += 1
                if self._consecutive_calib_errors >= 3:
                    logger.error("Disabling calibration after 3 failures")
                    self._calib_enabled = False

        return {
            "response": resp_text,
            "conversation_id": conversation_id,
            "raw_prob": raw_prob,
            "calibrated_prob": calibrated,
            "timestamp": datetime.now(UTC).isoformat(),
            "mode": mode,
            "service_status": status.value,
            "processing_time_ms": int((time.time() - started) * 1000),
        }

    # Additional utility methods
    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status information."""
        return {
            "mode": self._mode.value,
            "service_status": self._current_status.value,
            "circuit_breaker_state": self._circuit_breaker.state.value,
            "circuit_breaker_stats": self._circuit_breaker.get_stats(),
            "calibration_enabled": self._calib_enabled,
            "consecutive_calibration_errors": self._consecutive_calib_errors,
            "last_health_check": self._last_health_check,
            "twin_url": TWIN_URL,
            "offline_responses_enabled": OFFLINE_RESPONSES_ENABLED,
        }

    def force_mode_change(self, new_mode: str) -> bool:
        """Force change chat engine mode (for testing/admin purposes)."""
        try:
            self._mode = ChatMode(new_mode.lower())
            logger.info(f"Chat engine mode changed to: {self._mode.value}")
            return True
        except ValueError:
            logger.error(f"Invalid mode: {new_mode}. Valid modes: {[m.value for m in ChatMode]}")
            return False

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state (for admin purposes)."""
        self._circuit_breaker.failure_count = 0
        self._circuit_breaker.state = self._circuit_breaker.state.__class__.CLOSED
        logger.info("Circuit breaker reset to closed state")

    def get_conversation_history(self, conversation_id: str) -> list[dict]:
        """Get local conversation history (available in local/fallback modes)."""
        return self._local_processor.get_conversation_history(conversation_id)
