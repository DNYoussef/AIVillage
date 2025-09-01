"""
Helper utilities for P2P infrastructure.

Provides common utility functions and helpers used across
all P2P components for consistency and code reuse.

This module serves as a clean aggregator of specialized utility modules
to maintain backward compatibility while improving code organization.
"""

# Import all utilities from specialized modules
from .async_decorators import retry_on_failure, timeout_after
from .data_utils import deep_update, flatten_dict, safe_json_dumps, unflatten_dict
from .formatting import format_bytes, format_duration
from .identifiers import create_checksum, generate_session_id, normalize_peer_id
from .network_utils import find_free_port, get_local_ip
from .network_validation import is_local_address, parse_address, validate_address
from .timing import calculate_latency, estimate_bandwidth

# Re-export all functions for backward compatibility
__all__ = [
    # Timing utilities
    "calculate_latency",
    "estimate_bandwidth",
    # Formatting utilities
    "format_bytes",
    "format_duration",
    # Network validation
    "validate_address",
    "parse_address",
    "is_local_address",
    # Network utilities
    "get_local_ip",
    "find_free_port",
    # Identifier utilities
    "generate_session_id",
    "create_checksum",
    "normalize_peer_id",
    # Async decorators
    "timeout_after",
    "retry_on_failure",
    # Data utilities
    "safe_json_dumps",
    "deep_update",
    "flatten_dict",
    "unflatten_dict",
]
