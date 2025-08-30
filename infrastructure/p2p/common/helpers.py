"""
Helper utilities for P2P infrastructure.

Provides common utility functions and helpers used across
all P2P components for consistency and code reuse.

This module serves as a clean aggregator of specialized utility modules
to maintain backward compatibility while improving code organization.
"""

# Import all utilities from specialized modules
from .timing import calculate_latency, estimate_bandwidth
from .formatting import format_bytes, format_duration
from .network_validation import validate_address, parse_address, is_local_address
from .network_utils import get_local_ip, find_free_port
from .identifiers import generate_session_id, create_checksum, normalize_peer_id
from .async_decorators import timeout_after, retry_on_failure
from .data_utils import safe_json_dumps, deep_update, flatten_dict, unflatten_dict

# Re-export all functions for backward compatibility
__all__ = [
    # Timing utilities
    'calculate_latency',
    'estimate_bandwidth',
    
    # Formatting utilities
    'format_bytes',
    'format_duration',
    
    # Network validation
    'validate_address',
    'parse_address',
    'is_local_address',
    
    # Network utilities
    'get_local_ip',
    'find_free_port',
    
    # Identifier utilities
    'generate_session_id',
    'create_checksum',
    'normalize_peer_id',
    
    # Async decorators
    'timeout_after',
    'retry_on_failure',
    
    # Data utilities
    'safe_json_dumps',
    'deep_update',
    'flatten_dict',
    'unflatten_dict',
]