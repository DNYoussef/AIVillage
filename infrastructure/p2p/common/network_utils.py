"""
Network utility functions for P2P infrastructure.
"""

import socket
from typing import Optional


from .constants import (
    DEFAULT_DNS_SERVER, DEFAULT_TEST_PORT, DEFAULT_START_PORT, 
    DEFAULT_MAX_PORT_ATTEMPTS, LOOPBACK_IP
)

DEFAULT_TEST_ADDRESS = DEFAULT_DNS_SERVER  # Alias for backward compatibility


def get_local_ip() -> str:
    """Get local IP address (best guess).
    
    Returns:
        Local IP address string
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((DEFAULT_TEST_ADDRESS, DEFAULT_TEST_PORT))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"


def find_free_port(*, start_port: int = DEFAULT_START_PORT, max_attempts: int = DEFAULT_MAX_PORT_ATTEMPTS) -> Optional[int]:
    """Find a free port starting from start_port.
    
    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try
        
    Returns:
        Free port number or None if none found
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None