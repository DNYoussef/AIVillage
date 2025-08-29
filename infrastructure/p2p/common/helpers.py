"""
Helper utilities for P2P infrastructure.

Provides common utility functions and helpers used across
all P2P components for consistency and code reuse.
"""

import time
import hashlib
import socket
import uuid
import re
import ipaddress
from typing import Optional, Union, Tuple, Dict, Any
from urllib.parse import urlparse
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)


def calculate_latency(start_time: float, end_time: Optional[float] = None) -> float:
    """
    Calculate latency between two timestamps.
    
    Args:
        start_time: Start timestamp (from time.time())
        end_time: End timestamp (defaults to current time)
        
    Returns:
        Latency in milliseconds
    """
    if end_time is None:
        end_time = time.time()
    
    return (end_time - start_time) * 1000  # Convert to milliseconds


def estimate_bandwidth(bytes_transferred: int, duration_seconds: float) -> float:
    """
    Estimate bandwidth based on data transferred and time taken.
    
    Args:
        bytes_transferred: Number of bytes transferred
        duration_seconds: Duration in seconds
        
    Returns:
        Bandwidth in bytes per second
    """
    if duration_seconds <= 0:
        return 0.0
    
    return bytes_transferred / duration_seconds


def format_bytes(byte_count: int, decimal_places: int = 2) -> str:
    """
    Format byte count into human-readable string.
    
    Args:
        byte_count: Number of bytes
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted string (e.g., "1.23 KB", "4.56 MB")
    """
    if byte_count == 0:
        return "0 B"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(byte_count)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.{decimal_places}f} {units[unit_index]}"


def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        precision: Number of decimal places for sub-second durations
        
    Returns:
        Formatted string (e.g., "1.23s", "2m 30s", "1h 15m")
    """
    if seconds < 1:
        return f"{seconds:.{precision}f}s"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds:.0f}s"
        return f"{minutes}m"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if hours < 24:
        if remaining_minutes > 0:
            return f"{hours}h {remaining_minutes}m"
        return f"{hours}h"
    
    days = hours // 24
    remaining_hours = hours % 24
    
    if remaining_hours > 0:
        return f"{days}d {remaining_hours}h"
    return f"{days}d"


def validate_address(address: str) -> bool:
    """
    Validate network address format.
    
    Supports:
    - IPv4 addresses (with optional port)
    - IPv6 addresses (with optional port)
    - Hostnames (with optional port)
    - Multiaddresses (/ip4/127.0.0.1/tcp/4001)
    
    Args:
        address: Address to validate
        
    Returns:
        True if address is valid format
    """
    if not address or not isinstance(address, str):
        return False
    
    # Check for multiaddress format
    if address.startswith('/'):
        return _validate_multiaddress(address)
    
    # Check for URL format
    if '://' in address:
        try:
            parsed = urlparse(address)
            return bool(parsed.netloc)
        except Exception:
            return False
    
    # Check for host:port format
    if ':' in address:
        try:
            host, port_str = address.rsplit(':', 1)
            port = int(port_str)
            if not (1 <= port <= 65535):
                return False
            address = host  # Validate host part
        except ValueError:
            # Could be IPv6 without port
            pass
    
    # Validate IP address
    try:
        ipaddress.ip_address(address)
        return True
    except ValueError:
        pass
    
    # Validate hostname
    return _validate_hostname(address)


def parse_address(address: str) -> Optional[Tuple[str, Optional[int]]]:
    """
    Parse network address into host and port components.
    
    Args:
        address: Network address
        
    Returns:
        Tuple of (host, port) or None if invalid
    """
    if not validate_address(address):
        return None
    
    # Handle multiaddress format
    if address.startswith('/'):
        return _parse_multiaddress(address)
    
    # Handle URL format
    if '://' in address:
        try:
            parsed = urlparse(address)
            port = parsed.port
            return parsed.hostname, port
        except Exception:
            return None
    
    # Handle host:port format
    if ':' in address and not address.startswith('['):
        try:
            host, port_str = address.rsplit(':', 1)
            port = int(port_str)
            return host, port
        except ValueError:
            pass
    
    # Handle IPv6 with port [::1]:8080
    if address.startswith('[') and ']:' in address:
        try:
            bracket_end = address.index(']:')
            host = address[1:bracket_end]
            port = int(address[bracket_end + 2:])\n            return host, port\n        except (ValueError, IndexError):\n            return None\n    \n    # Just hostname/IP without port\n    return address, None\n\n\ndef generate_session_id(prefix: str = \"session\") -> str:\n    \"\"\"Generate a unique session identifier.\"\"\"\n    timestamp = int(time.time() * 1000)  # Milliseconds\n    unique_part = uuid.uuid4().hex[:8]\n    return f\"{prefix}_{timestamp}_{unique_part}\"\n\n\ndef create_checksum(data: bytes, algorithm: str = \"sha256\") -> str:\n    \"\"\"Create checksum/hash of data.\"\"\"\n    if algorithm == \"sha256\":\n        return hashlib.sha256(data).hexdigest()\n    elif algorithm == \"sha1\":\n        return hashlib.sha1(data).hexdigest()\n    elif algorithm == \"md5\":\n        return hashlib.md5(data).hexdigest()\n    elif algorithm == \"blake2b\":\n        return hashlib.blake2b(data).hexdigest()\n    else:\n        raise ValueError(f\"Unsupported hash algorithm: {algorithm}\")\n\n\ndef is_local_address(address: str) -> bool:\n    \"\"\"Check if address is a local/loopback address.\"\"\"\n    host, _ = parse_address(address) or (address, None)\n    \n    try:\n        ip = ipaddress.ip_address(host)\n        return ip.is_loopback or ip.is_private\n    except ValueError:\n        # Not an IP, check for localhost hostname\n        return host.lower() in ['localhost', '127.0.0.1', '::1']\n\n\ndef get_local_ip() -> str:\n    \"\"\"Get local IP address (best guess).\"\"\"\n    try:\n        # Connect to a remote address to determine local IP\n        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:\n            s.connect((\"8.8.8.8\", 80))\n            local_ip = s.getsockname()[0]\n        return local_ip\n    except Exception:\n        return \"127.0.0.1\"\n\n\ndef find_free_port(start_port: int = 8000, max_attempts: int = 100) -> Optional[int]:\n    \"\"\"Find a free port starting from start_port.\"\"\"\n    for port in range(start_port, start_port + max_attempts):\n        try:\n            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:\n                s.bind(('localhost', port))\n                return port\n        except OSError:\n            continue\n    return None\n\n\ndef normalize_peer_id(peer_id: str) -> str:\n    \"\"\"Normalize peer ID to consistent format.\"\"\"\n    # Remove common prefixes\n    for prefix in ['peer_', 'node_', 'id_']:\n        if peer_id.startswith(prefix):\n            peer_id = peer_id[len(prefix):]\n    \n    # Convert to lowercase and remove special characters\n    peer_id = re.sub(r'[^a-z0-9]', '', peer_id.lower())\n    \n    # Ensure minimum length\n    if len(peer_id) < 8:\n        peer_id = peer_id + hashlib.sha256(peer_id.encode()).hexdigest()[:8]\n    \n    return peer_id[:32]  # Limit to 32 characters\n\n\ndef timeout_after(seconds: float):\n    \"\"\"Decorator to add timeout to async functions.\"\"\"\n    def decorator(func):\n        async def wrapper(*args, **kwargs):\n            try:\n                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)\n            except asyncio.TimeoutError:\n                logger.warning(f\"Function {func.__name__} timed out after {seconds}s\")\n                raise\n        return wrapper\n    return decorator\n\n\ndef retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):\n    \"\"\"Decorator to retry function on failure.\"\"\"\n    def decorator(func):\n        async def wrapper(*args, **kwargs):\n            last_exception = None\n            \n            for attempt in range(max_attempts):\n                try:\n                    return await func(*args, **kwargs)\n                except Exception as e:\n                    last_exception = e\n                    if attempt < max_attempts - 1:\n                        wait_time = delay * (backoff ** attempt)\n                        logger.debug(f\"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}\")\n                        await asyncio.sleep(wait_time)\n                    else:\n                        logger.error(f\"All {max_attempts} attempts failed: {e}\")\n            \n            raise last_exception\n        return wrapper\n    return decorator\n\n\ndef safe_json_dumps(obj: Any, **kwargs) -> str:\n    \"\"\"JSON dumps with safe handling of non-serializable objects.\"\"\"\n    def default_handler(o):\n        if isinstance(o, datetime):\n            return o.isoformat()\n        elif isinstance(o, timedelta):\n            return o.total_seconds()\n        elif hasattr(o, '__dict__'):\n            return o.__dict__\n        elif hasattr(o, 'to_dict'):\n            return o.to_dict()\n        else:\n            return str(o)\n    \n    import json\n    return json.dumps(obj, default=default_handler, **kwargs)\n\n\ndef deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:\n    \"\"\"Deep update dictionary with nested merge.\"\"\"\n    result = base_dict.copy()\n    \n    for key, value in update_dict.items():\n        if key in result and isinstance(result[key], dict) and isinstance(value, dict):\n            result[key] = deep_update(result[key], value)\n        else:\n            result[key] = value\n    \n    return result\n\n\ndef flatten_dict(d: Dict[str, Any], separator: str = '.', prefix: str = '') -> Dict[str, Any]:\n    \"\"\"Flatten nested dictionary using separator.\"\"\"\n    items = []\n    \n    for key, value in d.items():\n        new_key = f\"{prefix}{separator}{key}\" if prefix else key\n        \n        if isinstance(value, dict):\n            items.extend(flatten_dict(value, separator, new_key).items())\n        else:\n            items.append((new_key, value))\n    \n    return dict(items)\n\n\ndef unflatten_dict(d: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:\n    \"\"\"Unflatten dictionary using separator.\"\"\"\n    result = {}\n    \n    for key, value in d.items():\n        parts = key.split(separator)\n        current = result\n        \n        for part in parts[:-1]:\n            if part not in current:\n                current[part] = {}\n            current = current[part]\n        \n        current[parts[-1]] = value\n    \n    return result\n\n\n# Private helper functions\ndef _validate_multiaddress(address: str) -> bool:\n    \"\"\"Validate multiaddress format.\"\"\"\n    # Basic multiaddress validation\n    if not address.startswith('/'):\n        return False\n    \n    # Split into components\n    parts = address.split('/')[1:]  # Remove empty first part\n    \n    # Must have even number of parts (protocol/value pairs)\n    if len(parts) % 2 != 0:\n        return False\n    \n    # Check for valid protocol/value pairs\n    valid_protocols = ['ip4', 'ip6', 'tcp', 'udp', 'ws', 'wss', 'p2p']\n    for i in range(0, len(parts), 2):\n        protocol = parts[i]\n        if protocol not in valid_protocols:\n            return False\n        # Could add more specific validation for each protocol\n    \n    return True\n\n\ndef _parse_multiaddress(address: str) -> Optional[Tuple[str, Optional[int]]]:\n    \"\"\"Parse multiaddress format.\"\"\"\n    parts = address.split('/')[1:]  # Remove empty first part\n    \n    host = None\n    port = None\n    \n    for i in range(0, len(parts), 2):\n        protocol = parts[i]\n        value = parts[i + 1] if i + 1 < len(parts) else None\n        \n        if protocol in ['ip4', 'ip6'] and value:\n            host = value\n        elif protocol in ['tcp', 'udp'] and value:\n            try:\n                port = int(value)\n            except ValueError:\n                pass\n    \n    return (host, port) if host else None\n\n\ndef _validate_hostname(hostname: str) -> bool:\n    \"\"\"Validate hostname format.\"\"\"\n    if len(hostname) > 253:\n        return False\n    \n    # Check each label\n    labels = hostname.split('.')\n    for label in labels:\n        if not label or len(label) > 63:\n            return False\n        if not re.match(r'^[a-zA-Z0-9-]+$', label):\n            return False\n        if label.startswith('-') or label.endswith('-'):\n            return False\n    \n    return True