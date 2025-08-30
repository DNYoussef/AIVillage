"""
Network address validation utilities for P2P infrastructure.
"""

import ipaddress
import re
from typing import Optional, Tuple
from urllib.parse import urlparse


from .constants import (
    VALID_MULTIADDR_PROTOCOLS, LOCALHOST_ADDRESSES, MAX_HOSTNAME_LENGTH, 
    MAX_DNS_LABEL_LENGTH, PORT_MIN, PORT_MAX
)

LOCALHOST_VARIANTS = LOCALHOST_ADDRESSES  # Alias for backward compatibility
MAX_LABEL_LENGTH = MAX_DNS_LABEL_LENGTH   # Alias for backward compatibility


def validate_address(address: str) -> bool:
    """Validate network address format.
    
    Supports IPv4/IPv6, hostnames, multiaddresses, and URLs.
    
    Args:
        address: Address to validate
        
    Returns:
        True if address is valid format
    """
    if not address or not isinstance(address, str):
        return False
    
    if address.startswith('/'):
        return _validate_multiaddress(address)
    
    if '://' in address:
        return _validate_url(address)
    
    if ':' in address:
        host, port = _extract_host_port(address)
        if port is not None and not (PORT_MIN <= port <= PORT_MAX):
            return False
        address = host
    
    return _validate_ip_or_hostname(address)


def parse_address(address: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse network address into host and port components.
    
    Args:
        address: Network address
        
    Returns:
        Tuple of (host, port) or None if invalid
    """
    if not validate_address(address):
        return None
    
    if address.startswith('/'):
        return _parse_multiaddress(address)
    
    if '://' in address:
        return _parse_url(address)
    
    return _parse_host_port(address)


def is_local_address(address: str) -> bool:
    """Check if address is a local/loopback address.
    
    Args:
        address: Network address to check
        
    Returns:
        True if address is local/loopback
    """
    host, _ = parse_address(address) or (address, None)
    
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return host.lower() in LOCALHOST_VARIANTS


def _validate_multiaddress(address: str) -> bool:
    """Validate multiaddress format."""
    if not address.startswith('/'):
        return False
    
    parts = address.split('/')[1:]
    
    if len(parts) % 2 != 0:
        return False
    
    for i in range(0, len(parts), 2):
        protocol = parts[i]
        if protocol not in VALID_MULTIADDR_PROTOCOLS:
            return False
    
    return True


def _validate_url(address: str) -> bool:
    """Validate URL format."""
    try:
        parsed = urlparse(address)
        return bool(parsed.netloc)
    except Exception:
        return False


def _validate_ip_or_hostname(address: str) -> bool:
    """Validate IP address or hostname."""
    try:
        ipaddress.ip_address(address)
        return True
    except ValueError:
        return _validate_hostname(address)


def _validate_hostname(hostname: str) -> bool:
    """Validate hostname format."""
    if len(hostname) > MAX_HOSTNAME_LENGTH:
        return False
    
    labels = hostname.split('.')
    for label in labels:
        if not label or len(label) > MAX_LABEL_LENGTH:
            return False
        if not re.match(r'^[a-zA-Z0-9-]+$', label):
            return False
        if label.startswith('-') or label.endswith('-'):
            return False
    
    return True


def _extract_host_port(address: str) -> Tuple[str, Optional[int]]:
    """Extract host and port from address string."""
    if address.startswith('[') and ']:' in address:
        bracket_end = address.index(']:')
        host = address[1:bracket_end]
        try:
            port = int(address[bracket_end + 2:])
            return host, port
        except ValueError:
            return host, None
    
    if ':' in address:
        try:
            host, port_str = address.rsplit(':', 1)
            port = int(port_str)
            return host, port
        except ValueError:
            pass
    
    return address, None


def _parse_multiaddress(address: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse multiaddress format."""
    parts = address.split('/')[1:]
    
    host = None
    port = None
    
    for i in range(0, len(parts), 2):
        protocol = parts[i]
        value = parts[i + 1] if i + 1 < len(parts) else None
        
        if protocol in {'ip4', 'ip6'} and value:
            host = value
        elif protocol in {'tcp', 'udp'} and value:
            try:
                port = int(value)
            except ValueError:
                pass
    
    return (host, port) if host else None


def _parse_url(address: str) -> Optional[Tuple[str, Optional[int]]]:
    """Parse URL format."""
    try:
        parsed = urlparse(address)
        return parsed.hostname, parsed.port
    except Exception:
        return None


def _parse_host_port(address: str) -> Tuple[str, Optional[int]]:
    """Parse host:port format."""
    host, port = _extract_host_port(address)
    return host, port