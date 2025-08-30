"""
Identifier generation and normalization for P2P infrastructure.
"""

import hashlib
import re
import time
import uuid


PEER_ID_PREFIXES = {'peer_', 'node_', 'id_'}
MIN_PEER_ID_LENGTH = 8
MAX_PEER_ID_LENGTH = 32


def generate_session_id(*, prefix: str = "session") -> str:
    """Generate a unique session identifier.
    
    Args:
        prefix: Prefix for the session ID
        
    Returns:
        Unique session identifier
    """
    timestamp = int(time.time() * 1000)
    unique_part = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique_part}"


def create_checksum(data: bytes, *, algorithm: str = "sha256") -> str:
    """Create checksum/hash of data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm to use
        
    Returns:
        Hex digest of the hash
        
    Raises:
        ValueError: If algorithm is not supported
    """
    hash_algorithms = {
        "sha256": hashlib.sha256,
        "sha1": hashlib.sha1,
        "md5": hashlib.md5,
        "blake2b": hashlib.blake2b,
    }
    
    if algorithm not in hash_algorithms:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    return hash_algorithms[algorithm](data).hexdigest()


def normalize_peer_id(peer_id: str) -> str:
    """Normalize peer ID to consistent format.
    
    Args:
        peer_id: Peer ID to normalize
        
    Returns:
        Normalized peer ID
    """
    for prefix in PEER_ID_PREFIXES:
        if peer_id.startswith(prefix):
            peer_id = peer_id[len(prefix):]
    
    peer_id = re.sub(r'[^a-z0-9]', '', peer_id.lower())
    
    if len(peer_id) < MIN_PEER_ID_LENGTH:
        peer_id = peer_id + hashlib.sha256(peer_id.encode()).hexdigest()[:8]
    
    return peer_id[:MAX_PEER_ID_LENGTH]