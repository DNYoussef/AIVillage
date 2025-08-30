"""
Timing and performance utilities for P2P infrastructure.
"""

import time
from typing import Optional


def calculate_latency(start_time: float, *, end_time: Optional[float] = None) -> float:
    """Calculate latency between two timestamps.
    
    Args:
        start_time: Start timestamp (from time.time())
        end_time: End timestamp (defaults to current time)
        
    Returns:
        Latency in milliseconds
    """
    if end_time is None:
        end_time = time.time()
    
    return (end_time - start_time) * 1000


def estimate_bandwidth(bytes_transferred: int, *, duration_seconds: float) -> float:
    """Estimate bandwidth based on data transferred and time taken.
    
    Args:
        bytes_transferred: Number of bytes transferred
        duration_seconds: Duration in seconds
        
    Returns:
        Bandwidth in bytes per second
    """
    if duration_seconds <= 0:
        return 0.0
    
    return bytes_transferred / duration_seconds