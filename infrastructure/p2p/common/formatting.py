"""
Text formatting utilities for P2P infrastructure.
"""


def format_bytes(byte_count: int, *, decimal_places: int = 2) -> str:
    """Format byte count into human-readable string.
    
    Args:
        byte_count: Number of bytes
        decimal_places: Number of decimal places to show
        
    Returns:
        Formatted string (e.g., "1.23 KB", "4.56 MB")
    """
    if byte_count == 0:
        return "0 B"
    
    BYTE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(byte_count)
    
    while size >= 1024 and unit_index < len(BYTE_UNITS) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.{decimal_places}f} {BYTE_UNITS[unit_index]}"


def format_duration(seconds: float, *, precision: int = 2) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        precision: Number of decimal places for sub-second durations
        
    Returns:
        Formatted string (e.g., "1.23s", "2m 30s", "1h 15m")
    """
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600  
    SECONDS_PER_DAY = 86400
    
    if seconds < 1:
        return f"{seconds:.{precision}f}s"
    
    if seconds < SECONDS_PER_MINUTE:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // SECONDS_PER_MINUTE)
    remaining_seconds = seconds % SECONDS_PER_MINUTE
    
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