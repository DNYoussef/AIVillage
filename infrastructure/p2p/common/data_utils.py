"""
Data manipulation utilities for P2P infrastructure.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """JSON dumps with safe handling of non-serializable objects.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation
    """
    def default_handler(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        elif isinstance(o, timedelta):
            return o.total_seconds()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        elif hasattr(o, 'to_dict'):
            return o.to_dict()
        else:
            return str(o)
    
    return json.dumps(obj, default=default_handler, **kwargs)


def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Deep update dictionary with nested merge.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], *, separator: str = '.', prefix: str = '') -> Dict[str, Any]:
    """Flatten nested dictionary using separator.
    
    Args:
        d: Dictionary to flatten
        separator: Separator string for keys
        prefix: Prefix for keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in d.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator=separator, prefix=new_key).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_dict(d: Dict[str, Any], *, separator: str = '.') -> Dict[str, Any]:
    """Unflatten dictionary using separator.
    
    Args:
        d: Flattened dictionary
        separator: Separator string used in keys
        
    Returns:
        Unflattened nested dictionary
    """
    result = {}
    
    for key, value in d.items():
        parts = key.split(separator)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result