# Core Common Utilities
# Production-ready utility functions for the AIVillage system

import hashlib
import json
import logging
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from functools import wraps
import threading

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class SystemArchitecture(Enum):
    """System architecture types."""
    X86_64 = "x86_64"
    ARM64 = "arm64"
    ARM32 = "arm32"
    UNKNOWN = "unknown"


class OperatingSystem(Enum):
    """Operating system types."""
    LINUX = "linux"
    WINDOWS = "windows"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    UNKNOWN = "unknown"


@dataclass
class SystemInfo:
    """System information structure."""
    
    hostname: str = field(default_factory=socket.gethostname)
    os_type: OperatingSystem = OperatingSystem.UNKNOWN
    os_version: str = "unknown"
    architecture: SystemArchitecture = SystemArchitecture.UNKNOWN
    cpu_count: int = 1
    memory_total_gb: float = 0.0
    python_version: str = field(default_factory=lambda: platform.python_version())
    
    def __post_init__(self):
        if self.os_type == OperatingSystem.UNKNOWN:
            self.os_type = self._detect_os()
        if self.os_version == "unknown":
            self.os_version = platform.platform()
        if self.architecture == SystemArchitecture.UNKNOWN:
            self.architecture = self._detect_architecture()
        if psutil and self.cpu_count == 1:
            self.cpu_count = psutil.cpu_count(logical=False) or 1
        if psutil and self.memory_total_gb == 0.0:
            self.memory_total_gb = psutil.virtual_memory().total / (1024**3)
    
    def _detect_os(self) -> OperatingSystem:
        system = platform.system().lower()
        if system == "linux":
            return OperatingSystem.LINUX
        elif system == "windows":
            return OperatingSystem.WINDOWS
        elif system == "darwin":
            return OperatingSystem.MACOS
        return OperatingSystem.UNKNOWN
    
    def _detect_architecture(self) -> SystemArchitecture:
        arch = platform.machine().lower()
        if arch in ["x86_64", "amd64"]:
            return SystemArchitecture.X86_64
        elif arch in ["arm64", "aarch64"]:
            return SystemArchitecture.ARM64
        elif arch in ["arm", "armv7l"]:
            return SystemArchitecture.ARM32
        return SystemArchitecture.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "os_type": self.os_type.value,
            "os_version": self.os_version,
            "architecture": self.architecture.value,
            "cpu_count": self.cpu_count,
            "memory_total_gb": round(self.memory_total_gb, 2),
            "python_version": self.python_version
        }


def generate_id(prefix: str = "", length: int = 8) -> str:
    """Generate a unique identifier."""
    unique_id = str(uuid.uuid4()).replace('-', '')[:length]
    return f"{prefix}{unique_id}" if prefix else unique_id


def generate_hash(data: Union[str, bytes, Dict[str, Any]], algorithm: str = "sha256") -> str:
    """Generate hash of data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_func = hashlib.new(algorithm)
    hash_func.update(data)
    return hash_func.hexdigest()


def safe_json_loads(data: str, default: Any = None) -> Any:
    """Safely parse JSON data."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.debug(f"Failed to parse JSON: {e}")
        return default


def safe_json_dumps(data: Any, default: str = "{}") -> str:
    """Safely serialize data to JSON."""
    try:
        return json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.debug(f"Failed to serialize JSON: {e}")
        return default


def get_local_ip() -> str:
    """Get local IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def is_port_open(host: str, port: int, timeout: float = 5.0) -> bool:
    """Check if a port is open on a host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.error, socket.timeout):
        return False


def find_free_port(start_port: int = 8000, end_port: int = 9000) -> Optional[int]:
    """Find a free port in the given range."""
    for port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, backoff_factor: float = 2.0):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.debug(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


def format_bytes(bytes_value: int) -> str:
    """Format bytes value in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def get_timestamp(utc: bool = True) -> str:
    """Get current timestamp in ISO format."""
    if utc:
        return datetime.now(timezone.utc).isoformat()
    else:
        return datetime.now().isoformat()


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def safe_file_write(file_path: Union[str, Path], content: str, backup: bool = True) -> bool:
    """Safely write content to file with optional backup."""
    path_obj = Path(file_path)
    
    try:
        if backup and path_obj.exists():
            backup_path = path_obj.with_suffix(path_obj.suffix + ".bak")
            path_obj.rename(backup_path)
        
        temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
        temp_path.write_text(content, encoding='utf-8')
        temp_path.rename(path_obj)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        return False


def safe_file_read(file_path: Union[str, Path], default: str = "") -> str:
    """Safely read content from file."""
    try:
        return Path(file_path).read_text(encoding='utf-8')
    except Exception as e:
        logger.debug(f"Failed to read file {file_path}: {e}")
        return default


class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
    
    def allow_call(self) -> bool:
        """Check if a call is allowed under the rate limit."""
        current_time = time.time()
        
        with self.lock:
            self.calls = [call_time for call_time in self.calls if current_time - call_time < self.time_window]
            
            if len(self.calls) < self.max_calls:
                self.calls.append(current_time)
                return True
            
            return False


class EventEmitter:
    """Simple event emitter implementation."""
    
    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
    
    def on(self, event: str, listener: Callable):
        """Add event listener."""
        with self.lock:
            if event not in self.listeners:
                self.listeners[event] = []
            self.listeners[event].append(listener)
    
    def emit(self, event: str, *args, **kwargs):
        """Emit event to all listeners."""
        listeners_copy = []
        
        with self.lock:
            if event in self.listeners:
                listeners_copy = self.listeners[event].copy()
        
        for listener in listeners_copy:
            try:
                listener(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event listener for '{event}': {e}")


class HealthChecker:
    """Health check utility."""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        results = {
            "status": "healthy",
            "timestamp": get_timestamp(),
            "checks": {},
            "system_info": SystemInfo().to_dict()
        }
        
        all_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration = time.time() - start_time
                
                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "duration_ms": round(duration * 1000, 2)
                }
                
                if not is_healthy:
                    all_healthy = False
                    
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                all_healthy = False
        
        if not all_healthy:
            results["status"] = "unhealthy"
        
        self.results = results
        return results


# Global health checker instance
health_checker = HealthChecker()


# Register default health checks
def _check_basic_functionality() -> bool:
    """Check basic system functionality."""
    return True


health_checker.register_check("basic_functionality", _check_basic_functionality)


# Backward compatibility - try to import from actual infrastructure locations first
try:
    from core.common import *
except ImportError:
    # Use the implementations defined above
    pass
