"""Unified Configuration System."""

from typing import Dict, Any, Optional, List
import os
import threading
import yaml
from dataclasses import dataclass, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

@dataclass
class UnifiedConfig:
    """Configuration class for the RAG system."""
    
    # Core settings
    model_name: str = field(default="gpt-3.5-turbo")
    embedding_model: str = field(default="text-embedding-ada-002")
    
    # System parameters
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # RAG specific settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_doc: int = 10
    similarity_threshold: float = 0.7
    
    # Performance settings
    batch_size: int = 32
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    
    # Paths and storage
    data_dir: str = "data"
    cache_dir: str = "cache"
    index_path: str = "indexes"
    
    # Monitoring and logging
    log_level: str = "INFO"
    enable_monitoring: bool = True
    metrics_port: int = 8000

class ConfigWatcher(FileSystemEventHandler):
    """Watches for changes in configuration files and updates settings."""
    
    def __init__(self, config_path: str, callback):
        """
        Initialize config watcher.
        
        Args:
            config_path: Path to config file
            callback: Function to call when config changes
        """
        self.config_path = config_path
        self.callback = callback
        self._observer = None
        self._lock = threading.Lock()

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path == self.config_path:
            with self._lock:
                self.callback()

    def start(self):
        """Start watching config file."""
        if self._observer is None:
            self._observer = Observer()
            self._observer.schedule(self, os.path.dirname(self.config_path), recursive=False)
            self._observer.start()

    def stop(self):
        """Stop watching config file."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

class ConfigManager:
    """Manages configuration loading, validation, and updates."""
    
    def __init__(self, config_path: str):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.config = UnifiedConfig()
        self._watcher = ConfigWatcher(config_path, self.reload_config)
        self._lock = threading.Lock()
        self.load_config()

    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            with self._lock:
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                    
            self._validate_config()
        except Exception as e:
            raise ValueError(f"Error loading config: {str(e)}")

    def reload_config(self):
        """Reload configuration from file."""
        self.load_config()

    def _validate_config(self):
        """Validate configuration values."""
        if self.config.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not (0 <= self.config.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")
        if not (0 <= self.config.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")
        if self.config.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def start_watching(self):
        """Start watching config file for changes."""
        self._watcher.start()

    def stop_watching(self):
        """Stop watching config file."""
        self._watcher.stop()

    def get_config(self) -> UnifiedConfig:
        """
        Get current configuration.
        
        Returns:
            Current configuration object
        """
        with self._lock:
            return self.config

    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        with self._lock:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self._validate_config()
            
            # Save updates to file
            current_config = {
                key: getattr(self.config, key)
                for key in self.config.__annotations__
            }
            
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(current_config, f)

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Returns:
            Dictionary of model configuration
        """
        return {
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty
        }

    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get RAG-specific configuration.
        
        Returns:
            Dictionary of RAG configuration
        """
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "max_chunks_per_doc": self.config.max_chunks_per_doc,
            "similarity_threshold": self.config.similarity_threshold
        }

    def get_performance_config(self) -> Dict[str, Any]:
        """
        Get performance-related configuration.
        
        Returns:
            Dictionary of performance configuration
        """
        return {
            "batch_size": self.config.batch_size,
            "max_concurrent_requests": self.config.max_concurrent_requests,
            "request_timeout": self.config.request_timeout
        }

    def get_storage_paths(self) -> Dict[str, str]:
        """
        Get storage path configuration.
        
        Returns:
            Dictionary of storage paths
        """
        return {
            "data_dir": self.config.data_dir,
            "cache_dir": self.config.cache_dir,
            "index_path": self.config.index_path
        }

    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get monitoring configuration.
        
        Returns:
            Dictionary of monitoring configuration
        """
        return {
            "log_level": self.config.log_level,
            "enable_monitoring": self.config.enable_monitoring,
            "metrics_port": self.config.metrics_port
        }
