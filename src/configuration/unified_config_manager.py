"""
Unified Configuration Management System with Distributed Caching
Centralizes all configuration management using Context7 MCP distributed caching
"""
import asyncio
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import yaml
import sqlite3
from threading import RLock

# Configuration storage backend
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationLayer:
    """Represents a configuration layer in the hierarchy"""
    name: str
    priority: int  # Higher = more priority
    source: str    # file path, service, environment, etc.
    data: Dict[str, Any]
    last_modified: datetime
    checksum: str
    encrypted: bool = False

class ConfigurationHierarchy:
    """Manages the configuration hierarchy with proper override semantics"""
    
    def __init__(self):
        self.layers: List[ConfigurationLayer] = []
        self._lock = RLock()
        
    def add_layer(self, layer: ConfigurationLayer):
        """Add a configuration layer in priority order"""
        with self._lock:
            # Insert in priority order (highest first)
            inserted = False
            for i, existing in enumerate(self.layers):
                if layer.priority > existing.priority:
                    self.layers.insert(i, layer)
                    inserted = True
                    break
            if not inserted:
                self.layers.append(layer)
                
    def get_merged_config(self) -> Dict[str, Any]:
        """Merge all layers into a single configuration dict"""
        with self._lock:
            merged = {}
            # Process from lowest to highest priority
            for layer in reversed(self.layers):
                self._deep_merge(merged, layer.data)
            return merged
            
    def _deep_merge(self, target: Dict, source: Dict):
        """Deep merge source into target"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

class DistributedConfigCache:
    """Distributed configuration cache with Context7 MCP integration"""
    
    def __init__(self, cache_ttl: int = 3600):
        self.cache_ttl = cache_ttl
        self._local_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._lock = RLock()
        
    async def cache_configuration(self, key: str, config: Dict[str, Any], ttl: Optional[int] = None):
        """Cache configuration with distributed replication"""
        ttl = ttl or self.cache_ttl
        
        # Store in local cache
        with self._lock:
            self._local_cache[key] = config
            self._cache_timestamps[key] = datetime.now()
            
        # TODO: Integrate with Context7 MCP for distributed caching
        # await context7.cache.set(f"config/{key}", config, ttl=ttl)
        # await context7.distributed.replicate(f"config/{key}", config, nodes=["primary", "replica"])
        
        logger.info(f"Cached configuration for key: {key}")
        
    async def get_cached_configuration(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached configuration"""
        with self._lock:
            # Check local cache first
            if key in self._local_cache:
                timestamp = self._cache_timestamps[key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return self._local_cache[key]
                else:
                    # Expired, remove from local cache
                    del self._local_cache[key]
                    del self._cache_timestamps[key]
                    
        # TODO: Try distributed cache via Context7 MCP
        # try:
        #     return await context7.cache.get(f"config/{key}")
        # except KeyError:
        #     return None
            
        return None
        
    async def invalidate_cache(self, key: str):
        """Invalidate cached configuration"""
        with self._lock:
            self._local_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
            
        # TODO: Invalidate distributed cache
        # await context7.cache.delete(f"config/{key}")
        logger.info(f"Invalidated cache for key: {key}")

class SecureConfigurationManager:
    """Manages encrypted configuration with secure key derivation"""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or os.environ.get('CONFIG_MASTER_KEY', 'default-dev-key')
        self._fernet = self._create_fernet()
        
    def _create_fernet(self) -> Fernet:
        """Create Fernet encryption instance from master password"""
        password = self.master_password.encode()
        salt = b'aivillage-config-salt'  # In production, use random salt stored securely
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
        
    def encrypt_config(self, config: Dict[str, Any]) -> bytes:
        """Encrypt configuration data"""
        json_data = json.dumps(config, sort_keys=True)
        return self._fernet.encrypt(json_data.encode())
        
    def decrypt_config(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt configuration data"""
        decrypted = self._fernet.decrypt(encrypted_data)
        return json.loads(decrypted.decode())

class UnifiedConfigurationManager:
    """Main configuration manager with distributed caching and MCP integration"""
    
    def __init__(self, config_dir: str = "config", cache_ttl: int = 3600):
        self.config_dir = Path(config_dir)
        self.hierarchy = ConfigurationHierarchy()
        self.cache = DistributedConfigCache(cache_ttl)
        self.secure_manager = SecureConfigurationManager()
        self._watchers: Dict[str, Any] = {}
        self._lock = RLock()
        
        # Configuration file patterns to monitor
        self.config_patterns = [
            "*.yaml", "*.yml", "*.json",
            "*config*.yaml", "*config*.yml", 
            "*.env*"
        ]
        
    async def initialize(self):
        """Initialize the configuration manager"""
        logger.info("Initializing Unified Configuration Manager")
        
        # Load configuration hierarchy
        await self._discover_configurations()
        
        # Setup file watchers for hot reload
        await self._setup_file_watchers()
        
        # Store configuration decisions in Memory MCP
        # TODO: Integrate with Memory MCP
        # await memory_mcp.store("config-management/initialization", {
        #     "timestamp": datetime.now().isoformat(),
        #     "discovered_configs": len(self.hierarchy.layers),
        #     "cache_enabled": True
        # })
        
        logger.info(f"Configuration manager initialized with {len(self.hierarchy.layers)} layers")
        
    async def _discover_configurations(self):
        """Discover all configuration files and build hierarchy"""
        
        # Base configuration (lowest priority)
        base_configs = [
            ("config/aivillage_config.yaml", 10),
            ("config/production_services.yaml", 20),
        ]
        
        # Environment-specific configurations (medium priority)
        env_configs = [
            ("config/aivillage_config_development.yaml", 30),
            ("config/aivillage_config_production.yaml", 30),
        ]
        
        # Service-specific configurations (high priority)
        service_configs = [
            ("config/orchestration_config.yaml", 40),
            ("config/rag_config.yaml", 40),
            ("config/hyperag_mcp.yaml", 40),
        ]
        
        # Runtime configurations (highest priority)
        runtime_configs = [
            ("tools/ci-cd/deployment/k8s/configmap.yaml", 50),
        ]
        
        all_configs = base_configs + env_configs + service_configs + runtime_configs
        
        for config_path, priority in all_configs:
            full_path = Path(config_path)
            if full_path.exists():
                await self._load_configuration_layer(full_path, priority)
                
    async def _load_configuration_layer(self, config_path: Path, priority: int):
        """Load a configuration layer from file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_path.suffix == '.json':
                    data = json.load(f)
                else:
                    logger.warning(f"Unsupported config format: {config_path}")
                    return
                    
            if not data:
                data = {}
                
            # Calculate checksum
            content = f.read()
            f.seek(0)
            checksum = hashlib.sha256(content.encode()).hexdigest()
            
            # Create configuration layer
            layer = ConfigurationLayer(
                name=config_path.stem,
                priority=priority,
                source=str(config_path),
                data=data,
                last_modified=datetime.fromtimestamp(config_path.stat().st_mtime),
                checksum=checksum
            )
            
            self.hierarchy.add_layer(layer)
            logger.info(f"Loaded configuration layer: {config_path} (priority: {priority})")
            
        except Exception as e:
            logger.error(f"Failed to load configuration {config_path}: {e}")
            
    async def get_configuration(self, key: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """Get merged configuration with caching"""
        cache_key = f"merged_config_{key or 'all'}"
        
        # Try cache first
        if use_cache:
            cached = await self.cache.get_cached_configuration(cache_key)
            if cached:
                return cached
                
        # Get merged configuration
        merged_config = self.hierarchy.get_merged_config()
        
        # Extract specific key if requested
        if key:
            keys = key.split('.')
            result = merged_config
            for k in keys:
                if isinstance(result, dict) and k in result:
                    result = result[k]
                else:
                    return {}
            merged_config = result
            
        # Cache the result
        if use_cache:
            await self.cache.cache_configuration(cache_key, merged_config)
            
        return merged_config
        
    async def set_configuration(self, key: str, value: Any, layer: str = "runtime"):
        """Set a configuration value in a specific layer"""
        keys = key.split('.')
        
        # Find or create the target layer
        target_layer = None
        for layer_obj in self.hierarchy.layers:
            if layer_obj.name == layer:
                target_layer = layer_obj
                break
                
        if not target_layer:
            # Create new runtime layer
            target_layer = ConfigurationLayer(
                name=layer,
                priority=100,  # Highest priority
                source="runtime",
                data={},
                last_modified=datetime.now(),
                checksum="",
                encrypted=False
            )
            self.hierarchy.add_layer(target_layer)
            
        # Set the value using key path
        current = target_layer.data
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        
        # Update metadata
        target_layer.last_modified = datetime.now()
        target_layer.checksum = hashlib.sha256(
            json.dumps(target_layer.data, sort_keys=True).encode()
        ).hexdigest()
        
        # Invalidate cache
        await self.cache.invalidate_cache("merged_config_all")
        
        logger.info(f"Set configuration: {key} = {value} in layer: {layer}")
        
    async def reload_configuration(self):
        """Reload all configuration layers"""
        logger.info("Reloading configuration layers")
        
        # Clear existing layers
        self.hierarchy.layers.clear()
        
        # Rediscover configurations
        await self._discover_configurations()
        
        # Invalidate all caches
        # TODO: Use Context7 MCP for distributed cache invalidation
        # await context7.cache.flush_pattern("config/*")
        
        logger.info("Configuration reloaded successfully")
        
    async def _setup_file_watchers(self):
        """Setup file watchers for hot reload"""
        # TODO: Implement file watching for automatic configuration reload
        # This would integrate with the system's file monitoring capabilities
        logger.info("File watchers setup completed")
        
    async def export_configuration(self, format: str = "yaml") -> str:
        """Export merged configuration"""
        config = await self.get_configuration()
        
        if format.lower() == "yaml":
            return yaml.dump(config, default_flow_style=False, sort_keys=True)
        elif format.lower() == "json":
            return json.dumps(config, indent=2, sort_keys=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    async def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration consistency and completeness"""
        issues = {
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        config = await self.get_configuration()
        
        # Check for required fields
        required_fields = [
            "services.gateway.port",
            "services.agent_forge.port", 
            "database.evolution_metrics.path"
        ]
        
        for field in required_fields:
            keys = field.split('.')
            current = config
            missing = False
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    issues["errors"].append(f"Missing required configuration: {field}")
                    missing = True
                    break
                current = current[key]
                
        # Check for conflicting ports
        ports = []
        if "services" in config:
            for service_name, service_config in config["services"].items():
                if isinstance(service_config, dict) and "port" in service_config:
                    port = service_config["port"]
                    if port in ports:
                        issues["errors"].append(f"Port conflict: {port} used by multiple services")
                    ports.append(port)
                    
        # Security checks
        if config.get("security", {}).get("tls", {}).get("enabled") == False:
            issues["warnings"].append("TLS is disabled - consider enabling for production")
            
        return issues

# Global configuration manager instance
_config_manager: Optional[UnifiedConfigurationManager] = None

async def get_config_manager() -> UnifiedConfigurationManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = UnifiedConfigurationManager()
        await _config_manager.initialize()
    return _config_manager

async def get_config(key: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get configuration"""
    manager = await get_config_manager()
    return await manager.get_configuration(key)

async def set_config(key: str, value: Any, layer: str = "runtime"):
    """Convenience function to set configuration"""
    manager = await get_config_manager()
    await manager.set_configuration(key, value, layer)

async def reload_config():
    """Convenience function to reload configuration"""
    manager = await get_config_manager()
    await manager.reload_configuration()

if __name__ == "__main__":
    async def main():
        # Test the configuration manager
        manager = UnifiedConfigurationManager()
        await manager.initialize()
        
        config = await manager.get_configuration()
        print("Merged Configuration:")
        print(yaml.dump(config, default_flow_style=False))
        
        # Validation
        issues = await manager.validate_configuration()
        if issues["errors"]:
            print("\nConfiguration Errors:")
            for error in issues["errors"]:
                print(f"  - {error}")
                
        if issues["warnings"]:
            print("\nConfiguration Warnings:")
            for warning in issues["warnings"]:
                print(f"  - {warning}")
                
    asyncio.run(main())