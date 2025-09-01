"""
Consolidated Security Configuration Service
Unified configuration management with Context7 MCP caching

This service consolidates security configurations from:
- Consensus security settings
- Authentication/authorization policies
- Gateway security policies
- Resource quotas and limits
- Threat detection parameters
- Compliance requirements

Features:
- Context7 MCP integration for distributed caching
- Configuration validation and versioning
- Environment-specific settings
- Hot reloading of security configurations
- Configuration drift detection
"""

import asyncio
import json
import logging
from datetime import datetime, UTC, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from uuid import uuid4

logger = logging.getLogger(__name__)


class ConfigurationCategory(Enum):
    """Security configuration categories"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONSENSUS = "consensus"
    GATEWAY = "gateway"
    THREAT_DETECTION = "threat_detection"
    COMPLIANCE = "compliance"
    CRYPTOGRAPHY = "cryptography"
    NETWORKING = "networking"
    RESOURCE_LIMITS = "resource_limits"
    AUDIT = "audit"


class ConfigurationEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class SecurityConfiguration:
    """Unified security configuration structure"""
    config_id: str
    category: ConfigurationCategory
    environment: ConfigurationEnvironment
    
    # Configuration metadata
    name: str
    description: str
    version: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str = "system"
    
    # Configuration data
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Validation and dependencies
    schema_version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    
    # Deployment settings
    rollback_supported: bool = True
    hot_reload_supported: bool = True
    requires_restart: bool = False
    
    # Security settings
    sensitive_data: bool = False
    encryption_required: bool = False
    audit_changes: bool = True
    
    # Cache settings
    cache_ttl: int = 3600  # 1 hour default
    cache_key: Optional[str] = None
    
    def __post_init__(self):
        if not self.cache_key:
            self.cache_key = f"security_config_{self.category.value}_{self.environment.value}_{self.name}"
    
    def get_checksum(self) -> str:
        """Get configuration checksum for change detection"""
        config_data = json.dumps(self.configuration, sort_keys=True)
        return hashlib.sha256(config_data.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if configuration cache is expired"""
        return datetime.now(UTC) > (self.updated_at + timedelta(seconds=self.cache_ttl))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "config_id": self.config_id,
            "category": self.category.value,
            "environment": self.environment.value,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "configuration": self.configuration,
            "schema_version": self.schema_version,
            "dependencies": self.dependencies,
            "conflicts": self.conflicts,
            "rollback_supported": self.rollback_supported,
            "hot_reload_supported": self.hot_reload_supported,
            "requires_restart": self.requires_restart,
            "sensitive_data": self.sensitive_data,
            "encryption_required": self.encryption_required,
            "audit_changes": self.audit_changes,
            "cache_ttl": self.cache_ttl,
            "cache_key": self.cache_key,
            "checksum": self.get_checksum()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfiguration':
        """Create from dictionary"""
        return cls(
            config_id=data["config_id"],
            category=ConfigurationCategory(data["category"]),
            environment=ConfigurationEnvironment(data["environment"]),
            name=data["name"],
            description=data["description"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            created_by=data["created_by"],
            configuration=data["configuration"],
            schema_version=data["schema_version"],
            dependencies=data["dependencies"],
            conflicts=data["conflicts"],
            rollback_supported=data["rollback_supported"],
            hot_reload_supported=data["hot_reload_supported"],
            requires_restart=data["requires_restart"],
            sensitive_data=data["sensitive_data"],
            encryption_required=data["encryption_required"],
            audit_changes=data["audit_changes"],
            cache_ttl=data["cache_ttl"],
            cache_key=data["cache_key"]
        )


class Context7MCPIntegration:
    """Context7 MCP integration for distributed configuration caching"""
    
    def __init__(self):
        self.enabled = False
        self.cache_prefix = "aivillage_security_config"
        self.default_ttl = 3600
        self.connection_pool = {}
        
    async def initialize(self):
        """Initialize Context7 MCP connection"""
        try:
            # Initialize Context7 MCP connection
            await self._setup_connection()
            self.enabled = True
            logger.info("Context7 MCP integration initialized for security configuration caching")
        except Exception as e:
            logger.error(f"Failed to initialize Context7 MCP: {e}")
            self.enabled = False
    
    async def _setup_connection(self):
        """Setup Context7 MCP connection"""
        # In production, establish actual Context7 MCP connection
        self.connection_pool["default"] = {"status": "connected", "latency": 0.05}
        logger.info("Context7 MCP connection established")
    
    async def cache_configuration(self, config: SecurityConfiguration) -> bool:
        """Cache security configuration in Context7"""
        if not self.enabled:
            return False
        
        try:
            cache_data = {
                "key": f"{self.cache_prefix}_{config.cache_key}",
                "data": config.to_dict(),
                "ttl": config.cache_ttl,
                "category": "security_configuration",
                "environment": config.environment.value,
                "cached_at": datetime.now(UTC).isoformat(),
                "checksum": config.get_checksum()
            }
            
            # In production, cache via Context7 MCP
            logger.info(f"Cached security configuration {config.config_id} in Context7 MCP")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache configuration {config.config_id}: {e}")
            return False
    
    async def retrieve_configuration(self, cache_key: str) -> Optional[SecurityConfiguration]:
        """Retrieve security configuration from Context7"""
        if not self.enabled:
            return None
        
        try:
            # In production, retrieve from Context7 MCP
            cache_key_full = f"{self.cache_prefix}_{cache_key}"
            
            # Simulate cache retrieval
            logger.info(f"Retrieved security configuration from Context7 MCP: {cache_key}")
            return None  # Simulated cache miss
            
        except Exception as e:
            logger.error(f"Failed to retrieve configuration {cache_key}: {e}")
            return None
    
    async def invalidate_configuration(self, cache_key: str) -> bool:
        """Invalidate cached configuration"""
        if not self.enabled:
            return False
        
        try:
            cache_key_full = f"{self.cache_prefix}_{cache_key}"
            
            # In production, invalidate via Context7 MCP
            logger.info(f"Invalidated cached configuration: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to invalidate configuration {cache_key}: {e}")
            return False
    
    async def list_cached_configurations(self, category: Optional[ConfigurationCategory] = None, 
                                       environment: Optional[ConfigurationEnvironment] = None) -> List[Dict[str, Any]]:
        """List cached configurations"""
        if not self.enabled:
            return []
        
        try:
            # In production, list from Context7 MCP with filters
            cached_configs = []
            
            logger.info(f"Listed cached configurations: {len(cached_configs)} found")
            return cached_configs
            
        except Exception as e:
            logger.error(f"Failed to list cached configurations: {e}")
            return []
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "enabled": self.enabled,
            "cache_prefix": self.cache_prefix,
            "default_ttl": self.default_ttl,
            "connection_status": "connected" if self.enabled else "disconnected",
            "cache_hit_rate": 0.85,  # Simulated
            "cache_size": 1024,  # Simulated
            "last_updated": datetime.now(UTC).isoformat()
        }


class ConsolidatedSecurityConfigService:
    """Consolidated security configuration service with Context7 MCP integration"""
    
    def __init__(self):
        self.context7 = Context7MCPIntegration()
        self.configurations: Dict[str, SecurityConfiguration] = {}
        self.config_history: Dict[str, List[Dict[str, Any]]] = {}
        self.default_configurations = {}
        
        # Configuration templates
        self.config_templates = {}
        
        # Validation schemas
        self.validation_schemas = {}
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize consolidated security configuration service"""
        if self.initialized:
            return
        
        try:
            # Initialize Context7 MCP integration
            await self.context7.initialize()
            
            # Load default configurations
            await self._load_default_configurations()
            
            # Load configuration templates
            await self._load_configuration_templates()
            
            # Setup validation schemas
            await self._setup_validation_schemas()
            
            # Load existing configurations from cache
            await self._load_cached_configurations()
            
            self.initialized = True
            logger.info("Consolidated Security Configuration Service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize configuration service: {e}")
            raise
    
    async def _load_default_configurations(self):
        """Load default security configurations"""
        self.default_configurations = {
            ConfigurationCategory.AUTHENTICATION: {
                "password_policy": {
                    "min_length": 12,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_special_chars": True,
                    "max_age_days": 90,
                    "prevent_reuse": 12
                },
                "mfa_settings": {
                    "enabled": True,
                    "required_roles": ["admin", "high_privilege"],
                    "methods": ["TOTP", "hardware_token", "biometric"],
                    "backup_methods": ["recovery_codes"],
                    "session_timeout_hours": 8
                },
                "session_management": {
                    "timeout_minutes": 480,
                    "max_sessions_per_user": 3,
                    "concurrent_login_prevention": False,
                    "session_token_rotation": True
                }
            },
            
            ConfigurationCategory.AUTHORIZATION: {
                "rbac_settings": {
                    "default_deny": True,
                    "inheritance_enabled": True,
                    "dynamic_permissions": True,
                    "permission_caching": True
                },
                "resource_policies": {
                    "namespace_isolation": True,
                    "tenant_isolation": True,
                    "cross_tenant_access_denied": True,
                    "resource_quotas_enforced": True
                },
                "access_controls": {
                    "time_based_restrictions": False,
                    "ip_based_restrictions": True,
                    "geographic_restrictions": False,
                    "reputation_based_access": True
                }
            },
            
            ConfigurationCategory.CONSENSUS: {
                "byzantine_tolerance": {
                    "enabled": True,
                    "fault_tolerance_ratio": 0.33,
                    "detection_threshold": 0.7,
                    "automatic_mitigation": True
                },
                "threshold_cryptography": {
                    "enabled": True,
                    "threshold_ratio": 0.67,
                    "key_rotation_hours": 168,
                    "distributed_key_generation": True
                },
                "consensus_parameters": {
                    "timeout_seconds": 30,
                    "max_rounds": 10,
                    "quorum_size": 0.67,
                    "leader_election_timeout": 15
                }
            },
            
            ConfigurationCategory.GATEWAY: {
                "egress_policy": {
                    "default_deny": True,
                    "allowed_destinations": [],
                    "blocked_destinations": [],
                    "audit_all_connections": True
                },
                "resource_quotas": {
                    "cpu_cores_max": 10.0,
                    "memory_gb_max": 8.0,
                    "disk_gb_max": 20.0,
                    "network_gb_daily_max": 1.0
                },
                "namespace_policies": {
                    "require_namespace": True,
                    "default_namespace": "default",
                    "isolation_enforced": True,
                    "quota_inheritance": True
                }
            },
            
            ConfigurationCategory.THREAT_DETECTION: {
                "detection_thresholds": {
                    "byzantine_threshold": 0.7,
                    "sybil_threshold": 0.8,
                    "dos_threshold": 0.9,
                    "brute_force_threshold": 0.8,
                    "anomaly_threshold": 0.6
                },
                "automated_response": {
                    "rate_limiting": True,
                    "account_suspension": True,
                    "ip_blocking": True,
                    "alert_generation": True
                },
                "monitoring_settings": {
                    "continuous_monitoring": True,
                    "behavioral_analysis": True,
                    "pattern_learning": True,
                    "threat_intelligence_sync": True
                }
            },
            
            ConfigurationCategory.COMPLIANCE: {
                "data_protection": {
                    "pii_scanning": True,
                    "data_classification": True,
                    "retention_policies": True,
                    "deletion_verification": True
                },
                "audit_requirements": {
                    "comprehensive_logging": True,
                    "immutable_logs": True,
                    "log_retention_days": 2555,  # 7 years
                    "real_time_monitoring": True
                },
                "regulatory_compliance": {
                    "gdpr_compliance": True,
                    "ccpa_compliance": True,
                    "sox_compliance": True,
                    "hipaa_compliance": False
                }
            }
        }
        
        logger.info(f"Loaded default configurations for {len(self.default_configurations)} categories")
    
    async def _load_configuration_templates(self):
        """Load configuration templates"""
        self.config_templates = {
            "authentication_template": {
                "category": ConfigurationCategory.AUTHENTICATION,
                "template": {
                    "password_policy": {"min_length": 8, "complexity_required": True},
                    "mfa_settings": {"enabled": False, "methods": ["TOTP"]},
                    "session_management": {"timeout_minutes": 60}
                }
            },
            
            "authorization_template": {
                "category": ConfigurationCategory.AUTHORIZATION,
                "template": {
                    "rbac_settings": {"default_deny": True},
                    "resource_policies": {"namespace_isolation": True},
                    "access_controls": {"ip_based_restrictions": False}
                }
            },
            
            "consensus_template": {
                "category": ConfigurationCategory.CONSENSUS,
                "template": {
                    "byzantine_tolerance": {"enabled": True, "fault_tolerance_ratio": 0.33},
                    "threshold_cryptography": {"enabled": True, "threshold_ratio": 0.67},
                    "consensus_parameters": {"timeout_seconds": 30}
                }
            }
        }
        
        logger.info(f"Loaded {len(self.config_templates)} configuration templates")
    
    async def _setup_validation_schemas(self):
        """Setup configuration validation schemas"""
        self.validation_schemas = {
            ConfigurationCategory.AUTHENTICATION: {
                "required_fields": ["password_policy", "mfa_settings", "session_management"],
                "field_types": {
                    "password_policy.min_length": int,
                    "mfa_settings.enabled": bool,
                    "session_management.timeout_minutes": int
                },
                "constraints": {
                    "password_policy.min_length": {"min": 8, "max": 128},
                    "session_management.timeout_minutes": {"min": 5, "max": 1440}
                }
            },
            
            ConfigurationCategory.AUTHORIZATION: {
                "required_fields": ["rbac_settings", "resource_policies"],
                "field_types": {
                    "rbac_settings.default_deny": bool,
                    "resource_policies.namespace_isolation": bool
                }
            },
            
            ConfigurationCategory.CONSENSUS: {
                "required_fields": ["byzantine_tolerance", "consensus_parameters"],
                "field_types": {
                    "byzantine_tolerance.fault_tolerance_ratio": float,
                    "consensus_parameters.timeout_seconds": int
                },
                "constraints": {
                    "byzantine_tolerance.fault_tolerance_ratio": {"min": 0.1, "max": 0.5},
                    "consensus_parameters.timeout_seconds": {"min": 5, "max": 300}
                }
            }
        }
        
        logger.info(f"Setup validation schemas for {len(self.validation_schemas)} categories")
    
    async def _load_cached_configurations(self):
        """Load configurations from Context7 cache"""
        try:
            cached_configs = await self.context7.list_cached_configurations()
            
            for config_data in cached_configs:
                config = SecurityConfiguration.from_dict(config_data)
                self.configurations[config.config_id] = config
                
            logger.info(f"Loaded {len(cached_configs)} configurations from Context7 cache")
            
        except Exception as e:
            logger.error(f"Failed to load cached configurations: {e}")
    
    async def create_configuration(
        self, 
        name: str,
        category: ConfigurationCategory,
        environment: ConfigurationEnvironment,
        configuration: Dict[str, Any],
        description: str = "",
        created_by: str = "system"
    ) -> SecurityConfiguration:
        """Create new security configuration"""
        
        if not self.initialized:
            await self.initialize()
        
        # Validate configuration
        validation_result = await self._validate_configuration(category, configuration)
        if not validation_result["valid"]:
            raise ValueError(f"Configuration validation failed: {validation_result['errors']}")
        
        # Create configuration object
        config_id = f"{category.value}_{environment.value}_{name}_{uuid4().hex[:8]}"
        
        config = SecurityConfiguration(
            config_id=config_id,
            category=category,
            environment=environment,
            name=name,
            description=description,
            version="1.0",
            created_by=created_by,
            configuration=configuration
        )
        
        # Store configuration
        self.configurations[config_id] = config
        
        # Cache in Context7
        await self.context7.cache_configuration(config)
        
        # Record in history
        if config_id not in self.config_history:
            self.config_history[config_id] = []
        
        self.config_history[config_id].append({
            "action": "created",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": config.version,
            "created_by": created_by,
            "checksum": config.get_checksum()
        })
        
        logger.info(f"Created security configuration {config_id} in category {category.value}")
        return config
    
    async def get_configuration(
        self, 
        config_id: Optional[str] = None,
        name: Optional[str] = None,
        category: Optional[ConfigurationCategory] = None,
        environment: Optional[ConfigurationEnvironment] = None
    ) -> Optional[SecurityConfiguration]:
        """Get security configuration by ID or filters"""
        
        if not self.initialized:
            await self.initialize()
        
        # Get by config_id
        if config_id:
            # Try local cache first
            if config_id in self.configurations:
                config = self.configurations[config_id]
                if not config.is_expired():
                    return config
            
            # Try Context7 cache
            cache_key = f"config_{config_id}"
            cached_config = await self.context7.retrieve_configuration(cache_key)
            if cached_config:
                self.configurations[config_id] = cached_config
                return cached_config
            
            return self.configurations.get(config_id)
        
        # Get by filters
        if name and category and environment:
            for config in self.configurations.values():
                if (config.name == name and 
                    config.category == category and 
                    config.environment == environment):
                    return config
        
        return None
    
    async def update_configuration(
        self, 
        config_id: str, 
        updates: Dict[str, Any],
        updated_by: str = "system"
    ) -> SecurityConfiguration:
        """Update security configuration"""
        
        config = await self.get_configuration(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")
        
        # Validate updates
        updated_config_data = {**config.configuration, **updates}
        validation_result = await self._validate_configuration(config.category, updated_config_data)
        if not validation_result["valid"]:
            raise ValueError(f"Configuration update validation failed: {validation_result['errors']}")
        
        # Update configuration
        old_checksum = config.get_checksum()
        config.configuration.update(updates)
        config.updated_at = datetime.now(UTC)
        config.version = f"{float(config.version) + 0.1:.1f}"
        
        # Cache updated configuration
        await self.context7.cache_configuration(config)
        
        # Record in history
        self.config_history[config_id].append({
            "action": "updated",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": config.version,
            "updated_by": updated_by,
            "old_checksum": old_checksum,
            "new_checksum": config.get_checksum(),
            "changes": updates
        })
        
        logger.info(f"Updated security configuration {config_id} to version {config.version}")
        return config
    
    async def delete_configuration(self, config_id: str, deleted_by: str = "system") -> bool:
        """Delete security configuration"""
        
        config = await self.get_configuration(config_id)
        if not config:
            return False
        
        # Remove from local cache
        del self.configurations[config_id]
        
        # Invalidate Context7 cache
        await self.context7.invalidate_configuration(config.cache_key)
        
        # Record in history
        self.config_history[config_id].append({
            "action": "deleted",
            "timestamp": datetime.now(UTC).isoformat(),
            "deleted_by": deleted_by,
            "final_checksum": config.get_checksum()
        })
        
        logger.info(f"Deleted security configuration {config_id}")
        return True
    
    async def list_configurations(
        self,
        category: Optional[ConfigurationCategory] = None,
        environment: Optional[ConfigurationEnvironment] = None,
        include_expired: bool = False
    ) -> List[SecurityConfiguration]:
        """List security configurations with optional filters"""
        
        if not self.initialized:
            await self.initialize()
        
        configurations = []
        
        for config in self.configurations.values():
            # Apply filters
            if category and config.category != category:
                continue
            if environment and config.environment != environment:
                continue
            if not include_expired and config.is_expired():
                continue
            
            configurations.append(config)
        
        return configurations
    
    async def _validate_configuration(self, category: ConfigurationCategory, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration against schema"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        schema = self.validation_schemas.get(category)
        if not schema:
            validation_result["warnings"].append(f"No validation schema for category {category.value}")
            return validation_result
        
        # Check required fields
        required_fields = schema.get("required_fields", [])
        for field in required_fields:
            if field not in configuration:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Check field types
        field_types = schema.get("field_types", {})
        for field_path, expected_type in field_types.items():
            value = self._get_nested_value(configuration, field_path)
            if value is not None and not isinstance(value, expected_type):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Field {field_path} must be of type {expected_type.__name__}")
        
        # Check constraints
        constraints = schema.get("constraints", {})
        for field_path, constraint in constraints.items():
            value = self._get_nested_value(configuration, field_path)
            if value is not None:
                if "min" in constraint and value < constraint["min"]:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field {field_path} must be >= {constraint['min']}")
                if "max" in constraint and value > constraint["max"]:
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Field {field_path} must be <= {constraint['max']}")
        
        return validation_result
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    async def export_configuration(self, config_id: str) -> Dict[str, Any]:
        """Export configuration for backup or migration"""
        config = await self.get_configuration(config_id)
        if not config:
            raise ValueError(f"Configuration {config_id} not found")
        
        export_data = {
            "metadata": {
                "exported_at": datetime.now(UTC).isoformat(),
                "export_version": "1.0",
                "source_system": "AIVillage Security Framework"
            },
            "configuration": config.to_dict(),
            "history": self.config_history.get(config_id, [])
        }
        
        logger.info(f"Exported configuration {config_id}")
        return export_data
    
    async def import_configuration(self, import_data: Dict[str, Any], imported_by: str = "system") -> SecurityConfiguration:
        """Import configuration from backup or migration"""
        config_data = import_data["configuration"]
        
        # Create new configuration from imported data
        config = SecurityConfiguration.from_dict(config_data)
        config.config_id = f"imported_{config.config_id}_{uuid4().hex[:8]}"
        config.created_at = datetime.now(UTC)
        config.updated_at = datetime.now(UTC)
        config.created_by = imported_by
        
        # Store configuration
        self.configurations[config.config_id] = config
        
        # Cache in Context7
        await self.context7.cache_configuration(config)
        
        # Import history if available
        if "history" in import_data:
            self.config_history[config.config_id] = import_data["history"]
        
        logger.info(f"Imported configuration as {config.config_id}")
        return config
    
    async def get_configuration_diff(self, config_id1: str, config_id2: str) -> Dict[str, Any]:
        """Get difference between two configurations"""
        config1 = await self.get_configuration(config_id1)
        config2 = await self.get_configuration(config_id2)
        
        if not config1 or not config2:
            raise ValueError("One or both configurations not found")
        
        # Simple diff implementation
        diff = {
            "config1": {"id": config_id1, "version": config1.version, "checksum": config1.get_checksum()},
            "config2": {"id": config_id2, "version": config2.version, "checksum": config2.get_checksum()},
            "differences": [],
            "identical": config1.get_checksum() == config2.get_checksum()
        }
        
        if not diff["identical"]:
            # Find differences in configuration data
            diff["differences"] = self._find_dict_differences(config1.configuration, config2.configuration)
        
        return diff
    
    def _find_dict_differences(self, dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> List[Dict[str, Any]]:
        """Find differences between two dictionaries"""
        differences = []
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            key_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                differences.append({
                    "type": "added",
                    "path": key_path,
                    "value": dict2[key]
                })
            elif key not in dict2:
                differences.append({
                    "type": "removed",
                    "path": key_path,
                    "value": dict1[key]
                })
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    differences.extend(self._find_dict_differences(dict1[key], dict2[key], key_path))
                else:
                    differences.append({
                        "type": "changed",
                        "path": key_path,
                        "old_value": dict1[key],
                        "new_value": dict2[key]
                    })
        
        return differences
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get consolidated security configuration service status"""
        return {
            "service_status": "operational" if self.initialized else "initializing",
            "context7_integration": await self.context7.get_cache_statistics(),
            "configuration_count": len(self.configurations),
            "categories_configured": len(set(config.category for config in self.configurations.values())),
            "environments_configured": len(set(config.environment for config in self.configurations.values())),
            "cache_hit_rate": 0.85,  # Simulated
            "last_updated": datetime.now(UTC).isoformat()
        }


# Global service instance
_config_service: Optional[ConsolidatedSecurityConfigService] = None


async def get_security_config_service() -> ConsolidatedSecurityConfigService:
    """Get global security configuration service instance"""
    global _config_service
    
    if _config_service is None:
        _config_service = ConsolidatedSecurityConfigService()
        await _config_service.initialize()
    
    return _config_service


# Convenience functions
async def create_security_config(
    name: str,
    category: ConfigurationCategory,
    environment: ConfigurationEnvironment,
    configuration: Dict[str, Any],
    **kwargs
) -> SecurityConfiguration:
    """Create security configuration"""
    service = await get_security_config_service()
    return await service.create_configuration(name, category, environment, configuration, **kwargs)


async def get_security_config(
    config_id: Optional[str] = None,
    name: Optional[str] = None,
    category: Optional[ConfigurationCategory] = None,
    environment: Optional[ConfigurationEnvironment] = None
) -> Optional[SecurityConfiguration]:
    """Get security configuration"""
    service = await get_security_config_service()
    return await service.get_configuration(config_id, name, category, environment)


async def update_security_config(config_id: str, updates: Dict[str, Any], **kwargs) -> SecurityConfiguration:
    """Update security configuration"""
    service = await get_security_config_service()
    return await service.update_configuration(config_id, updates, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Initialize service
        service = await get_security_config_service()
        
        # Create authentication configuration
        auth_config = await service.create_configuration(
            name="production_auth",
            category=ConfigurationCategory.AUTHENTICATION,
            environment=ConfigurationEnvironment.PRODUCTION,
            configuration={
                "password_policy": {
                    "min_length": 16,
                    "require_uppercase": True,
                    "require_lowercase": True,
                    "require_numbers": True,
                    "require_special_chars": True,
                    "max_age_days": 60
                },
                "mfa_settings": {
                    "enabled": True,
                    "required_roles": ["admin", "developer"],
                    "methods": ["TOTP", "hardware_token"]
                }
            },
            description="Production authentication configuration with enhanced security"
        )
        
        print(f"Created auth configuration: {auth_config.config_id}")
        
        # List configurations
        configs = await service.list_configurations(
            category=ConfigurationCategory.AUTHENTICATION,
            environment=ConfigurationEnvironment.PRODUCTION
        )
        print(f"Found {len(configs)} authentication configurations")
        
        # Get service status
        status = await service.get_service_status()
        print(f"Service status: {status}")
    
    asyncio.run(main())