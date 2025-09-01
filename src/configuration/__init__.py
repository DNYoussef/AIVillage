"""
Unified Configuration Management System
Provides centralized, distributed configuration management with MCP integration
"""

# Core components
from .unified_config_manager import (
    UnifiedConfigurationManager,
    ConfigurationHierarchy,
    ConfigurationLayer,
    get_config_manager,
    get_config,
    set_config,
    reload_config
)

from .distributed_cache_integration import (
    DistributedCacheManager,
    CacheNode,
    CacheEntry,
    Context7MCPIntegration,
    create_distributed_cache_manager
)

from .sequential_config_analyzer import (
    SequentialConfigurationAnalyzer,
    ConfigurationAnalysisStep,
    ConfigurationHierarchyAnalysis,
    SequentialThinkingMCPIntegration
)

from .memory_mcp_integration import (
    ConfigurationMemoryManager,
    ConfigurationPattern,
    ConfigurationDecision,
    PerformanceMetrics,
    ConfigurationPatternType,
    create_memory_manager
)

from .github_mcp_deployment import (
    GitHubMCPConfigurationDeployment,
    ConfigurationDeployment,
    ConfigurationValidationResult,
    DeploymentEnvironment,
    create_github_deployment_manager
)

from .validation_and_hot_reload import (
    ConfigurationValidationSystem,
    ValidationRule,
    ValidationResult,
    ConfigurationChange,
    create_validation_system
)

from .centralized_service import (
    CentralizedConfigurationService,
    ServiceHealth,
    ConfigurationServiceMetrics,
    create_centralized_service,
    create_docker_compose,
    create_k8s_manifests
)

# Version information
__version__ = "1.0.0"
__author__ = "AIVillage Configuration Team"
__description__ = "Unified Configuration Management with MCP Integration"

# Main exports
__all__ = [
    # Core managers
    "UnifiedConfigurationManager",
    "DistributedCacheManager", 
    "SequentialConfigurationAnalyzer",
    "ConfigurationMemoryManager",
    "GitHubMCPConfigurationDeployment",
    "ConfigurationValidationSystem",
    "CentralizedConfigurationService",
    
    # Data classes
    "ConfigurationHierarchy",
    "ConfigurationLayer",
    "CacheNode",
    "CacheEntry",
    "ConfigurationAnalysisStep",
    "ConfigurationHierarchyAnalysis",
    "ConfigurationPattern",
    "ConfigurationDecision",
    "PerformanceMetrics",
    "ConfigurationPatternType",
    "ConfigurationDeployment",
    "ConfigurationValidationResult",
    "DeploymentEnvironment",
    "ValidationRule",
    "ValidationResult",
    "ConfigurationChange",
    "ServiceHealth",
    "ConfigurationServiceMetrics",
    
    # Integration classes
    "Context7MCPIntegration",
    "SequentialThinkingMCPIntegration",
    
    # Factory functions
    "get_config_manager",
    "create_distributed_cache_manager",
    "create_memory_manager",
    "create_github_deployment_manager",
    "create_validation_system",
    "create_centralized_service",
    
    # Utility functions
    "get_config",
    "set_config",
    "reload_config",
    "create_docker_compose",
    "create_k8s_manifests"
]

# Configuration service usage examples
USAGE_EXAMPLES = {
    "basic_usage": '''
# Basic configuration management
from src.configuration import get_config, set_config, reload_config

# Get configuration
config = await get_config("database.host")
all_config = await get_config()

# Set configuration 
await set_config("api.rate_limit", 1000)

# Reload all configurations
await reload_config()
''',
    
    "service_deployment": '''
# Deploy centralized configuration service
from src.configuration import create_centralized_service

config_dirs = ["config", "tools/ci-cd/deployment/k8s"]
redis_nodes = [
    {"node_id": "primary", "host": "localhost", "port": 6379, "weight": 100}
]

service = await create_centralized_service(
    config_directories=config_dirs,
    redis_nodes=redis_nodes
)

await service.start_service(host="0.0.0.0", port=8090)
''',
    
    "distributed_caching": '''
# Use distributed configuration caching
from src.configuration import create_distributed_cache_manager

redis_nodes = [
    {"node_id": "primary", "host": "localhost", "port": 6379, "weight": 100},
    {"node_id": "replica", "host": "localhost", "port": 6380, "weight": 50}
]

cache_manager = await create_distributed_cache_manager(
    redis_nodes=redis_nodes,
    replication_factor=2,
    consistency_level="eventual"
)

# Cache configuration
await cache_manager.cache_configuration("app_config", config_data, ttl=3600)

# Retrieve from cache
cached_config = await cache_manager.get_cached_configuration("app_config")
''',
    
    "validation_and_hot_reload": '''
# Setup validation and hot-reload
from src.configuration import create_validation_system

config_dirs = ["config", "tools/ci-cd/deployment"]
validation_system = await create_validation_system(config_dirs)

# Register hot-reload callback
def on_config_change(file_path: str, config_data: Dict[str, Any]):
    print(f"Configuration changed: {file_path}")
    # Reload application configuration
    
await validation_system.register_reload_callback(on_config_change)

# Validate configurations
summary = await validation_system.get_validation_summary()
print(f"Validation summary: {summary}")
''',
    
    "deployment_pipeline": '''
# Configuration deployment with GitHub integration
from src.configuration import create_github_deployment_manager

deployment_manager = await create_github_deployment_manager("aivillage", "AIVillage")

# Create deployment
config_files = ["config/production_services.yaml", "config/aivillage_config.yaml"]
deployment = await deployment_manager.create_configuration_deployment(
    environment="production",
    config_files=config_files,
    deployment_strategy="blue_green"
)

# Validate deployment
validation = await deployment_manager.validate_configuration_deployment(deployment.deployment_id)

# Deploy if validation passes
if validation.passed:
    success = await deployment_manager.deploy_configuration(deployment.deployment_id)
''',
    
    "pattern_learning": '''
# Configuration pattern learning with Memory MCP
from src.configuration import create_memory_manager, ConfigurationPattern, ConfigurationPatternType

memory_manager = await create_memory_manager()

# Store successful configuration pattern
pattern = ConfigurationPattern(
    pattern_id="prod_db_config",
    pattern_type=ConfigurationPatternType.PERFORMANCE_OPTIMIZATION,
    name="Production Database Configuration",
    description="Optimized database settings for production workloads",
    pattern_data={"max_connections": 100, "timeout": 30},
    success_rate=0.95,
    usage_count=5,
    created_at=datetime.now(),
    last_used=datetime.now(),
    confidence_score=0.9,
    tags=["database", "production", "performance"]
)

await memory_manager.store_configuration_pattern(pattern)

# Get recommendations
recommendations = await memory_manager.get_pattern_recommendations(
    context={"environment": "production", "service": "database"}
)
'''
}

# Configuration best practices
BEST_PRACTICES = {
    "hierarchy_design": [
        "Use clear hierarchy levels: base -> environment -> service -> runtime",
        "Higher priority numbers override lower priority configurations",
        "Keep base configurations minimal and environment-agnostic",
        "Use environment-specific overrides sparingly"
    ],
    
    "security": [
        "Never hardcode secrets in configuration files",
        "Use environment variables or secure vaults for sensitive data",
        "Enable encryption for sensitive configuration data",
        "Validate all configuration inputs"
    ],
    
    "performance": [
        "Enable distributed caching for frequently accessed configurations",
        "Use appropriate cache TTL values based on change frequency",
        "Implement configuration validation to catch errors early",
        "Monitor configuration reload performance"
    ],
    
    "deployment": [
        "Always validate configurations before deployment",
        "Use gradual rollout strategies for critical environments",
        "Maintain rollback capabilities for quick recovery",
        "Test configuration changes in staging first"
    ],
    
    "monitoring": [
        "Track configuration validation metrics",
        "Monitor cache hit rates and performance",
        "Set up alerts for configuration validation failures",
        "Log all configuration changes for audit trail"
    ]
}

def print_usage_guide():
    """Print usage guide for the configuration system"""
    
    print("=" * 80)
    print("AIVillage Unified Configuration Management System")
    print("=" * 80)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print()
    
    print("USAGE EXAMPLES:")
    print("-" * 40)
    for example_name, code in USAGE_EXAMPLES.items():
        print(f"\n{example_name.replace('_', ' ').title()}:")
        print(code)
    
    print("\nBEST PRACTICES:")
    print("-" * 40)
    for category, practices in BEST_PRACTICES.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for practice in practices:
            print(f"  • {practice}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_usage_guide()