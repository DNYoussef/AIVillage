"""
Configuration Management for Performance Benchmarks

Manages benchmark configurations, environment settings, and test scenarios
for comprehensive performance testing of monolithic vs microservices architectures.
"""

import json
import os
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

@dataclass
class BenchmarkConfig:
    """Configuration for a specific benchmark"""
    name: str
    duration: int  # seconds
    concurrent_connections: int
    requests_per_connection: int
    target_endpoints: List[str]
    custom_headers: Dict[str, str]
    timeout: int
    warmup_duration: int

@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str
    base_url: str
    websocket_url: str
    database_url: Optional[str]
    redis_url: Optional[str]
    auth_token: Optional[str]
    ssl_verify: bool
    custom_settings: Dict[str, Any]

@dataclass
class SystemConfig:
    """System resource limits and monitoring settings"""
    max_memory_mb: int
    max_cpu_percent: float
    monitoring_interval: float
    resource_limits: Dict[str, int]
    gc_settings: Dict[str, Any]

class ConfigManager:
    """
    Manages all benchmark configurations and environment settings
    """
    
    def __init__(self, config_dir: str = "swarm/phase2/benchmarks/backend/config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.configs = {}
        
        # Load default configurations
        self._load_default_configs()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup configuration manager logging"""
        logger = logging.getLogger('config_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_default_configs(self):
        """Load default benchmark configurations"""
        
        # Default benchmark configurations
        self.configs['benchmarks'] = {
            'training_throughput': BenchmarkConfig(
                name='training_throughput',
                duration=60,
                concurrent_connections=10,
                requests_per_connection=100,
                target_endpoints=['/api/train'],
                custom_headers={'Content-Type': 'application/json'},
                timeout=30,
                warmup_duration=10
            ),
            'websocket_latency': BenchmarkConfig(
                name='websocket_latency',
                duration=60,
                concurrent_connections=5,
                requests_per_connection=200,
                target_endpoints=['/ws/realtime'],
                custom_headers={},
                timeout=10,
                warmup_duration=5
            ),
            'api_response_time': BenchmarkConfig(
                name='api_response_time',
                duration=60,
                concurrent_connections=15,
                requests_per_connection=50,
                target_endpoints=['/api/health', '/api/status', '/api/models', '/api/predict'],
                custom_headers={'Accept': 'application/json'},
                timeout=15,
                warmup_duration=5
            ),
            'concurrent_load': BenchmarkConfig(
                name='concurrent_load',
                duration=120,
                concurrent_connections=50,
                requests_per_connection=20,
                target_endpoints=['/api/health'],
                custom_headers={},
                timeout=30,
                warmup_duration=15
            ),
            'memory_stress': BenchmarkConfig(
                name='memory_stress',
                duration=180,
                concurrent_connections=20,
                requests_per_connection=100,
                target_endpoints=['/api/heavy-compute'],
                custom_headers={},
                timeout=60,
                warmup_duration=20
            )
        }
        
        # Default environment configurations
        self.configs['environments'] = {
            'monolithic': EnvironmentConfig(
                name='monolithic',
                base_url='http://localhost:8080',
                websocket_url='ws://localhost:8080/ws',
                database_url='postgresql://localhost:5432/aivillage',
                redis_url='redis://localhost:6379',
                auth_token=None,
                ssl_verify=False,
                custom_settings={
                    'connection_pool_size': 20,
                    'keep_alive': True,
                    'compression': False
                }
            ),
            'microservices': EnvironmentConfig(
                name='microservices',
                base_url='http://localhost:8081',  # API Gateway
                websocket_url='ws://localhost:8082/ws',  # WebSocket service
                database_url='postgresql://localhost:5433/aivillage_micro',
                redis_url='redis://localhost:6380',
                auth_token=None,
                ssl_verify=False,
                custom_settings={
                    'connection_pool_size': 10,  # Smaller per service
                    'keep_alive': True,
                    'compression': True,  # Enable compression for microservices
                    'circuit_breaker': True,
                    'retry_attempts': 3
                }
            ),
            'production': EnvironmentConfig(
                name='production',
                base_url='https://api.aivillage.com',
                websocket_url='wss://api.aivillage.com/ws',
                database_url=None,  # Not exposed externally
                redis_url=None,
                auth_token='${AUTH_TOKEN}',  # Environment variable
                ssl_verify=True,
                custom_settings={
                    'connection_pool_size': 50,
                    'keep_alive': True,
                    'compression': True,
                    'rate_limiting': True,
                    'max_requests_per_minute': 1000
                }
            )
        }
        
        # Default system configurations
        self.configs['system'] = {
            'default': SystemConfig(
                max_memory_mb=2048,
                max_cpu_percent=80.0,
                monitoring_interval=1.0,
                resource_limits={
                    'max_open_files': 1024,
                    'max_connections': 100,
                    'max_threads': 50
                },
                gc_settings={
                    'enabled': True,
                    'threshold0': 700,
                    'threshold1': 10,
                    'threshold2': 10
                }
            ),
            'high_performance': SystemConfig(
                max_memory_mb=8192,
                max_cpu_percent=95.0,
                monitoring_interval=0.5,
                resource_limits={
                    'max_open_files': 4096,
                    'max_connections': 500,
                    'max_threads': 200
                },
                gc_settings={
                    'enabled': True,
                    'threshold0': 2000,
                    'threshold1': 25,
                    'threshold2': 25
                }
            ),
            'low_resource': SystemConfig(
                max_memory_mb=512,
                max_cpu_percent=50.0,
                monitoring_interval=2.0,
                resource_limits={
                    'max_open_files': 256,
                    'max_connections': 25,
                    'max_threads': 10
                },
                gc_settings={
                    'enabled': True,
                    'threshold0': 200,
                    'threshold1': 5,
                    'threshold2': 5
                }
            )
        }
    
    def get_benchmark_config(self, benchmark_name: str) -> Optional[BenchmarkConfig]:
        """Get configuration for a specific benchmark"""
        return self.configs.get('benchmarks', {}).get(benchmark_name)
    
    def get_environment_config(self, env_name: str) -> Optional[EnvironmentConfig]:
        """Get configuration for a specific environment"""
        return self.configs.get('environments', {}).get(env_name)
    
    def get_system_config(self, config_name: str = 'default') -> Optional[SystemConfig]:
        """Get system configuration"""
        return self.configs.get('system', {}).get(config_name)
    
    def create_benchmark_scenario(self, scenario_name: str,
                                 benchmark_configs: List[str],
                                 environment: str,
                                 system_config: str = 'default') -> Dict[str, Any]:
        """Create a complete benchmark scenario configuration"""
        
        scenario = {
            'name': scenario_name,
            'environment': self.get_environment_config(environment),
            'system': self.get_system_config(system_config),
            'benchmarks': {}
        }
        
        # Add benchmark configurations
        for benchmark_name in benchmark_configs:
            benchmark_config = self.get_benchmark_config(benchmark_name)
            if benchmark_config:
                scenario['benchmarks'][benchmark_name] = benchmark_config
            else:
                self.logger.warning(f"Benchmark config not found: {benchmark_name}")
        
        return scenario
    
    def save_config(self, config_name: str, config_data: Any, format_type: str = 'json'):
        """Save configuration to file"""
        
        file_path = self.config_dir / f"{config_name}.{format_type}"
        
        try:
            if format_type == 'json':
                with open(file_path, 'w') as f:
                    if hasattr(config_data, '__dict__'):
                        json.dump(asdict(config_data), f, indent=2)
                    else:
                        json.dump(config_data, f, indent=2, default=str)
            
            elif format_type == 'yaml':
                with open(file_path, 'w') as f:
                    if hasattr(config_data, '__dict__'):
                        yaml.dump(asdict(config_data), f, default_flow_style=False)
                    else:
                        yaml.dump(config_data, f, default_flow_style=False)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            self.logger.info(f"Config saved: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config {config_name}: {e}")
            raise
    
    def load_config(self, config_name: str, format_type: str = 'json') -> Any:
        """Load configuration from file"""
        
        file_path = self.config_dir / f"{config_name}.{format_type}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        try:
            if format_type == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            
            elif format_type == 'yaml':
                with open(file_path, 'r') as f:
                    return yaml.safe_load(f)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to load config {config_name}: {e}")
            raise
    
    def create_comparison_configs(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Create optimized configurations for monolithic vs microservices comparison"""
        
        # Monolithic configuration - optimized for single process
        monolithic_config = {
            'training': {
                'duration': 60,
                'concurrent_models': 8,
                'model_size': 'medium',
                'batch_size': 32,
                'optimization_level': 'high'
            },
            'websocket': {
                'websocket_url': 'ws://localhost:8080/ws',
                'message_count': 1000,
                'concurrent_connections': 8,
                'message_size': 1024,
                'keepalive': True
            },
            'api': {
                'api_url': 'http://localhost:8080',
                'endpoints': ['/health', '/api/status', '/api/models', '/api/predict'],
                'requests_per_endpoint': 100,
                'concurrent_requests': 15,
                'connection_reuse': True,
                'compression': False
            },
            'concurrent': {
                'api_url': 'http://localhost:8080',
                'max_concurrent': 75,
                'ramp_up_duration': 20,
                'steady_duration': 60,
                'endpoint': '/api/health',
                'connection_pooling': True
            },
            'system': {
                'max_memory_mb': 4096,
                'gc_optimization': 'throughput',
                'thread_pool_size': 20,
                'io_buffer_size': 8192
            }
        }
        
        # Microservices configuration - optimized for distributed architecture
        microservices_config = {
            'training': {
                'duration': 60,
                'concurrent_models': 10,  # Higher concurrency for distributed processing
                'model_size': 'medium',
                'batch_size': 16,  # Smaller batches for better distribution
                'optimization_level': 'balanced'
            },
            'websocket': {
                'websocket_url': 'ws://localhost:8081/ws',  # Gateway URL
                'message_count': 1000,
                'concurrent_connections': 10,  # Higher concurrency
                'message_size': 1024,
                'keepalive': True,
                'circuit_breaker': True
            },
            'api': {
                'api_url': 'http://localhost:8081',  # API Gateway
                'endpoints': ['/health', '/api/status', '/api/models', '/api/predict'],
                'requests_per_endpoint': 100,
                'concurrent_requests': 20,  # Higher concurrency
                'connection_reuse': True,
                'compression': True,  # Enable compression for network efficiency
                'retry_strategy': 'exponential_backoff',
                'circuit_breaker_enabled': True
            },
            'concurrent': {
                'api_url': 'http://localhost:8081',
                'max_concurrent': 100,  # Higher for better scalability testing
                'ramp_up_duration': 25,
                'steady_duration': 60,
                'endpoint': '/api/health',
                'connection_pooling': True,
                'load_balancing': True
            },
            'system': {
                'max_memory_mb': 2048,  # Lower per service
                'gc_optimization': 'latency',
                'thread_pool_size': 15,  # Smaller per service
                'io_buffer_size': 4096,
                'service_mesh_enabled': True
            }
        }
        
        return monolithic_config, microservices_config
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for completeness and correctness"""
        issues = []
        
        required_sections = ['training', 'websocket', 'api', 'concurrent']
        
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
                continue
            
            section_config = config[section]
            
            # Validate training config
            if section == 'training':
                required_fields = ['duration', 'concurrent_models']
                for field in required_fields:
                    if field not in section_config:
                        issues.append(f"Missing field in training config: {field}")
                
                if section_config.get('duration', 0) < 10:
                    issues.append("Training duration should be at least 10 seconds")
            
            # Validate websocket config
            elif section == 'websocket':
                if 'websocket_url' not in section_config:
                    issues.append("Missing websocket_url in websocket config")
                
                url = section_config.get('websocket_url', '')
                if not (url.startswith('ws://') or url.startswith('wss://')):
                    issues.append("Invalid websocket URL format")
            
            # Validate API config
            elif section == 'api':
                required_fields = ['api_url', 'endpoints']
                for field in required_fields:
                    if field not in section_config:
                        issues.append(f"Missing field in API config: {field}")
                
                endpoints = section_config.get('endpoints', [])
                if not isinstance(endpoints, list) or len(endpoints) == 0:
                    issues.append("API endpoints must be a non-empty list")
            
            # Validate concurrent config
            elif section == 'concurrent':
                max_concurrent = section_config.get('max_concurrent', 0)
                if max_concurrent < 1:
                    issues.append("max_concurrent must be at least 1")
                
                if max_concurrent > 1000:
                    issues.append("max_concurrent seems too high (>1000), may cause resource issues")
        
        return issues
    
    def get_optimized_config(self, architecture: str, workload_profile: str) -> Dict[str, Any]:
        """Get optimized configuration based on architecture and workload"""
        
        base_mono, base_micro = self.create_comparison_configs()
        
        if architecture == 'monolithic':
            config = base_mono.copy()
        else:
            config = base_micro.copy()
        
        # Apply workload-specific optimizations
        if workload_profile == 'high_throughput':
            # Optimize for maximum throughput
            config['training']['concurrent_models'] *= 1.5
            config['api']['concurrent_requests'] *= 1.2
            config['concurrent']['max_concurrent'] *= 1.3
            config['system']['thread_pool_size'] *= 1.2
        
        elif workload_profile == 'low_latency':
            # Optimize for minimal latency
            config['training']['batch_size'] = min(config['training'].get('batch_size', 16), 8)
            config['websocket']['concurrent_connections'] = min(config['websocket']['concurrent_connections'], 5)
            config['system']['gc_optimization'] = 'latency'
            config['system']['io_buffer_size'] = 16384
        
        elif workload_profile == 'memory_constrained':
            # Optimize for low memory usage
            config['training']['concurrent_models'] = max(1, config['training']['concurrent_models'] // 2)
            config['system']['max_memory_mb'] = min(config['system']['max_memory_mb'], 1024)
            config['system']['gc_optimization'] = 'memory'
            config['api']['concurrent_requests'] = max(5, config['api']['concurrent_requests'] // 2)
        
        elif workload_profile == 'stress_test':
            # Maximum load testing
            config['training']['concurrent_models'] *= 2
            config['api']['concurrent_requests'] *= 1.5
            config['concurrent']['max_concurrent'] *= 1.5
            config['concurrent']['steady_duration'] *= 2
            config['system']['max_memory_mb'] *= 1.5
        
        return config
    
    def export_configs(self, output_dir: str = None):
        """Export all configurations to files"""
        if output_dir is None:
            output_dir = self.config_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export benchmark configs
        benchmark_configs = {}
        for name, config in self.configs.get('benchmarks', {}).items():
            benchmark_configs[name] = asdict(config)
        
        self.save_config('benchmark_configs', benchmark_configs)
        
        # Export environment configs
        env_configs = {}
        for name, config in self.configs.get('environments', {}).items():
            env_configs[name] = asdict(config)
        
        self.save_config('environment_configs', env_configs)
        
        # Export system configs
        sys_configs = {}
        for name, config in self.configs.get('system', {}).items():
            sys_configs[name] = asdict(config)
        
        self.save_config('system_configs', sys_configs)
        
        # Export comparison configs
        mono_config, micro_config = self.create_comparison_configs()
        self.save_config('monolithic_config', mono_config)
        self.save_config('microservices_config', micro_config)
        
        self.logger.info(f"All configurations exported to {output_path}")

# Export main classes
__all__ = ['ConfigManager', 'BenchmarkConfig', 'EnvironmentConfig', 'SystemConfig']