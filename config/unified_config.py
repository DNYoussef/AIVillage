"""Unified configuration management system for AI Village."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import logging
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of models that can be used."""
    FRONTIER = "frontier"
    LOCAL = "local"

class AgentType(Enum):
    """Types of agents in the system."""
    KING = "king"
    SAGE = "sage"
    MAGI = "magi"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    type: ModelType
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    rate_limit: int = 50
    retry_delay: int = 1
    max_retries: int = 3

    def update(self, updates: Dict[str, Any]):
        """Update model configuration."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass
class AgentConfig:
    """Configuration for a specific agent."""
    type: AgentType
    frontier_model: ModelConfig
    local_model: ModelConfig
    description: str
    capabilities: list = field(default_factory=list)
    performance_threshold: float = 0.7
    complexity_threshold: float = 0.6
    evolution_rate: float = 0.1

    def update(self, updates: Dict[str, Any]):
        """Update agent configuration."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.frontier_model, key):
                self.frontier_model.update({key: value})
                self.local_model.update({key: value})

@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    retrieval_depth: int = 3
    relevance_threshold: float = 0.7
    feedback_enabled: bool = True
    exploration_weight: float = 1.0
    max_context_length: int = 2000

@dataclass
class DatabaseConfig:
    """Configuration for the database system."""
    path: str = "data/agent_data.db"
    backup_interval: int = 24  # hours
    max_backup_count: int = 7
    vacuum_threshold: int = 1000  # rows

@dataclass
class PerformanceConfig:
    """Configuration for performance tracking."""
    metrics: Dict[str, float] = field(default_factory=lambda: {
        "response_quality": 0.4,
        "task_completion": 0.3,
        "efficiency": 0.2,
        "creativity": 0.1
    })
    dpo_batch_size: int = 16
    learning_rate: float = 1e-5
    beta: float = 0.1
    update_interval: int = 100

class UnifiedConfig:
    """
    Centralized configuration management system.
    Handles loading, validation, and access to all configuration settings.
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize UnifiedConfig.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config: Dict[str, Any] = {}
        self.agents: Dict[str, AgentConfig] = {}
        self.rag_config: RAGConfig = RAGConfig()
        self.db_config: DatabaseConfig = DatabaseConfig()
        self.performance_config: PerformanceConfig = PerformanceConfig()
        
        # Load configurations
        self._load_configs()
        
        logger.info("Initialized UnifiedConfig")
    
    def update(self, config: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            config: Dictionary containing configuration updates
        """
        self.config.update(config)
        
        # Update component configs if present
        if 'agents' in config:
            self._process_agent_config({'agents': config['agents']})
        if 'rag_system' in config:
            for key, value in config['rag_system'].items():
                if hasattr(self.rag_config, key):
                    setattr(self.rag_config, key, value)
        if 'database' in config:
            for key, value in config['database'].items():
                if hasattr(self.db_config, key):
                    setattr(self.db_config, key, value)
        if 'performance' in config:
            for key, value in config['performance'].items():
                if hasattr(self.performance_config, key):
                    setattr(self.performance_config, key, value)
    
    def _load_configs(self):
        """Load all configuration files."""
        try:
            # Load default config
            default_config = self._load_yaml("default.yaml")
            self.config.update(default_config)
            
            # Load agent-specific config
            agent_config = self._load_yaml("openrouter_agents.yaml")
            self._process_agent_config(agent_config)
            
            # Load environment variables
            self._load_env_vars()
            
            # Validate configurations
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Error loading configurations: {str(e)}")
            raise
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        file_path = self.config_dir / filename
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {filename}")
            return {}
            
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _process_agent_config(self, config: Dict[str, Any]):
        """Process agent-specific configurations."""
        for agent_name, agent_data in config.get('agents', {}).items():
            # Create model configs
            frontier_model = ModelConfig(
                name=agent_data['frontier_model'],
                type=ModelType.FRONTIER,
                temperature=agent_data['settings'].get('temperature', 0.7),
                max_tokens=agent_data['settings'].get('max_tokens', 1000),
                system_prompt=agent_data['settings'].get('system_prompt')
            )
            
            local_model = ModelConfig(
                name=agent_data['local_model'],
                type=ModelType.LOCAL,
                temperature=agent_data['settings'].get('temperature', 0.7),
                max_tokens=agent_data['settings'].get('max_tokens', 1000)
            )
            
            # Create agent config
            self.agents[agent_name] = AgentConfig(
                type=AgentType[agent_name.upper()],
                frontier_model=frontier_model,
                local_model=local_model,
                description=agent_data['description']
            )
    
    def _load_env_vars(self):
        """Load configuration from environment variables."""
        # Load environment variables from .env file (already done at module level)
        
        # API keys
        self.config['openrouter_api_key'] = os.getenv('OPENROUTER_API_KEY')
        
        # UI Configuration
        self.config['ui_port'] = int(os.getenv('UI_PORT', '8080'))
        self.config['ui_host'] = os.getenv('UI_HOST', '0.0.0.0')
        
        # System Configuration
        self.config['environment'] = os.getenv('ENVIRONMENT', 'development')
        self.config['log_level'] = os.getenv('LOG_LEVEL', 'INFO')
        
        # Model Configuration
        self.config['default_temperature'] = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
        self.config['max_tokens'] = int(os.getenv('MAX_TOKENS', '2000'))
        
        # Override configurations from environment variables
        for key, value in os.environ.items():
            if key.startswith('AIVILLAGE_'):
                config_key = key[10:].lower()  # Remove AIVILLAGE_ prefix
                self.config[config_key] = value
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        # In development mode, don't require API key
        if self.is_development():
            if not self.config.get('openrouter_api_key'):
                logger.warning("No OpenRouter API key found. Running in development mode with limited functionality.")
                self.config['openrouter_api_key'] = 'development_key'
        else:
            # In production, require API key
            if not self.config.get('openrouter_api_key'):
                raise ValueError("OpenRouter API key is required in production mode")
            
        # Validate agent configurations
        if not self.agents:
            raise ValueError("No agent configurations found")
            
        # Validate model configurations
        for agent_name, agent_config in self.agents.items():
            if not agent_config.frontier_model.name:
                raise ValueError(f"Missing frontier model name for agent {agent_name}")
            if not agent_config.local_model.name:
                raise ValueError(f"Missing local model name for agent {agent_name}")
    
    def get_agent_config(self, agent_type: str) -> AgentConfig:
        """
        Get configuration for a specific agent.
        
        Args:
            agent_type: Type of agent ("king", "sage", or "magi")
            
        Returns:
            AgentConfig for the specified agent
        """
        agent_config = self.agents.get(agent_type.lower())
        if not agent_config:
            raise ValueError(f"Invalid agent type: {agent_type}")
        return agent_config
    
    def update_agent_config(self, agent_type: str, updates: Dict[str, Any]):
        """
        Update configuration for a specific agent.
        
        Args:
            agent_type: Type of agent
            updates: Dictionary of updates to apply
        """
        agent_config = self.get_agent_config(agent_type)
        agent_config.update(updates)
        self._save_agent_config(agent_type)
    
    def _save_agent_config(self, agent_type: str):
        """Save updated agent configuration to file."""
        config_file = self.config_dir / "openrouter_agents.yaml"
        
        try:
            # Load current config
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update agent config
            agent_config = self.agents[agent_type]
            config['agents'][agent_type] = {
                'frontier_model': agent_config.frontier_model.name,
                'local_model': agent_config.local_model.name,
                'description': agent_config.description,
                'settings': {
                    'temperature': agent_config.frontier_model.temperature,
                    'max_tokens': agent_config.frontier_model.max_tokens,
                    'system_prompt': agent_config.frontier_model.system_prompt
                }
            }
            
            # Save updated config
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            logger.info(f"Saved updated configuration for agent {agent_type}")
            
        except Exception as e:
            logger.error(f"Error saving agent configuration: {str(e)}")
            raise
    
    def get_rag_config(self) -> RAGConfig:
        """Get RAG system configuration."""
        return self.rag_config
    
    def get_db_config(self) -> DatabaseConfig:
        """Get database configuration."""
        return self.db_config
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance tracking configuration."""
        return self.performance_config
    
    def get_api_key(self) -> str:
        """Get OpenRouter API key."""
        api_key = self.config.get('openrouter_api_key')
        if not api_key:
            raise ValueError("OpenRouter API key not found in configuration")
        return api_key
    
    def get_environment(self) -> str:
        """Get current environment (development, staging, production)."""
        return self.config.get('environment', 'development')
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.get_environment() == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get_environment() == 'production'
