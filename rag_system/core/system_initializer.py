import logging
import os
from typing import Dict, Any, List
import asyncio
from pathlib import Path
from rag_system.core.unified_config import unified_config
from rag_system.error_handling.error_handler import error_handler
from rag_system.error_handling.performance_monitor import performance_monitor

logger = logging.getLogger(__name__)

class SystemInitializer:
    def __init__(self):
        self.initialized_components: Dict[str, bool] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        self._setup_logging()

    def _setup_logging(self):
        """Set up system-wide logging configuration."""
        log_config = unified_config.get('system.logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # Create log directory if it doesn't exist
        log_dir = Path(log_config.get('handlers', [])[0].get('filename', 'logs/system.log')).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    async def initialize_system(self):
        """Initialize the entire system."""
        try:
            logger.info("Starting system initialization")
            
            # Load configuration
            config_path = os.getenv('CONFIG_PATH', 'config/default.yaml')
            if not unified_config.load_config(config_path):
                raise RuntimeError("Failed to load configuration")

            # Create required directories
            self._create_directories()

            # Initialize core components
            await self._initialize_core_components()

            # Initialize remaining components in dependency order
            components = unified_config.config.get('components', {})
            for component_name, component_config in components.items():
                if component_config.get('enabled', True):
                    await self._initialize_component(component_name, component_config)

            logger.info("System initialization completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error during system initialization: {str(e)}")
            return False

    def _create_directories(self):
        """Create required system directories."""
        directories = [
            unified_config.get('system.resources.data_directory', 'data/'),
            unified_config.get('system.resources.temp_directory', 'tmp/'),
            'logs/',
            'config/'
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    async def _initialize_core_components(self):
        """Initialize core system components."""
        # Initialize error handling
        error_handler_config = unified_config.get('error_handler.settings', {})
        error_handler.setup_logging()
        logger.info("Initialized error handling system")

        # Initialize performance monitoring
        performance_monitor.start_monitoring()
        logger.info("Initialized performance monitoring system")

        # Start periodic tasks
        asyncio.create_task(error_handler.monitor_performance())
        asyncio.create_task(performance_monitor.monitor_performance())

    async def _initialize_component(self, component_name: str, component_config: Dict[str, Any]):
        """Initialize a system component."""
        if component_name in self.initialized_components:
            return self.initialized_components[component_name]

        try:
            # Check dependencies
            dependencies = component_config.get('dependencies', [])
            for dependency in dependencies:
                if not await self._initialize_component(dependency, unified_config.get(f'components.{dependency}')):
                    raise RuntimeError(f"Failed to initialize dependency: {dependency}")

            # Initialize component
            logger.info(f"Initializing component: {component_name}")
            
            # Component-specific initialization logic would go here
            # For example:
            if component_name == 'task_manager':
                await self._initialize_task_manager(component_config)
            elif component_name == 'planning':
                await self._initialize_planning(component_config)
            elif component_name == 'knowledge_tracker':
                await self._initialize_knowledge_tracker(component_config)

            self.initialized_components[component_name] = True
            logger.info(f"Successfully initialized component: {component_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing component {component_name}: {str(e)}")
            self.initialized_components[component_name] = False
            return False

    async def _initialize_task_manager(self, config: Dict[str, Any]):
        """Initialize task management system."""
        settings = config.get('settings', {})
        # Task manager initialization logic would go here
        pass

    async def _initialize_planning(self, config: Dict[str, Any]):
        """Initialize planning system."""
        settings = config.get('settings', {})
        # Planning system initialization logic would go here
        pass

    async def _initialize_knowledge_tracker(self, config: Dict[str, Any]):
        """Initialize knowledge tracking system."""
        settings = config.get('settings', {})
        # Knowledge tracker initialization logic would go here
        pass

    def get_initialization_status(self) -> Dict[str, bool]:
        """Get the initialization status of all components."""
        return dict(self.initialized_components)

# Create singleton instance
system_initializer = SystemInitializer()
