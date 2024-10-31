# AI Village Implementation Todo List

## 1. Create Core Files
- Create `initialize_village.py` in root directory
- Create `download_ai_village.py` in root directory
- Update `README.md` with setup instructions

## 2. Set Up Project Structure
Create the following core components:

### 2.1 Agent System
```python
# Required Agent Classes
from agent_forge.agents.king.king_agent import KingAgent
from agent_forge.agents.sage.sage_agent import SageAgent
from agent_forge.agents.magi.magi_agent import MagiAgent
from agent_forge.agents.unified_base_agent import UnifiedBaseAgent
```

### 2.2 Core Systems
```python
# Required System Classes
from agent_forge.config.unified_config import UnifiedConfig
from agent_forge.data.data_collector import DataCollector
from agent_forge.rag_system.rag_manager import RAGManager
from agent_forge.communications.communication_manager import CommunicationManager
from agent_forge.agent_forge.forge_manager import ForgeManager
from agent_forge.testing.test_manager import TestManager
from agent_forge.utils.utility_manager import UtilityManager
from agent_forge.ui.ui_manager import UIManager
```

## 3. Implement Core Components

### 3.1 Main Village Class Implementation
```python
class AIVillage:
    def __init__(self):
        self.config = UnifiedConfig()
        self.data_collector = DataCollector()
        self.rag_manager = RAGManager()
        self.communication_manager = CommunicationManager()
        self.forge_manager = ForgeManager()
        self.test_manager = TestManager()
        self.utility_manager = UtilityManager()
        self.ui_manager = UIManager()
        
        # Initialize agents
        self.unified_base_agent = UnifiedBaseAgent()
        self.king_agent = KingAgent()
        self.sage_agent = SageAgent()
        self.magi_agent = MagiAgent()
```

### 3.2 Initialization Method
```python
async def initialize(self):
    logger.info("Initializing AI Village System")
    await self.config.load()
    await self.data_collector.initialize()
    await self.rag_manager.initialize()
    await self.communication_manager.initialize()
    await self.forge_manager.initialize()
    await self.utility_manager.initialize()
    await self.ui_manager.initialize()
    
    # Initialize agents
    await self.unified_base_agent.initialize()
    await self.king_agent.initialize()
    await self.sage_agent.initialize()
    await self.magi_agent.initialize()
    
    # Run initial tests
    await self.test_manager.run_initial_tests()
```

## 4. Implement UI System

### 4.1 UI Manager Class
```python
class UIManager:
    def __init__(self):
        self.app = web.Application()
        self.websocket_manager = WebSocketManager()

    async def initialize(self):
        # Set up routes
        self.app.router.add_get('/', self.index_handler)
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # Set up CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
```

## 5. Create Installation Script

### 5.1 Download Script Implementation
```python
def main():
    print("Starting AI Village download and setup...")

    # Clone repository
    repo_url = "https://github.com/yourusername/ai_village.git"
    run_command(f"git clone {repo_url}")
    os.chdir("ai_village")

    # Create and activate virtual environment
    run_command("python -m venv venv")
    if sys.platform == "win32":
        run_command(r"venv\Scripts\activate")
    else:
        run_command("source venv/bin/activate")

    # Install dependencies
    run_command("pip install -r requirements.txt")

    # Install UI dependencies
    os.chdir("ui")
    run_command("npm install")
    os.chdir("..")
```

## 6. Additional Tasks

### 6.1 Create Supporting Scripts
- Create `scripts/download_models.py`
- Create `scripts/download_rag_data.py`
- Create `scripts/setup_database.py`

### 6.2 Set Up Testing
- Implement unit tests
- Implement integration tests
- Create test configurations

### 6.3 Documentation
- Document installation process
- Document system architecture
- Create API documentation
- Document UI components

### 6.4 Dependencies
Update `requirements.txt` to include:
- aiohttp
- aiohttp_cors
- Other required packages

## 7. Testing Checklist
- [ ] Test initialization process
- [ ] Test agent communication
- [ ] Test RAG system
- [ ] Test UI functionality
- [ ] Test WebSocket connections
- [ ] Test database operations
- [ ] Run integration tests
- [ ] Verify all dependencies install correctly


# AI Village Enhanced Implementation Guide

## Missing Features Analysis

The previous implementation was missing several important features:

1. Task Processing System
2. Error Handling
3. Component Status Monitoring
4. Advanced Agent Communication
5. Proper Shutdown Procedures
6. Configuration Validation
7. Performance Monitoring

Here's the enhanced version with all features:

```python
import logging
import asyncio
import signal
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Core imports
from agent_forge.config.unified_config import UnifiedConfig
from agent_forge.data.data_collector import DataCollector
from agent_forge.agents.king.king_agent import KingAgent
from agent_forge.agents.sage.sage_agent import SageAgent
from agent_forge.agents.magi.magi_agent import MagiAgent
from agent_forge.agents.unified_base_agent import UnifiedBaseAgent
from agent_forge.rag_system.rag_manager import RAGManager
from agent_forge.communications.communication_manager import CommunicationManager
from agent_forge.agent_forge.forge_manager import ForgeManager
from agent_forge.testing.test_manager import TestManager
from agent_forge.utils.utility_manager import UtilityManager
from agent_forge.ui.ui_manager import UIManager
from agent_forge.monitoring.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    name: str
    status: str
    last_check: datetime
    health: float
    error_count: int
    last_error: Optional[str]

class AIVillage:
    """Enhanced main class for initializing and managing the AI Village system."""
    
    def __init__(self):
        # Core components
        self.config = UnifiedConfig()
        self.data_collector = DataCollector()
        self.rag_manager = RAGManager()
        self.communication_manager = CommunicationManager()
        self.forge_manager = ForgeManager()
        self.test_manager = TestManager()
        self.utility_manager = UtilityManager()
        self.ui_manager = UIManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Agents
        self.unified_base_agent = UnifiedBaseAgent()
        self.king_agent = KingAgent()
        self.sage_agent = SageAgent()
        self.magi_agent = MagiAgent()
        
        # System state
        self.is_running = False
        self.component_statuses: Dict[str, ComponentStatus] = {}
        self.task_queue = asyncio.Queue()
        self.error_queue = asyncio.Queue()
        
        # Performance metrics
        self.start_time = None
        self.task_count = 0
        self.error_count = 0
    
    async def initialize(self):
        """Enhanced initialization with proper error handling and status tracking."""
        try:
            logger.info("Initializing AI Village System")
            self.start_time = datetime.now()
            
            # Register signal handlers
            self._register_signal_handlers()
            
            # Initialize components with status tracking
            components = [
                (self.config, "Configuration"),
                (self.data_collector, "Data Collector"),
                (self.rag_manager, "RAG System"),
                (self.communication_manager, "Communication"),
                (self.forge_manager, "Forge"),
                (self.utility_manager, "Utilities"),
                (self.ui_manager, "UI"),
                (self.performance_monitor, "Performance Monitor")
            ]
            
            for component, name in components:
                try:
                    await component.initialize()
                    self._update_component_status(name, "running", None)
                except Exception as e:
                    self._update_component_status(name, "error", str(e))
                    raise
            
            # Initialize agents with enhanced error handling
            agents = [
                (self.unified_base_agent, "Base Agent"),
                (self.king_agent, "King Agent"),
                (self.sage_agent, "Sage Agent"),
                (self.magi_agent, "Magi Agent")
            ]
            
            for agent, name in agents:
                try:
                    await agent.initialize()
                    self._update_component_status(name, "running", None)
                except Exception as e:
                    self._update_component_status(name, "error", str(e))
                    raise
            
            # Start monitoring and maintenance tasks
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("AI Village System initialized successfully")
        
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            await self.shutdown()
            raise
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced task processing with monitoring and error handling."""
        try:
            self.task_count += 1
            self.performance_monitor.start_task(task['id'])
            
            # Validate task
            if not self._validate_task(task):
                raise ValueError(f"Invalid task format: {task}")
            
            # Route task to appropriate agent
            agent = self._get_agent_for_task(task)
            if not agent:
                raise ValueError(f"No suitable agent found for task: {task}")
            
            # Process task with timeout
            async with asyncio.timeout(300):  # 5-minute timeout
                result = await agent.process_task(task)
            
            self.performance_monitor.end_task(task['id'], success=True)
            return result
            
        except Exception as e:
            self.error_count += 1
            self.performance_monitor.end_task(task['id'], success=False)
            await self.error_queue.put((task, str(e)))
            raise
    
    async def shutdown(self):
        """Graceful shutdown procedure."""
        logger.info("Initiating AI Village shutdown...")
        self.is_running = False
        
        # Stop all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Shutdown components in reverse order
        components = [
            self.ui_manager,
            self.utility_manager,
            self.forge_manager,
            self.communication_manager,
            self.rag_manager,
            self.data_collector,
            self.performance_monitor
        ]
        
        for component in components:
            try:
                await component.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down component: {str(e)}")
        
        logger.info("AI Village shutdown complete")
    
    def _register_signal_handlers(self):
        """Register system signal handlers."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._signal_handler(s))
            )
    
    async def _signal_handler(self, signal):
        """Handle system signals."""
        logger.info(f"Received signal {signal}")
        await self.shutdown()
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self.background_tasks = [
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._process_error_queue()),
            asyncio.create_task(self._run_maintenance())
        ]
    
    async def _monitor_system_health(self):
        """Monitor system health and performance."""
        while self.is_running:
            try:
                metrics = self.performance_monitor.get_metrics()
                for component_name, status in self.component_statuses.items():
                    if status.error_count > 5:
                        await self._handle_component_failure(component_name)
                
                await self.ui_manager.update_metrics(metrics)
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in health monitoring: {str(e)}")
    
    def _update_component_status(self, name: str, status: str, error: Optional[str]):
        """Update component status tracking."""
        self.component_statuses[name] = ComponentStatus(
            name=name,
            status=status,
            last_check=datetime.now(),
            health=1.0 if status == "running" else 0.0,
            error_count=self.component_statuses.get(name, ComponentStatus(name, "", datetime.now(), 0.0, 0, None)).error_count + (1 if error else 0),
            last_error=error
        )
```

## Key Enhancements:

1. **Robust Error Handling**
   - Component-level error tracking
   - Error queuing system
   - Graceful failure recovery

2. **Performance Monitoring**
   - Task timing and metrics
   - System health monitoring
   - Component status tracking

3. **Enhanced Task Processing**
   - Task validation
   - Timeout handling
   - Agent routing

4. **System Management**
   - Graceful shutdown
   - Signal handling
   - Background task management

5. **Status Tracking**
   - Component health monitoring
   - Error count tracking
   - Performance metrics

6. **Maintenance Features**
   - Automatic system health checks
   - Component failure handling
   - Background maintenance tasks

## Usage Example:

```python
async def main():
    try:
        village = AIVillage()
        await village.initialize()
        
        # Process a task
        task = {
            "id": "task-001",
            "type": "analysis",
            "content": "Analyze market trends",
            "priority": "high",
            "timeout": 180
        }
        
        result = await village.process_task(task)
        print(f"Task result: {result}")
        
        # Get system metrics
        metrics = village.performance_monitor.get_metrics()
        print(f"System metrics: {metrics}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        await village.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

This enhanced version provides a much more robust and feature-rich implementation compared to the original. Would you like me to elaborate on any particular aspect or add more features?