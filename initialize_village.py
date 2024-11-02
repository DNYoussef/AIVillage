"""Initialize AI Village system with core agents and components."""

import asyncio
import logging
from pathlib import Path
import os
from typing import Dict, Any, Optional

# Core imports
from config.unified_config import UnifiedConfig
from agent_forge.data.data_collector import DataCollector
from agent_forge.agents.king.king_agent import KingAgent
from agent_forge.agents.sage.sage_agent import SageAgent
from agent_forge.agents.magi.magi_agent import MagiAgent
from agent_forge.agents.openrouter_agent import OpenRouterAgent
from communications.community_hub import CommunityHub
from communications.protocol import StandardCommunicationProtocol
from communications.message import Message, MessageType, Priority
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.retrieval.vector_store import VectorStore
from rag_system.retrieval.graph_store import GraphStore
from ui.ui_manager import UIManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIVillage:
    """Main AI Village system orchestrator."""
    
    def __init__(self):
        """Initialize AI Village components."""
        self.config = UnifiedConfig()
        self.initialized = False
        self.agents = {}
        self.systems = {}
        self.openrouter_agents = {}
    
    async def initialize(self):
        """Initialize all AI Village components."""
        try:
            logger.info("Initializing AI Village...")
            
            # Initialize core systems
            await self._initialize_core_systems()
            
            # Initialize communication protocol
            self.systems['communication_protocol'] = StandardCommunicationProtocol()
            
            # Initialize OpenRouter agents first
            await self._initialize_openrouter_agents()
            
            # Initialize specialized agents
            await self._initialize_agents()
            
            # Initialize communication hub
            await self._initialize_communication()
            
            # Initialize UI system
            await self._initialize_ui()
            
            # Run initial tests
            await self._run_initial_tests()
            
            self.initialized = True
            logger.info("AI Village initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing AI Village: {str(e)}")
            raise
    
    async def _initialize_core_systems(self):
        """Initialize core system components."""
        logger.info("Initializing core systems...")
        
        try:
            # Initialize data systems
            self.systems['data_collector'] = DataCollector(config=self.config)
            await self.systems['data_collector'].initialize()
            
            self.systems['vector_store'] = VectorStore()
            await self.systems['vector_store'].initialize()
            
            self.systems['graph_store'] = GraphStore(self.config)
            await self.systems['graph_store'].initialize()
            
            # Initialize RAG system
            self.systems['cognitive_nexus'] = CognitiveNexus()
            await self.systems['cognitive_nexus'].initialize()
            
            # Initialize storage directories
            self._initialize_directories()
            
            logger.info("Core systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing core systems: {str(e)}")
            raise
    
    async def _initialize_openrouter_agents(self):
        """Initialize OpenRouter agent instances."""
        logger.info("Initializing OpenRouter agents...")
        
        try:
            api_key = self.config.get_api_key()
            
            # Create and initialize OpenRouter agent instances using config
            for agent_name, agent_config in self.config.agents.items():
                self.openrouter_agents[agent_name] = OpenRouterAgent(
                    api_key=api_key,
                    model=agent_config.frontier_model.name,
                    local_model=agent_config.local_model.name,
                    config=self.config
                )
            
            # Initialize each OpenRouter agent
            for agent_name, agent in self.openrouter_agents.items():
                await agent.initialize()
            
            logger.info("OpenRouter agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OpenRouter agents: {str(e)}")
            raise
    
    async def _initialize_agents(self):
        """Initialize specialized agent components."""
        logger.info("Initializing specialized agents...")
        
        try:
            # Initialize specialized agents
            self.agents['king'] = KingAgent(
                openrouter_agent=self.openrouter_agents['king'],
                config=self.config
            )
            await self.agents['king'].initialize()
            
            self.agents['sage'] = SageAgent(
                openrouter_agent=self.openrouter_agents['sage'],
                config=self.config
            )
            await self.agents['sage'].initialize()
            
            self.agents['magi'] = MagiAgent(
                openrouter_agent=self.openrouter_agents['magi'],
                config=self.config
            )
            await self.agents['magi'].initialize()
            
            logger.info("Specialized agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing specialized agents: {str(e)}")
            raise
    
    async def _initialize_communication(self):
        """Initialize communication system."""
        logger.info("Initializing communication system...")
        
        try:
            self.systems['community_hub'] = CommunityHub(
                communication_protocol=self.systems['communication_protocol']
            )
            
            # Initialize community hub
            await self.systems['community_hub'].initialize()
            
            # Register agents with communication hub
            for agent_name, agent in self.agents.items():
                await self.systems['community_hub'].register_agent(
                    agent_id=agent_name,
                    capabilities=agent.info if hasattr(agent, 'info') else {}
                )
            
            logger.info("Communication system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing communication system: {str(e)}")
            raise
    
    async def _initialize_ui(self):
        """Initialize UI system."""
        logger.info("Initializing UI system...")
        
        try:
            # Create UI Manager instance
            self.systems['ui_manager'] = UIManager()
            
            # Initialize UI
            await self.systems['ui_manager'].initialize()
            
            # Start UI server
            await self.systems['ui_manager'].start(
                host=self.config.config.get('ui', {}).get('host', '0.0.0.0'),
                port=self.config.config.get('ui', {}).get('port', 8080)
            )
            
            logger.info("UI system initialized successfully")
            logger.info(f"UI available at http://localhost:{self.config.config.get('ui', {}).get('port', 8080)}")
            
        except Exception as e:
            logger.error(f"Error initializing UI system: {str(e)}")
            raise
    
    def _initialize_directories(self):
        """Initialize required directories."""
        directories = [
            'data',
            'data/backups',
            'logs',
            'logs/agents',
            'logs/tasks',
            'cache',
            'ui/static'  # Add static files directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def _run_initial_tests(self):
        """Run initial system tests."""
        logger.info("Running initial tests...")
        
        try:
            # Test agent communication
            test_message = Message(
                type=MessageType.NOTIFICATION,
                sender="system",
                receiver="king",  # Send to King agent first
                content={"message": "Test message from initialization"},
                priority=Priority.HIGH
            )
            await self.systems['communication_protocol'].send_message(test_message)
            
            # Test RAG system
            test_query = "Test query for initialization"
            await self.systems['cognitive_nexus'].query(
                content=test_query,
                embeddings=[0.1] * 10,  # Dummy embeddings
                entities=["test"]  # Dummy entities
            )
            
            # Test UI WebSocket
            test_metrics = {
                "task_success_rate": 1.0,
                "response_quality": 0.95,
                "system_load": 0.3
            }
            await self.systems['ui_manager'].update_metrics(test_metrics)
            
            logger.info("Initial tests completed successfully")
            
        except Exception as e:
            logger.error(f"Error running initial tests: {str(e)}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown AI Village system."""
        logger.info("Initiating AI Village shutdown...")
        
        try:
            # Shutdown UI first to stop accepting new connections
            if 'ui_manager' in self.systems:
                logger.info("Shutting down UI system...")
                await self.systems['ui_manager'].shutdown()
            
            # Shutdown specialized agents
            for agent_name, agent in self.agents.items():
                logger.info(f"Shutting down {agent_name} agent...")
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            
            # Shutdown OpenRouter agents
            for agent_name, agent in self.openrouter_agents.items():
                logger.info(f"Shutting down OpenRouter {agent_name} agent...")
                if hasattr(agent, 'shutdown'):
                    await agent.shutdown()
            
            # Shutdown remaining systems
            for system_name, system in self.systems.items():
                if system_name != 'ui_manager':  # UI already shut down
                    logger.info(f"Shutting down {system_name}...")
                    if hasattr(system, 'shutdown'):
                        await system.shutdown()
            
            self.initialized = False
            logger.info("AI Village shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise

async def main():
    """Main entry point for AI Village initialization."""
    village = AIVillage()
    
    try:
        await village.initialize()
        logger.info("AI Village is ready")
        logger.info("Access the UI at http://localhost:8080")
        
        # Keep the system running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            await village.shutdown()
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        if village.initialized:
            await village.shutdown()
        raise

if __name__ == "__main__":
    asyncio.run(main())
