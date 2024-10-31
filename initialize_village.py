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
    
    async def initialize(self):
        """Initialize all AI Village components."""
        try:
            logger.info("Initializing AI Village...")
            
            # Initialize core systems
            await self._initialize_core_systems()
            
            # Initialize communication protocol
            self.systems['communication_protocol'] = StandardCommunicationProtocol()
            
            # Initialize agents
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
            self.systems['vector_store'] = VectorStore()
            self.systems['graph_store'] = GraphStore(self.config)
            
            # Initialize RAG system
            self.systems['cognitive_nexus'] = CognitiveNexus()
            await self.systems['cognitive_nexus'].initialize()
            
            # Initialize storage directories
            self._initialize_directories()
            
            logger.info("Core systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing core systems: {str(e)}")
            raise
    
    async def _initialize_agents(self):
        """Initialize agent components."""
        logger.info("Initializing agents...")
        
        try:
            api_key = self.config.get_api_key()
            
            # Create OpenRouter agent instances for each agent type
            openrouter_agents = {
                'king': OpenRouterAgent(
                    api_key=api_key,
                    model="nvidia/llama-3.1-nemotron-70b-instruct",
                    local_model="Qwen/Qwen2.5-3B-Instruct",
                    config=self.config
                ),
                'sage': OpenRouterAgent(
                    api_key=api_key,
                    model="anthropic/claude-3.5-sonnet",
                    local_model="deepseek-ai/Janus-1.3B",
                    config=self.config
                ),
                'magi': OpenRouterAgent(
                    api_key=api_key,
                    model="openai/o1-mini-2024-09-12",
                    local_model="ibm-granite/granite-3b-code-instruct-128k",
                    config=self.config
                )
            }
            
            # Initialize specialized agents
            self.agents['king'] = KingAgent(
                openrouter_agent=openrouter_agents['king'],
                config=self.config
            )
            
            self.agents['sage'] = SageAgent(
                openrouter_agent=openrouter_agents['sage'],
                config=self.config
            )
            
            self.agents['magi'] = MagiAgent(
                openrouter_agent=openrouter_agents['magi'],
                config=self.config
            )
            
            logger.info("Agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    async def _initialize_communication(self):
        """Initialize communication system."""
        logger.info("Initializing communication system...")
        
        try:
            self.systems['community_hub'] = CommunityHub(
                communication_protocol=self.systems['communication_protocol']
            )
            
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
            
            # Shutdown agents
            for agent_name, agent in self.agents.items():
                logger.info(f"Shutting down {agent_name} agent...")
                # Add agent-specific shutdown logic here
            
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
