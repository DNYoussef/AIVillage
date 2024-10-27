
import os
import yaml
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
from .openrouter_agent import OpenRouterAgent
from .king.king_agent import KingAgent
from .sage.sage_agent import SageAgent
from .magi.magi_agent import MagiAgent

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Manages the three main agents (King, Sage, Magi) using OpenRouter for frontier models
    and tracking their interactions for training local models.
    """
    
    def __init__(self, config_path: str = "config/openrouter_agents.yaml"):
        """
        Initialize AgentManager.
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize OpenRouter agents and specialized agents
        self.openrouter_agents: Dict[str, OpenRouterAgent] = {}
        self.agents: Dict[str, Any] = {}  # Holds KingAgent, SageAgent, MagiAgent instances
        self._initialize_agents()
        
        logger.info("AgentManager initialized with agents: " + ", ".join(self.agents.keys()))
    
    def _initialize_agents(self):
        """Initialize King, Sage, and Magi agents with their respective models."""
        for agent_name, agent_config in self.config['agents'].items():
            # Create OpenRouter agent for this agent type
            openrouter_agent = OpenRouterAgent(
                api_key=self.api_key,
                model=agent_config['frontier_model'],
                local_model=agent_config['local_model']
            )
            self.openrouter_agents[agent_name] = openrouter_agent
            
            # Create specialized agent instance
            if agent_name == "king":
                self.agents[agent_name] = KingAgent(openrouter_agent)
            elif agent_name == "sage":
                self.agents[agent_name] = SageAgent(openrouter_agent)
            elif agent_name == "magi":
                self.agents[agent_name] = MagiAgent(openrouter_agent)
            
            logger.info(f"Initialized {agent_name} agent with models:")
            logger.info(f"  Frontier: {agent_config['frontier_model']}")
            logger.info(f"  Local: {agent_config['local_model']}")
    
    async def process_task(self, task: str, agent_type: str, **kwargs) -> Dict[str, Any]:
        """
        Process a task using the specified agent.
        
        Args:
            task: The task to process
            agent_type: Type of agent to use ("king", "sage", or "magi")
            **kwargs: Additional arguments specific to each agent type
            
        Returns:
            Dictionary containing the response and metadata
        """
        agent = self.agents.get(agent_type.lower())
        if not agent:
            raise ValueError(f"Invalid agent type: {agent_type}")
        
        # Route to appropriate agent method based on type
        if agent_type == "king":
            interaction = await agent.process_task(task, **kwargs)
        elif agent_type == "sage":
            interaction = await agent.conduct_research(task, **kwargs)
        elif agent_type == "magi":
            interaction = await agent.generate_code(task, **kwargs)
        
        return {
            "response": interaction.response,
            "model_used": interaction.model,
            "metadata": interaction.metadata
        }
    
    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent instance by name.
        
        Args:
            name: Agent name ("king", "sage", or "magi")
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(name.lower())
    
    def get_agent_config(self, name: str) -> Optional[Dict]:
        """
        Get an agent's configuration.
        
        Args:
            name: Agent name ("king", "sage", or "magi")
            
        Returns:
            Agent configuration dictionary or None if not found
        """
        return self.config['agents'].get(name.lower())
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all agents.
        
        Returns:
            Dictionary mapping agent names to their performance metrics
        """
        return {
            name: agent.get_performance_metrics()
            for name, agent in self.agents.items()
        }
    
    def get_training_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get training data for all agents' local models.
        
        Returns:
            Dictionary mapping agent names to their training data
        """
        return {
            name: agent.get_training_data()
            for name, agent in self.agents.items()
        }
    
    def get_dpo_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get DPO metrics for all agents.
        
        Returns:
            Dictionary mapping agent names to their DPO metrics
        """
        return {
            name: openrouter.get_dpo_metrics()
            for name, openrouter in self.openrouter_agents.items()
        }
