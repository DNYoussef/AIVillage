"""Enhanced agent management system with unified configuration."""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from datetime import datetime

from config.unified_config import UnifiedConfig, AgentConfig
from .openrouter_agent import OpenRouterAgent, AgentInteraction
from .king.king_agent import KingAgent
from .sage.sage_agent import SageAgent
from .magi.magi_agent import MagiAgent
from ..data.data_collector import DataCollector
from ..data.complexity_evaluator import ComplexityEvaluator

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Enhanced management system for King, Sage, and Magi agents with
    unified configuration, data collection, and complexity evaluation.
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize AgentManager with unified configuration.
        
        Args:
            config: UnifiedConfig instance
        """
        self.config = config
        self.api_key = config.get_api_key()
        
        # Initialize support systems
        self.data_collector = DataCollector(config)
        self.complexity_evaluator = ComplexityEvaluator(config)
        
        # Initialize agent containers
        self.openrouter_agents: Dict[str, OpenRouterAgent] = {}
        self.agents: Dict[str, Any] = {}
        
        # Initialize agents
        self._initialize_agents()
        
        # Start performance monitoring
        self._start_monitoring()
        
        logger.info("Initialized AgentManager with agents: " + ", ".join(self.agents.keys()))
    
    def _initialize_agents(self):
        """Initialize all agent types with their respective configurations."""
        for agent_type, agent_config in self.config.agents.items():
            try:
                # Create OpenRouter agent
                openrouter_agent = OpenRouterAgent(
                    api_key=self.api_key,
                    model=agent_config.frontier_model.name,
                    local_model=agent_config.local_model.name
                )
                self.openrouter_agents[agent_type] = openrouter_agent
                
                # Create specialized agent instance
                if agent_type == "king":
                    self.agents[agent_type] = KingAgent(openrouter_agent, self.config)
                elif agent_type == "sage":
                    self.agents[agent_type] = SageAgent(openrouter_agent, self.config)
                elif agent_type == "magi":
                    self.agents[agent_type] = MagiAgent(openrouter_agent, self.config)
                
                logger.info(f"Initialized {agent_type} agent with models:")
                logger.info(f"  Frontier: {agent_config.frontier_model.name}")
                logger.info(f"  Local: {agent_config.local_model.name}")
                
            except Exception as e:
                logger.error(f"Error initializing {agent_type} agent: {str(e)}")
                raise
    
    def _start_monitoring(self):
        """Start performance monitoring tasks."""
        async def monitoring_loop():
            while True:
                try:
                    # Collect and analyze performance metrics
                    metrics = self.get_performance_metrics()
                    
                    # Store metrics
                    for agent_type, agent_metrics in metrics.items():
                        await self.data_collector.store_performance_metrics(
                            agent_type=agent_type,
                            model_type="frontier",
                            metrics=agent_metrics
                        )
                    
                    # Adjust complexity thresholds based on performance
                    for agent_type in self.agents:
                        self.complexity_evaluator.adjust_thresholds(
                            agent_type=agent_type,
                            performance_metrics=metrics[agent_type]
                        )
                    
                    # Wait for next monitoring interval
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {str(e)}")
                    await asyncio.sleep(30)  # Wait before retrying
        
        # Start monitoring loop
        asyncio.create_task(monitoring_loop())
    
    async def process_task(self, 
                          task: str, 
                          agent_type: str, 
                          **kwargs) -> Dict[str, Any]:
        """
        Process a task using the specified agent with enhanced tracking.
        
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
        
        try:
            # Evaluate task complexity
            complexity_evaluation = self.complexity_evaluator.evaluate_complexity(
                agent_type=agent_type,
                task=task,
                context=kwargs.get('context')
            )
            
            # Record start time for performance tracking
            start_time = datetime.now()
            
            # Process task with appropriate agent
            if agent_type == "king":
                interaction = await agent.process_task(
                    task=task,
                    system_prompt=kwargs.get('system_prompt')
                )
            elif agent_type == "sage":
                interaction = await agent.conduct_research(
                    task=task,
                    depth=kwargs.get('depth', 3)
                )
            elif agent_type == "magi":
                interaction = await agent.generate_code(
                    task=task,
                    language=kwargs.get('language', 'python')
                )
            
            # Calculate performance metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            performance_metrics = {
                "response_time": duration,
                "token_usage": interaction.metadata.get("usage", {}).get("total_tokens", 0),
                "was_complex": complexity_evaluation["is_complex"]
            }
            
            # Store interaction data
            await self.data_collector.store_interaction(
                agent_type=agent_type,
                interaction=interaction.__dict__,
                was_complex=complexity_evaluation["is_complex"],
                performance_metrics=performance_metrics
            )
            
            # Update complexity evaluator with performance data
            self.complexity_evaluator.record_performance(
                agent_type=agent_type,
                task_complexity=complexity_evaluation,
                performance_metrics=performance_metrics
            )
            
            # Return interaction as dictionary
            return {
                "response": interaction.response,
                "model_used": interaction.model,
                "complexity_analysis": complexity_evaluation,
                "performance_metrics": performance_metrics,
                "metadata": interaction.metadata,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing task with {agent_type} agent: {str(e)}")
            raise
    
    def get_agent(self, name: str) -> Optional[Any]:
        """
        Get an agent instance by name.
        
        Args:
            name: Agent name ("king", "sage", or "magi")
            
        Returns:
            Agent instance or None if not found
        """
        return self.agents.get(name.lower())
    
    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """
        Get an agent's configuration.
        
        Args:
            name: Agent name ("king", "sage", or "magi")
            
        Returns:
            AgentConfig instance or None if not found
        """
        return self.config.get_agent_config(name.lower())
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive performance metrics for all agents.
        
        Returns:
            Dictionary mapping agent names to their performance metrics
        """
        metrics = {}
        for name, agent in self.agents.items():
            agent_metrics = agent.get_performance_metrics()
            
            # Add complexity evaluation metrics
            complexity_analysis = self.complexity_evaluator.get_threshold_analysis(name)
            
            # Combine metrics
            metrics[name] = {
                **agent_metrics,
                "complexity_threshold": complexity_analysis["current_threshold"],
                "complex_task_ratio": complexity_analysis["complex_task_ratio"],
                "performance_by_complexity": complexity_analysis["performance_by_complexity"]
            }
        
        return metrics
    
    async def get_training_data(self, 
                              agent_type: Optional[str] = None,
                              min_quality: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get training data for local models with filtering.
        
        Args:
            agent_type: Optional agent type to filter by
            min_quality: Optional minimum quality score
            
        Returns:
            Dictionary mapping agent names to their training data
        """
        if agent_type:
            if agent_type not in self.agents:
                raise ValueError(f"Invalid agent type: {agent_type}")
            agents_to_check = [agent_type]
        else:
            agents_to_check = list(self.agents.keys())
        
        training_data = {}
        for name in agents_to_check:
            data = await self.data_collector.get_training_data(
                agent_type=name,
                min_quality=min_quality
            )
            training_data[name] = data
        
        return training_data
    
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
    
    async def export_agent_data(self, 
                              export_dir: Optional[str] = None,
                              format: str = "json") -> Dict[str, str]:
        """
        Export all agent data and metrics.
        
        Args:
            export_dir: Optional directory for export files
            format: Export format ("json" or "csv")
            
        Returns:
            Dictionary mapping data types to export file paths
        """
        return await self.data_collector.export_data(
            export_dir=export_dir,
            format=format
        )
