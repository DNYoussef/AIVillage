"""Main entry point for the AI Village system."""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml
from .agents.agent_manager import AgentManager
from .data.data_collector import DataCollector
from .data.complexity_evaluator import ComplexityEvaluator
from config.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

class AIVillage:
    """
    Main system orchestrator that manages agents, data collection,
    and complexity evaluation for the AI Village system.
    """
    
    def __init__(self, config: Optional[UnifiedConfig] = None):
        """
        Initialize AIVillage.
        
        Args:
            config: Optional UnifiedConfig instance. If not provided,
                   will load from default config path.
        """
        # Ensure OpenRouter API key is set
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Initialize configuration
        self.config = config if config else UnifiedConfig()
        
        # Initialize components with config
        self.agent_manager = AgentManager(self.config)
        self.data_collector = DataCollector(self.config)
        self.complexity_evaluator = ComplexityEvaluator(self.config)
        
        # Task queue
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info("Initialized AI Village system")
    
    async def process_task(self, task: str, agent_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a task using the appropriate agent.
        
        Args:
            task: The task to process
            agent_type: Optional specific agent to use
            **kwargs: Additional arguments for the agent
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Determine appropriate agent if not specified
            if not agent_type:
                agent_type = self._determine_agent_type(task)
            
            # Evaluate task complexity
            complexity_analysis = await self.complexity_evaluator.evaluate_complexity(
                agent_type=agent_type,
                task=task,
                context=kwargs.get("context")
            )
            
            # Process task with appropriate agent
            result = await self.agent_manager.process_task(
                task=task,
                agent_type=agent_type,
                **kwargs
            )
            
            # Store interaction data
            await self.data_collector.store_interaction(
                agent_type=agent_type,
                interaction=result,
                was_complex=complexity_analysis["is_complex"]
            )
            
            # If frontier model was used, store training data
            agent_config = self.config.get_agent_config(agent_type)
            if result["model"] == agent_config.frontier_model.name:
                await self.data_collector.store_training_example(
                    agent_type=agent_type,
                    frontier_model=agent_config.frontier_model.name,
                    local_model=agent_config.local_model.name,
                    prompt=task,
                    response=result["response"],
                    quality_score=result.metadata.get("quality_score", 0.0)
                )
            
            # Store performance metrics
            agent = self.agent_manager.get_agent(agent_type)
            await self.data_collector.store_performance_metrics(
                agent_type=agent_type,
                model_type="frontier" if complexity_analysis["is_complex"] else "local",
                metrics=agent.get_performance_metrics()
            )
            
            return {
                "response": result["response"],
                "model_used": result["model"],
                "complexity_analysis": complexity_analysis,
                "performance_metrics": agent.get_performance_metrics(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _determine_agent_type(self, task: str) -> str:
        """
        Determine the most appropriate agent type for a task.
        
        Args:
            task: The task to analyze
            
        Returns:
            Agent type ("king", "sage", or "magi")
        """
        task_lower = task.lower()
        
        # Check for code-related indicators
        code_indicators = [
            "code", "program", "function", "implement",
            "algorithm", "debug", "compile", "programming"
        ]
        if any(indicator in task_lower for indicator in code_indicators):
            return "magi"
        
        # Check for research-related indicators
        research_indicators = [
            "research", "analyze", "study", "investigate",
            "examine", "review", "survey", "evaluate"
        ]
        if any(indicator in task_lower for indicator in research_indicators):
            return "sage"
        
        # Default to king for general tasks
        return "king"
    
    async def run_task_loop(self):
        """Run the main task processing loop."""
        logger.info("Starting task processing loop")
        
        while True:
            try:
                # Get task from queue
                task_data = await self.task_queue.get()
                
                # Process task
                result = await self.process_task(**task_data)
                
                # Update complexity thresholds periodically
                await self._update_complexity_thresholds()
                
                # Mark task as done
                self.task_queue.task_done()
                
                logger.info(f"Processed task using {result['model_used']}")
                
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
                continue
    
    async def _update_complexity_thresholds(self):
        """Update complexity thresholds based on recent performance."""
        for agent_type in ["king", "sage", "magi"]:
            try:
                # Get recent performance data
                agent = self.agent_manager.get_agent(agent_type)
                performance_metrics = agent.get_performance_metrics()
                
                # Get recent complexity history
                complexity_history = await self.data_collector.get_performance_history(
                    agent_type=agent_type,
                    model_type="local",
                    metric_name="complexity_score",
                    days=7
                )
                
                # Adjust thresholds
                new_threshold = await self.complexity_evaluator.adjust_thresholds(
                    agent_type=agent_type,
                    performance_metrics=performance_metrics,
                    complexity_history=complexity_history
                )
                
                if new_threshold is not None:
                    logger.info(f"Updated complexity threshold for {agent_type}: {new_threshold}")
                
            except Exception as e:
                logger.error(f"Error updating complexity thresholds for {agent_type}: {str(e)}")
    
    async def add_task(self, task: str, **kwargs):
        """
        Add a task to the processing queue.
        
        Args:
            task: The task to process
            **kwargs: Additional arguments for processing
        """
        await self.task_queue.put({
            "task": task,
            **kwargs
        })
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and metrics.
        
        Returns:
            Dictionary containing system status information
        """
        agent_metrics = {}
        for agent_type in ["king", "sage", "magi"]:
            try:
                agent = self.agent_manager.get_agent(agent_type)
                metrics = agent.get_performance_metrics()
                agent_metrics[agent_type] = {
                    "task_success_rate": metrics.get("task_success_rate", 0.0),
                    "local_model_performance": metrics.get("local_model_performance", 0.0)
                }
            except Exception as e:
                logger.error(f"Error getting metrics for {agent_type}: {str(e)}")
                agent_metrics[agent_type] = {
                    "task_success_rate": 0.0,
                    "local_model_performance": 0.0
                }
        
        return {
            "queue_size": self.task_queue.qsize(),
            "agent_metrics": agent_metrics,
            "complexity_thresholds": {
                agent_type: self.complexity_evaluator.get_threshold(agent_type)
                for agent_type in ["king", "sage", "magi"]
            },
            "training_data_counts": {
                agent_type: len(self.data_collector.get_training_data(agent_type=agent_type))
                for agent_type in ["king", "sage", "magi"]
            },
            "system_health": {
                "status": "healthy",
                "last_error": None
            }
        }

async def main():
    """Main entry point for the AI Village system."""
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize AI Village
        village = AIVillage()
        
        # Start task processing loop
        task_loop = asyncio.create_task(village.run_task_loop())
        
        # Example tasks for testing
        test_tasks = [
            "Analyze the implications of quantum computing on cryptography",
            "Write a Python function to implement merge sort",
            "Develop a strategic plan for AI implementation in healthcare"
        ]
        
        # Add test tasks to queue
        for task in test_tasks:
            await village.add_task(task)
        
        # Wait for tasks to complete
        await village.task_queue.join()
        
        # Cancel task loop
        task_loop.cancel()
        
        # Print final status
        status = village.get_system_status()
        logger.info("Final system status:")
        logger.info(yaml.dump(status, default_flow_style=False))
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
