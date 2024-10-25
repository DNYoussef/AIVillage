"""MAGI Agent implementation."""

from typing import Dict, Any, List, Optional
import logging
import asyncio
import os
import json
from datetime import datetime

from ...utils.logging import get_logger
from ...utils.exceptions import AIVillageException
from ..utils.task_management import TaskManager
from ..core.quality_assurance_layer import QualityAssuranceLayer
from ..core.evolution_manager import EvolutionManager
from ..core.learning_system import LearningSystem
from ..core.graph_manager import GraphManager
from ..core.planning_system import PlanningSystem
from ..tools.persistence_manager import PersistenceManager
from ..tools.tool_creator import ToolCreator
from ..tools.tool_manager import ToolManager
from ..tools.tool_optimizer import ToolOptimizer
from ..core.knowledge_manager import KnowledgeManager

logger = get_logger(__name__)

class MAGIAgent:
    """
    MAGI (Multi-Agent General Intelligence) Agent implementation.
    Coordinates multiple specialized sub-agents and systems for complex task handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MAGI Agent with configuration.
        
        Args:
            config: Configuration dictionary for the agent
        """
        self.config = config
        self.task_manager = TaskManager()
        self.quality_assurance = QualityAssuranceLayer()
        self.evolution_manager = EvolutionManager()
        self.learning_system = LearningSystem()
        self.graph_manager = GraphManager()
        self.planning_system = PlanningSystem()
        self.persistence_manager = PersistenceManager()
        self.tool_creator = ToolCreator()
        self.tool_manager = ToolManager()
        self.tool_optimizer = ToolOptimizer()
        self.knowledge_manager = KnowledgeManager()
        
        logger.info("MAGI Agent initialized with all subsystems")

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task through the MAGI system.
        
        Args:
            task: Task description and parameters
            
        Returns:
            Task results and metadata
        """
        try:
            # Initial task analysis
            task_analysis = await self.task_manager.analyze_task(task)
            
            # Quality check
            qa_result = await self.quality_assurance.check_task(task_analysis)
            if not qa_result['passed']:
                raise AIVillageException(f"Task failed quality check: {qa_result['reason']}")
            
            # Generate execution plan
            plan = await self.planning_system.create_plan(task_analysis)
            
            # Execute plan with continuous monitoring
            result = await self._execute_plan_with_monitoring(plan)
            
            # Post-execution analysis and learning
            await self._post_execution_processing(task, result)
            
            return result
        except Exception as e:
            logger.exception(f"Error processing task: {str(e)}")
            raise AIVillageException(f"Error processing task: {str(e)}")

    async def _execute_plan_with_monitoring(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a plan with continuous monitoring and adaptation.
        
        Args:
            plan: Execution plan details
            
        Returns:
            Execution results
        """
        try:
            execution_context = await self._prepare_execution_context(plan)
            
            while not await self._is_execution_complete(execution_context):
                # Execute next step
                step_result = await self._execute_step(execution_context)
                
                # Monitor and adapt
                if await self._needs_adaptation(step_result):
                    execution_context = await self._adapt_execution(execution_context, step_result)
                
                # Update progress
                execution_context['progress'] = await self._update_progress(execution_context)
            
            return await self._finalize_execution(execution_context)
        except Exception as e:
            logger.exception(f"Error executing plan: {str(e)}")
            raise AIVillageException(f"Error executing plan: {str(e)}")

    async def _prepare_execution_context(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the execution context for a plan.
        
        Args:
            plan: Plan to execute
            
        Returns:
            Execution context
        """
        return {
            'plan': plan,
            'current_step': 0,
            'results': [],
            'progress': 0.0,
            'start_time': datetime.now(),
            'metrics': {}
        }

    async def _is_execution_complete(self, context: Dict[str, Any]) -> bool:
        """
        Check if execution is complete.
        
        Args:
            context: Current execution context
            
        Returns:
            True if execution is complete, False otherwise
        """
        return context['current_step'] >= len(context['plan']['steps'])

    async def _execute_step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step in the plan.
        
        Args:
            context: Current execution context
            
        Returns:
            Step execution results
        """
        step = context['plan']['steps'][context['current_step']]
        result = await self.tool_manager.execute_tool(step['tool'], step['params'])
        context['current_step'] += 1
        context['results'].append(result)
        return result

    async def _needs_adaptation(self, step_result: Dict[str, Any]) -> bool:
        """
        Check if execution needs adaptation based on step results.
        
        Args:
            step_result: Results from the last executed step
            
        Returns:
            True if adaptation is needed, False otherwise
        """
        return step_result.get('needs_adaptation', False)

    async def _adapt_execution(self, context: Dict[str, Any], step_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt the execution based on monitoring results.
        
        Args:
            context: Current execution context
            step_result: Results from the last executed step
            
        Returns:
            Updated execution context
        """
        adaptation_plan = await self.evolution_manager.generate_adaptation(context, step_result)
        return await self.evolution_manager.apply_adaptation(context, adaptation_plan)

    async def _update_progress(self, context: Dict[str, Any]) -> float:
        """
        Update execution progress.
        
        Args:
            context: Current execution context
            
        Returns:
            Updated progress value
        """
        return context['current_step'] / len(context['plan']['steps'])

    async def _finalize_execution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize execution and prepare results.
        
        Args:
            context: Execution context
            
        Returns:
            Final execution results
        """
        return {
            'success': True,
            'results': context['results'],
            'metrics': context['metrics'],
            'execution_time': (datetime.now() - context['start_time']).total_seconds()
        }

    async def _post_execution_processing(self, task: Dict[str, Any], result: Dict[str, Any]):
        """
        Perform post-execution processing and learning.
        
        Args:
            task: Original task
            result: Execution results
        """
        try:
            # Update knowledge base
            await self.knowledge_manager.update_knowledge(task, result)
            
            # Learn from execution
            await self.learning_system.learn_from_execution(task, result)
            
            # Optimize tools based on usage
            await self.tool_optimizer.optimize_tools(self.tool_manager.get_tool_usage_stats())
            
            # Persist relevant data
            await self.persistence_manager.save_execution_data(task, result)
        except Exception as e:
            logger.exception(f"Error in post-execution processing: {str(e)}")
            # Don't raise exception here to avoid affecting the main task result

    async def save_state(self, path: str):
        """
        Save agent state to disk.
        
        Args:
            path: Path to save state
        """
        try:
            state = {
                'config': self.config,
                'task_manager': await self.task_manager.get_state(),
                'quality_assurance': await self.quality_assurance.get_state(),
                'evolution_manager': await self.evolution_manager.get_state(),
                'learning_system': await self.learning_system.get_state(),
                'graph_manager': await self.graph_manager.get_state(),
                'planning_system': await self.planning_system.get_state(),
                'tool_manager': await self.tool_manager.get_state(),
                'knowledge_manager': await self.knowledge_manager.get_state()
            }
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Agent state saved to {path}")
        except Exception as e:
            logger.exception(f"Error saving agent state: {str(e)}")
            raise AIVillageException(f"Error saving agent state: {str(e)}")

    async def load_state(self, path: str):
        """
        Load agent state from disk.
        
        Args:
            path: Path to load state from
        """
        try:
            with open(path, 'r') as f:
                state = json.load(f)
                
            self.config = state['config']
            await self.task_manager.load_state(state['task_manager'])
            await self.quality_assurance.load_state(state['quality_assurance'])
            await self.evolution_manager.load_state(state['evolution_manager'])
            await self.learning_system.load_state(state['learning_system'])
            await self.graph_manager.load_state(state['graph_manager'])
            await self.planning_system.load_state(state['planning_system'])
            await self.tool_manager.load_state(state['tool_manager'])
            await self.knowledge_manager.load_state(state['knowledge_manager'])
            
            logger.info(f"Agent state loaded from {path}")
        except Exception as e:
            logger.exception(f"Error loading agent state: {str(e)}")
            raise AIVillageException(f"Error loading agent state: {str(e)}")

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Status information for the agent and all subsystems
        """
        try:
            return {
                'task_manager': await self.task_manager.get_status(),
                'quality_assurance': await self.quality_assurance.get_status(),
                'evolution_manager': await self.evolution_manager.get_status(),
                'learning_system': await self.learning_system.get_status(),
                'graph_manager': await self.graph_manager.get_status(),
                'planning_system': await self.planning_system.get_status(),
                'tool_manager': await self.tool_manager.get_status(),
                'knowledge_manager': await self.knowledge_manager.get_status()
            }
        except Exception as e:
            logger.exception(f"Error getting agent status: {str(e)}")
            raise AIVillageException(f"Error getting agent status: {str(e)}")
