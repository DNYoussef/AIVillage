"""MAGI Agent implementation."""

from typing import Dict, Any, List, Optional
import logging
import asyncio
import os
import json
from datetime import datetime

from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from agents.utils.task import Task as LangroidTask
from agents.utils.logging import get_logger
from agents.utils.exceptions import AIVillageException

# MAGI-specific imports
from agents.magi.core.task_research import TaskResearch
from agents.magi.core.quality_assurance_layer import QualityAssuranceLayer
from agents.magi.core.evolution_manager import EvolutionManager
from agents.magi.core.continuous_learner import ContinuousLearner
from agents.magi.core.magi_planning import GraphManager, MagiPlanning
from agents.magi.core.project_planner import ProjectPlanner
from agents.magi.core.knowledge_manager import KnowledgeManager

# Tool-related imports
from agents.magi.tools.tool_persistence import ToolPersistence
from agents.magi.tools.tool_creator import ToolCreator
from agents.magi.tools.tool_management import ToolManager
from agents.magi.tools.tool_optimization import ToolOptimizer
from communications.protocol import StandardCommunicationProtocol

logger = get_logger(__name__)

class MAGIAgent(UnifiedBaseAgent):
    """
    MAGI (Multi-Agent General Intelligence) Agent implementation.
    Inherits from UnifiedBaseAgent while maintaining specialized capabilities.
    """
    
    def __init__(
        self,
        config: UnifiedAgentConfig,
        communication_protocol: StandardCommunicationProtocol
    ):
        super().__init__(config, communication_protocol)
        
        # Initialize MAGI-specific components
        self.task_manager = TaskResearch()
        self.magi_planning = MagiPlanning(
            communication_protocol,
            self.quality_assurance_layer,
            self.graph_manager
        )
        self.project_planner = ProjectPlanner()
        self.tool_persistence = ToolPersistence("tools_storage")
        self.tool_creator = ToolCreator()
        self.tool_manager = ToolManager()
        self.tool_optimizer = ToolOptimizer()
        
        # Load persisted tools
        self.load_persisted_tools()
        
        # Add MAGI-specific capabilities
        self.add_capability("tool_creation", {
            "description": "Dynamic tool creation and optimization",
            "confidence": 0.95
        })
        self.add_capability("project_planning", {
            "description": "Advanced project planning and execution",
            "confidence": 0.9
        })
        self.add_capability("knowledge_synthesis", {
            "description": "Complex knowledge integration and analysis",
            "confidence": 0.9
        })

    def load_persisted_tools(self):
        """Load previously persisted tools."""
        persisted_tools = self.tool_persistence.load_all_tools()
        for tool_name, tool_data in persisted_tools.items():
            self.tools[tool_name] = tool_data

    async def _process_task(self, task: LangroidTask) -> Dict[str, Any]:
        """
        Implement core task processing logic specific to MAGIAgent.
        """
        try:
            # Initial task analysis
            task_analysis = await self.task_manager.analyze_task(task)
            
            # Generate execution plan
            plan = await self.magi_planning.create_plan(task_analysis)
            
            # Execute plan with continuous monitoring
            result = await self._execute_plan_with_monitoring(plan)
            
            # Post-execution processing
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
            await self.continuous_learner.learn_from_execution(task, result)
            
            # Optimize tools based on usage
            await self.tool_optimizer.optimize_tools(self.tool_manager.get_tool_usage_stats())
            
            # Persist relevant data
            await self.tool_persistence.save_execution_data(task, result)
        except Exception as e:
            logger.exception(f"Error in post-execution processing: {str(e)}")

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
                'quality_assurance': await self.quality_assurance_layer.get_state(),
                'evolution_manager': await self.evolution_manager.get_state(),
                'continuous_learner': await self.continuous_learner.get_state(),
                'graph_manager': await self.graph_manager.get_state(),
                'magi_planning': await self.magi_planning.get_state(),
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
            await self.quality_assurance_layer.load_state(state['quality_assurance'])
            await self.evolution_manager.load_state(state['evolution_manager'])
            await self.continuous_learner.load_state(state['continuous_learner'])
            await self.graph_manager.load_state(state['graph_manager'])
            await self.magi_planning.load_state(state['magi_planning'])
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
                'quality_assurance': await self.quality_assurance_layer.get_status(),
                'evolution_manager': await self.evolution_manager.get_status(),
                'continuous_learner': await self.continuous_learner.get_status(),
                'graph_manager': await self.graph_manager.get_status(),
                'magi_planning': await self.magi_planning.get_status(),
                'tool_manager': await self.tool_manager.get_status(),
                'knowledge_manager': await self.knowledge_manager.get_status()
            }
        except Exception as e:
            logger.exception(f"Error getting agent status: {str(e)}")
            raise AIVillageException(f"Error getting agent status: {str(e)}")
