"""MAGI Agent core functionality."""

from typing import Dict, Any, List, Optional
import asyncio
import logging
import numpy as np
from queue import PriorityQueue
from collections import defaultdict
from dataclasses import dataclass, field
from ..tools.tool_creator import ToolCreator
from ..tools.tool_management import ToolManager
from ..tools.tool_optimization import ToolOptimizer
from ..tools.reverse_engineer import ReverseEngineer
from .continuous_learner import ContinuousLearner
from .evolution_manager import EvolutionManager
from .quality_assurance_layer import QualityAssuranceLayer
from .magi_planning import MagiPlanning, GraphManager
from .knowledge_manager import KnowledgeManager
from ..tools.tool_persistence import ToolPersistence
from ..utils.task import Task as LangroidTask
from communications.protocol import StandardCommunicationProtocol
from rag_system.core.config import RAGConfig
from rag_system.core.pipeline import EnhancedRAGPipeline
from langroid.vector_store.base import VectorStore

logger = logging.getLogger(__name__)

@dataclass
class MagiAgentConfig:
    """Configuration for MagiAgent."""
    name: str
    description: str
    capabilities: List[str]
    vector_store: Any
    model: str
    instructions: str
    development_capabilities: List[str] = field(default_factory=list)

class MagiAgent:
    def __init__(
        self,
        config: MagiAgentConfig,
        communication_protocol: StandardCommunicationProtocol,
        rag_config: RAGConfig,
        vector_store: VectorStore,
    ):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.capabilities = config.capabilities + config.development_capabilities
        self.model = config.model
        self.instructions = config.instructions
        self.communication_protocol = communication_protocol
        self.communication_protocol.subscribe(self.name, self.handle_message)
        self.vector_store = vector_store
        self.rag_system = EnhancedRAGPipeline(rag_config)
        
        # Initialize core components
        self.qa_layer = QualityAssuranceLayer()
        self.evolution_manager = EvolutionManager()
        self.continuous_learner = ContinuousLearner(self.qa_layer, self.evolution_manager)
        self.graph_manager = GraphManager()
        self.magi_planning = MagiPlanning(communication_protocol, self.qa_layer, self.graph_manager)
        
        # Initialize tool-related components
        self.tool_persistence = ToolPersistence("tools_storage")
        self.tool_creator = ToolCreator(llm=self.model, continuous_learner=self.continuous_learner)
        self.tool_manager = ToolManager(
            llm=self.model,
            tool_persistence=self.tool_persistence,
            tool_creator=self.tool_creator,
            continuous_learner=self.continuous_learner
        )
        self.tool_optimizer = ToolOptimizer(
            llm=self.model,
            tool_persistence=self.tool_persistence,
            tool_creator=self.tool_creator,
            continuous_learner=self.continuous_learner
        )
        self.reverse_engineer = ReverseEngineer(
            llm=self.model,
            continuous_learner=self.continuous_learner
        )
        
        # Initialize knowledge management
        self.knowledge_manager = KnowledgeManager(
            llm=self.model,
            vector_store=self.vector_store
        )
        
        # Initialize state
        self.tools: Dict[str, Any] = {}
        self.development_capabilities = config.development_capabilities
        self.llm = config.model
        self.task_queue = PriorityQueue()
        self.monitoring_data = []
        self.task_history: List[Dict[str, Any]] = []
        self.tool_usage_count = defaultdict(int)
        
        self.load_persisted_tools()

    async def generate(self, prompt: str) -> str:
        """Generate text using the language model."""
        response = await self.llm.complete(prompt)
        return response.text

    async def evolve(self):
        """Evolve the agent's capabilities and parameters."""
        await self.evolution_manager.evolve(self)
        await self.magi_planning.evolve()
        await self.continuous_learner.evolve()
        await self.qa_layer.evolve()
        logger.info(f"MagiAgent evolved to generation {self.evolution_manager.generation}")

    async def self_reflect(self):
        """Perform self-reflection and improvement."""
        await self.enhanced_self_reflection()
        await self.analyze_long_term_trends()
        await self.analyze_capabilities()
        await self.analyze_tool_effectiveness()

    async def handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages."""
        if message.get('type') == 'task':
            task = LangroidTask(self, message['content'])
            result = await self.execute_task(task)
            await self.communication_protocol.send_message({
                'type': 'response',
                'content': result,
                'sender': self.name,
                'receiver': message['sender']
            })

    async def create_dynamic_tool(
        self,
        name: str,
        code: str,
        description: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new tool dynamically."""
        try:
            # Create tool using ToolCreator
            tool_data = await self.tool_creator.create_tool(
                name=name,
                description=description,
                parameters=parameters,
                code=code
            )
            
            # Store tool
            self.tools[name] = tool_data['function']
            self.tool_persistence.save_tool(
                name=name,
                code=code,
                description=description,
                parameters=parameters
            )
            
            return tool_data
            
        except Exception as e:
            logger.error(f"Error creating tool {name}: {e}")
            raise

    async def reverse_engineer(self, program_path: str) -> Dict[str, Any]:
        """Perform reverse engineering analysis."""
        try:
            # Use ReverseEngineer component
            return await self.reverse_engineer.analyze_program(program_path)
        except Exception as e:
            logger.error(f"Error reverse engineering {program_path}: {e}")
            raise

    async def execute_task(self, task: LangroidTask) -> Dict[str, Any]:
        # Quality Assurance Layer check
        is_safe, metrics = self.qa_layer.check_task_safety(task)
        if not is_safe:
            logger.warning(f"Task '{task.content}' deemed unsafe: {metrics}")
            return {"error": "Task deemed unsafe", "metrics": metrics}

        # Process the task through RAG pipeline
        rag_result = await self.rag_system.process_query(task.content)
        task.content += f"\nRAG Context: {rag_result}"

        # Use MagiPlanning to generate a plan
        plan = await self.magi_planning.generate_plan(task.content, {"problem_analysis": rag_result})

        # Execute the plan
        result = await self.execute_plan(plan)

        # Continuous Learning Layer update
        await self.continuous_learner.update_embeddings(task, result)
        await self.continuous_learner.learn_from_task_execution(task, result, list(self.tools.keys()))

        # Adjust learning rate and trigger evolution if needed
        self.continuous_learner.adjust_learning_rate()

        # Calculate performance metric
        performance = await self.calculate_performance(task, result)

        # Apply feedback and update task history
        await self.apply_feedback(task, result, performance)

        return result

    async def calculate_performance(self, task: LangroidTask, result: Dict[str, Any]) -> float:
        # Implement a more sophisticated performance calculation
        execution_time = result.get('execution_time', 1.0)
        output_quality = await self.assess_output_quality(task, result)
        resource_efficiency = await self.assess_resource_efficiency(result)

        # Combine metrics (you can adjust weights as needed)
        performance = (
            0.4 * output_quality +
            0.3 * (1 / execution_time) +  # Inverse of execution time, faster is better
            0.3 * resource_efficiency
        )

        return performance

    async def assess_output_quality(self, task: LangroidTask, result: Dict[str, Any]) -> float:
        # Implement logic to assess the quality of the output
        quality_prompt = f"Assess the quality of this output for the given task on a scale of 0 to 1:\n\nTask: {task.content}\n\nOutput: {result.get('output', '')}"
        quality_assessment = await self.generate(quality_prompt)
        return float(quality_assessment)

    async def assess_resource_efficiency(self, result: Dict[str, Any]) -> float:
        # Implement logic to assess resource efficiency
        return 0.8  # Placeholder value

    async def apply_feedback(self, task: LangroidTask, result: Dict[str, Any], performance: float):
        # Update task history
        self.task_history.append({
            "task": task.content,
            "result": result,
            "performance": performance
        })

        # Update evolution manager
        await self.evolution_manager.update(task, result, performance)

        # Update graph manager
        self.graph_manager.update_agent_experience(self.name, task.id, performance)

        # Trigger learning from feedback
        await self.continuous_learner.learn_from_feedback([{
            "task_content": task.content,
            "performance": performance
        }])

        # Analyze long-term trends
        await self.analyze_long_term_trends()

    async def analyze_long_term_trends(self):
        if len(self.task_history) % 50 == 0:  # Analyze every 50 tasks
            recent_performances = [task['performance'] for task in self.task_history[-50:]]
            avg_performance = sum(recent_performances) / len(recent_performances)
            performance_trend = self.calculate_trend(recent_performances)

            logger.info(f"Long-term performance analysis: Average: {avg_performance:.2f}, Trend: {performance_trend}")

            # Adjust learning rate based on performance trend
            await self.adjust_learning_rate(avg_performance, performance_trend)

            # Analyze capability usage and effectiveness
            await self.analyze_capabilities()

            # Analyze tool effectiveness
            await self.analyze_tool_effectiveness()

    async def analyze_capabilities(self):
        capability_usage = defaultdict(int)
        capability_performance = defaultdict(list)

        for task in self.task_history[-50:]:
            for capability in self.capabilities:
                if capability.lower() in task['task'].lower():
                    capability_usage[capability] += 1
                    capability_performance[capability].append(task['performance'])

        for capability, usage in capability_usage.items():
            avg_performance = sum(capability_performance[capability]) / len(capability_performance[capability])
            logger.info(f"Capability '{capability}': Usage: {usage}, Avg Performance: {avg_performance:.2f}")

            if usage < 5 and avg_performance < 0.6:
                await self.refine_capability(capability)
            elif usage > 10 and avg_performance > 0.8:
                await self.enhance_capability(capability)
            elif usage == 0:
                await self.consider_removing_capability(capability)

        await self.consider_new_capabilities(capability_usage)

    async def analyze_tool_effectiveness(self):
        tool_usage = defaultdict(int)
        tool_performance = defaultdict(list)

        for task in self.task_history[-50:]:
            for tool in task.get('tools_used', []):
                tool_usage[tool] += 1
                tool_performance[tool].append(task['performance'])

        for tool, usage in tool_usage.items():
            avg_performance = sum(tool_performance[tool]) / len(tool_performance[tool])
            logger.info(f"Tool '{tool}': Usage: {usage}, Avg Performance: {avg_performance:.2f}")

            if usage < 3 and avg_performance < 0.5:
                await self.tool_manager.consider_removing_tool(tool)
            elif usage > 10 and avg_performance > 0.8:
                await self.tool_optimizer.optimize_tool(tool)

        recent_tasks = [task['task'] for task in self.task_history[-50:]]
        await self.tool_manager.consider_new_tools(tool_usage, recent_tasks)

    async def periodic_maintenance(self):
        """Perform periodic maintenance tasks to improve MAGI's performance."""
        await self.evolve()
        await self.self_reflect()
        await self.knowledge_manager.optimize_knowledge_base()
        await self.tool_manager.prune_unused_tools(self.tool_usage_count)
        logger.info("Completed periodic maintenance")

    async def enhanced_self_reflection(self):
        recent_tasks = self.task_history[-20:]
        performance_trend = self.calculate_trend([task['performance'] for task in recent_tasks])
        
        reflection_prompt = f"""
        Perform a comprehensive self-reflection analysis considering the following aspects:
        1. Performance trend: {performance_trend}
        2. Most successful strategies: {self.get_top_strategies(recent_tasks)}
        3. Most challenging task types: {self.get_challenging_tasks(recent_tasks)}
        4. Most effective tools: {self.get_effective_tools(recent_tasks)}
        5. Recent experiment results: {self.get_recent_experiment_results()}
        6. Current knowledge base status: {self.knowledge_manager.get_knowledge_base_status()}
        7. Capability utilization: {self.get_capability_utilization()}

        Based on this comprehensive analysis:
        1. What are the key strengths to reinforce?
        2. What are the main areas for improvement?
        3. What new capabilities or tools should be developed?
        4. How can decision-making processes be enhanced?
        5. What experiments should be prioritized?
        6. Suggest 3-5 concrete actions to enhance overall performance and adaptability.
        """
        
        reflection = await self.generate(reflection_prompt)
        logger.info(f"Enhanced self-reflection insights:\n{reflection}")
        
        await self.implement_reflection_insights(reflection)

    async def implement_reflection_insights(self, reflection: str):
        # Parse the reflection and implement suggested actions
        actions = reflection.split("Suggest 3-5 concrete actions to enhance overall performance and adaptability.")[-1].split("\n")
        for action in actions:
            if action.strip():
                await self.implement_action(action.strip())

    async def implement_action(self, action: str):
        logger.info(f"Implementing action: {action}")
        if "create new tool" in action.lower():
            await self.tool_manager.create_new_tool(action)
        elif "adjust parameter" in action.lower():
            await self.adjust_parameter(action)
        elif "refine strategy" in action.lower():
            await self.refine_strategy(action.split("refine strategy")[-1].strip())
        elif "develop new capability" in action.lower():
            await self.develop_new_capability(action)
        elif "optimize decision process" in action.lower():
            await self.optimize_decision_process(action)
        elif "prioritize experiment" in action.lower():
            await self.prioritize_experiment(action)
        else:
            # For actions that don't fit predefined categories, use a more general approach
            await self.execute_general_action(action)

    async def develop_new_capability(self, action: str):
        capability_prompt = f"Design a new capability based on this action: {action}. Provide a name, description, and high-level implementation plan."
        capability_design = await self.generate(capability_prompt)
        # Implement logic to add the new capability
        pass

    async def optimize_decision_process(self, action: str):
        optimization_prompt = f"Suggest specific improvements to the decision-making process based on this action: {action}. Include changes to algorithms, criteria, or evaluation methods."
        optimization_plan = await self.generate(optimization_prompt)
        # Implement logic to update the decision-making process
        pass

    async def prioritize_experiment(self, action: str):
        experiment_prompt = f"Design a high-priority experiment based on this action: {action}. Include hypothesis, methodology, and expected outcomes."
        experiment_design = await self.generate(experiment_prompt)
        # Implement logic to add the experiment to a priority queue
        pass

    async def execute_general_action(self, action: str):
        execution_prompt = f"Provide a step-by-step plan to implement this action: {action}"
        execution_plan = await self.generate(execution_prompt)
        # Implement logic to execute the general action plan
        pass

    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from a list of values."""
        if len(values) < 2:
            return "Not enough data"
        
        x = np.arange(len(values))
        y = np.array(values)
        z = np.polyfit(x, y, 1)
        slope = z[0]
        
        if slope > 0.1:
            return "Strongly Improving"
        elif slope > 0.05:
            return "Improving"
        elif slope < -0.1:
            return "Strongly Declining"
        elif slope < -0.05:
            return "Declining"
        else:
            return "Stable"

    def get_top_strategies(self, recent_tasks: List[Dict[str, Any]]) -> List[str]:
        """Get most successful strategies from recent tasks."""
        strategy_performance = defaultdict(list)
        for task in recent_tasks:
            if 'strategy' in task:
                strategy_performance[task['strategy']].append(task['performance'])
        
        avg_performance = {
            strategy: sum(perfs) / len(perfs)
            for strategy, perfs in strategy_performance.items()
        }
        
        return sorted(
            avg_performance.keys(),
            key=lambda x: avg_performance[x],
            reverse=True
        )[:3]

    def get_challenging_tasks(self, recent_tasks: List[Dict[str, Any]]) -> List[str]:
        """Get types of tasks that have been challenging."""
        task_type_performance = defaultdict(list)
        for task in recent_tasks:
            task_type = self._infer_task_type(task['task'])
            task_type_performance[task_type].append(task['performance'])
        
        avg_performance = {
            task_type: sum(perfs) / len(perfs)
            for task_type, perfs in task_type_performance.items()
        }
        
        return [
            task_type for task_type, perf in avg_performance.items()
            if perf < 0.6
        ]

    def _infer_task_type(self, task_content: str) -> str:
        """Infer the type of a task from its content."""
        if "code" in task_content.lower():
            return "coding"
        elif "debug" in task_content.lower():
            return "debugging"
        elif "review" in task_content.lower():
            return "review"
        else:
            return "general"

    def get_effective_tools(self, recent_tasks: List[Dict[str, Any]]) -> List[str]:
        """Get most effective tools from recent tasks."""
        tool_performance = defaultdict(list)
        for task in recent_tasks:
            for tool in task.get('tools_used', []):
                tool_performance[tool].append(task['performance'])
        
        avg_performance = {
            tool: sum(perfs) / len(perfs)
            for tool, perfs in tool_performance.items()
        }
        
        return sorted(
            avg_performance.keys(),
            key=lambda x: avg_performance[x],
            reverse=True
        )[:5]

    def get_recent_experiment_results(self) -> List[Dict[str, Any]]:
        """Get results from recent experiments."""
        return [
            task for task in self.task_history[-10:]
            if task.get('type') == 'experiment'
        ]

    def get_capability_utilization(self) -> Dict[str, float]:
        """Get utilization metrics for each capability."""
        capability_usage = defaultdict(int)
        total_tasks = len(self.task_history[-50:])
        
        for task in self.task_history[-50:]:
            for capability in self.capabilities:
                if capability.lower() in task['task'].lower():
                    capability_usage[capability] += 1
        
        return {
            capability: count / total_tasks
            for capability, count in capability_usage.items()
        }

    async def adjust_learning_rate(self, avg_performance: float, trend: str):
        """Adjust learning rate based on performance."""
        if trend in ["Strongly Improving", "Improving"]:
            self.continuous_learner.learning_rate *= 0.9
        elif trend in ["Strongly Declining", "Declining"]:
            self.continuous_learner.learning_rate *= 1.1
        
        # Keep learning rate within reasonable bounds
        self.continuous_learner.learning_rate = max(
            0.001,
            min(0.1, self.continuous_learner.learning_rate)
        )

    async def refine_capability(self, capability: str):
        """Refine an underperforming capability."""
        refinement_prompt = f"""
        Analyze and suggest improvements for the capability: {capability}
        
        Recent performance has been suboptimal. Consider:
        1. Current implementation weaknesses
        2. Potential improvements
        3. Required changes
        4. Success metrics
        """
        refinement_plan = await self.generate(refinement_prompt)
        # Implement refinement plan
        pass

    async def enhance_capability(self, capability: str):
        """Enhance a well-performing capability."""
        enhancement_prompt = f"""
        Design enhancements for the successful capability: {capability}
        
        Consider:
        1. Current strengths
        2. Potential extensions
        3. New features
        4. Integration opportunities
        """
        enhancement_plan = await self.generate(enhancement_prompt)
        # Implement enhancement plan
        pass

    async def consider_removing_capability(self, capability: str):
        """Consider removing an unused capability."""
        analysis_prompt = f"""
        Analyze the implications of removing capability: {capability}
        
        Consider:
        1. Historical usage
        2. Dependencies
        3. Future potential
        4. Maintenance cost
        """
        analysis = await self.generate(analysis_prompt)
        # Make decision based on analysis
        pass

    async def consider_new_capabilities(self, capability_usage: Dict[str, int]):
        """Consider adding new capabilities."""
        analysis_prompt = f"""
        Analyze current capabilities and suggest new ones.
        
        Current usage:
        {capability_usage}
        
        Consider:
        1. Missing functionality
        2. Common patterns
        3. Future needs
        4. Integration requirements
        """
        suggestions = await self.generate(analysis_prompt)
        # Implement new capabilities
        pass

    async def adjust_parameter(self, action: str):
        """Adjust system parameters based on action."""
        param_name = action.split("adjust parameter")[-1].strip()
        if hasattr(self, param_name):
            adjustment_prompt = f"Suggest optimal value for parameter: {param_name}"
            new_value = await self.generate(adjustment_prompt)
            setattr(self, param_name, new_value)

    async def refine_strategy(self, strategy: str):
        """Refine a strategy based on performance data."""
        refinement_prompt = f"""
        Refine strategy: {strategy}
        
        Consider:
        1. Historical performance
        2. Success patterns
        3. Failure points
        4. Optimization opportunities
        """
        refinement = await self.generate(refinement_prompt)
        # Implement refinement
        pass

    async def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generated plan."""
        result = {"steps": []}
        
        for step in plan.get('steps', []):
            step_result = await self._execute_step(step)
            result["steps"].append(step_result)
            
            if not step_result.get('success', False):
                break
        
        result["success"] = all(step.get('success', False) for step in result["steps"])
        return result

    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step of a plan."""
        try:
            if step.get('type') == 'tool':
                tool = self.tools.get(step['tool'])
                if tool:
                    result = await tool(**step.get('parameters', {}))
                    return {"success": True, "result": result}
            elif step.get('type') == 'task':
                result = await self.execute_task(LangroidTask(self, step['content']))
                return {"success": True, "result": result}
            
            return {"success": False, "error": "Invalid step type"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def load_persisted_tools(self):
        """Load persisted tools from storage."""
        tools = self.tool_persistence.load_all_tools()
        for name, tool_data in tools.items():
            self.tools[name] = self._create_tool_function(tool_data)

    def _create_tool_function(self, tool_data: Dict[str, Any]) -> Any:
        """Create a function from tool data."""
        # Create namespace for function
        namespace = {}
        
        # Execute code in namespace
        exec(tool_data['code'], namespace)
        
        # Get function from namespace
        if tool_data['name'] not in namespace:
            raise ValueError(f"Function '{tool_data['name']}' not found in code")
        
        return namespace[tool_data['name']]

# Example usage
if __name__ == "__main__":
    vector_store = VectorStore()  # Placeholder, implement actual VectorStore
    communication_protocol = StandardCommunicationProtocol()
    rag_config = RAGConfig()
    
    magi_config = MagiAgentConfig(
        name="MagiAgent",
        description="A development and coding agent",
        capabilities=["coding", "debugging", "code_review"],
        vector_store=vector_store,
        model="gpt-4",
        instructions="You are a Magi agent capable of writing, debugging, and reviewing code."
    )
    
    magi_agent = MagiAgent(magi_config, communication_protocol, rag_config, vector_store)
    
    # Use the magi_agent to process tasks and perform reverse engineering
    program_path = "path/to/program"
    asyncio.run(magi_agent.reverse_engineer(program_path))

    # Perform periodic maintenance
    asyncio.run(magi_agent.periodic_maintenance())

    print("MagiAgent has been evolved and improved.")
