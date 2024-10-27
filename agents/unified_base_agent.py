"""Unified base agent implementation."""

from types import SimpleNamespace
from typing import Dict, Any, List, Optional, Callable, Tuple
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import random
import numpy as np
from pydantic import BaseModel, Field

# Core imports
from agents.utils.task import Task as LangroidTask
from agents.language_models.openai_gpt import OpenAIGPTConfig
from langroid.vector_store.base import VectorStore

# RAG system imports
from rag_system.core.config import UnifiedConfig
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.processing.reasoning_engine import UncertaintyAwareReasoningEngine
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker
from rag_system.core.exploration_mode import ExplorationMode
from rag_system.utils.advanced_analytics import AdvancedAnalytics
from rag_system.evaluation.comprehensive_evaluation import ComprehensiveEvaluationFramework

# Communication imports
from communications.protocol import StandardCommunicationProtocol, Message, MessageType, Priority
from utils.standardized_formats import OutputFormat, create_standardized_output, create_standardized_prompt

@dataclass
class UnifiedAgentConfig:
    """Enhanced configuration for unified agent implementation."""
    # Basic configuration
    name: str
    description: str
    capabilities: List[str]
    rag_config: UnifiedConfig
    vector_store: VectorStore
    model: str
    instructions: str
    
    # Performance configuration
    max_retries: int = 3
    timeout: float = 30.0
    batch_size: int = 32
    memory_size: int = 1000
    
    # Learning configuration
    learning_rate: float = 0.01
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Evolution configuration
    evolution_rate: float = 0.1
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    
    # Planning configuration
    planning_depth: int = 3
    exploration_weight: float = 1.0
    simulation_depth: int = 10
    
    # Quality assurance configuration
    quality_threshold: float = 0.7
    safety_threshold: float = 0.8
    uncertainty_threshold: float = 0.3
    
    # Resource configuration
    max_memory_usage: float = 0.9
    max_cpu_usage: float = 0.8
    
    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

class UnifiedBaseAgent:
    """Enhanced base agent with standardized implementation and RAG system integration."""
    
    def __init__(self, config: UnifiedAgentConfig, communication_protocol: StandardCommunicationProtocol):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.capabilities = config.capabilities
        self.vector_store = config.vector_store
        self.model = config.model
        self.instructions = config.instructions
        self.tools: Dict[str, Callable] = {}
        self.communication_protocol = communication_protocol
        self.communication_protocol.subscribe(self.name, self.handle_message)
        
        # Initialize LLM
        self.llm = OpenAIGPTConfig(
            chat_model=self.model,
            temperature=config.temperature,
            top_p=config.top_p
        ).create()

        # Initialize RAG system components
        self.rag_pipeline = EnhancedRAGPipeline(config.rag_config)
        self.cognitive_nexus = CognitiveNexus()
        self.hybrid_retriever = HybridRetriever(config.rag_config)
        self.reasoning_engine = UncertaintyAwareReasoningEngine(config.rag_config)
        self.knowledge_tracker = UnifiedKnowledgeTracker(config.vector_store, None)  # Graph store to be added if needed
        self.exploration_mode = ExplorationMode(None, self.llm, None)  # Graph store and NLP to be added if needed
        self.advanced_analytics = AdvancedAnalytics()
        self.evaluation_framework = ComprehensiveEvaluationFramework(self.advanced_analytics)

        # Initialize enhanced layers
        self.quality_assurance_layer = QualityAssuranceLayer(upo_threshold=0.7)
        self.foundational_layer = FoundationalLayer(self.vector_store)
        self.continuous_learning_layer = ContinuousLearningLayer(self.vector_store)
        self.agent_architecture_layer = AgentArchitectureLayer()
        self.decision_making_layer = DecisionMakingLayer()

        # Initialize memory and state
        self.memory = []
        self.state = SimpleNamespace(
            last_task=None,
            last_result=None,
            performance_metrics={},
            error_count=0
        )

        # Add RAG-specific tools
        self.add_tool("query_rag", self.query_rag, "Query the RAG system for information")
        self.add_tool("explore_knowledge", self.explore_knowledge, "Explore knowledge connections")
        self.add_tool("track_knowledge", self.track_knowledge, "Track and update knowledge")
        self.add_tool("reason_with_uncertainty", self.reason_with_uncertainty, "Perform uncertainty-aware reasoning")

    async def query_rag(self, query: str) -> Dict[str, Any]:
        """Query the RAG system with uncertainty handling."""
        try:
            # Start timing for analytics
            start_time = datetime.now()

            # Create standardized prompt
            prompt = create_standardized_prompt(query)

            # Process through RAG pipeline
            retrieval_results = await self.hybrid_retriever.retrieve(prompt)
            reasoning_results = await self.reasoning_engine.reason(prompt, retrieval_results)
            
            # Track knowledge updates
            await self.knowledge_tracker.track_query(query, retrieval_results)
            
            # Process through cognitive nexus
            enhanced_results = await self.cognitive_nexus.process(
                query=query,
                retrieval_results=retrieval_results,
                reasoning_results=reasoning_results
            )

            # Create standardized output
            output = create_standardized_output(enhanced_results, OutputFormat.DETAILED)

            # Record analytics
            end_time = datetime.now()
            self.advanced_analytics.record_query(
                query=query,
                duration=(end_time - start_time).total_seconds(),
                result_count=len(retrieval_results),
                quality_score=enhanced_results.get('quality_score', 0)
            )

            return output

        except Exception as e:
            self.state.error_count += 1
            return {
                "error": str(e),
                "status": "failed",
                "query": query
            }

    async def explore_knowledge(self, start_point: str, depth: int = 2) -> Dict[str, Any]:
        """Explore knowledge connections using the exploration mode."""
        try:
            exploration_results = await self.exploration_mode.explore(
                start_point=start_point,
                depth=depth,
                analytics=self.advanced_analytics
            )
            
            await self.knowledge_tracker.track_exploration(
                start_point=start_point,
                results=exploration_results
            )
            
            return exploration_results

        except Exception as e:
            self.state.error_count += 1
            return {
                "error": str(e),
                "status": "failed",
                "start_point": start_point
            }

    async def track_knowledge(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Track and update knowledge in the system."""
        try:
            tracking_result = await self.knowledge_tracker.track_update(content)
            
            # Update analytics
            self.advanced_analytics.record_knowledge_update(
                content_type=content.get('type', 'unknown'),
                update_size=len(str(content)),
                success=tracking_result.get('success', False)
            )
            
            return tracking_result

        except Exception as e:
            self.state.error_count += 1
            return {
                "error": str(e),
                "status": "failed",
                "content_type": content.get('type', 'unknown')
            }

    async def reason_with_uncertainty(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform uncertainty-aware reasoning."""
        try:
            reasoning_result = await self.reasoning_engine.reason_with_uncertainty(
                query=query,
                context=context
            )
            
            # Track reasoning process
            await self.knowledge_tracker.track_reasoning(
                query=query,
                result=reasoning_result
            )
            
            return reasoning_result

        except Exception as e:
            self.state.error_count += 1
            return {
                "error": str(e),
                "status": "failed",
                "query": query
            }

    async def execute_task(self, task: LangroidTask) -> Dict[str, Any]:
        """Enhanced task execution with RAG integration."""
        try:
            # Pre-execution checks
            if not self.quality_assurance_layer.check_task_safety(task):
                return {"error": "Task deemed unsafe", "status": "rejected"}

            # Get relevant knowledge from RAG system
            rag_results = await self.query_rag(task.content)
            
            # Enhance task with RAG knowledge
            enhanced_task = await self.enhance_task_with_rag(task, rag_results)
            
            # Process through layers
            prepared_task = await self.foundational_layer.process_task(enhanced_task)
            
            # Core execution with retries
            for attempt in range(self.config.max_retries):
                try:
                    result = await asyncio.wait_for(
                        self._process_task(prepared_task),
                        timeout=self.config.timeout
                    )
                    break
                except asyncio.TimeoutError:
                    if attempt == self.config.max_retries - 1:
                        return {"error": "Task execution timeout", "status": "failed"}
                    continue
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        return {"error": str(e), "status": "failed"}
                    continue

            # Post-processing with RAG enhancement
            processed_result = await self.agent_architecture_layer.process_result(result)
            decision = await self.decision_making_layer.make_decision(enhanced_task, processed_result)
            
            # Update knowledge tracking
            await self.track_knowledge({
                "type": "task_execution",
                "task": task.content,
                "result": decision,
                "timestamp": datetime.now().isoformat()
            })
            
            # Learning and updates
            await self.continuous_learning_layer.update(enhanced_task, decision)
            self._update_memory(enhanced_task, decision)
            self._update_metrics(enhanced_task, decision)

            return {
                "status": "success",
                "result": decision,
                "metrics": self.state.performance_metrics,
                "rag_enhancement": rag_results
            }

        except Exception as e:
            self.state.error_count += 1
            return {
                "error": str(e),
                "status": "failed",
                "error_count": self.state.error_count
            }

    async def enhance_task_with_rag(self, task: LangroidTask, rag_results: Dict[str, Any]) -> LangroidTask:
        """Enhance a task with RAG system results."""
        enhanced_content = f"""
        Original Task: {task.content}
        
        Retrieved Knowledge:
        {rag_results.get('knowledge', 'No knowledge retrieved')}
        
        Reasoning:
        {rag_results.get('reasoning', 'No reasoning available')}
        
        Confidence: {rag_results.get('confidence', 0)}
        """
        
        task.content = enhanced_content
        task.metadata = {
            **(task.metadata or {}),
            "rag_enhanced": True,
            "rag_confidence": rag_results.get('confidence', 0),
            "enhancement_timestamp": datetime.now().isoformat()
        }
        
        return task

    async def _process_task(self, task: LangroidTask) -> Dict[str, Any]:
        """Core task processing logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_task method")

    def _update_memory(self, task: LangroidTask, result: Any):
        """Update agent's memory with task and result."""
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "task": task.content,
            "result": result
        })
        if len(self.memory) > self.config.memory_size:
            self.memory.pop(0)

    def _update_metrics(self, task: LangroidTask, result: Any):
        """Update performance metrics."""
        self.state.performance_metrics.update({
            "tasks_completed": self.state.performance_metrics.get("tasks_completed", 0) + 1,
            "error_rate": self.state.error_count / (self.state.performance_metrics.get("tasks_completed", 1)),
            "last_task_timestamp": datetime.now().isoformat()
        })

    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced message processing with standardized communication."""
        task = LangroidTask(self, message['content'])
        result = await self.execute_task(task)
        
        # Standardized response format
        return {
            "agent": self.name,
            "task_id": message.get('id'),
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "status": result.get("status", "unknown")
        }

    async def handle_message(self, message: Message):
        """Standardized message handling."""
        if message.type == MessageType.TASK:
            result = await self.process_message(message.content)
            response = Message(
                type=MessageType.RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=result,
                parent_id=message.id,
                priority=message.priority
            )
            await self.communication_protocol.send_message(response)

    # Tool management methods
    def add_tool(self, name: str, tool: Callable, description: str = ""):
        """Add a tool with metadata."""
        self.tools[name] = {
            "function": tool,
            "description": description,
            "added": datetime.now().isoformat()
        }

    def remove_tool(self, name: str):
        """Remove a tool."""
        if name in self.tools:
            del self.tools[name]

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool with metadata."""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with metadata."""
        return [{
            "name": name,
            "description": tool["description"],
            "added": tool["added"]
        } for name, tool in self.tools.items()]

    # Capability management
    def add_capability(self, capability: str, metadata: Dict[str, Any] = None):
        """Add a capability with metadata."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            if metadata:
                self.state.performance_metrics[f"capability_{capability}"] = metadata

    def remove_capability(self, capability: str):
        """Remove a capability."""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            if f"capability_{capability}" in self.state.performance_metrics:
                del self.state.performance_metrics[f"capability_{capability}"]

    @property
    def info(self) -> Dict[str, Any]:
        """Enhanced agent information."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "model": self.model,
            "tools": self.list_tools(),
            "metrics": self.state.performance_metrics,
            "memory_size": len(self.memory),
            "error_count": self.state.error_count
        }

    async def evolve(self):
        """Enhanced evolution process."""
        print(f"Evolving agent: {self.name}")
        
        # Evolve all layers
        await self.quality_assurance_layer.evolve()
        await self.foundational_layer.evolve()
        await self.continuous_learning_layer.evolve()
        await self.agent_architecture_layer.evolve()
        await self.decision_making_layer.evolve()
        
        # Update configuration based on performance
        if self.state.performance_metrics.get("error_rate", 1.0) > 0.2:
            self.config.max_retries += 1
        
        if self.state.performance_metrics.get("timeout_rate", 0.0) > 0.1:
            self.config.timeout *= 1.5
        
        print(f"Agent {self.name} evolution complete.")
        print(f"Updated configuration: max_retries={self.config.max_retries}, timeout={self.config.timeout}")

# New layer implementations

class QualityAssuranceLayer:
    def __init__(self, upo_threshold: float = 0.7):
        self.upo_threshold = upo_threshold

    def check_task_safety(self, task: LangroidTask) -> bool:
        uncertainty = self.estimate_uncertainty(task)
        return uncertainty < self.upo_threshold

    def estimate_uncertainty(self, task: LangroidTask) -> float:
        # Implement UPO (Uncertainty-enhanced Preference Optimization)
        return random.random()  # Placeholder implementation

    async def evolve(self):
        self.upo_threshold = max(0.5, min(0.9, self.upo_threshold * (1 + (random.random() - 0.5) * 0.1)))

class FoundationalLayer:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    async def process_task(self, task: LangroidTask) -> LangroidTask:
        baked_knowledge = await self.bake_knowledge(task.content)
        task.content = f"{task.content}\nBaked Knowledge: {baked_knowledge}"
        return task

    async def bake_knowledge(self, content: str) -> str:
        # Implement Prompt Baking mechanism
        return f"Baked: {content}"

    async def evolve(self):
        # Implement evolution logic for Foundational Layer
        pass

class ContinuousLearningLayer:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    async def update(self, task: LangroidTask, result: Any):
        # Implement SELF-PARAM (rapid parameter updating)
        learned_info = self.extract_learning(task, result)
        await self.vector_store.add_texts([learned_info])

    def extract_learning(self, task: LangroidTask, result: Any) -> str:
        return f"Learned: Task '{task.content}' resulted in '{result}'"

    async def evolve(self):
        # Implement evolution logic for Continuous Learning Layer
        pass

class AgentArchitectureLayer:
    def __init__(self):
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()

    async def process_result(self, result: Any) -> Any:
        # Implement SAGE framework (Self-Aware Generative Engine)
        evaluation = await self.evaluate_result(result)
        if evaluation["quality"] < 0.9:
            result = await self.revise_result(result, evaluation)
        return result

    async def evaluate_result(self, result: Any) -> Dict[str, Any]:
        evaluation_prompt = f"Evaluate the following result: '{result}'. Provide a quality score between 0 and 1."
        evaluation = await self.llm.complete(evaluation_prompt)
        return {"quality": float(evaluation.text)}

    async def revise_result(self, result: Any, evaluation: Dict[str, Any]) -> Any:
        revision_prompt = f"Revise the following result to improve its quality: '{result}'"
        revision = await self.llm.complete(revision_prompt)
        return revision.text

    async def evolve(self):
        # Implement evolution logic for Agent Architecture Layer
        pass

class DecisionMakingLayer:
    def __init__(self):
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()

    async def make_decision(self, task: LangroidTask, context: Any) -> Any:
        # Implement Agent Q (Monte Carlo Tree Search and Direct Preference Optimization)
        mcts_result = self.monte_carlo_tree_search(task, context)
        dpo_result = await self.direct_preference_optimization(task, context)
        
        decision_prompt = f"""
        Task: {task.content}
        Context: {context}
        MCTS Result: {mcts_result}
        DPO Result: {dpo_result}
        Based on the MCTS and DPO results, make a final decision for the task.
        """
        decision = await self.llm.complete(decision_prompt)
        return decision.text

    def monte_carlo_tree_search(self, task: LangroidTask, context: str) -> str:
        options = ["Option A", "Option B", "Option C"]
        scores = [self.simulate(task, context, option) for option in options]
        best_option = options[np.argmax(scores)]
        return f"MCTS suggests: {best_option}"

    def simulate(self, task: LangroidTask, context: str, option: str) -> float:
        return random.random()

    async def direct_preference_optimization(self, task: LangroidTask, context: Any) -> str:
        # Implement DPO logic
        return "DPO placeholder result"

    async def direct_preference_optimization(self, task: LangroidTask, context: str) -> str:
        options = ["Approach X", "Approach Y", "Approach Z"]
        preferences = await self.get_preferences(task, context, options)
        best_approach = max(preferences, key=preferences.get)
        return f"DPO suggests: {best_approach}"

    async def get_preferences(self, task: LangroidTask, context: str, options: List[str]) -> Dict[str, float]:
        prompt = f"""
        Task: {task.content}
        Context: {context}
        Options: {', '.join(options)}
        Assign a preference score (0-1) to each option based on its suitability for the task and context.
        """
        response = await self.llm.complete(prompt)
        lines = response.text.split('\n')
        preferences = {}
        for line in lines:
            if ':' in line:
                option, score = line.split(':')
                preferences[option.strip()] = float(score.strip())
        return preferences

    async def process_query(self, query: str, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        # Implement query processing logic here
        retrieval_results = await self.rag_pipeline.retrieve(query, timestamp=timestamp)
        reasoning_result = await self.rag_pipeline.reason(query, retrieval_results)
        return reasoning_result

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Implement task execution logic here
        task_type = task.get('type', 'default')
        task_content = task.get('content', '')
        
        if task_type == 'query':
            return await self.process_query(task_content)
        elif task_type == 'analysis':
            # Implement analysis logic
            pass
        elif task_type == 'generation':
            # Implement generation logic
            pass
        else:
            raise ValueError(f"Unknown task type: {task_type}")

class MCTSConfig:
    def __init__(self):
        self.exploration_weight = 1.0
        self.simulation_depth = 10

class SelfEvolvingSystem:
    def __init__(self, agents: List[UnifiedBaseAgent]):
        self.agents = agents

    async def process_task(self, task: LangroidTask) -> Dict[str, Any]:
        for agent in self.agents:
            if task.type in agent.capabilities:
                return await agent.execute_task(task)
        return {"error": "No suitable agent found for the task"}

    async def evolve(self):
        print("Starting system-wide evolution...")
        for agent in self.agents:
            await agent.evolve()
        print("System-wide evolution complete.")

    async def evolve_agent(self, agent: UnifiedBaseAgent):
        print(f"Evolving agent: {agent.name}")
        performance = await self.analyze_agent_performance(agent)
        new_capabilities = await self.generate_new_capabilities(agent, performance)
        for capability in new_capabilities:
            agent.add_capability(capability)
        print(f"Agent {agent.name} evolution complete. New capabilities: {new_capabilities}")

    async def analyze_agent_performance(self, agent: UnifiedBaseAgent) -> Dict[str, float]:
        print(f"Analyzing performance of agent: {agent.name}")
        performance = {capability: random.uniform(0.4, 1.0) for capability in agent.capabilities}
        print(f"Performance analysis for {agent.name}: {performance}")
        return performance

    async def generate_new_capabilities(self, agent: UnifiedBaseAgent, performance: Dict[str, float]) -> List[str]:
        print(f"Generating new capabilities for agent: {agent.name}")
        low_performing = [cap for cap, score in performance.items() if score < 0.6]
        prompt = f"Agent {agent.name} is underperforming in {', '.join(low_performing)}. Suggest 2-3 new capabilities to improve performance."
        response = await self.sage_framework.assistant_response(prompt)
        new_capabilities = [cap.strip() for cap in response.split(',')]
        print(f"Suggested new capabilities for {agent.name}: {new_capabilities}")
        return new_capabilities

    async def evolve_decision_maker(self):
        print("Evolving decision maker...")
        self.mcts.exploration_weight *= 1.05
        self.mcts.simulation_depth += 1

        recent_decisions = self.get_recent_decisions()
        if recent_decisions:
            X = np.array([d[0] for d in recent_decisions])
            y = np.array([d[1] for d in recent_decisions])
            self.dpo.fit(X, y)

        print("Decision maker evolution complete.")

    async def optimize_upo_threshold(self) -> float:
        print("Optimizing UPO threshold...")
        safety_checks = await self.quality_assurance.get_recent_safety_checks()
        
        if safety_checks:
            safety_scores = [check.safety_score for check in safety_checks]
            mean_score = np.mean(safety_scores)
            std_score = np.std(safety_scores)
            
            new_threshold = mean_score - (1.5 * std_score)
            new_threshold = max(0.5, min(0.9, new_threshold))
        else:
            new_threshold = self.quality_assurance.upo_threshold * (1 + (random.random() - 0.5) * 0.1)

        print(f"New UPO threshold: {new_threshold:.4f}")
        return new_threshold

    def get_recent_decisions(self) -> List[tuple]:
        return [(np.random.rand(5), random.choice([0, 1])) for _ in range(100)]

    async def add_decision(self, features: np.array, outcome: int):
        self.recent_decisions.append((features, outcome))
        if len(self.recent_decisions) > 1000:
            self.recent_decisions.pop(0)

def create_agent(agent_type: str, config: UnifiedAgentConfig, communication_protocol: StandardCommunicationProtocol) -> UnifiedBaseAgent:
    """Factory function to create different types of agents."""
    return UnifiedBaseAgent(config, communication_protocol)

# Example usage (keeping existing example)
if __name__ == "__main__":
    vector_store = VectorStore()  # Placeholder, implement actual VectorStore
    communication_protocol = StandardCommunicationProtocol()
    
    agent_config = UnifiedAgentConfig(
        name="ExampleAgent",
        description="An example agent",
        capabilities=["general_task"],
        vector_store=vector_store,
        model="gpt-4",
        instructions="You are an example agent capable of handling general tasks."
    )
    
    agent = create_agent("ExampleAgent", agent_config, communication_protocol)
    
    self_evolving_system = SelfEvolvingSystem([agent], vector_store)
    
    # Use the self_evolving_system to process tasks and evolve the system


def create_agent(agent_type: str, config: UnifiedAgentConfig, communication_protocol: StandardCommunicationProtocol) -> UnifiedBaseAgent:
    """
    Factory function to create different types of agents.
    """
    return UnifiedBaseAgent(config, communication_protocol)

# Example usage
if __name__ == "__main__":
    vector_store = VectorStore()  # Placeholder, implement actual VectorStore
    communication_protocol = StandardCommunicationProtocol()
    
    agent_config = UnifiedAgentConfig(
        name="ExampleAgent",
        description="An example agent",
        capabilities=["general_task"],
        vector_store=vector_store,
        model="gpt-4",
        instructions="You are an example agent capable of handling general tasks."
    )
    
    agent = create_agent("ExampleAgent", agent_config, communication_protocol)
    
    self_evolving_system = SelfEvolvingSystem([agent], vector_store)
    
    # Use the self_evolving_system to process tasks and evolve the system

