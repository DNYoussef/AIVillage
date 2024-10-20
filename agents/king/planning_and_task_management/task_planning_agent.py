import logging
from typing import Dict, Any, List
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.error_handling.error_handler import error_handler, safe_execute, AIVillageException
import networkx as nx
import matplotlib.pyplot as plt
import io

logger = logging.getLogger(__name__)

class TaskPlanningAgent:
    def __init__(self, llm_config: OpenAIGPTConfig):
        self.llm = llm_config.create()

    @error_handler.handle_error
    async def generate_task_plan(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a task plan based on the interpreted intent and key concepts.

        Args:
            interpreted_intent (Dict[str, Any]): The interpreted user intent.
            key_concepts (Dict[str, Any]): The extracted key concepts.

        Returns:
            Dict[str, Any]: A dictionary containing the generated task plan.
        """
        prompt = self._create_task_planning_prompt(interpreted_intent, key_concepts)
        response = await self.llm.complete(prompt)
        task_plan = self._parse_task_plan_response(response.text)
        task_graph = self._build_task_graph(task_plan)
        return {
            "plan": task_plan,
            "graph": task_graph
        }

    def _create_task_planning_prompt(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> str:
        return f"""
        Based on the following interpreted user intent and key concepts, generate a detailed task plan:

        Interpreted Intent: {interpreted_intent}
        Key Concepts: {key_concepts}

        Please provide a structured task plan that includes:
        1. Main Goal: The overall objective of the plan.
        2. Sub-tasks: Break down the main goal into smaller, actionable tasks.
        3. Priority: Assign a priority level (High, Medium, Low) to each task.
        4. Time Estimate: Provide an estimated time to complete each task.
        5. Required Resources: List any resources or skills needed for each task.
        6. Dependencies: Identify any tasks that depend on the completion of others.
        7. Milestones: Define key milestones or checkpoints in the plan.

        Provide your task plan in a structured JSON format, where each task is an object with properties for priority, time estimate, resources, and dependencies.
        """

    def _parse_task_plan_response(self, response: str) -> Dict[str, Any]:
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse task plan response: {response}")
            raise AIVillageException("Failed to parse task plan response")

    def _build_task_graph(self, task_plan: Dict[str, Any]) -> nx.DiGraph:
        G = nx.DiGraph()
        for task_id, task_data in task_plan['sub_tasks'].items():
            G.add_node(task_id, **task_data)
            for dependency in task_data.get('dependencies', []):
                G.add_edge(dependency, task_id)
        return G

    @error_handler.handle_error
    async def optimize_task_plan(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize the generated task plan.

        Args:
            task_plan (Dict[str, Any]): The initial task plan.

        Returns:
            Dict[str, Any]: An optimized version of the task plan.
        """
        prompt = self._create_optimization_prompt(task_plan)
        response = await self.llm.complete(prompt)
        optimized_plan = self._parse_task_plan_response(response.text)
        return optimized_plan

    def _create_optimization_prompt(self, task_plan: Dict[str, Any]) -> str:
        return f"""
        Analyze and optimize the following task plan:

        Task Plan: {task_plan}

        Please optimize the task plan considering the following factors:
        1. Efficiency: Identify any redundant or unnecessary tasks.
        2. Parallelization: Determine which tasks can be performed concurrently.
        3. Resource Allocation: Suggest better ways to allocate resources across tasks.
        4. Time Management: Propose ways to reduce the overall time required.
        5. Risk Mitigation: Identify potential risks and suggest mitigation strategies.

        Provide your optimized task plan in the same JSON format as the original plan, with additional comments explaining your optimizations.
        """

    @safe_execute
    async def process_input(self, interpreted_intent: Dict[str, Any], key_concepts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the interpreted intent and key concepts to generate and optimize a task plan.

        Args:
            interpreted_intent (Dict[str, Any]): The interpreted user intent.
            key_concepts (Dict[str, Any]): The extracted key concepts.

        Returns:
            Dict[str, Any]: A dictionary containing the original and optimized task plans.
        """
        initial_plan = await self.generate_task_plan(interpreted_intent, key_concepts)
        optimized_plan = await self.optimize_task_plan(initial_plan['plan'])
        
        return {
            "initial_plan": initial_plan['plan'],
            "optimized_plan": optimized_plan,
            "task_graph": initial_plan['graph']
        }

    def visualize_task_graph(self, graph: nx.DiGraph) -> bytes:
        """
        Visualize the task graph and return the image as bytes.

        Args:
            graph (nx.DiGraph): The NetworkX graph of tasks.

        Returns:
            bytes: The PNG image of the graph visualization as bytes.
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color='lightblue')
        
        # Draw edges
        nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=8, font_weight="bold")
        
        # Add priority and time estimate as node labels
        node_labels = {node: f"{data['priority']}\n{data['time_estimate']}" for node, data in graph.nodes(data=True)}
        pos_attrs = {node: (coord[0], coord[1] - 0.1) for node, coord in pos.items()}
        nx.draw_networkx_labels(graph, pos_attrs, labels=node_labels, font_size=6)
        
        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        llm_config = OpenAIGPTConfig(chat_model="gpt-4")
        planner = TaskPlanningAgent(llm_config)
        
        interpreted_intent = {
            "primary_intent": "Improve team performance",
            "secondary_intents": ["Enhance productivity", "Develop communication skills"],
            "key_entities": ["team", "productivity", "communication skills"],
            "sentiment": "Determined",
            "urgency": "Medium",
            "context": "Workplace improvement"
        }
        
        key_concepts = {
            "team_performance": {
                "attributes": {"current_level": "moderate", "desired_level": "high"},
                "relationships": [
                    {"concept": "productivity", "type": "contributes_to"},
                    {"concept": "communication_skills", "type": "contributes_to"}
                ]
            },
            "productivity": {
                "attributes": {"importance": "high", "current_level": "moderate"},
                "relationships": [
                    {"concept": "time_management", "type": "improves"},
                    {"concept": "task_prioritization", "type": "improves"}
                ]
            },
            "communication_skills": {
                "attributes": {"importance": "high", "current_level": "low"},
                "relationships": [
                    {"concept": "team_collaboration", "type": "improves"},
                    {"concept": "conflict_resolution", "type": "improves"}
                ]
            }
        }
        
        result = await planner.process_input(interpreted_intent, key_concepts)
        
        print("Initial Task Plan:")
        print(result["initial_plan"])
        print("\nOptimized Task Plan:")
        print(result["optimized_plan"])
        
        # Visualize the task graph
        graph_image = planner.visualize_task_graph(result["task_graph"])
        with open("task_graph.png", "wb") as f:
            f.write(graph_image)
        print("\nTask graph visualization saved as 'task_graph.png'")

    asyncio.run(main())
