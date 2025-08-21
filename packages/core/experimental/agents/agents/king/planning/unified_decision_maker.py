import asyncio
import io
import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import torch
from core.error_handling import AIVillageException, StandardCommunicationProtocol
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rag_system.core.pipeline import EnhancedRAGPipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .quality_assurance_layer import QualityAssuranceLayer

logger = logging.getLogger(__name__)


class MCTSNode:
    def __init__(self, state, parent=None) -> None:
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


class UnifiedDecisionMaker:
    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        rag_system: EnhancedRAGPipeline,
        agent,
        quality_assurance_layer: QualityAssuranceLayer,
        exploration_weight=1.0,
        max_depth=10,
    ) -> None:
        self.communication_protocol = communication_protocol
        self.rag_system = rag_system
        self.agent = agent
        self.quality_assurance_layer = quality_assurance_layer
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.stats = defaultdict(lambda: {"visits": 0, "value": 0})
        self.available_agents = []
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    async def make_decision(self, content: str, eudaimonia_score: float) -> dict[str, Any]:
        try:
            rag_info = await self.rag_system.process_query(content)
            task_vector = self.quality_assurance_layer.eudaimonia_triangulator.get_embedding(content)
            rule_compliance = self.quality_assurance_layer.evaluate_rule_compliance(task_vector)

            plan = await self.generate_plan(content, {"content": content, "rag_info": rag_info})

            decision = plan["decision"]
            best_alternative = plan["best_alternative"]
            implementation_plan = await self._create_implementation_plan(plan)

            return {
                "decision": decision,
                "eudaimonia_score": eudaimonia_score,
                "rule_compliance": rule_compliance,
                "rag_info": rag_info,
                "best_alternative": best_alternative,
                "implementation_plan": implementation_plan,
                "full_plan": plan,
            }
        except Exception as e:
            logger.exception("Error making decision: %s", e)
            msg = f"Error making decision: {e!s}"
            raise AIVillageException(msg) from e

    async def generate_plan(self, goal: str, problem_analysis: dict[str, Any]) -> dict[str, Any]:
        try:
            current_resources = await self._get_current_resources()
            plan_tree = await self._create_plan_tree(goal, current_resources)
            await self._extract_tasks(plan_tree)
            await self._optimize_tasks(plan_tree)

            success_likelihood = 0
            iteration = 0
            max_iterations = 10
            success_threshold = 0.95

            while success_likelihood < success_threshold and iteration < max_iterations:
                await self._conduct_premortem(plan_tree)
                await self._assess_antifragility(plan_tree)
                await self._develop_xanatos_gambits(plan_tree)
                plan_tree = await self._update_plan(plan_tree)
                success_likelihood = self._calculate_success_likelihood(plan_tree)
                iteration += 1

            gaps = self._identify_capability_gaps(plan_tree, current_resources)
            checkpoints = await self._plan_checkpoints(self._extract_all_tasks(plan_tree))
            swot_analysis = await self._perform_swot_analysis(plan_tree, problem_analysis)
            metrics = self._calculate_plan_metrics(plan_tree)

            plan_data = {
                "goal": goal,
                "plan_tree": self._tree_to_dict(plan_tree),
                "capability_gaps": gaps,
                "checkpoints": checkpoints,
                "swot_analysis": swot_analysis,
                "success_likelihood": success_likelihood,
                "iterations": iteration,
                "metrics": metrics,
            }

            visualization = self._create_plan_visualization(plan_tree)

            return {**plan_data, "visualization": visualization}
        except Exception as e:
            logger.exception("Error in plan generation: %s", e)
            msg = f"Error in plan generation: {e!s}"
            raise AIVillageException(msg)

    async def mcts_search(self, task, problem_analyzer, plan_generator, iterations=1000):
        root = MCTSNode(task)

        for _ in range(iterations):
            node = self.select(root)
            if node.visits < 1 or len(node.children) == 0:
                child = await self.expand(node, problem_analyzer)
            else:
                child = self.best_uct_child(node)
            result = await self.simulate(child, plan_generator)
            self.backpropagate(child, result)

        return self.best_child(root).state

    def select(self, node):
        path = []
        while True:
            path.append(node)
            if not node.children or len(path) > self.max_depth:
                return node
            unexplored = [child for child in node.children if child.visits == 0]
            if unexplored:
                return random.choice(unexplored)
            node = self.best_uct_child(node)

    async def expand(self, node, problem_analyzer):
        if problem_analyzer:
            new_states = await problem_analyzer.generate_possible_states(node.state)
        else:
            new_states = [node.state]  # Placeholder for when problem_analyzer is not provided
        for state in new_states:
            if state not in [child.state for child in node.children]:
                new_node = MCTSNode(state, parent=node)
                node.children.append(new_node)
                return new_node
        return random.choice(node.children)

    async def simulate(self, node, plan_generator):
        return await plan_generator.evaluate(node.state)

    def backpropagate(self, node, result) -> None:
        while node:
            self.stats[node.state]["visits"] += 1
            self.stats[node.state]["value"] += result
            node.visits += 1
            node.value += result
            node = node.parent

    def best_uct_child(self, node):
        log_n_visits = math.log(self.stats[node.state]["visits"])
        return max(
            node.children,
            key=lambda c: (self.stats[c.state]["value"] / self.stats[c.state]["visits"])
            + self.exploration_weight * math.sqrt(log_n_visits / self.stats[c.state]["visits"]),
        )

    def best_child(self, node):
        return max(node.children, key=lambda c: self.stats[c.state]["visits"])

    async def update_model(self, task: dict[str, Any], result: Any) -> None:
        try:
            logger.info("Updating unified decision maker model with task result: %s", result)
            await self.mcts_update(task, result)
            await self.quality_assurance_layer.update_task_history(
                task, result.get("performance", 0.5), result.get("uncertainty", 0.5)
            )
        except Exception as e:
            logger.exception("Error updating unified decision maker model: %s", e)
            msg = f"Error updating unified decision maker model: {e!s}"
            raise AIVillageException(msg) from e

    async def mcts_update(self, task, result) -> None:
        self.stats[task]["visits"] += 1
        self.stats[task]["value"] += result

    async def mcts_prune(self, node, threshold) -> None:
        node.children = [child for child in node.children if self.stats[child.state]["visits"] > threshold]
        for child in node.children:
            await self.mcts_prune(child, threshold)

    async def parallel_mcts_search(self, task, problem_analyzer, plan_generator, iterations=1000, num_workers=4):
        root = MCTSNode(task)
        semaphore = asyncio.Semaphore(num_workers)

        async def worker() -> None:
            async with semaphore:
                node = self.select(root)
                if node.visits < 1 or len(node.children) == 0:
                    child = await self.expand(node, problem_analyzer)
                else:
                    child = self.best_uct_child(node)
                result = await self.simulate(child, plan_generator)
                self.backpropagate(child, result)

        await asyncio.gather(*[worker() for _ in range(iterations)])
        return self.best_child(root).state

    def update_agent_list(self, agent_list: list[str]) -> None:
        self.available_agents = agent_list
        logger.info("Updated available agents: %s", self.available_agents)

    async def save_models(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)

            with open(os.path.join(path, "mcts_stats.json"), "w") as f:
                json.dump(dict(self.stats), f)

            await self.quality_assurance_layer.save(os.path.join(path, "quality_assurance_layer.json"))

            data = {
                "exploration_weight": self.exploration_weight,
                "max_depth": self.max_depth,
                "available_agents": self.available_agents,
            }
            with open(os.path.join(path, "unified_decision_maker_data.json"), "w") as f:
                json.dump(data, f)

            logger.info("Models saved successfully to %s", path)
        except Exception as e:
            logger.exception("Error saving models: %s", e)
            msg = f"Error saving models: {e!s}"
            raise AIVillageException(msg) from e

    async def load_models(self, path: str) -> None:
        try:
            with open(os.path.join(path, "mcts_stats.json")) as f:
                self.stats = defaultdict(lambda: {"visits": 0, "value": 0}, json.load(f))

            await self.quality_assurance_layer.load(os.path.join(path, "quality_assurance_layer.json"))

            with open(os.path.join(path, "unified_decision_maker_data.json")) as f:
                data = json.load(f)
                self.exploration_weight = data["exploration_weight"]
                self.max_depth = data["max_depth"]
                self.available_agents = data["available_agents"]

            logger.info("Models loaded successfully from %s", path)
        except Exception as e:
            logger.exception("Error loading models: %s", e)
            msg = f"Error loading models: {e!s}"
            raise AIVillageException(msg) from e

    async def introspect(self) -> dict[str, Any]:
        return {
            "type": "UnifiedDecisionMaker",
            "description": "Makes decisions based on task content, RAG information, eudaimonia score, and rule compliance",
            "available_agents": self.available_agents,
            "quality_assurance_info": self.quality_assurance_layer.get_info(),
            "mcts_stats": dict(self.stats),
        }

    async def _get_current_resources(self) -> dict[str, Any]:
        prompt = "List the current resources, position, and tools available for the AI Village project. Output as a JSON dictionary with keys 'resources', 'position', and 'tools'."
        try:
            return await self.agent.generate_structured_response(prompt)
        except Exception as e:
            logger.exception("Error getting current resources: %s", e)
            msg = f"Error getting current resources: {e!s}"
            raise AIVillageException(msg)

    async def _create_plan_tree(self, goal: str, current_resources: dict[str, Any]) -> dict[str, Any]:
        try:
            prompt = f"""
            Create a detailed plan tree for the goal: "{goal}"
            Consider the following current resources: {json.dumps(current_resources)}

            The plan tree should include:
            1. Main goal
            2. Sub-goals (multiple levels if necessary)
            3. Tasks for each sub-goal
            4. Dependencies between tasks and sub-goals
            5. Estimated time and resources required for each task
            6. Potential risks and mitigation strategies

            Output the plan tree as a nested JSON structure.
            """
            plan_tree = await self.agent.generate_structured_response(prompt)
            return plan_tree
        except Exception as e:
            logger.exception("Error creating plan tree: %s", e)
            msg = f"Error creating plan tree: {e!s}"
            raise AIVillageException(msg)

    async def _extract_tasks(self, plan_tree: dict[str, Any]) -> list[dict[str, Any]]:
        tasks = []

        def extract_tasks_recursive(node) -> None:
            if "tasks" in node:
                tasks.extend(node["tasks"])
            for sub_goal in node.get("sub_goals", []):
                extract_tasks_recursive(sub_goal)

        extract_tasks_recursive(plan_tree)
        return tasks

    async def _optimize_tasks(self, plan_tree: dict[str, Any]) -> dict[str, Any]:
        try:
            tasks = await self._extract_tasks(plan_tree)
            prompt = f"""
            Optimize the following set of tasks:
            {json.dumps(tasks, indent=2)}

            Consider the following aspects for optimization:
            1. Task dependencies and potential parallelization
            2. Resource allocation and efficiency
            3. Risk mitigation and contingency planning
            4. Time estimation accuracy

            Provide an optimized version of the tasks, maintaining the same structure.
            """
            optimized_tasks = await self.agent.generate_structured_response(prompt)

            # Update the plan tree with optimized tasks
            def update_tasks_recursive(node) -> None:
                if "tasks" in node:
                    node["tasks"] = [task for task in optimized_tasks if task["id"] in [t["id"] for t in node["tasks"]]]
                for sub_goal in node.get("sub_goals", []):
                    update_tasks_recursive(sub_goal)

            update_tasks_recursive(plan_tree)
            return plan_tree
        except Exception as e:
            logger.exception("Error optimizing tasks: %s", e)
            msg = f"Error optimizing tasks: {e!s}"
            raise AIVillageException(msg)

    async def _conduct_premortem(self, plan_tree: dict[str, Any]) -> dict[str, Any]:
        try:
            prompt = f"""
            Conduct a premortem analysis for the following plan:
            {json.dumps(plan_tree, indent=2)}

            Imagine that the plan has failed spectacularly. Identify:
            1. Potential points of failure
            2. Unforeseen risks or challenges
            3. Assumptions that might prove false
            4. External factors that could negatively impact the plan

            For each identified issue, suggest preventive measures or plan adjustments.
            Output the results as a JSON structure with 'issues' and 'preventive_measures' keys.
            """
            premortem_results = await self.agent.generate_structured_response(prompt)

            # Update the plan tree with premortem insights
            plan_tree["premortem_analysis"] = premortem_results
            return plan_tree
        except Exception as e:
            logger.exception("Error conducting premortem: %s", e)
            msg = f"Error conducting premortem: {e!s}"
            raise AIVillageException(msg)

    async def _assess_antifragility(self, plan_tree: dict[str, Any]) -> dict[str, Any]:
        try:
            prompt = f"""
            Assess the antifragility of the following plan:
            {json.dumps(plan_tree, indent=2)}

            For each component of the plan, evaluate:
            1. How it responds to volatility, randomness, and chaos
            2. Its potential to gain from disorder
            3. Its resilience and adaptability

            Provide an antifragility score (-10 to 10) for each component, where:
            - Negative scores indicate fragility
            - Scores around 0 indicate robustness
            - Positive scores indicate antifragility

            Suggest improvements to increase the overall antifragility of the plan.
            Output the results as a JSON structure with 'antifragility_scores' and 'improvement_suggestions' keys.
            """
            antifragility_assessment = await self.agent.generate_structured_response(prompt)

            # Update the plan tree with antifragility assessment
            plan_tree["antifragility_assessment"] = antifragility_assessment
            return plan_tree
        except Exception as e:
            logger.exception("Error assessing antifragility: %s", e)
            msg = f"Error assessing antifragility: {e!s}"
            raise AIVillageException(msg)

    async def _develop_xanatos_gambits(self, plan_tree: dict[str, Any]) -> dict[str, Any]:
        try:
            prompt = f"""
            Develop Xanatos Gambits for the following plan:
            {json.dumps(plan_tree, indent=2)}

            A Xanatos Gambit is a plan that benefits the planner regardless of the outcome.
            For each major component or goal in the plan:
            1. Identify potential failure scenarios
            2. Devise strategies that turn these failures into alternative successes
            3. Explain how each gambit benefits the overall goal, even if the original plan fails

            Output the results as a JSON structure with 'gambits' key, where each gambit includes:
            - 'component': The part of the plan it applies to
            - 'failure_scenario': The potential failure
            - 'alternative_success': How the failure is turned into a success
            - 'benefit_explanation': How this gambit benefits the overall goal
            """
            xanatos_gambits = await self.agent.generate_structured_response(prompt)

            # Update the plan tree with Xanatos Gambits
            plan_tree["xanatos_gambits"] = xanatos_gambits
            return plan_tree
        except Exception as e:
            logger.exception("Error developing Xanatos Gambits: %s", e)
            msg = f"Error developing Xanatos Gambits: {e!s}"
            raise AIVillageException(msg)

    async def _update_plan(self, plan_tree: dict[str, Any]) -> dict[str, Any]:
        try:
            prompt = f"""
            Update the following plan based on the premortem analysis, antifragility assessment, and Xanatos Gambits:
            {json.dumps(plan_tree, indent=2)}

            Integrate the insights and strategies from these analyses to create an improved plan that is:
            1. More robust against potential failures
            2. More antifragile and able to benefit from volatility
            3. Structured to succeed in multiple scenarios

            Maintain the original structure of the plan tree, but update and add elements as necessary.
            Provide explanations for significant changes made to the plan.
            """
            updated_plan = await self.agent.generate_structured_response(prompt)
            return updated_plan
        except Exception as e:
            logger.exception("Error updating plan: %s", e)
            msg = f"Error updating plan: {e!s}"
            raise AIVillageException(msg)

    def _calculate_success_likelihood(self, plan_tree: dict[str, Any]) -> float:
        try:
            # This is a simplified calculation and should be expanded based on your specific requirements
            antifragility_scores = plan_tree.get("antifragility_assessment", {}).get("antifragility_scores", {})
            if not antifragility_scores:
                return 0.5  # Default to 50% if no scores are available

            avg_antifragility = sum(antifragility_scores.values()) / len(antifragility_scores)
            success_likelihood = (avg_antifragility + 10) / 20  # Convert -10 to 10 scale to 0 to 1 scale
            return max(0, min(1, success_likelihood))  # Ensure the result is between 0 and 1
        except Exception as e:
            logger.exception("Error calculating success likelihood: %s", e)
            return 0.5  # Return a default value in case of error

    def _identify_capability_gaps(self, plan_tree: dict[str, Any], current_resources: dict[str, Any]) -> list[str]:
        try:
            required_resources = set()
            for task in self._extract_all_tasks(plan_tree):
                required_resources.update(task.get("required_resources", []))

            available_resources = set(current_resources.get("resources", []) + current_resources.get("tools", []))

            return list(required_resources - available_resources)
        except Exception as e:
            logger.exception("Error identifying capability gaps: %s", e)
            return []

    async def _plan_checkpoints(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            prompt = f"""
            Plan checkpoints for the following set of tasks:
            {json.dumps(tasks, indent=2)}

            Create a list of checkpoints that:
            1. Cover critical milestones in the plan
            2. Allow for progress evaluation and course correction
            3. Are spaced appropriately throughout the timeline

            For each checkpoint, provide:
            - Description: What should be achieved by this point
            - Evaluation criteria: How to measure progress
            - Potential adjustments: Actions to take if progress is off-track

            Output the checkpoints as a JSON list of dictionaries.
            """
            checkpoints = await self.agent.generate_structured_response(prompt)
            return checkpoints
        except Exception as e:
            logger.exception("Error planning checkpoints: %s", e)
            msg = f"Error planning checkpoints: {e!s}"
            raise AIVillageException(msg)

    async def _perform_swot_analysis(
        self, plan_tree: dict[str, Any], problem_analysis: dict[str, Any]
    ) -> dict[str, list[str]]:
        try:
            prompt = f"""
            Perform a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis for the following plan and problem analysis:

            Plan: {json.dumps(plan_tree, indent=2)}
            Problem Analysis: {json.dumps(problem_analysis, indent=2)}

            Provide a comprehensive SWOT analysis that considers:
            1. Internal factors (Strengths and Weaknesses)
            2. External factors (Opportunities and Threats)
            3. How the plan addresses the problem
            4. Potential future developments

            Output the analysis as a JSON dictionary with 'strengths', 'weaknesses', 'opportunities', and 'threats' keys,
            each containing a list of relevant points.
            """
            swot_analysis = await self.agent.generate_structured_response(prompt)
            return swot_analysis
        except Exception as e:
            logger.exception("Error performing SWOT analysis: %s", e)
            msg = f"Error performing SWOT analysis: {e!s}"
            raise AIVillageException(msg)

    def _calculate_plan_metrics(self, plan_tree: dict[str, Any]) -> dict[str, Any]:
        try:
            tasks = self._extract_all_tasks(plan_tree)
            total_time = sum(task.get("estimated_time", 0) for task in tasks)
            total_resources = sum(len(task.get("required_resources", [])) for task in tasks)
            risk_levels = [task.get("risk_level", 0) for task in tasks if "risk_level" in task]
            avg_risk = sum(risk_levels) / len(risk_levels) if risk_levels else 0

            return {
                "total_tasks": len(tasks),
                "total_estimated_time": total_time,
                "total_required_resources": total_resources,
                "average_risk_level": avg_risk,
            }
        except Exception as e:
            logger.exception("Error calculating plan metrics: %s", e)
            return {}

    def _tree_to_dict(self, node: dict[str, Any]) -> dict[str, Any]:
        result = {
            "name": node.get("name", ""),
            "description": node.get("description", ""),
            "tasks": node.get("tasks", []),
            "antifragility_score": node.get("antifragility_score", 0),
            "xanatos_factor": node.get("xanatos_factor", 0),
        }
        if "sub_goals" in node:
            result["children"] = [self._tree_to_dict(child) for child in node["sub_goals"]]
        return result

    def _create_plan_visualization(self, plan_tree: dict[str, Any]) -> str:
        G = nx.DiGraph()

        def add_nodes(node, parent=None) -> None:
            node_id = node["name"]
            G.add_node(
                node_id,
                description=node.get("description", ""),
                antifragility=node.get("antifragility_score", 0),
                xanatos_factor=node.get("xanatos_factor", 0),
            )
            if parent:
                G.add_edge(parent, node_id)
            for child in node.get("sub_goals", []):
                add_nodes(child, node_id)

        add_nodes(plan_tree)

        pos = nx.spring_layout(G)
        plt.figure(figsize=(20, 20))

        node_colors = [
            (
                "red"
                if G.nodes[node]["antifragility"] < -3
                else "green"
                if G.nodes[node]["antifragility"] > 3
                else "yellow"
            )
            for node in G.nodes()
        ]

        [
            ("s" if G.nodes[node]["xanatos_factor"] < -3 else "^" if G.nodes[node]["xanatos_factor"] > 3 else "o")
            for node in G.nodes()
        ]

        nx.draw(
            G,
            pos,
            node_color=node_colors,
            node_shape="o",
            node_size=3000,
            with_labels=False,
        )

        nx.draw_networkx_labels(G, pos, {node: node for node in G.nodes()}, font_size=8, font_weight="bold")

        node_labels = nx.get_node_attributes(G, "description")
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1] + 0.08)
        nx.draw_networkx_labels(G, pos_attrs, labels=node_labels, font_size=6)

        plt.title(
            "Plan Tree Visualization\nColors: Red (Fragile), Yellow (Robust), Green (Antifragile)\nShapes: Square (Negative Xanatos), Circle (Neutral Xanatos), Triangle (Positive Xanatos)"
        )
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        plt.close()

        return buf.getvalue()

    def _extract_all_tasks(self, plan_tree: dict[str, Any]) -> list[dict[str, Any]]:
        tasks = []

        def extract_tasks_recursive(node) -> None:
            tasks.extend(node.get("tasks", []))
            for sub_goal in node.get("sub_goals", []):
                extract_tasks_recursive(sub_goal)

        extract_tasks_recursive(plan_tree)
        return tasks

    async def _create_implementation_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        try:
            prompt = f"""
            Create an implementation strategy for the following plan:
            {json.dumps(plan, indent=2)}

            The implementation strategy should include:
            1. Monitoring steps to track progress and alignment with eudaimonia
            2. Feedback analysis to continuously improve the plan
            3. Troubleshooting steps to address potential issues
            4. Adaptive measures to adjust the plan based on new information or changing circumstances
            5. Resource allocation and timeline
            6. Risk management strategies
            7. Communication and coordination plans

            Output the implementation strategy as a JSON dictionary with appropriate keys for each section.
            """
            implementation_plan = await self.agent.generate_structured_response(prompt)
            return implementation_plan
        except Exception as e:
            logger.exception("Error creating implementation plan: %s", e)
            msg = f"Error creating implementation plan: {e!s}"
            raise AIVillageException(msg)


if __name__ == "__main__":
    msg = "Run 'agents/orchestration.py' to start the decision making subsystem."
    raise SystemExit(msg)
