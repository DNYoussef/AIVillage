import logging
from typing import Any

from agents.utils.task import Task as LangroidTask

logger = logging.getLogger(__name__)


class TaskExecutor:
    def __init__(self, agent):
        self.agent = agent

    async def execute_task(self, task):
        try:
            # Process the task through the foundational layer
            task = await self.agent.foundational_layer.process_task(task)

            # Streamlined query processing pipeline
            processed_task = await self.agent.query_processor.process_query(
                task["content"]
            )
            task["content"] = processed_task

            # Check if the task is complex and needs to be broken down into subgoals
            if await self._is_complex_task(task):
                result = await self._execute_with_subgoals(task)
            # Proceed with regular task execution
            elif task["type"] in self.agent.research_capabilities:
                handler = getattr(self.agent, f"handle_{task['type']}", None)
                if handler:
                    result = await handler(task)
                else:
                    langroid_task = LangroidTask(
                        self.agent,
                        task.get("content"),
                        task.get("id", ""),
                        task.get("priority", 1),
                    )
                    langroid_task.type = task.get("type", "general")
                    result = await self.agent.execute_task(langroid_task)
            else:
                langroid_task = LangroidTask(
                    self.agent,
                    task.get("content"),
                    task.get("id", ""),
                    task.get("priority", 1),
                )
                langroid_task.type = task.get("type", "general")
                result = await self.agent.execute_task(langroid_task)

            # Apply self-consistency check
            result = await self.agent.apply_self_consistency(task, result)

            # Update cognitive nexus with the task and result
            await self.agent.update_cognitive_nexus(task, result)

            # Update continuous learning layer
            await self.agent.continuous_learning_layer.update(task, result)

            # Return the result
            return result
        except Exception as e:
            logger.error(f"Error executing task: {e!s}")
            return {"error": str(e)}

    async def _is_complex_task(self, task):
        try:
            # Flagged directly as complex
            if task.get("is_complex"):
                return True

            content = str(task.get("content", ""))
            task_type = task.get("type", "")

            # Consider content length
            if len(content) > 200:
                return True

            # Some task types are inherently complex
            complex_types = {
                "data_analysis",
                "information_synthesis",
                "exploration_mode",
            }
            if task_type in complex_types:
                return True

            return False
        except Exception as e:
            logger.error(f"Error determining task complexity: {e!s}")
            return False

    async def _execute_with_subgoals(self, task) -> dict[str, Any]:
        try:
            subgoals = await self.generate_subgoals(task["content"])
            results = []
            for subgoal in subgoals:
                subtask = {
                    "type": task["type"],
                    "content": subgoal,
                    "priority": task.get("priority", 1),
                    "id": task.get("id", ""),
                }
                langroid_subtask = LangroidTask(
                    self.agent,
                    subtask["content"],
                    subtask["id"],
                    subtask["priority"],
                )
                langroid_subtask.type = subtask["type"]
                subtask_result = await self.execute_task(langroid_subtask)
                results.append(subtask_result)
            final_result = await self.summarize_results(task, subgoals, results)
            return final_result
        except Exception as e:
            logger.error(f"Error executing task with subgoals: {e!s}")
            return {"error": str(e)}

    async def generate_subgoals(self, content: str) -> list[str]:
        try:
            cognitive_context = await self.agent.query_cognitive_nexus(content)
            subgoals_text = await self.agent.tree_of_thoughts.process(
                f"Break down the following task into subgoals, considering this context:\n"
                f"Context: {cognitive_context}\n"
                f"Task: {content}"
            )
            subgoals = subgoals_text.strip().split("\n")
            return subgoals
        except Exception as e:
            logger.error(f"Error generating subgoals: {e!s}")
            return []

    async def summarize_results(
        self, task, subgoals: list[str], results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        summary_prompt = f"""
        Original task: {task["content"]}
        Subgoals and their results:
        {self._format_subgoals_and_results(subgoals, results)}
        Provide a comprehensive summary addressing the original task.
        """
        summary = await self.agent.generate(summary_prompt)
        return {"summary": summary, "subgoal_results": results}

    def _format_subgoals_and_results(
        self, subgoals: list[str], results: list[dict[str, Any]]
    ) -> str:
        formatted = ""
        for subgoal, result in zip(subgoals, results, strict=False):
            formatted += f"Subgoal: {subgoal}\nResult: {result}\n\n"
        return formatted
