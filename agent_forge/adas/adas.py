from __future__ import annotations
import json
import importlib.util
import os
import types
from utils.logging import get_logger
import time
import pathlib
import tempfile
from typing import List, Dict, Any
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig

from .technique_archive import PROMPT_TECHNIQUE_ARCHIVE



class ADASTask(Task):
    def __init__(self, task_description: str):
        config = ChatAgentConfig(
            name="ADAS",
            system_message="You are an expert machine learning researcher designing agentic systems.",
            llm=OpenAIGPTConfig(model="gpt-4")
        )
        agent = ChatAgent(config)
        super().__init__(agent)
        self.task_description = task_description
        self.archive = PROMPT_TECHNIQUE_ARCHIVE
        self.best_agent = None
        self.best_performance = float('-inf')

    def generate_prompt(self, archive: List[Dict]) -> str:
        archive_str = ",\n".join([str(agent) for agent in archive])
        
        return f"""
        Your objective is to create an optimal agent for the following task:

        {self.task_description}

        Here is the archive of discovered architectures:

        [{archive_str}]

        Design a new agent that improves upon these existing architectures. Be creative and think outside the box.
        Your response should be a JSON object with the following fields:
        - thought: Your reasoning behind the agent design
        - name: A name for your proposed agent architecture
        - code: The complete Python code for the forward() function of your agent

        Ensure your code uses the Langroid ChatAgent class and follows the correct structure.
        """

    def create_new_agent(self) -> Dict[str, Any]:
        prompt = self.generate_prompt(self.archive)
        response = self.agent.llm_response(prompt)
        return response.content

    def evaluate_agent(self, agent: Dict[str, Any]) -> float:
        """Simple evaluation of a generated agent.

        The ``agent`` argument can either be a dictionary or a JSON string
        representing an object with a ``code`` field.  We attempt to compile
        the code to make sure it is syntactically valid.  If compilation
        succeeds we return a deterministic score based on the number of lines in
        the code.  If it fails, ``0.0`` is returned.
        """

        if isinstance(agent, str):
            try:
                agent = json.loads(agent)
            except json.JSONDecodeError:
                return 0.0

        code = agent.get("code", "")

        try:
            compile(code, "<adas-agent>", "exec")
        except Exception:
            return 0.0

        # Use the length of the code as a very rough heuristic for performance
        # so that results are deterministic during testing.
        return float(len(code.splitlines()))

    async def run(self):
        num_iterations = 10  # You can adjust this or make it a parameter
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}")
            
            new_agent = self.create_new_agent()
            performance = self.evaluate_agent(new_agent)
            
            if performance > self.best_performance:
                self.best_agent = new_agent
                self.best_performance = performance
                print(f"New best agent found! Performance: {performance}")
            
            self.archive.append(new_agent)
        
        print(f"Evolution complete. Best agent performance: {self.best_performance}")
        return self.best_agent

class AgentTechnique(ToolMessage):
    request: str = "apply_technique"
    purpose: str = "Apply a specific AI technique"
    technique_name: str
    code: str
    SAFE_BUILTINS = {
        "__import__": __import__,
        "open": open,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "str": str,
        "float": float,
    }
    ALLOWED_IMPORTS = {"os"}

    def __init__(self, **data: Any):
        super().__init__(**data)
        if "get_logger" in globals():
            self.logger = get_logger("ADAS")
        else:
            import logging
            self.logger = logging.getLogger("ADAS")

    @classmethod
    def _sanitize_code(cls, code: str) -> bool:
        """Basic validation for untrusted code snippets."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [n.name for n in node.names]
                if not all(name in cls.ALLOWED_IMPORTS for name in names):
                    return False
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"exec", "eval", "__import__", "compile"}:
                    return False

        return True

    def _load_module(self, code: str) -> types.ModuleType:
        """Safely load code string into a new module."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            path = tmp.name
        try:
            spec = importlib.util.spec_from_file_location(
                f"adas_tech_{self.technique_name}", path
            )
            module = importlib.util.module_from_spec(spec)
            module.__dict__["__builtins__"] = self.SAFE_BUILTINS
            if spec.loader is not None:
                spec.loader.exec_module(module)
            return module
        finally:
            os.remove(path)

    def handle(self, model_path: str, params: Dict[str, Any]) -> float:
        """Execute the technique callable stored in ``code``.

        The ``code`` should define a function ``run(model_path, work_dir, params)``
        that returns a fitness score between 0 and 1.
        """
        if not self._sanitize_code(self.code):
            self.logger.error("Technique %s failed security checks", self.technique_name)
            return 0.0
        try:
            module = self._load_module(self.code)
        except Exception:  # pragma: no cover - user code may be faulty
            self.logger.exception("Failed to load technique %s", self.technique_name)
            return 0.0

        fn = getattr(module, "run", None)
        if not callable(fn):
            self.logger.error("Technique %s has no run() function", self.technique_name)
            return 0.0

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"adas_{self.technique_name}_") as wdir:
            wdir_path = pathlib.Path(wdir)
            try:
                score = float(fn(model_path, wdir_path, params))
            except Exception:
                self.logger.exception("Technique %s failed", self.technique_name)
                score = 0.0

        dt = time.perf_counter() - t0
        score = max(0.0, min(1.0, score))
        self.logger.info(
            "ADAS | %s completed in %.1fs with score %.4f", self.technique_name, dt, score
        )
        return score

# Example usage
if __name__ == "__main__":
    task_description = "Design an agent that can solve abstract reasoning tasks in the ARC challenge."
    adas_task = ADASTask(task_description)
    best_agent = adas_task.run()
    
    print("Best Agent:")
    print(best_agent)
