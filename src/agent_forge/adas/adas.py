from __future__ import annotations

import ast
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile

try:
    import resource

    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

    class resource:
        RLIMIT_AS = "RLIMIT_AS"
        RLIMIT_CPU = "RLIMIT_CPU"

        @staticmethod
        def setrlimit(resource_type, limits):
            """Setrlimit - Planned feature not yet implemented.

            This functionality is part of the Atlantis roadmap.
            """
            raise NotImplementedError(
                "'setrlimit' is not yet implemented. Track progress: https://github.com/DNYoussef/AIVillage/issues/feature-setrlimit"
            )


from contextlib import contextmanager
from typing import Any

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig

from rag_system.utils.logging import setup_logger as get_logger

from .technique_archive import PROMPT_TECHNIQUE_ARCHIVE


class ADASTask(Task):
    """ADAS Task for evolutionary agent development."""

    def __init__(self, task_description: str):
        config = ChatAgentConfig(
            name="ADAS",
            system_message="You are an expert machine learning researcher designing agentic systems.",
            llm=OpenAIGPTConfig(model="gpt-4"),
        )
        agent = ChatAgent(config)
        super().__init__(agent)
        self.task_description = task_description
        self.archive = PROMPT_TECHNIQUE_ARCHIVE
        self.best_agent = None
        self.best_performance = float("-inf")

    def generate_prompt(self, archive: list[dict]) -> str:
        archive_str = ",\n".join([str(agent) for agent in archive])
        return f"\n        Your objective is to create an optimal agent for the following task:\n\n        {
            self.task_description}\n\n        Here is the archive of discovered architectures:\n\n        [{archive_str}]\n\n        Design a new agent that improves upon these existing architectures. Be creative and think outside the box.\n        Your response should be a JSON object with the following fields:\n        - thought: Your reasoning behind the agent design\n        - name: A name for your proposed agent architecture\n        - code: The complete Python code for the forward() function of your agent\n\n        Ensure your code uses the Langroid ChatAgent class and follows the correct structure.\n        "

    def create_new_agent(self) -> dict[str, Any]:
        prompt = self.generate_prompt(self.archive)
        response = self.agent.llm_response(prompt)
        return response.content

    def evaluate_agent(self, agent: dict[str, Any]) -> float:
        """Static evaluation of a generated agent without execution.

        Uses code complexity metrics and static analysis to score agents.
        """
        if isinstance(agent, str):
            try:
                agent = json.loads(agent)
            except json.JSONDecodeError:
                return 0.0
        code = agent.get("code", "")
        score = 0.0
        try:
            import ast

            tree = ast.parse(code)
            score += 0.2
        except SyntaxError:
            return 0.0
        lines = code.splitlines()
        if 10 <= len(lines) <= 200:
            score += 0.2
        if any('"""' in line or "'''" in line for line in lines):
            score += 0.1
        if any("->" in line or ": " in line for line in lines):
            score += 0.1
        if any(keyword in code for keyword in ["try:", "except:", "finally:"]):
            score += 0.2
        if "log" in code.lower():
            score += 0.1
        if "def forward(" in code or "def run(" in code:
            score += 0.1
        return min(score, 1.0)

    async def run(self):
        num_iterations = 10
        for i in range(num_iterations):
            print(f"Iteration {i + 1}/{num_iterations}")
            new_agent = self.create_new_agent()
            performance = self.evaluate_agent(new_agent)
            if performance > self.best_performance:
                self.best_agent = new_agent
                self.best_performance = performance
                print(f"New best agent found! Performance: {performance}")
            self.archive.append(new_agent)
        print(f"Evolution complete. Best agent performance: {self.best_performance}")
        return self.best_agent


class SecureCodeRunner:
    """Secure code execution using subprocess isolation."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger("SecureCodeRunner")

    @contextmanager
    def _timeout(self, seconds: int):
        """Context manager for timeout (Unix/Linux only)."""
        if platform.system() == "Windows":
            yield
        else:

            def timeout_handler(signum, frame):
                raise TimeoutError(f"Code execution exceeded {seconds} seconds")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

    def run_code_sandbox(
        self,
        code: str,
        model_path: str,
        params: dict[str, Any],
        timeout: int = 30,
        memory_limit_mb: int = 512,
    ) -> float:
        """Run code in an isolated subprocess with resource limits.

        Args:
            code: Python code to execute
            model_path: Path to model files
            params: Parameters to pass to the code
            timeout: Maximum execution time in seconds
            memory_limit_mb: Maximum memory usage in MB

        Returns:
            Score between 0.0 and 1.0
        """
        wrapper_code = f"""\nimport sys\nimport json\nimport os\nimport platform\n\n# Set resource limits (Unix/Linux only)\ntry:\n    import resource\n    resource.setrlimit(resource.RLIMIT_AS, ({memory_limit_mb} * 1024 * 1024, {memory_limit_mb} * 1024 * 1024))\n    resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))\nexcept ImportError:\n    # Windows doesn't support resource limits - use timeout instead\n    pass\n\n# Restrict file system access (cross-platform)\nif platform.system() == "Windows":\n    import tempfile\n    os.chdir(tempfile.gettempdir())\nelse:\n    os.chdir("/tmp")\n\n# User code\n{code}\n\n# Execute the run function\nif __name__ == "__main__":\n    import tempfile\n    model_path = sys.argv[1]\n    params = json.loads(sys.argv[2])\n\n    with tempfile.TemporaryDirectory() as work_dir:\n        try:\n            score = run(model_path, work_dir, params)\n            print(json.dumps({{"success": True, "score": float(score)}}))\n        except Exception as e:\n            print(json.dumps({{"success": False, "error": str(e)}}))\n"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper_code)
            script_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, "-u", script_path, model_path, json.dumps(params)],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
            )
            if result.returncode != 0:
                self.logger.error(f"Code execution failed: {result.stderr}")
                return 0.0
            try:
                output = json.loads(result.stdout)
                if output.get("success"):
                    return max(0.0, min(1.0, output.get("score", 0.0)))
                self.logger.error(f"Code error: {output.get('error')}")
                return 0.0
            except json.JSONDecodeError:
                self.logger.error(f"Invalid output: {result.stdout}")
                return 0.0
        except subprocess.TimeoutExpired:
            self.logger.error("Code execution timeout")
            return 0.0
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return 0.0
        finally:
            os.unlink(script_path)


class AgentTechnique(ToolMessage):
    """Secure version of AgentTechnique using sandboxed execution."""

    request: str = "apply_technique"
    purpose: str = "Apply a specific AI technique"
    technique_name: str
    code: str

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.logger = get_logger("ADAS")
        self.runner = SecureCodeRunner(self.logger)

    def validate_code(self, code: str) -> bool:
        """Validate code for basic safety checks."""
        import ast

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        has_run_function = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run":
                has_run_function = True
                break
        if not has_run_function:
            self.logger.error(
                "Code must define a run(model_path, work_dir, params) function"
            )
            return False
        dangerous_patterns = [
            "__import__",
            "eval",
            "exec",
            "compile",
            "open(/etc",
            'open("/etc',
            "subprocess",
            "os.system",
            "socket",
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                self.logger.error(
                    f"Code contains potentially dangerous pattern: {pattern}"
                )
                return False
        return True

    def handle(self, model_path: str, params: dict[str, Any]) -> float:
        """Execute the technique in a secure sandbox.

        The code should define a function `run(model_path, work_dir, params)`
        that returns a fitness score between 0 and 1.
        """
        if not self.validate_code(self.code):
            self.logger.error("Technique %s failed validation", self.technique_name)
            return 0.0
        try:
            score = self.runner.run_code_sandbox(
                self.code, model_path, params, timeout=30, memory_limit_mb=512
            )
            self.logger.info(
                "ADAS | %s completed with score %.4f", self.technique_name, score
            )
            return score
        except Exception as e:
            self.logger.error(f"Failed to run technique {self.technique_name}: {e}")
            return 0.0


if __name__ == "__main__":
    task_description = (
        "Design an agent that can solve abstract reasoning tasks in the ARC challenge."
    )
    adas_task = ADASTask(task_description)
    best_agent = adas_task.run()
    print("Best Agent:")
    print(best_agent)
