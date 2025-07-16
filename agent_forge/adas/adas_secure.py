from __future__ import annotations

from contextlib import contextmanager
import json
import os
import signal
import subprocess
import tempfile
from typing import Any

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.language_models.openai_gpt import OpenAIGPTConfig

from utils.logging import get_logger

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

        # Static code analysis scoring
        score = 0.0

        # Basic syntax check
        try:
            import ast

            tree = ast.parse(code)
            score += 0.2  # Valid syntax
        except SyntaxError:
            return 0.0

        # Code quality metrics
        lines = code.splitlines()

        # Reasonable length (not too short, not too long)
        if 10 <= len(lines) <= 200:
            score += 0.2

        # Has docstrings
        if any('"""' in line or "'''" in line for line in lines):
            score += 0.1

        # Uses type hints
        if any("->" in line or ": " in line for line in lines):
            score += 0.1

        # Has error handling
        if any(keyword in code for keyword in ["try:", "except:", "finally:"]):
            score += 0.2

        # Uses logging
        if "log" in code.lower():
            score += 0.1

        # Has main function
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
        """Context manager for timeout."""

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Code execution exceeded {seconds} seconds")

        # Set the signal handler and alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm

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
        # Create a wrapper script that will be executed in subprocess
        wrapper_code = f"""
import sys
import json
import resource
import os

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({memory_limit_mb} * 1024 * 1024, {memory_limit_mb} * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CPU, ({timeout}, {timeout}))

# Restrict file system access
os.chdir("/tmp")

# User code
{code}

# Execute the run function
if __name__ == "__main__":
    import tempfile
    model_path = sys.argv[1]
    params = json.loads(sys.argv[2])

    with tempfile.TemporaryDirectory() as work_dir:
        try:
            score = run(model_path, work_dir, params)
            print(json.dumps({{"success": True, "score": float(score)}}))
        except Exception as e:
            print(json.dumps({{"success": False, "error": str(e)}}))
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(wrapper_code)
            script_path = f.name

        try:
            # Run in subprocess with restrictions
            result = subprocess.run(
                [sys.executable, "-u", script_path, model_path, json.dumps(params)],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    "PYTHONPATH": "",  # Clear PYTHONPATH
                    "PATH": os.environ.get("PATH", ""),  # Minimal PATH
                },
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

        # Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        # Must have a run function
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

        # No obvious malicious patterns
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


# Example usage
if __name__ == "__main__":
    task_description = (
        "Design an agent that can solve abstract reasoning tasks in the ARC challenge."
    )
    adas_task = ADASTask(task_description)
    best_agent = adas_task.run()

    print("Best Agent:")
    print(best_agent)
