from pathlib import Path
import sys
import tempfile
import unittest

# Add the project root to the Python path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

# Now we can import properly
try:
    from agent_forge.adas.adas import AgentTechnique
except ImportError:
    # Skip the test if dependencies are missing
    import unittest

    class TestAgentTechniqueHandle(unittest.TestCase):
        def test_handle_executes_code(self):
            self.skipTest("AgentTechnique not available - dependencies missing")

    if __name__ == "__main__":
        unittest.main()

    import sys

# REMOVED:     sys.exit(0)


class TestAgentTechniqueHandle(unittest.TestCase):
    """Test the secure AgentTechnique implementation."""

    def test_handle_executes_code(self):
        """Test that valid code executes correctly."""
        code = """
import os

def run(model_path, work_dir, params):
    '''Test function that creates a file and returns a score.'''
    # Create a test file in the work directory
    test_file = os.path.join(work_dir, 'test_output.txt')
    with open(test_file, 'w') as f:
        f.write('Test successful')

    # Return the score from params
    return params.get('score', 0.0)
"""
        technique = AgentTechnique(technique_name="demo", code=code)

        with tempfile.TemporaryDirectory() as tmp:
            result = technique.handle(tmp, {"score": 0.7})
            assert result == 0.7

    def test_invalid_code_returns_zero(self):
        """Test that invalid code returns 0.0."""
        code = "This is not valid Python code!"
        technique = AgentTechnique(technique_name="invalid", code=code)

        with tempfile.TemporaryDirectory() as tmp:
            result = technique.handle(tmp, {})
            assert result == 0.0

    def test_code_without_run_function_returns_zero(self):
        """Test that code without a run function returns 0.0."""
        code = """
def some_other_function():
    return 42
"""
        technique = AgentTechnique(technique_name="no_run", code=code)

        with tempfile.TemporaryDirectory() as tmp:
            result = technique.handle(tmp, {})
            assert result == 0.0

    def test_dangerous_code_rejected(self):
        """Test that potentially dangerous code is rejected."""
        dangerous_codes = [
            # Trying to use eval
            """
def run(model_path, work_dir, params):
    return eval('1 + 1')
""",
            # Trying to use exec
            """
def run(model_path, work_dir, params):
    exec('x = 1')
    return 0.5
""",
            # Trying to access system files
            """
def run(model_path, work_dir, params):
    with open('/etc/passwd', 'r') as f:
        data = f.read()
    return 0.5
""",
            # Trying to import subprocess
            """
import subprocess

def run(model_path, work_dir, params):
    subprocess.run(['ls', '/'])
    return 0.5
""",
        ]

        for i, code in enumerate(dangerous_codes):
            technique = AgentTechnique(technique_name=f"dangerous_{i}", code=code)
            with tempfile.TemporaryDirectory() as tmp:
                result = technique.handle(tmp, {})
                assert result == 0.0, f"Dangerous code {i} should return 0.0"

    def test_code_with_exception_returns_zero(self):
        """Test that code that raises exceptions returns 0.0."""
        code = """
def run(model_path, work_dir, params):
    # This will raise a ZeroDivisionError
    return 1 / 0
"""
        technique = AgentTechnique(technique_name="exception", code=code)

        with tempfile.TemporaryDirectory() as tmp:
            result = technique.handle(tmp, {})
            assert result == 0.0

    def test_score_clamping(self):
        """Test that scores are clamped to [0.0, 1.0]."""
        # Test score > 1.0
        code = """
def run(model_path, work_dir, params):
    return 2.5  # Above maximum
"""
        technique = AgentTechnique(technique_name="high_score", code=code)

        with tempfile.TemporaryDirectory() as tmp:
            result = technique.handle(tmp, {})
            assert result == 1.0

        # Test score < 0.0
        code = """
def run(model_path, work_dir, params):
    return -0.5  # Below minimum
"""
        technique = AgentTechnique(technique_name="low_score", code=code)

        with tempfile.TemporaryDirectory() as tmp:
            result = technique.handle(tmp, {})
            assert result == 0.0


if __name__ == "__main__":
    unittest.main()
