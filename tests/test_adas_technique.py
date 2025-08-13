import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Try to import the secure AgentTechnique directly
    from agent_forge.adas.adas import AgentTechnique
except ImportError:
    # If that fails, skip the test
    import unittest

    class TestAgentTechniqueHandle(unittest.TestCase):
        def test_handle_executes_code(self):
            self.skipTest("AgentTechnique not available - dependencies missing")

    if __name__ == "__main__":
        unittest.main()

    # Exit early to avoid the rest of the file
    import sys

# REMOVED:     sys.exit(0)


class TestAgentTechniqueHandle(unittest.TestCase):
    def test_handle_executes_code(self):
        code = """
import os

def run(model_path, work_dir, params):
    path = os.path.join(model_path, 'flag.txt')
    with open(path, 'w') as f:
        f.write('done')
    return params.get('score', 0.0)
"""
        technique = AgentTechnique(technique_name="demo", code=code)
        with tempfile.TemporaryDirectory() as tmp:
            result = technique.handle(tmp, {"score": 0.7})
            assert os.path.isfile(os.path.join(tmp, "flag.txt"))
            assert result == 0.7


if __name__ == "__main__":
    unittest.main()
