import unittest
import tempfile
import os
from pathlib import Path
import ast

# Load only the AgentTechnique class from the source file to avoid heavy deps
repo_root = Path(__file__).resolve().parents[1]
source = (repo_root / "agent_forge" / "adas" / "adas.py").read_text()
module = ast.parse(source)
class_src = None
for node in module.body:
    if isinstance(node, ast.ClassDef) and node.name == "AgentTechnique":
        lines = source.splitlines()[node.lineno - 1: node.end_lineno]
        class_src = "\n".join(lines)
        break

import typing, logging, time, tempfile, pathlib, importlib, os, types

class DummyToolMessage:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)
local_ns = {}
exec(
    class_src,
    {
        "ToolMessage": DummyToolMessage,
        "Any": typing.Any,
        "Dict": typing.Dict,
        "logging": logging,
        "time": time,
        "tempfile": tempfile,
        "pathlib": pathlib,
        "importlib": importlib,
        "os": os,
        "types": types,
    },
    local_ns,
)
AgentTechnique = local_ns["AgentTechnique"]


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
            self.assertTrue(os.path.isfile(os.path.join(tmp, 'flag.txt')))
            self.assertEqual(result, 0.7)


if __name__ == "__main__":
    unittest.main()
