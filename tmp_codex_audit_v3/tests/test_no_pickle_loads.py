import ast
from pathlib import Path

ALLOWED_DIRS = {"tests", "tmp_codex_audit_v3"}

def test_no_pickle_loads():
    for path in Path("src").rglob("*.py"):
        if any(part in ALLOWED_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(
                path.read_text(encoding="utf-8", errors="ignore"), filename=str(path)
            )
        except SyntaxError:
            # Skip files that cannot be parsed
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(getattr(node.func, "attr", None), str)
            ):
                func = node.func
                if getattr(func, "attr", None) == "loads" and getattr(getattr(func, "value", None), "id", None) == "pickle":
                    raise AssertionError(f"pickle.loads found in {path}:{node.lineno}")
