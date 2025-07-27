#!/usr/bin/env python3
"""Transform misleading stubs into honest NotImplementedError declarations.
Integrates with pre-commit for ongoing enforcement.
"""

import ast
from pathlib import Path


class StubFixer(ast.NodeTransformer):
    def __init__(self, issue_base_url="https://github.com/DNYoussef/AIVillage/issues/"):
        self.issue_base_url = issue_base_url
        self.changes_made = []

    def visit_FunctionDef(self, node):
        """Transform misleading stubs into proper NotImplementedError."""
        if self._is_misleading_stub(node):
            # Preserve docstring if it exists
            docstring = None
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            ):
                docstring = node.body[0]
            else:
                # Create informative docstring
                docstring = ast.Expr(
                    value=ast.Constant(
                        value=f"{node.name} - Planned feature not yet implemented.\n\n"
                        f"This functionality is part of the Atlantis roadmap."
                    )
                )

            # Create proper NotImplementedError
            raise_stmt = ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                    args=[
                        ast.Constant(
                            value=f"'{node.name}' is not yet implemented. "
                            f"Track progress: {self.issue_base_url}feature-{node.name}"
                        )
                    ],
                    keywords=[],
                )
            )

            node.body = [docstring, raise_stmt]
            self.changes_made.append(node.name)

        return self.generic_visit(node)

    def _is_misleading_stub(self, node) -> bool:
        """Detect misleading stub implementations."""
        if len(node.body) == 1:
            stmt = node.body[0]

            # Misleading logging statements
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if hasattr(stmt.value.func, "attr"):
                    if stmt.value.func.attr in ["info", "debug"]:
                        # Check for misleading content
                        for arg in stmt.value.args:
                            if isinstance(arg, ast.Constant):
                                if "evolving" in str(arg.value).lower():
                                    return True

            # Empty pass statements
            if isinstance(stmt, ast.Pass):
                return True

        return False


def process_file(filepath: Path) -> bool:
    """Process a single file to fix stubs."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
    except (UnicodeDecodeError, OSError):
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    fixer = StubFixer()
    new_tree = fixer.visit(tree)

    if fixer.changes_made:
        # Generate new source
        new_source = ast.unparse(new_tree)

        # Write back
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_source)

        print(f"Fixed {len(fixer.changes_made)} stubs in {filepath}")
        return True

    return False


# Main execution
if __name__ == "__main__":
    fixed_count = 0

    # Focus on key directories to avoid timeout
    target_dirs = ["agents", "agent_forge", "rag_system"]

    for target_dir in target_dirs:
        target_path = Path(target_dir)
        if target_path.exists():
            for py_file in target_path.rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    if process_file(py_file):
                        fixed_count += 1

    print(f"\nFixed stubs in {fixed_count} files")
    print("Run 'pre-commit run --all-files' to verify formatting")
