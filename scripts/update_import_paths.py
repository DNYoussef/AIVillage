#!/usr/bin/env python3
"""Automatically fix legacy import paths after src/ restructure.

This script scans Python files, rewrites outdated imports to the new
``src`` package layout, and attempts to validate the changes.  Any file
that cannot be processed is reported as an ambiguous case for manual
follow-up.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

MODULE_PREFIXES = {
    "agent_forge": "src.agent_forge",
    "communications": "src.communications",
    "production": "src.production",
    "digital_twin": "src.digital_twin",
    "core": "src.core",
}

SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git"}


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def replace_segment(lines: list[str], node: ast.AST, new_segment: str) -> None:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return
    start, end = node.lineno - 1, node.end_lineno - 1
    prefix = lines[start][: node.col_offset]
    suffix = lines[end][node.end_col_offset :]
    lines[start : end + 1] = [prefix + new_segment + suffix]


def handle_import(node: ast.Import, lines: list[str], ambiguous: list[tuple[Path, str]]) -> bool:
    modified = False
    new_parts: list[str] = []
    for alias in node.names:
        name = alias.name
        as_part = f" as {alias.asname}" if alias.asname else ""
        if name.startswith("forge_"):
            new_parts.append(f"from src.production.agent_forge import {name}{as_part}")
            modified = True
            continue
        for old, new in MODULE_PREFIXES.items():
            if name == old or name.startswith(old + "."):
                new_name = name.replace(old, new, 1)
                new_parts.append(f"import {new_name}{as_part}")
                modified = True
                break
        else:
            new_parts.append(f"import {name}{as_part}")
    if modified:
        replace_segment(lines, node, "; ".join(new_parts))
    return modified


def handle_import_from(node: ast.ImportFrom, lines: list[str]) -> bool:
    if not node.module:
        return False
    module = node.module
    for old, new in MODULE_PREFIXES.items():
        if module == old or module.startswith(old + "."):
            new_module = module.replace(old, new, 1)
            names = ", ".join([ast.unparse(n) for n in node.names])
            replace_segment(lines, node, f"from {new_module} import {names}")
            return True
    return False


def process_file(path: Path, ambiguous: list[tuple[Path, str]]) -> bool:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        ambiguous.append((path, "syntax error"))
        return False
    lines = source.splitlines()
    modified = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if handle_import(node, lines, ambiguous):
                modified = True
        elif isinstance(node, ast.ImportFrom):
            if handle_import_from(node, lines):
                modified = True
    if modified:
        path.write_text("\n".join(lines) + ("\n" if source.endswith("\n") else ""))
    return modified


def validate_modules(repo_root: Path, modules: Iterable[str]) -> list[str]:
    """Attempt to import each module to ensure the new paths resolve."""
    project_root = repo_root
    if repo_root.name == "src":
        project_root = repo_root.parent
    sys.path.insert(0, str(project_root))
    failures = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            failures.append(f"{mod}: {exc}")
    return failures


def main(root: str = ".") -> None:
    repo_root = Path(root).resolve()
    ambiguous: list[tuple[Path, str]] = []
    modified_files = []
    for py_file in iter_python_files(repo_root):
        if process_file(py_file, ambiguous):
            modified_files.append(py_file)
    if modified_files:
        print("Modified files:")
        for path in modified_files:
            print(f"  {path}")
    else:
        print("No files required changes.")
    if ambiguous:
        print("\nAmbiguous cases:")
        for path, reason in ambiguous:
            print(f"  {path}: {reason}")
    failures = validate_modules(repo_root, MODULE_PREFIXES.values())
    if failures:
        print("\nImport validation failures:")
        for msg in failures:
            print(f"  {msg}")
    else:
        print("\nAll mapped modules imported successfully.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)
