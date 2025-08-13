#!/usr/bin/env python
"""Generate repository statistics and update docs/status.md."""

import json
import subprocess
from datetime import date
from pathlib import Path


def collect_stats() -> tuple[int, int]:
    """Return (files, code_lines) using cloc."""
    result = subprocess.run(
        ["cloc", ".", "--json", "--exclude-dir=.git"],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    files = data["header"]["n_files"]
    code = data["SUM"]["code"]
    return files, code


def main() -> None:
    files, code = collect_stats()
    content = (
        "# Repository Status\n\n"
        f"_Last updated: {date.today()}_\n\n"
        "| Metric | Value |\n"
        "| ------ | ----- |\n"
        f"| Files | {files} |\n"
        f"| Lines of code | {code} |\n"
    )
    Path("docs/status.md").write_text(content)
    print(content)


if __name__ == "__main__":
    main()
