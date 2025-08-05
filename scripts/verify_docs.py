"""Verify README + docs reflect current code reality.
Produces a drift % (lines changed) and exits non-zero if > threshold.
Intended for CI gate.
"""

from __future__ import annotations

import difflib
from pathlib import Path
import sys
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
TMP = tempfile.gettempdir()
THRESHOLD = 0.02  # 2 % drift

# Tag markers copied from README & docs pages
START, END = "<!--feature-matrix-start-->", "<!--feature-matrix-end-->"

# ---------------------------------------------------------------------------
# Helpers


def current_feature_matrix() -> str:
    """Load matrix block from README.md"""
    content = (ROOT / "README.md").read_text().splitlines()
    if START not in content or END not in content:
        sys.exit("Feature matrix tags missing from README.md")
    start, end = content.index(START), content.index(END)
    return "\n".join(content[start + 1 : end])


def generate_live_matrix() -> str:
    """Create a markdown table from codebase inspection."""

    def has_path(p: str) -> bool:
        return (ROOT / p).exists()

    status_map = {
        "Twin Runtime": "✅" if has_path("twin_runtime") else "🔴",
        "King / Sage / Magi": "✅" if has_path("agents") else "🔴",
        "Self‑Evolving System": ("✅" if has_path("agent_forge/self_evolving_system.py") else "🔴"),
        "HippoRAG": "✅" if has_path("rag_system/hipporag.py") else "🔴",
        "Mesh Credits": "✅" if has_path("communications/credits.py") else "🔴",
        "ADAS Optimisation": "✅" if has_path("agent_forge/adas") else "🔴",
        "ConfidenceEstimator": ("✅" if has_path("rag_system/processing/confidence_estimator.py") else "🔴"),
    }

    lines = ["| Sub-system | Status |", "|------------|--------|"]
    for k, v in status_map.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def diff_ratio(a: str, b: str) -> float:
    diff = list(difflib.unified_diff(a.splitlines(), b.splitlines()))
    changed = sum(1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
    total = max(len(a.splitlines()), 1)
    return changed / total


# ---------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    readme_block = current_feature_matrix()
    live_block = generate_live_matrix()

    ratio = diff_ratio(readme_block, live_block)
    if ratio > THRESHOLD:
        tmp = Path(TMP) / "live_matrix.md"
        tmp.write_text(textwrap.dedent(f"""{START}\n{live_block}\n{END}\n"""))
        print(f"Doc drift {ratio:.1%} exceeds threshold.\nUpdated matrix saved to {tmp}")
        sys.exit(1)
    print("DocSync drift OK (\u2714\ufe0f)")
