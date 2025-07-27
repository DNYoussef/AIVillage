"""Very simple JSONL event logger."""
import json
from pathlib import Path


def log_event(path: str, event: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
