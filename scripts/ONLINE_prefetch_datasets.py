import argparse
import os
from pathlib import Path

import yaml
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
a = p.parse_args()

root = Path(os.environ.get("AIV_ROOT", r"D:\AIVillage"))
bundle = Path(a.bundle)
ds_cache = bundle / "hf_cache" / "datasets"
ds_cache.mkdir(parents=True, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = str(ds_cache)
os.environ["HF_DATASETS_OFFLINE"] = "0"

suites = root / "benchmarks" / "suites"
tasks = set()
for y in suites.glob("*.yaml"):
    cfg = yaml.safe_load(y.read_text(encoding="utf-8"))
    for g in cfg.get("task_groups", []):
        tasks.update(g.get("tasks", []))

map_ = {
    "hellaswag": ("hellaswag", None),
    "winogrande": ("winogrande", "winogrande_xl"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge"),
    "boolq": ("boolq", None),
    "gsm8k": ("gsm8k", "main"),
    "mbpp": ("mbpp", "sanitized"),
    "humaneval": ("openai_humaneval", None),
}

for t in sorted(tasks):
    if t in map_:
        name, conf = map_[t]
        print(f"[datasets] warm {t} -> load_dataset({name!r}, {conf!r})")
        _ = load_dataset(name, conf) if conf else load_dataset(name)
print("[OK] datasets warmed into bundle cache.")
