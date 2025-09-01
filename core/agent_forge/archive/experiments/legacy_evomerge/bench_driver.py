from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml


def suite_for(model_dir: Path) -> str:
    # read models/models.yaml to map type -> suite
    m = (Path("models") / "models.yaml").read_text(encoding="utf-8")
    entries = yaml.safe_load(m)
    for e in entries:
        rid = e["id"].replace("/", "__")
        if model_dir.name.startswith(rid):
            t = e.get("type", "general")
            return {"coding": "coding", "math": "math", "logic": "logic"}.get(t, "general")
    return "general"


def main():
    aiv_root = Path(os.environ.get("AIV_ROOT", r"D:\AIVillage"))
    models = Path(os.environ.get("AIV_MODELS_DIR", str(aiv_root / "models")))
    out = Path(os.environ.get("AIV_BENCHMARKS_DIR", str(aiv_root / "benchmarks" / "results")))
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"

    # W&B safe defaults
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_DIR", str(aiv_root / "wandb"))

    # Iterate each model dir and run lm-eval with the appropriate suite
    for md in sorted([p for p in models.iterdir() if p.is_dir()]):
        suite = suite_for(md)
        tasks = yaml.safe_load((Path("benchmarks") / "suites" / f"{suite}.yaml").read_text(encoding="utf-8"))[
            "task_groups"
        ]
        task_csv = ",".join(sum([g["tasks"] for g in tasks], []))
        outdir = out / "G0001" / "seeds" / md.name
        outdir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={str(md)},trust_remote_code=True,torch_dtype=float16",
            "--tasks",
            task_csv,
            "--batch_size",
            "auto",
            "--device",
            device,
            "--output_path",
            str(outdir),
            "--wandb_args",
            f"project=AIVillage-EvoMerge,group={md.name},name={md.name}-G0001-seeds,tags=generation:G0001;suite:{suite};phase:seeds",
        ]
        print("[RUN]", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[WARN] lm-eval failed on {md.name} (rc={rc})", file=sys.stderr)


if __name__ == "__main__":
    main()
