"""Download the seed models required for the Agent Forge pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Default seed models used by the pipeline.  These are intentionally small so
# they can be downloaded in tests without exhausting resources.
SEED_MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2-1.5B",
    "microsoft/phi-1_5",
]


def download_seed_models(
    models: Iterable[str] | None = None,
    target_dir: Path | str = "D:/agent_forge_models",
) -> dict[str, Path]:
    """Download the seed models from Hugging Face.

    Parameters
    ----------
    models:
        Iterable of model identifiers.  If ``None`` the ``SEED_MODELS`` list is
        used.
    target_dir:
        Directory where the models should be stored.  The directory is created
        if it does not already exist.

    Returns
    -------
    dict
        Mapping of model identifier to local path.
    """

    models = list(models or SEED_MODELS)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    local_paths: dict[str, Path] = {}
    for model in models:
        logger.info("Downloading %s", model)
        # Each model gets its own sub directory under the target dir.  Using
        # ``local_dir_use_symlinks=False`` avoids symlink issues on Windows.
        model_dir = target_dir / model.replace("/", "_")
        path = snapshot_download(
            repo_id=model,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
        )
        local_paths[model] = Path(path)
    return local_paths
