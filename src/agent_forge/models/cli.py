"""Command line helpers for model management.

The CLI intentionally provides just a few small commands so that the behaviour
of the model helpers can be demonstrated without the full production
infrastructure.

Usage examples::

    python -m agent_forge.models.cli download-seeds
    python -m agent_forge.models.cli start-evomerge --generations 3
    python -m agent_forge.models.cli run-pipeline
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click

from .benchmark import benchmark_model
from .seed_downloader import SEED_MODELS, download_seed_models
from .storage import cleanup_storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Model management helpers for Agent Forge."""
    logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# download-seeds
# ---------------------------------------------------------------------------


@main.command("download-seeds")
@click.option(
    "--target-dir",
    type=click.Path(path_type=Path),
    default=Path("D:/agent_forge_models"),
    help="Directory where seed models will be stored",
)
@click.option(
    "--model",
    "models",
    type=str,
    multiple=True,
    help="Additional model identifiers to download",
)
def download_seeds_cmd(target_dir: Path, models: tuple[str, ...]) -> None:
    """Download the default set of seed models."""

    model_list = list(SEED_MODELS) + list(models)
    paths = download_seed_models(model_list, target_dir)
    cleanup_storage(target_dir)
    for model, path in paths.items():
        click.echo(f"Downloaded {model} -> {path}")


# ---------------------------------------------------------------------------
# start-evomerge
# ---------------------------------------------------------------------------


@main.command("start-evomerge")
@click.option("--generations", default=1, show_default=True, type=int)
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path),
    default=Path("D:/agent_forge_models"),
)
def start_evomerge_cmd(generations: int, models_dir: Path) -> None:
    """Kick off the EvoMerge pipeline if available."""

    try:  # pragma: no cover - heavy pipeline import
        from production.evolution.evomerge_pipeline import (
            EvolutionConfig,
            EvoMergePipeline,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        click.echo(f"EvoMerge pipeline unavailable: {exc}")
        return

    config = EvolutionConfig(base_models=list(SEED_MODELS), generations=generations)
    pipeline = EvoMergePipeline(config, models_dir=models_dir)
    asyncio.run(pipeline.run())


# ---------------------------------------------------------------------------
# run-pipeline
# ---------------------------------------------------------------------------


@main.command("run-pipeline")
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path),
    default=Path("D:/agent_forge_models"),
    help="Directory for downloaded models",
)
@click.option("--export", type=click.Path(path_type=Path), default=None)
def run_pipeline_cmd(models_dir: Path, export: Path | None) -> None:
    """Run a minimal seed → benchmark pipeline."""

    paths = download_seed_models(target_dir=models_dir)
    cleanup_storage(models_dir)
    results = {model: benchmark_model(path) for model, path in paths.items()}

    if export:
        with open(export, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to {export}")
    else:
        click.echo(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
