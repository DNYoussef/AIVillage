"""Unified CLI for Agent Forge.

Combines all Agent Forge commands into a single interface:
- forge evo: Run evolutionary model merging (EvoMerge)
- forge bake-quietstar: Bake Quiet-STaR reasoning into models
- forge dashboard: Launch monitoring dashboard
"""

import logging
import subprocess
import sys
from pathlib import Path

import click
import torch

try:
    from .version import __version__
except ImportError:  # pragma: no cover - fallback
    __version__ = "1.0.0"

logger = logging.getLogger(__name__)

# Import command groups from submodules
try:
    from agent_forge.adas.runner import specialize as adas_commands
except ImportError as e:  # pragma: no cover - surface missing modules
    raise ImportError("ADAS specialization commands not available") from e

try:
    from production.compression.compression_pipeline import forge as compression_cli
except ImportError as e:  # pragma: no cover - surface missing modules
    raise ImportError("Compression pipeline module not available") from e

try:
    from agent_forge.curriculum.cli import curriculum_cli
except ImportError as e:  # pragma: no cover - surface missing modules
    raise ImportError("Curriculum engine module not available") from e

try:
    from production.evolution.evomerge_pipeline import forge as evomerge_cli
except ImportError as e:  # pragma: no cover - surface missing modules
    raise ImportError("Evomerge pipeline module not available") from e

try:
    from agent_forge.quietstar_baker import forge as quietstar_cli
except ImportError as e:  # pragma: no cover - surface missing modules
    raise ImportError("QuietSTaR baker module not available") from e

try:
    from agent_forge.training.cli_commands import commands as training_commands
except ImportError as e:  # pragma: no cover - surface missing modules
    raise ImportError("Training commands module not available") from e

try:
    from agent_forge.unified_pipeline import forge as unified_cli
except ImportError as e:  # pragma: no cover - surface missing modules
    raise ImportError("Unified pipeline module not available") from e


@click.group()
def forge() -> None:
    """Agent Forge CLI - Advanced AI Agent Development Platform.

    Commands:
        evo             Run evolutionary model merging
        bake-quietstar  Bake reasoning into model weights
        compress        Apply BitNet compression
        run-pipeline    Run complete unified pipeline
        train           Run Forge training loop (NEW)
        analyze         Analyze training checkpoints
        test            Test trained models
        validate        Validate training configs
        specialize      Run ADAS×Transformer² specialization search (NEW)
        curriculum      Frontier curriculum engine (NEW)
        dashboard       Launch monitoring dashboard
    """


# Register pipeline commands
try:
    forge.add_command(evomerge_cli.commands["evo"])
    forge.add_command(quietstar_cli.commands["bake-quietstar"])
    forge.add_command(compression_cli.commands["compress"])
    forge.add_command(unified_cli.commands["run-pipeline"])

    # Register new training commands
    for cmd_name, cmd_func in training_commands.items():
        forge.add_command(cmd_func, name=cmd_name)

    # Register ADAS specialization commands
    forge.add_command(adas_commands, name="specialize")

    # Register curriculum engine commands
    forge.add_command(curriculum_cli, name="curriculum")

except (KeyError, AttributeError) as e:
    logger.warning("Could not register some commands: %s", e)


@forge.command()
@click.option("--port", default=8501, help="Dashboard port")
@click.option("--host", default="localhost", help="Dashboard host")
def dashboard(port: int, host: str) -> None:
    """Launch Agent Forge monitoring dashboard."""
    try:
        dashboard_script = Path(__file__).parent.parent / "scripts" / "run_dashboard.py"

        if not dashboard_script.exists():
            click.echo(f"Error: Dashboard script not found at {dashboard_script}")
            sys.exit(1)

        click.echo(f"🚀 Launching Agent Forge Dashboard at https://{host}:{port}")

        subprocess.run(
            [
                sys.executable,
                str(dashboard_script),
                "--port",
                str(port),
                "--host",
                host,
            ],
            check=False,
        )

    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped by user")
    except Exception as e:
        click.echo(f"❌ Dashboard error: {e}")
        sys.exit(1)


@forge.command()
def version() -> None:
    """Show Agent Forge version."""
    click.echo(f"Agent Forge v{__version__}")


@forge.command()
def status() -> None:
    """Check Agent Forge system status."""
    click.echo("🔍 Agent Forge System Status")
    click.echo("=" * 40)

    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    click.echo(f"Python: {python_version}")

    # Check PyTorch
    click.echo(f"PyTorch: {torch.__version__}")

    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        click.echo(f"CUDA: ✅ Available ({device_name}, {memory_gb:.1f} GB)")
    else:
        click.echo("CUDA: ❌ Not available")

    # Check directories
    dirs_to_check = [
        ("Models", Path("D:/agent_forge_models")),
        ("Benchmarks", Path("./benchmarks")),
        ("Output", Path("./evomerge_output")),
        ("Checkpoints", Path("./evomerge_checkpoints")),
    ]

    click.echo("\nDirectories:")
    for name, path in dirs_to_check:
        if path.exists():
            click.echo(f"  {name}: ✅ {path}")
        else:
            click.echo(f"  {name}: ❌ Not found")

    click.echo("\nAvailable Commands:")
    for cmd in forge.commands:
        click.echo(f"  forge {cmd}")

    click.echo("=" * 40)


def main() -> None:
    """Main CLI entry point."""
    forge()


if __name__ == "__main__":
    main()
