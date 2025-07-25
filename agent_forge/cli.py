#!/usr/bin/env python3
"""
Unified CLI for Agent Forge

Combines all Agent Forge commands into a single interface:
- forge evo: Run evolutionary model merging (EvoMerge)
- forge bake-quietstar: Bake Quiet-STaR reasoning into models
- forge dashboard: Launch monitoring dashboard
"""

import click
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Import command groups from submodules
try:
    from evomerge_pipeline import forge as evomerge_cli
    from quietstar_baker import forge as quietstar_cli
    from compression_pipeline import forge as compression_cli
    from unified_pipeline import forge as unified_cli
    imports_available = True
except ImportError as e:
    logger.warning(f"Some pipeline modules not available: {e}")
    imports_available = False

@click.group()
def forge():
    """
    Agent Forge CLI - Advanced AI Agent Development Platform

    Commands:
        evo             Run evolutionary model merging
        bake-quietstar  Bake reasoning into model weights
        compress        Apply BitNet compression
        run-pipeline    Run complete unified pipeline
        dashboard       Launch monitoring dashboard
    """
    pass

# Register pipeline commands if available
if imports_available:
    try:
        forge.add_command(evomerge_cli.commands['evo'])
        forge.add_command(quietstar_cli.commands['bake-quietstar'])
        forge.add_command(compression_cli.commands['compress'])
        forge.add_command(unified_cli.commands['run-pipeline'])
    except (KeyError, AttributeError) as e:
        logger.warning(f"Could not register some commands: {e}")

@forge.command()
@click.option('--port', default=8501, help='Dashboard port')
@click.option('--host', default='localhost', help='Dashboard host')
def dashboard(port, host):
    """Launch Agent Forge monitoring dashboard"""
    import subprocess

    try:
        dashboard_script = Path(__file__).parent.parent / "scripts" / "run_dashboard.py"

        if not dashboard_script.exists():
            click.echo(f"Error: Dashboard script not found at {dashboard_script}")
            sys.exit(1)

        click.echo(f"🚀 Launching Agent Forge Dashboard at http://{host}:{port}")

        subprocess.run([
            sys.executable,
            str(dashboard_script),
            "--port", str(port),
            "--host", host
        ])

    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped by user")
    except Exception as e:
        click.echo(f"❌ Dashboard error: {e}")
        sys.exit(1)

@forge.command()
def version():
    """Show Agent Forge version"""
    try:
        from version import __version__
        click.echo(f"Agent Forge v{__version__}")
    except:
        click.echo("Agent Forge v1.0.0")

@forge.command()
def status():
    """Check Agent Forge system status"""
    import torch
    from pathlib import Path

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
        ("Checkpoints", Path("./evomerge_checkpoints"))
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

def main():
    """Main CLI entry point"""
    forge()

if __name__ == "__main__":
    main()
