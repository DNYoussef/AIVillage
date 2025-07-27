#!/usr/bin/env python3
"""Agent Forge Environment Setup Script

Comprehensive setup script that prepares the environment for Agent Forge:
1. Downloads models optimized for RTX 2060 SUPER
2. Downloads benchmark datasets
3. Installs dependencies
4. Sets up directory structure
5. Validates GPU/CUDA setup
6. Creates configuration files
"""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_gpu_setup():
    """Check GPU and CUDA availability"""
    logger.info("Checking GPU setup...")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()

        if cuda_available:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory
            memory_gb = memory_total / (1024**3)

            logger.info(f"✅ CUDA Available: {cuda_available}")
            logger.info(f"✅ Device Count: {device_count}")
            logger.info(f"✅ Current Device: {device_name}")
            logger.info(f"✅ GPU Memory: {memory_gb:.1f} GB")

            # Validate RTX 2060 SUPER compatibility
            if "2060" in device_name and memory_gb >= 7.5:
                logger.info("✅ RTX 2060 SUPER detected - optimal for 1.5B models")
                return True, {"device": device_name, "memory_gb": memory_gb}
            if memory_gb >= 6.0:
                logger.info(f"✅ GPU has sufficient memory ({memory_gb:.1f} GB)")
                return True, {"device": device_name, "memory_gb": memory_gb}
            logger.warning(f"⚠️ GPU memory may be insufficient ({memory_gb:.1f} GB)")
            return False, {"device": device_name, "memory_gb": memory_gb}
        logger.warning("⚠️ CUDA not available - will use CPU")
        return False, {"device": "CPU", "memory_gb": 0}

    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False, {"device": "Unknown", "memory_gb": 0}


def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")

    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "huggingface_hub>=0.17.0",
        "wandb>=0.15.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pytest>=7.4.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
    ]

    try:
        for req in requirements:
            logger.info(f"Installing {req}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", req],
                check=True,
                capture_output=True,
            )

        logger.info("✅ All dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Dependency installation failed: {e}")
        return False


def setup_directory_structure():
    """Create necessary directory structure"""
    logger.info("Setting up directory structure...")

    directories = [
        "D:/agent_forge_models",
        "benchmarks",
        "forge_output_enhanced",
        "forge_checkpoints_enhanced",
        "tests/integration",
        "logs",
        "wandb",
    ]

    created_dirs = []

    for dir_path in directories:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            created_dirs.append(dir_path)
            logger.info(f"✅ Created directory: {dir_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create {dir_path}: {e}")

    return created_dirs


def create_config_files():
    """Create configuration files"""
    logger.info("Creating configuration files...")

    # Create environment config
    env_config = {
        "models_directory": "D:/agent_forge_models",
        "benchmarks_directory": "./benchmarks",
        "output_directory": "./forge_output_enhanced",
        "checkpoints_directory": "./forge_checkpoints_enhanced",
        "wandb_project": "agent-forge-rtx2060",
        "device_optimization": "rtx_2060_super",
        "max_model_size_gb": 4.0,
        "recommended_models": [
            "Qwen/Qwen2.5-Math-1.5B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ],
    }

    config_path = Path("agent_forge_config.json")
    with open(config_path, "w") as f:
        json.dump(env_config, f, indent=2)

    logger.info(f"✅ Created configuration: {config_path}")

    # Create pre-commit configuration
    precommit_config = """repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: local
    hooks:
      - id: agent-forge-tests
        name: Agent Forge Tests
        entry: python -m pytest agent_forge/evomerge/tests/ -v
        language: system
        always_run: true
        pass_filenames: false
"""

    precommit_path = Path(".pre-commit-config.yaml")
    with open(precommit_path, "w") as f:
        f.write(precommit_config)

    logger.info(f"✅ Created pre-commit config: {precommit_path}")

    return [config_path, precommit_path]


def download_models_and_benchmarks():
    """Download models and benchmarks"""
    logger.info("Downloading models and benchmarks...")

    success = True

    # Download models
    try:
        logger.info("Downloading models...")
        subprocess.run(
            [sys.executable, "scripts/download_models.py", "--check-space"], check=True
        )
        logger.info("✅ Models downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Model download failed: {e}")
        success = False

    # Download benchmarks
    try:
        logger.info("Downloading benchmarks...")
        subprocess.run([sys.executable, "scripts/download_benchmarks.py"], check=True)
        logger.info("✅ Benchmarks downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Benchmark download failed: {e}")
        success = False

    return success


def validate_setup():
    """Validate the complete setup"""
    logger.info("Validating setup...")

    validation_results = {
        "gpu_setup": False,
        "dependencies": False,
        "directories": False,
        "models": False,
        "benchmarks": False,
        "configuration": False,
    }

    # Check GPU setup
    gpu_ok, gpu_info = check_gpu_setup()
    validation_results["gpu_setup"] = gpu_ok

    # Check dependencies
    try:
        import datasets
        import torch
        import transformers

        import wandb

        validation_results["dependencies"] = True
        logger.info("✅ Key dependencies available")
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")

    # Check directories
    required_dirs = ["D:/agent_forge_models", "benchmarks", "forge_output_enhanced"]
    dirs_exist = all(Path(d).exists() for d in required_dirs)
    validation_results["directories"] = dirs_exist

    if dirs_exist:
        logger.info("✅ Directory structure validated")
    else:
        logger.error("❌ Some directories missing")

    # Check models
    model_manifest = Path("D:/agent_forge_models/model_manifest.json")
    if model_manifest.exists():
        with open(model_manifest) as f:
            manifest = json.load(f)
        model_count = len([m for m in manifest["models"].values() if m["downloaded"]])
        validation_results["models"] = model_count >= 2
        logger.info(f"✅ {model_count} models available")
    else:
        logger.warning("⚠️ Model manifest not found")

    # Check benchmarks
    benchmark_manifest = Path("benchmarks/benchmark_manifest.json")
    if benchmark_manifest.exists():
        validation_results["benchmarks"] = True
        logger.info("✅ Benchmarks available")
    else:
        logger.warning("⚠️ Benchmark manifest not found")

    # Check configuration
    config_file = Path("agent_forge_config.json")
    validation_results["configuration"] = config_file.exists()

    if config_file.exists():
        logger.info("✅ Configuration files created")
    else:
        logger.error("❌ Configuration files missing")

    return validation_results


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Set up Agent Forge environment")
    parser.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Skip model and benchmark downloads",
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only run validation"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AGENT FORGE ENVIRONMENT SETUP")
    logger.info("=" * 60)

    if args.validate_only:
        validation_results = validate_setup()

        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)

        for check, result in validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{check.replace('_', ' ').title()}: {status}")

        overall_success = all(validation_results.values())

        if overall_success:
            logger.info("\n🎉 Agent Forge is ready to run!")
            logger.info("Next steps:")
            logger.info("  1. cd agent_forge")
            logger.info("  2. python enhanced_orchestrator.py")
        else:
            logger.error("\n❌ Setup incomplete. Please run full setup.")
            return 1

        return 0

    setup_success = True

    # Step 1: Check GPU
    gpu_ok, gpu_info = check_gpu_setup()
    if not gpu_ok:
        logger.warning("GPU setup issues detected, but continuing...")

    # Step 2: Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            setup_success = False
            logger.error("Dependency installation failed")

    # Step 3: Setup directories
    created_dirs = setup_directory_structure()
    if len(created_dirs) < 5:
        logger.warning("Some directories could not be created")

    # Step 4: Create configuration
    config_files = create_config_files()
    if len(config_files) < 2:
        logger.warning("Some configuration files could not be created")

    # Step 5: Download models and benchmarks
    if not args.skip_downloads:
        if not download_models_and_benchmarks():
            setup_success = False
            logger.error("Downloads failed")

    # Step 6: Final validation
    validation_results = validate_setup()

    logger.info("\n" + "=" * 60)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 60)

    for check, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{check.replace('_', ' ').title()}: {status}")

    overall_success = all(validation_results.values()) and setup_success

    if overall_success:
        logger.info("\n🎉 Agent Forge setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. cd agent_forge")
        logger.info("  2. python enhanced_orchestrator.py")
        logger.info("\nFor development:")
        logger.info("  1. Install pre-commit: pip install pre-commit")
        logger.info("  2. Install hooks: pre-commit install")
        logger.info("  3. Run tests: pytest agent_forge/evomerge/tests/")
    else:
        logger.error("\n❌ Setup completed with errors")
        logger.error("Please review the logs and resolve issues")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
