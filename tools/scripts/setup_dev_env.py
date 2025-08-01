#!/usr/bin/env python3
"""
Development Environment Setup Script
Consolidates environment setup, dependency management, and validation.
"""

import argparse
import logging
import os
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DevEnvironmentSetup:
    """Development environment setup and validation."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.requirements_files = [
            "requirements.txt",
            "requirements-dev.txt", 
            "pyproject.toml"
        ]
        
    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                cwd=self.project_root
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            return e.returncode, e.stdout or "", e.stderr or ""
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        logger.info("Checking Python version...")
        version = sys.version_info
        if version < (3, 10):
            logger.error(f"Python 3.10+ required, found {version.major}.{version.minor}")
            return False
        logger.info(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    
    def check_gpu_cuda(self) -> Dict[str, bool]:
        """Check GPU and CUDA availability."""
        logger.info("Checking GPU/CUDA setup...")
        status = {"cuda": False, "gpu": False}
        
        try:
            import torch
            status["cuda"] = torch.cuda.is_available()
            status["gpu"] = torch.cuda.device_count() > 0
            if status["cuda"]:
                logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
            else:
                logger.warning("CUDA not available - CPU mode only")
        except ImportError:
            logger.warning("PyTorch not installed - skipping GPU check")
        
        return status
    
    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        logger.info("Installing dependencies...")
        
        # Try poetry first
        if (self.project_root / "pyproject.toml").exists():
            returncode, _, stderr = self.run_command(["poetry", "install", "--with", "dev"], check=False)
            if returncode == 0:
                logger.info("Dependencies installed via poetry")
                return True
            else:
                logger.warning("Poetry install failed, trying pip...")
        
        # Fallback to pip
        for req_file in self.requirements_files:
            req_path = self.project_root / req_file
            if req_path.exists() and req_file.endswith('.txt'):
                returncode, _, _ = self.run_command([
                    sys.executable, "-m", "pip", "install", "-r", str(req_path)
                ], check=False)
                if returncode != 0:
                    logger.error(f"Failed to install {req_file}")
                    return False
                logger.info(f"Installed {req_file}")
        
        return True
    
    def setup_directories(self) -> bool:
        """Create necessary directory structure."""
        logger.info("Setting up directory structure...")
        
        directories = [
            "data",
            "logs", 
            "models",
            "outputs",
            "wandb",
            "forge_output",
            "test_output"
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
        
        return True
    
    def create_env_file(self) -> bool:
        """Create .env file if it doesn't exist."""
        env_file = self.project_root / ".env"
        if env_file.exists():
            logger.info(".env file already exists")
            return True
        
        logger.info("Creating .env file...")
        env_content = """# AIVillage Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
PYTHONPATH=src:${PYTHONPATH}

# Database URLs (development)
DATABASE_URL=sqlite:///./data/aivillage.db
REDIS_URL=redis://localhost:6379/0
NEO4J_URL=bolt://localhost:7687

# API Keys (set your own)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
WANDB_API_KEY=your_wandb_key_here

# Model Paths
MODELS_DIR=./models
DATA_DIR=./data
OUTPUTS_DIR=./outputs
"""
        
        env_file.write_text(env_content)
        logger.info("Created .env file - please update with your API keys")
        return True
    
    def validate_setup(self) -> bool:
        """Validate the complete setup."""
        logger.info("Validating setup...")
        
        # Check core imports
        try:
            import fastapi
            import torch
            import transformers
            import numpy as np
            logger.info("Core dependencies imported successfully")
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return False
        
        # Check project structure
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            logger.error("src/ directory not found")
            return False
        
        # Check if tests can be discovered
        returncode, stdout, _ = self.run_command([
            sys.executable, "-m", "pytest", "--collect-only", "-q"
        ], check=False)
        
        if returncode == 0:
            logger.info("Test discovery successful")
        else:
            logger.warning("Test discovery failed - some tests may not run")
        
        return True
    
    def run_setup(self, skip_deps: bool = False, skip_validation: bool = False) -> bool:
        """Run the complete setup process."""
        logger.info("Starting development environment setup...")
        
        if not self.check_python_version():
            return False
        
        if not skip_deps and not self.install_dependencies():
            return False
        
        if not self.setup_directories():
            return False
        
        if not self.create_env_file():
            return False
        
        self.check_gpu_cuda()
        
        if not skip_validation and not self.validate_setup():
            return False
        
        logger.info("✅ Development environment setup complete!")
        logger.info("Next steps:")
        logger.info("1. Update .env file with your API keys")
        logger.info("2. Run 'make test' to verify everything works")
        logger.info("3. Run 'make dev-up' to start development services")
        
        return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up AIVillage development environment")
    parser.add_argument(
        "--skip-deps", 
        action="store_true", 
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-validation", 
        action="store_true", 
        help="Skip setup validation"
    )
    parser.add_argument(
        "--project-root", 
        type=Path, 
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    setup = DevEnvironmentSetup(args.project_root)
    success = setup.run_setup(args.skip_deps, args.skip_validation)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())