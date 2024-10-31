"""Download and setup script for AI Village system."""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json
import shutil
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIVillageSetup:
    """Handles the setup and download process for AI Village."""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.venv_path = self.root_dir / "venv"
        self.requirements_file = self.root_dir / "requirements.txt"
    
    def setup(self):
        """Run the complete setup process."""
        try:
            logger.info("Starting AI Village setup...")
            
            # Create directory structure
            self._create_directory_structure()
            
            # Create virtual environment
            self._create_virtual_environment()
            
            # Install dependencies
            self._install_dependencies()
            
            # Download required models
            self._download_models()
            
            # Initialize configuration
            self._initialize_config()
            
            logger.info("AI Village setup completed successfully")
            self._print_success_message()
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            raise
    
    def _create_directory_structure(self):
        """Create the required directory structure."""
        logger.info("Creating directory structure...")
        
        directories = [
            "agent_forge",
            "agent_forge/agents",
            "agent_forge/data",
            "agent_forge/model_compression",
            "agent_forge/utils",
            "agents",
            "agents/king",
            "agents/sage",
            "agents/magi",
            "agents/utils",
            "communications",
            "config",
            "data",
            "data/backups",
            "docs",
            "logs",
            "logs/agents",
            "logs/tasks",
            "rag_system",
            "rag_system/core",
            "rag_system/retrieval",
            "rag_system/processing",
            "tests",
            "ui",
            "utils"
        ]
        
        for directory in directories:
            (self.root_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def _create_virtual_environment(self):
        """Create a Python virtual environment."""
        logger.info("Creating virtual environment...")
        
        try:
            if not self.venv_path.exists():
                subprocess.run([sys.executable, "-m", "venv", str(self.venv_path)], check=True)
            
            logger.info("Virtual environment created successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {str(e)}")
            raise
    
    def _install_dependencies(self):
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        
        # Create requirements.txt if it doesn't exist
        if not self.requirements_file.exists():
            self._create_requirements_file()
        
        try:
            # Get path to pip in virtual environment
            if sys.platform == "win32":
                pip_path = self.venv_path / "Scripts" / "pip"
            else:
                pip_path = self.venv_path / "bin" / "pip"
            
            # Install requirements
            subprocess.run([
                str(pip_path),
                "install",
                "-r",
                str(self.requirements_file)
            ], check=True)
            
            logger.info("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {str(e)}")
            raise
    
    def _create_requirements_file(self):
        """Create requirements.txt with necessary dependencies."""
        requirements = [
            "aiohttp>=3.8.0",
            "aiohttp-cors>=0.7.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "pydantic>=1.8.0",
            "PyYAML>=6.0",
            "requests>=2.26.0",
            "torch>=1.9.0",
            "transformers>=4.11.0",
            "scikit-learn>=0.24.0",
            "pytest>=6.2.0",
            "pytest-asyncio>=0.16.0",
            "python-dotenv>=0.19.0",
            "tqdm>=4.62.0",
            "networkx>=2.6.0",
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=2.1.0"
        ]
        
        with open(self.requirements_file, "w") as f:
            f.write("\n".join(requirements))
    
    def _download_models(self):
        """Download required AI models."""
        logger.info("Downloading required models...")
        
        try:
            # Create models directory
            models_dir = self.root_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Download models using HuggingFace transformers
            from transformers import AutoTokenizer, AutoModel
            
            models = [
                "Qwen/Qwen2.5-3B-Instruct",  # Local model for KingAgent
                "deepseek-ai/Janus-1.3B",    # Local model for SageAgent
                "ibm-granite/granite-3b-code-instruct-128k"  # Local model for MagiAgent
            ]
            
            for model in models:
                logger.info(f"Downloading {model}...")
                AutoTokenizer.from_pretrained(model, cache_dir=str(models_dir))
                AutoModel.from_pretrained(model, cache_dir=str(models_dir))
            
            logger.info("Models downloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to download models: {str(e)}")
            raise
    
    def _initialize_config(self):
        """Initialize configuration files."""
        logger.info("Initializing configuration...")
        
        config_files = {
            "default.yaml": self._get_default_config(),
            "rag_config.yaml": self._get_rag_config(),
            "openrouter_agents.yaml": self._get_agent_config()
        }
        
        config_dir = self.root_dir / "config"
        
        for filename, content in config_files.items():
            file_path = config_dir / filename
            if not file_path.exists():
                with open(file_path, "w") as f:
                    json.dump(content, f, indent=2)
        
        logger.info("Configuration initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            "environment": "development",
            "system": {
                "log_level": "INFO",
                "max_memory_usage": 0.9,
                "max_cpu_usage": 0.8
            },
            "api": {
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": 30,
                "max_retries": 3
            }
        }
    
    def _get_rag_config(self) -> Dict[str, Any]:
        """Get RAG system configuration settings."""
        return {
            "model_name": "gpt-3.5-turbo",
            "embedding_model": "text-embedding-ada-002",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "similarity_threshold": 0.7
        }
    
    def _get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration settings."""
        return {
            "agents": {
                "king": {
                    "frontier_model": "nvidia/llama-3.1-nemotron-70b-instruct",
                    "local_model": "Qwen/Qwen2.5-3B-Instruct",
                    "description": "Strategic planning and decision making"
                },
                "sage": {
                    "frontier_model": "anthropic/claude-3.5-sonnet",
                    "local_model": "deepseek-ai/Janus-1.3B",
                    "description": "Research and knowledge synthesis"
                },
                "magi": {
                    "frontier_model": "openai/o1-mini-2024-09-12",
                    "local_model": "ibm-granite/granite-3b-code-instruct-128k",
                    "description": "Code generation and experimentation"
                }
            }
        }
    
    def _print_success_message(self):
        """Print success message with next steps."""
        message = """
AI Village Setup Complete!

Next steps:
1. Set your OpenRouter API key in environment:
   - Create a .env file in the root directory
   - Add: OPENROUTER_API_KEY=your_api_key_here

2. Initialize the AI Village:
   python initialize_village.py

3. Check the documentation in docs/ for usage instructions

For more information, see the README.md file.
"""
        print(message)

def main():
    """Main entry point for AI Village setup."""
    setup = AIVillageSetup()
    
    try:
        setup.setup()
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
