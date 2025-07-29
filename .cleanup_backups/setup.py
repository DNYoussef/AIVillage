#!/usr/bin/env python3
"""
Setup script for Agent Forge with CLI integration
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read README
readme_path = Path(__file__).parent / "README_AGENT_FORGE.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "agent_forge" / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
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
        "click>=8.0.0",
        "tqdm>=4.65.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pandas>=2.0.0",
        "psutil>=5.9.0"
    ]

setup(
    name="agent-forge",
    version="1.0.0",
    description="Advanced evolutionary AI agent development platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Agent Forge Team",
    author_email="contact@agent-forge.ai",
    url="https://github.com/your-org/agent-forge",
    packages=find_packages(include=["agent_forge", "agent_forge.*"]),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0"
        ],
        "cuda": [
            "torch[cuda]>=2.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "forge=agent_forge.cli:main",
            "agent-forge=scripts.run_agent_forge:main",
            "forge-dashboard=scripts.run_dashboard:main"
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords="ai, machine learning, evolutionary algorithms, model merging, transformers",
    project_urls={
        "Bug Reports": "https://github.com/your-org/agent-forge/issues",
        "Source": "https://github.com/your-org/agent-forge",
        "Documentation": "https://agent-forge.readthedocs.io/"
    }
)
