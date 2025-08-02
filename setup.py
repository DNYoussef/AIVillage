#!/usr/bin/env python3
"""
Legacy setup.py for backward compatibility.
All configuration is now in pyproject.toml.
"""

import warnings

from setuptools import setup

warnings.warn(
    "Using setup.py is deprecated. "
    "All project configuration is now in pyproject.toml. "
    "Use 'pip install -e .' instead of 'python setup.py install'.",
    DeprecationWarning,
    stacklevel=2,
)

# This setup.py exists only for backward compatibility
# All configuration is now in pyproject.toml
setup()
