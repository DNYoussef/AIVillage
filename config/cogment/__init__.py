"""
Cogment Configuration Package.

Unified configuration system for all Cogment components replacing HRRM's fragmented approach.
Provides Option A configuration (~25M parameters) with 4-stage curriculum and GrokFast integration.
"""

from .config_loader import CogmentConfigLoader
from .config_validation import CogmentConfigValidator

__all__ = ["CogmentConfigLoader", "CogmentConfigValidator"]
