"""Cogment core module: RefinementCore + ACT halting mechanism."""

from .act_halting import ACTHalting, ACTLoss, compute_halt_mask
from .config import CogmentConfig
from .model import Cogment, CogmentOutput
from .refinement_core import MemoryGate, RefinementCore, RefinementOutput

__all__ = [
    "CogmentConfig",
    "Cogment",
    "CogmentOutput",
    "RefinementCore",
    "RefinementOutput",
    "MemoryGate",
    "ACTHalting",
    "ACTLoss",
    "compute_halt_mask",
]
