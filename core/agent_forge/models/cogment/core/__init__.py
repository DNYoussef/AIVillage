"""Cogment core module: RefinementCore + ACT halting mechanism."""

from .config import CogmentConfig
from .model import Cogment, CogmentOutput
from .refinement_core import RefinementCore, RefinementOutput, MemoryGate
from .act_halting import ACTHalting, ACTLoss, compute_halt_mask

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
