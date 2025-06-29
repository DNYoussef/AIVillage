"""Utility modules for Agent Forge."""

from .grokfast import AugmentedAdam
from .svf import SVFLinear, replace_linear_with_svf
from .expert_vector import ExpertVector
from .seedlm import find_best_seed, regenerate_block
from .vptq import VPTQQuantizer
from .hypercomp import fit_hyperfunc, reconstruct
from .quiet_star import QuietSTAR
from .self_model import HiddenPredictor
from .adas import mutate_config, select_best
from .prompt_baking import PromptBank, bake_prompts

__all__ = [
    "AugmentedAdam",
    "SVFLinear",
    "replace_linear_with_svf",
    "ExpertVector",
    "find_best_seed",
    "regenerate_block",
    "VPTQQuantizer",
    "fit_hyperfunc",
    "reconstruct",
    "QuietSTAR",
    "HiddenPredictor",
    "mutate_config",
    "select_best",
    "PromptBank",
    "bake_prompts",
]
