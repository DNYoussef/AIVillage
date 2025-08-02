"""Compatibility wrapper for hyper-function compression.

Re-exports the :class:`HyperCompressionEncoder` from the production
implementation so that the ``agent_forge`` package exposes a hyper
compression component alongside other stages.
"""

from production.compression.compression.hyperfn import HyperCompressionEncoder

__all__ = ["HyperCompressionEncoder"]
