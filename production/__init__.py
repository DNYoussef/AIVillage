"""Deprecated shim for legacy ``production`` package.

Use :mod:`src.production` instead.
"""
import warnings as _w

_w.warn(
    "Deprecated module: 'production' -> 'src.production'",
    DeprecationWarning,
    stacklevel=2,
)

from src.production import *  # noqa: F401,F403
