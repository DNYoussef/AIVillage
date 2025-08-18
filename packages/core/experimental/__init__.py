"""Experimental AI Village components.

These components are under active development and APIs may change without notice.
"""

import warnings


class ExperimentalWarning(UserWarning):
    """Warning for experimental features."""


def warn_experimental(feature_name) -> None:
    """Issue experimental warning."""
    warnings.warn(
        f"{feature_name} is experimental and may change without notice.",
        ExperimentalWarning,
        stacklevel=3,
    )
