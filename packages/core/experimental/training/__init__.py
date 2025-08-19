"""
Experimental training modules for AIVillage.

Contains cutting-edge training techniques including:
- Self-modeling networks
- Sleep and dream consolidation
- GrokFast acceleration
- Model sharding and distributed inference
"""

try:
    from packages.core.experimental import warn_experimental

    warn_experimental(__name__)
except ImportError:
    # Experimental warning system not available
    pass
