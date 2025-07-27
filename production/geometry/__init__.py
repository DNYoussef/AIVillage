"""Production geometry components organized by Sprint 2."""

# Export main geometry classes
try:
    from .geometry_feedback import GeometryFeedback
    
    __all__ = ['GeometryFeedback']
except ImportError:
    # Handle missing dependencies gracefully
    GeometryFeedback = None
    __all__ = []
