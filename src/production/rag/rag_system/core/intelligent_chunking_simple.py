"""DEPRECATED: Simplified Intelligent Chunking using Sliding Window Similarity Analysis.

This module has been consolidated into intelligent_chunking.py which provides
the same functionality with additional features and optimizations.

The canonical implementation provides all features from this simple version plus:
- Enhanced document type detection
- Improved content type handling
- Better edge case handling
- Optimized performance

Please update your imports to use:
  from src.production.rag.rag_system.core.intelligent_chunking import IntelligentChunker

This file will be removed in a future version.
"""

import warnings

from .intelligent_chunking import ContentType as _ContentType
from .intelligent_chunking import DocumentType as _DocumentType
from .intelligent_chunking import IdeaBoundary as _IdeaBoundary
from .intelligent_chunking import IntelligentChunk as _IntelligentChunk
from .intelligent_chunking import IntelligentChunker as _IntelligentChunker
from .intelligent_chunking import SlidingWindow as _SlidingWindow

warnings.warn(
    "intelligent_chunking_simple is deprecated. "
    "Use intelligent_chunking instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Export the canonical implementations with deprecation warning
IntelligentChunker = _IntelligentChunker
DocumentType = _DocumentType
ContentType = _ContentType
SlidingWindow = _SlidingWindow
IdeaBoundary = _IdeaBoundary
IntelligentChunk = _IntelligentChunk


# Backward compatibility function aliases
def test_simple_intelligent_chunking(*args, **kwargs):
    """DEPRECATED: Use test_intelligent_chunking from intelligent_chunking module."""
    warnings.warn(
        "test_simple_intelligent_chunking is deprecated. "
        "Use test_intelligent_chunking from intelligent_chunking module.",
        DeprecationWarning,
        stacklevel=2,
    )
    from .intelligent_chunking import test_intelligent_chunking

    return test_intelligent_chunking(*args, **kwargs)
