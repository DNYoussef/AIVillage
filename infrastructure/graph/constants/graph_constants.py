"""
Graph Processing Constants
Centralized constants for graph analysis and ML operations.
"""

from enum import Enum


# Graph Analysis Types
class GapType(Enum):
    """Types of knowledge gaps in graphs."""

    MISSING_NODE = "missing_node"
    MISSING_RELATIONSHIP = "missing_relationship"
    INCONSISTENT_RELATIONSHIP = "inconsistent_relationship"
    INCOMPLETE_INFORMATION = "incomplete_information"
    OUTDATED_INFORMATION = "outdated_information"


class ConfidenceLevel(Enum):
    """Confidence levels for analysis results."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# Performance Constants
SEMANTIC_SIMILARITY_BATCH_SIZE = 32
MAX_GRAPH_NODES = 100000  # Maximum nodes for processing
GAP_DETECTION_TIMEOUT = 30000  # 30 seconds in ms
PROPOSAL_GENERATION_TIMEOUT = 15000  # 15 seconds in ms
VALIDATION_TIMEOUT = 10000  # 10 seconds in ms

# Algorithm Configuration
SIMILARITY_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.8
MAX_PROPOSALS_PER_GAP = 5
GRAPH_ANALYSIS_DEPTH = 3

# ML Model Constants
EMBEDDING_DIMENSIONS = 768
NEURAL_NETWORK_LAYERS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 100

# Caching Configuration
CACHE_TTL_SECONDS = 3600  # 1 hour
CACHE_MAX_SIZE = 10000  # Maximum cached items
CACHE_EVICTION_POLICY = "LRU"

# GPU Optimization
CUDA_DEVICE_COUNT = 1
MEMORY_FRACTION = 0.8  # Use 80% of GPU memory
ALLOW_MEMORY_GROWTH = True

# Performance Targets
TARGET_PROCESSING_TIME = {
    "gap_detection": 30,  # seconds for 1000 nodes
    "proposal_generation": 15,  # seconds
    "validation": 10,  # seconds
    "graph_analysis": 20,  # seconds
}

# Graph Processing Limits
MAX_NODES_PER_BATCH = 1000
MAX_EDGES_PER_BATCH = 5000
MAX_CONCURRENT_OPERATIONS = 4

# Algorithm Constants
PAGERANK_ITERATIONS = 100
PAGERANK_DAMPING = 0.85
CLUSTERING_ALGORITHM = "louvain"
CENTRALITY_ALGORITHMS = ["betweenness", "closeness", "eigenvector"]

# Trust Graph Configuration
DEFAULT_TRUST_SCORE = 0.5
TRUST_DECAY_RATE = 0.1
MIN_TRUST_THRESHOLD = 0.3
MAX_TRUST_THRESHOLD = 0.9


# Node Types
class NodeType(Enum):
    """Types of nodes in the graph."""

    ENTITY = "entity"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    ATTRIBUTE = "attribute"
    VALUE = "value"


# Relationship Types
RELATIONSHIP_TYPES = [
    "is_a",
    "part_of",
    "related_to",
    "influences",
    "causes",
    "enables",
    "requires",
    "conflicts_with",
    "similar_to",
    "derived_from",
]

# Data Quality Metrics
QUALITY_THRESHOLDS = {"completeness": 0.8, "consistency": 0.9, "accuracy": 0.85, "timeliness": 0.7, "relevance": 0.75}

# Error Messages
GRAPH_ERROR_MESSAGES = {
    "NODE_NOT_FOUND": "Graph node not found",
    "INVALID_GRAPH_STRUCTURE": "Invalid graph structure",
    "PROCESSING_TIMEOUT": "Graph processing timeout",
    "INSUFFICIENT_MEMORY": "Insufficient memory for graph operations",
    "GPU_NOT_AVAILABLE": "GPU acceleration not available",
    "MODEL_LOAD_FAILED": "ML model loading failed",
    "CACHE_MISS": "Cache miss during operation",
    "SIMILARITY_CALCULATION_FAILED": "Semantic similarity calculation failed",
}


# Status Codes
class GraphStatusCode(Enum):
    """Status codes for graph operations."""

    SUCCESS = "success"
    PROCESSING = "processing"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    MODEL_ERROR = "model_error"


# Configuration Defaults
DEFAULT_GRAPH_CONFIG = {
    "processing": {
        "batch_size": SEMANTIC_SIMILARITY_BATCH_SIZE,
        "max_nodes": MAX_GRAPH_NODES,
        "timeout_ms": GAP_DETECTION_TIMEOUT,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    },
    "ml": {
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "neural_layers": NEURAL_NETWORK_LAYERS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    },
    "caching": {"ttl_seconds": CACHE_TTL_SECONDS, "max_size": CACHE_MAX_SIZE, "eviction_policy": CACHE_EVICTION_POLICY},
    "gpu": {"device_count": CUDA_DEVICE_COUNT, "memory_fraction": MEMORY_FRACTION, "allow_growth": ALLOW_MEMORY_GROWTH},
}

__all__ = [
    "GapType",
    "ConfidenceLevel",
    "NodeType",
    "GraphStatusCode",
    "SEMANTIC_SIMILARITY_BATCH_SIZE",
    "MAX_GRAPH_NODES",
    "GAP_DETECTION_TIMEOUT",
    "PROPOSAL_GENERATION_TIMEOUT",
    "VALIDATION_TIMEOUT",
    "SIMILARITY_THRESHOLD",
    "CONFIDENCE_THRESHOLD",
    "MAX_PROPOSALS_PER_GAP",
    "GRAPH_ANALYSIS_DEPTH",
    "EMBEDDING_DIMENSIONS",
    "NEURAL_NETWORK_LAYERS",
    "LEARNING_RATE",
    "BATCH_SIZE",
    "EPOCHS",
    "CACHE_TTL_SECONDS",
    "CACHE_MAX_SIZE",
    "CACHE_EVICTION_POLICY",
    "TARGET_PROCESSING_TIME",
    "MAX_NODES_PER_BATCH",
    "MAX_EDGES_PER_BATCH",
    "RELATIONSHIP_TYPES",
    "QUALITY_THRESHOLDS",
    "GRAPH_ERROR_MESSAGES",
    "DEFAULT_GRAPH_CONFIG",
]
