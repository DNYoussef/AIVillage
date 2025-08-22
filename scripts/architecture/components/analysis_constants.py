"""Architectural analysis constants following connascence principles.

Centralizes all magic numbers and hardcoded values from the architectural
analysis system to eliminate Connascence of Meaning.
"""

from enum import Enum


class AnalysisConstants:
    """Core architectural analysis constants."""

    # File analysis limits
    MAX_FILE_LINES = 500
    MAX_FUNCTION_COMPLEXITY = 10

    # Quality thresholds
    DEFAULT_MAINTAINABILITY_INDEX = 70
    DEFAULT_TECHNICAL_DEBT_RATIO = 5
    DEFAULT_MAX_COUPLING_THRESHOLD = 0.3

    # Visualization settings
    VISUALIZATION_DPI = 300
    DEPENDENCY_GRAPH_SIZE = (12, 8)
    COUPLING_METRICS_SIZE = (15, 10)
    CONNASCENCE_HEATMAP_SIZE = (10, 6)
    TECHNICAL_DEBT_SIZE = (15, 10)

    # Spring layout parameters
    SPRING_LAYOUT_K = 2
    SPRING_LAYOUT_ITERATIONS = 50

    # Graph drawing parameters
    NODE_SIZE = 1000
    NODE_ALPHA = 0.7
    EDGE_ALPHA = 0.5
    ARROW_SIZE = 20
    FONT_SIZE = 10
    FONT_WEIGHT = "bold"

    # Directory exclusions
    EXCLUDED_DIRS = ["__pycache__", "test", ".", "codex-audit"]

    # File patterns
    PYTHON_EXTENSION = ".py"

    # Network analysis
    MAX_CYCLES_TO_DETECT = 100


class ConnascenceConstants:
    """Connascence detection constants."""

    # Magic number exclusions (commonly acceptable values)
    ACCEPTABLE_NUMBERS = {0, 1, -1, 2}

    # String analysis
    MIN_MAGIC_STRING_LENGTH = 10
    URL_PREFIXES = ("http://", "https://")

    # Position connascence
    MAX_POSITIONAL_ARGS = 3

    # Algorithm detection
    MIN_COMPLEX_FUNCTION_CONDITIONS = 2
    MIN_LOOPS_FOR_COMPLEXITY = 1
    MIN_EXCEPTION_HANDLING = 0

    # Severity calculation
    CRITICAL_THRESHOLD = 10
    HIGH_THRESHOLD = 5
    MEDIUM_THRESHOLD = 3

    # Surprise word detection
    SURPRISE_WORDS = [
        "unexpected",
        "surprising",
        "unusual",
        "novel",
        "strange",
        "shocking",
        "remarkable",
        "extraordinary",
    ]


class CouplingConstants:
    """Coupling analysis constants."""

    # Martin's metrics thresholds
    HIGH_INSTABILITY_THRESHOLD = 0.7
    OPTIMAL_DISTANCE_FROM_MAIN_SEQUENCE = 0.0

    # Visualization limits
    TOP_COUPLED_MODULES_COUNT = 10

    # Statistical analysis
    HISTOGRAM_BINS = 20


class TechnicalDebtConstants:
    """Technical debt analysis constants."""

    # Maintainability calculation
    MAINTAINABILITY_MULTIPLIER = 0.5
    DEBT_EFFORT_MULTIPLIER = 2

    # Risk level thresholds
    HIGH_RISK_MAINTAINABILITY_THRESHOLD = 50

    # Overall debt calculation
    HIGH_DEBT_RATIO_THRESHOLD = 20
    MAX_HIGH_DEBT_ITEMS = 5


class QualityGateConstants:
    """Quality gate evaluation constants."""

    # Gate thresholds
    MAX_HIGH_COUPLING_COUNT = 0
    MAX_CRITICAL_CONNASCENCE_COUNT = 0
    MAX_HIGH_DEBT_COUNT = 5
    MAX_CRITICAL_DRIFT_SEVERITY = 0.5

    # Circular dependency tolerance
    MAX_CIRCULAR_DEPENDENCIES = 0


class ReportingConstants:
    """Report generation constants."""

    # File naming
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    JSON_REPORT_PREFIX = "architecture_report_"
    HTML_REPORT_PREFIX = "architecture_report_"

    # Export limits
    MAX_COUPLING_METRICS_IN_HTML = 10

    # Visualization file names
    DEPENDENCY_GRAPH_FILE = "dependency_graph.png"
    COUPLING_METRICS_FILE = "coupling_metrics.png"
    CONNASCENCE_HEATMAP_FILE = "connascence_heatmap.png"
    TECHNICAL_DEBT_FILE = "technical_debt.png"


# Enum classes for type safety
class ConnascenceType(Enum):
    """Types of connascence violations."""

    NAME = "name"
    TYPE = "type"
    POSITION = "position"
    ALGORITHM = "algorithm"


class ConnascenceStrength(Enum):
    """Strength levels of connascence."""

    WEAK = "weak"
    STRONG = "strong"


class ConnascenceLocality(Enum):
    """Locality levels of connascence."""

    SAME_FUNCTION = "same_function"
    SAME_CLASS = "same_class"
    SAME_MODULE = "same_module"
    CROSS_MODULE = "cross_module"


class SeverityLevel(Enum):
    """Severity levels for violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(Enum):
    """Types of architectural drift."""

    DEPENDENCY_VIOLATION = "dependency_violation"
    COMPLEXITY_DRIFT = "complexity_drift"
    SIZE_DRIFT = "size_drift"


class DebtCategory(Enum):
    """Categories of technical debt."""

    MAINTAINABILITY = "maintainability"
    OVERALL = "overall"
    COMPLEXITY = "complexity"
    COUPLING = "coupling"


class RiskLevel(Enum):
    """Risk levels for technical debt."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Default configuration dictionary
DEFAULT_CONFIG = {
    "max_coupling_threshold": AnalysisConstants.DEFAULT_MAX_COUPLING_THRESHOLD,
    "max_file_lines": AnalysisConstants.MAX_FILE_LINES,
    "max_function_complexity": AnalysisConstants.MAX_FUNCTION_COMPLEXITY,
    "quality_thresholds": {
        "maintainability_index": AnalysisConstants.DEFAULT_MAINTAINABILITY_INDEX,
        "technical_debt_ratio": AnalysisConstants.DEFAULT_TECHNICAL_DEBT_RATIO,
    },
    "visualization": {
        "dpi": AnalysisConstants.VISUALIZATION_DPI,
        "figure_sizes": {
            "dependency_graph": AnalysisConstants.DEPENDENCY_GRAPH_SIZE,
            "coupling_metrics": AnalysisConstants.COUPLING_METRICS_SIZE,
            "connascence_heatmap": AnalysisConstants.CONNASCENCE_HEATMAP_SIZE,
            "technical_debt": AnalysisConstants.TECHNICAL_DEBT_SIZE,
        },
    },
}

# HTML template styling constants
HTML_TEMPLATE_STYLES = {
    "body_font": "Arial, sans-serif",
    "body_margin": "20px",
    "header_bg_color": "#f0f0f0",
    "header_padding": "20px",
    "header_border_radius": "5px",
    "section_margin": "20px 0",
    "section_padding": "15px",
    "section_border": "1px solid #ddd",
    "section_border_radius": "5px",
    "metric_display": "inline-block",
    "metric_margin": "10px",
    "metric_padding": "10px",
    "metric_bg_color": "#f9f9f9",
    "metric_border_radius": "3px",
    "violation_bg_color": "#ffebee",
    "violation_padding": "10px",
    "violation_margin": "5px 0",
    "violation_border_radius": "3px",
    "recommendation_bg_color": "#e8f5e8",
    "recommendation_padding": "10px",
    "recommendation_margin": "5px 0",
    "recommendation_border_radius": "3px",
    "passed_color": "green",
    "failed_color": "red",
    "font_weight_bold": "bold",
    "table_border_collapse": "collapse",
    "table_width": "100%",
    "table_cell_border": "1px solid #ddd",
    "table_cell_padding": "8px",
    "table_cell_text_align": "left",
    "table_header_bg_color": "#f2f2f2",
}

# Export all constants
__all__ = [
    "AnalysisConstants",
    "ConnascenceConstants",
    "CouplingConstants",
    "TechnicalDebtConstants",
    "QualityGateConstants",
    "ReportingConstants",
    "ConnascenceType",
    "ConnascenceStrength",
    "ConnascenceLocality",
    "SeverityLevel",
    "DriftType",
    "DebtCategory",
    "RiskLevel",
    "DEFAULT_CONFIG",
    "HTML_TEMPLATE_STYLES",
]
