"""Coupling analysis component for architectural analysis.

Implements Martin's coupling metrics and main sequence analysis.
Extracted from ArchitecturalAnalyzer following single responsibility principle.
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from .analysis_constants import CouplingConstants

logger = logging.getLogger(__name__)


@dataclass
class CouplingMetric:
    """Represents coupling measurements for a module."""

    module: str
    efferent_coupling: int  # Dependencies going out (Ce)
    afferent_coupling: int  # Dependencies coming in (Ca)
    instability: float  # Ce / (Ce + Ca)
    abstractness: float  # Abstract classes / Total classes
    distance: float  # Distance from main sequence |A + I - 1|

    def to_dict(self) -> dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "module": self.module,
            "efferent_coupling": self.efferent_coupling,
            "afferent_coupling": self.afferent_coupling,
            "instability": self.instability,
            "abstractness": self.abstractness,
            "distance": self.distance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> "CouplingMetric":
        """Create from dictionary."""
        return cls(
            module=data["module"],
            efferent_coupling=data["efferent_coupling"],
            afferent_coupling=data["afferent_coupling"],
            instability=data["instability"],
            abstractness=data["abstractness"],
            distance=data["distance"],
        )


class CouplingAnalyzer:
    """Analyzes coupling metrics using Martin's metrics.

    Single responsibility: Calculate and analyze coupling metrics
    including instability, abstractness, and distance from main sequence.
    """

    def __init__(self, project_root: Path):
        """Initialize coupling analyzer.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self._coupling_metrics: list[CouplingMetric] = []

        logger.info("Coupling analyzer initialized")

    def calculate_coupling_metrics(self, dependency_graph: nx.DiGraph) -> list[CouplingMetric]:
        """Calculate coupling metrics for all modules in the dependency graph.

        Args:
            dependency_graph: Module dependency graph

        Returns:
            List of coupling metrics
        """
        logger.info("Calculating coupling metrics...")

        self._coupling_metrics = []

        for module in dependency_graph.nodes():
            if not module.startswith("packages."):
                continue

            metric = self._calculate_module_coupling(module, dependency_graph)
            if metric:
                self._coupling_metrics.append(metric)

        logger.info(f"Calculated coupling metrics for {len(self._coupling_metrics)} modules")
        return self._coupling_metrics

    def _calculate_module_coupling(self, module: str, dependency_graph: nx.DiGraph) -> CouplingMetric | None:
        """Calculate coupling metrics for a single module.

        Args:
            module: Module name
            dependency_graph: Dependency graph

        Returns:
            CouplingMetric or None if calculation fails
        """
        try:
            # Efferent coupling (dependencies going out)
            efferent = len(list(dependency_graph.successors(module)))

            # Afferent coupling (dependencies coming in)
            afferent = len(list(dependency_graph.predecessors(module)))

            # Instability (Ce / (Ce + Ca))
            total_coupling = efferent + afferent
            instability = efferent / total_coupling if total_coupling > 0 else 0

            # Abstractness (requires code analysis)
            abstractness = self._calculate_abstractness(module)

            # Distance from main sequence |A + I - 1|
            distance = abs(abstractness + instability - 1)

            return CouplingMetric(
                module=module,
                efferent_coupling=efferent,
                afferent_coupling=afferent,
                instability=instability,
                abstractness=abstractness,
                distance=distance,
            )

        except Exception as e:
            logger.warning(f"Failed to calculate coupling for {module}: {e}")
            return None

    def _calculate_abstractness(self, module: str) -> float:
        """Calculate abstractness of a module.

        Abstractness = Abstract classes / Total classes

        Args:
            module: Module name

        Returns:
            Abstractness score (0.0-1.0)
        """
        try:
            module_path = self._module_name_to_path(module)
            if not module_path or not module_path.exists():
                return 0.0

            with open(module_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            total_classes = 0
            abstract_classes = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total_classes += 1

                    # Check if class is abstract
                    if self._is_abstract_class(node):
                        abstract_classes += 1

            return abstract_classes / total_classes if total_classes > 0 else 0.0

        except Exception as e:
            logger.debug(f"Failed to calculate abstractness for {module}: {e}")
            return 0.0

    def _is_abstract_class(self, class_node: ast.ClassDef) -> bool:
        """Check if a class is abstract.

        Args:
            class_node: AST class definition node

        Returns:
            True if class is abstract
        """
        # Check for abstract methods
        for node in ast.walk(class_node):
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                        return True

        # Check if inherits from ABC
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id == "ABC":
                return True
            elif isinstance(base, ast.Attribute) and base.attr == "ABC":
                return True

        return False

    def _module_name_to_path(self, module_name: str) -> Path | None:
        """Convert module name back to file path.

        Args:
            module_name: Module name in dot notation

        Returns:
            Path to module file or None if not found
        """
        if not module_name.startswith("packages."):
            return None

        # Remove 'packages' prefix
        parts = module_name.split(".")[1:]

        # Try different path combinations
        potential_paths = [
            self.project_root / "packages" / Path(*parts).with_suffix(".py"),
            self.project_root / "packages" / Path(*parts) / "__init__.py",
        ]

        for path in potential_paths:
            if path.exists():
                return path

        return None

    def get_high_coupling_modules(self, threshold: float | None = None) -> list[CouplingMetric]:
        """Get modules with high coupling (high instability).

        Args:
            threshold: Instability threshold (defaults to configured value)

        Returns:
            List of highly coupled modules
        """
        threshold = threshold or CouplingConstants.HIGH_INSTABILITY_THRESHOLD

        return [metric for metric in self._coupling_metrics if metric.instability > threshold]

    def get_modules_far_from_main_sequence(self, threshold: float = 0.5) -> list[CouplingMetric]:
        """Get modules that are far from the main sequence.

        Args:
            threshold: Distance threshold

        Returns:
            List of modules far from main sequence
        """
        return [metric for metric in self._coupling_metrics if metric.distance > threshold]

    def get_most_coupled_modules(self, count: int = 10) -> list[CouplingMetric]:
        """Get the most coupled modules by efferent coupling.

        Args:
            count: Number of modules to return

        Returns:
            List of most coupled modules
        """
        sorted_metrics = sorted(self._coupling_metrics, key=lambda m: m.efferent_coupling, reverse=True)

        return sorted_metrics[:count]

    def get_most_depended_upon_modules(self, count: int = 10) -> list[CouplingMetric]:
        """Get the most depended upon modules by afferent coupling.

        Args:
            count: Number of modules to return

        Returns:
            List of most depended upon modules
        """
        sorted_metrics = sorted(self._coupling_metrics, key=lambda m: m.afferent_coupling, reverse=True)

        return sorted_metrics[:count]

    def analyze_main_sequence_adherence(self) -> dict[str, any]:
        """Analyze how well modules adhere to the main sequence.

        Returns:
            Dictionary with main sequence analysis
        """
        if not self._coupling_metrics:
            return {
                "total_modules": 0,
                "on_main_sequence": 0,
                "average_distance": 0.0,
                "worst_violations": [],
            }

        # Calculate statistics
        total_modules = len(self._coupling_metrics)
        distances = [metric.distance for metric in self._coupling_metrics]
        average_distance = sum(distances) / total_modules

        # Count modules close to main sequence (distance < 0.1)
        on_main_sequence = sum(1 for d in distances if d < 0.1)

        # Find worst violations
        worst_violations = sorted(self._coupling_metrics, key=lambda m: m.distance, reverse=True)[:5]

        return {
            "total_modules": total_modules,
            "on_main_sequence": on_main_sequence,
            "percentage_on_main_sequence": (on_main_sequence / total_modules) * 100,
            "average_distance": average_distance,
            "worst_violations": [
                {
                    "module": metric.module.split(".")[-1],  # Just the module name
                    "distance": metric.distance,
                    "instability": metric.instability,
                    "abstractness": metric.abstractness,
                }
                for metric in worst_violations
            ],
        }

    def get_coupling_distribution(self) -> dict[str, any]:
        """Get distribution statistics for coupling metrics.

        Returns:
            Dictionary with coupling distribution statistics
        """
        if not self._coupling_metrics:
            return {}

        instabilities = [metric.instability for metric in self._coupling_metrics]
        abstractnesses = [metric.abstractness for metric in self._coupling_metrics]
        distances = [metric.distance for metric in self._coupling_metrics]
        efferent_couplings = [metric.efferent_coupling for metric in self._coupling_metrics]
        afferent_couplings = [metric.afferent_coupling for metric in self._coupling_metrics]

        return {
            "instability": {
                "mean": sum(instabilities) / len(instabilities),
                "min": min(instabilities),
                "max": max(instabilities),
                "std": self._calculate_std(instabilities),
            },
            "abstractness": {
                "mean": sum(abstractnesses) / len(abstractnesses),
                "min": min(abstractnesses),
                "max": max(abstractnesses),
                "std": self._calculate_std(abstractnesses),
            },
            "distance": {
                "mean": sum(distances) / len(distances),
                "min": min(distances),
                "max": max(distances),
                "std": self._calculate_std(distances),
            },
            "efferent_coupling": {
                "mean": sum(efferent_couplings) / len(efferent_couplings),
                "min": min(efferent_couplings),
                "max": max(efferent_couplings),
                "std": self._calculate_std(efferent_couplings),
            },
            "afferent_coupling": {
                "mean": sum(afferent_couplings) / len(afferent_couplings),
                "min": min(afferent_couplings),
                "max": max(afferent_couplings),
                "std": self._calculate_std(afferent_couplings),
            },
        }

    def _calculate_std(self, values: list[float]) -> float:
        """Calculate standard deviation.

        Args:
            values: List of numeric values

        Returns:
            Standard deviation
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance**0.5

    def export_metrics(self) -> list[dict[str, any]]:
        """Export coupling metrics as dictionaries.

        Returns:
            List of coupling metric dictionaries
        """
        return [metric.to_dict() for metric in self._coupling_metrics]

    @property
    def coupling_metrics(self) -> list[CouplingMetric]:
        """Get all coupling metrics."""
        return self._coupling_metrics.copy()

    def get_coupling_summary(self) -> dict[str, any]:
        """Get summary of coupling analysis.

        Returns:
            Dictionary with coupling summary
        """
        if not self._coupling_metrics:
            return {
                "total_modules": 0,
                "highly_coupled_count": 0,
                "average_instability": 0.0,
                "modules_on_main_sequence": 0,
            }

        high_coupling = self.get_high_coupling_modules()
        main_sequence_analysis = self.analyze_main_sequence_adherence()
        distribution = self.get_coupling_distribution()

        return {
            "total_modules": len(self._coupling_metrics),
            "highly_coupled_count": len(high_coupling),
            "average_instability": distribution.get("instability", {}).get("mean", 0.0),
            "average_distance_from_main_sequence": distribution.get("distance", {}).get("mean", 0.0),
            "modules_on_main_sequence": main_sequence_analysis["on_main_sequence"],
            "percentage_on_main_sequence": main_sequence_analysis["percentage_on_main_sequence"],
        }
