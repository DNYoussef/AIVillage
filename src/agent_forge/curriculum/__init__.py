"""Frontier Curriculum Engine for Agent Forge.

A curriculum system that uses frontier LLMs to find a model's edge-of-chaos band
and generate targeted problems for productive struggle at ~55-75% accuracy.

Core Components:
- Edge Finder: Assess model difficulty boundaries
- Problem Generator: Create ~1000 problems at edge
- Variant Synthesizer: Generate cosmetic variants preserving skill
- Grader: Auto-grade final answers (no chain-of-thought)
- Hint System: Generate ≤25 token hints from wrong answers
- Mastery Tracker: Track understanding via 3-variant-pass rule
- Controller: Maintain accuracy in target band via edge nudging
"""

from .controller import EdgeController, control_curriculum_edge

# Import all core components
from .edge_finder import EdgeFinder, find_model_edge
from .grader import Grader, grade_code_solution
from .hints import HintGenerator, generate_coding_hint
from .mastery import MasteryTracker, track_student_mastery
from .openrouter import OpenRouterLLM
from .orchestrator import CurriculumOrchestrator, run_full_curriculum_pipeline
from .problem_generator import ProblemGenerator, generate_coding_problems
from .schemas import *
from .variant_maker import VariantMaker, create_problem_variants

__version__ = "1.0.0"

__all__ = [
    # Core client
    "OpenRouterLLM",
    # Component classes
    "EdgeFinder",
    "ProblemGenerator",
    "VariantMaker",
    "Grader",
    "HintGenerator",
    "MasteryTracker",
    "EdgeController",
    "CurriculumOrchestrator",
    # Convenience functions
    "find_model_edge",
    "generate_coding_problems",
    "create_problem_variants",
    "grade_code_solution",
    "generate_coding_hint",
    "track_student_mastery",
    "control_curriculum_edge",
    "run_full_curriculum_pipeline",
    # All schemas
    "EdgeAssessmentRequest",
    "EdgeAssessmentResponse",
    "ProblemGenerationRequest",
    "ProblemGenerationResponse",
    "Problem",
    "ProblemVariant",
    "VariantRequest",
    "VariantResponse",
    "GradingRequest",
    "GradingResponse",
    "HintRequest",
    "HintResponse",
    "MasteryRequest",
    "MasteryResponse",
    "ControllerRequest",
    "ControllerResponse",
    "ConductorRequest",
    "ConductorResponse",
    "EdgeWindow",
    "TopicMix",
    "DistributionPoint",
    "GenerationPlan",
    "TelemetryEntry",
    "DifficultyScale",
    "EdgeConstraints",
    "VariantPolicy",
    "NumericJitterPolicy",
    "PeerSummary",
    "HintType",
    "AttemptRecord",
    "LastResult",
    "MasteryStatus",
    "MasteryAction",
    "EdgeDelta",
    "QueueBacklog",
    "MasteryStats",
    "BatchOperation",
    "BatchItem",
]
