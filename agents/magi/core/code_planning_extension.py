"""Code-specific extension for MagiPlanning."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from .magi_planning import MagiPlanning
from agents.core.techniques.registry import TechniqueRegistry
from agents.core.techniques.multi_path_exploration import MultiPathExploration
from agents.core.techniques.pattern_integration import PatternIntegration
from agents.core.techniques.solution_unit_manipulation import SolutionUnitManipulation

logger = logging.getLogger(__name__)

@dataclass
class CodePlanningContext:
    """Context for code planning operations."""
    existing_code: Dict[str, Any]
    style_guide: Optional[Dict[str, Any]] = None
    performance_requirements: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None

class CodePlanningExtension:
    """
    Extension that adds code-specific capabilities to MagiPlanning.
    Integrates with existing MagiPlanning features while adding specialized code handling.
    """
    
    def __init__(self, magi_planning: MagiPlanning):
        """
        Initialize the extension.
        
        Args:
            magi_planning: The MagiPlanning instance to extend
        """
        self.magi_planning = magi_planning
        self._register_code_techniques()
        
    def _register_code_techniques(self):
        """Register code-specific techniques with MagiPlanning's registry."""
        # Add code-specific tags to existing techniques
        self.magi_planning.technique_registry.register(
            MultiPathExploration,
            tags=["code", "exploration"],
            parameters={
                "max_paths": 5,
                "convergence_threshold": 0.8
            }
        )
        
        self.magi_planning.technique_registry.register(
            PatternIntegration,
            tags=["code", "patterns"],
            parameters={
                "confidence_threshold": 0.7,
                "similarity_threshold": 0.6
            }
        )
        
        self.magi_planning.technique_registry.register(
            SolutionUnitManipulation,
            tags=["code", "refactoring"],
            parameters={
                "min_unit_size": 3,
                "max_unit_size": 10,
                "human_factor_weights": {
                    "comprehensibility": 0.4,
                    "maintainability": 0.3,
                    "reusability": 0.3
                }
            }
        )
    
    async def enhance_plan_for_code(
        self,
        plan: Dict[str, Any],
        context: CodePlanningContext
    ) -> Dict[str, Any]:
        """
        Enhance a MagiPlanning plan with code-specific considerations.
        
        Args:
            plan: Original plan from MagiPlanning
            context: Code-specific context
            
        Returns:
            Enhanced plan with code-specific details
        """
        # Add code pattern analysis
        patterns = await self._analyze_code_patterns(
            plan,
            context.existing_code
        )
        
        # Add code-specific quality checks
        quality_checks = await self._create_code_quality_checks(
            patterns,
            context.style_guide
        )
        
        # Add performance considerations
        performance_plan = await self._plan_performance_optimization(
            plan,
            context.performance_requirements
        )
        
        # Add dependency management
        dependency_plan = await self._plan_dependency_management(
            plan,
            context.dependencies
        )
        
        # Enhance the original plan
        enhanced_plan = {
            **plan,
            'code_patterns': patterns,
            'quality_checks': quality_checks,
            'performance_optimization': performance_plan,
            'dependency_management': dependency_plan,
            'code_specific_metrics': await self._calculate_code_metrics(plan)
        }
        
        return enhanced_plan
    
    async def _analyze_code_patterns(
        self,
        plan: Dict[str, Any],
        existing_code: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze and identify code patterns."""
        # Use PatternIntegration technique through MagiPlanning
        pattern_result = await self.magi_planning.technique_registry.execute(
            {'plan': plan, 'existing_code': existing_code},
            tags=['code', 'patterns']
        )
        
        return pattern_result.output
    
    async def _create_code_quality_checks(
        self,
        patterns: List[Dict[str, Any]],
        style_guide: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create code-specific quality checks."""
        checks = []
        
        # Pattern-based checks
        for pattern in patterns:
            checks.append({
                'type': 'pattern_conformance',
                'pattern': pattern['name'],
                'criteria': pattern.get('validation_rules', [])
            })
        
        # Style guide checks
        if style_guide:
            checks.extend([
                {
                    'type': 'style_check',
                    'rule': rule,
                    'criteria': criteria
                }
                for rule, criteria in style_guide.items()
            ])
        
        # Add standard code quality checks
        checks.extend([
            {
                'type': 'complexity',
                'metric': 'cyclomatic_complexity',
                'threshold': 10
            },
            {
                'type': 'maintainability',
                'metric': 'maintainability_index',
                'threshold': 20
            },
            {
                'type': 'test_coverage',
                'metric': 'line_coverage',
                'threshold': 0.8
            }
        ])
        
        return checks
    
    async def _plan_performance_optimization(
        self,
        plan: Dict[str, Any],
        requirements: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Plan performance optimizations."""
        if not requirements:
            return {}
            
        return {
            'optimization_targets': requirements,
            'analysis_points': await self._identify_performance_hotspots(plan),
            'optimization_techniques': await self._select_optimization_techniques(
                plan,
                requirements
            )
        }
    
    async def _plan_dependency_management(
        self,
        plan: Dict[str, Any],
        dependencies: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Plan dependency management."""
        if not dependencies:
            return {}
            
        return {
            'required_dependencies': dependencies,
            'version_constraints': await self._analyze_version_constraints(
                plan,
                dependencies
            ),
            'integration_points': await self._identify_integration_points(
                plan,
                dependencies
            )
        }
    
    async def _calculate_code_metrics(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate code-specific metrics."""
        return {
            'estimated_complexity': await self._estimate_complexity(plan),
            'maintainability_score': await self._estimate_maintainability(plan),
            'reusability_score': await self._estimate_reusability(plan),
            'test_coverage_estimate': await self._estimate_test_coverage(plan)
        }
    
    async def _identify_performance_hotspots(
        self,
        plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential performance hotspots."""
        # Use MultiPathExploration to identify critical paths
        exploration_result = await self.magi_planning.technique_registry.execute(
            plan,
            tags=['code', 'exploration']
        )
        
        return [
            {
                'location': path['location'],
                'risk_level': path['metrics']['performance_risk'],
                'optimization_potential': path['metrics']['optimization_potential']
            }
            for path in exploration_result.output['critical_paths']
        ]
    
    async def _select_optimization_techniques(
        self,
        plan: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select appropriate optimization techniques."""
        techniques = []
        
        if 'time_complexity' in requirements:
            techniques.append({
                'type': 'algorithmic_optimization',
                'target': 'time_complexity',
                'techniques': ['memoization', 'dynamic_programming']
            })
            
        if 'memory_usage' in requirements:
            techniques.append({
                'type': 'memory_optimization',
                'target': 'memory_usage',
                'techniques': ['pooling', 'lazy_loading']
            })
            
        if 'response_time' in requirements:
            techniques.append({
                'type': 'response_optimization',
                'target': 'response_time',
                'techniques': ['caching', 'asynchronous_processing']
            })
            
        return techniques
    
    async def _analyze_version_constraints(
        self,
        plan: Dict[str, Any],
        dependencies: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze version constraints for dependencies."""
        constraints = {}
        
        for dep_name, dep_info in dependencies.items():
            constraints[dep_name] = {
                'min_version': dep_info.get('min_version'),
                'max_version': dep_info.get('max_version'),
                'compatibility_issues': await self._check_compatibility(
                    dep_name,
                    dep_info,
                    plan
                )
            }
            
        return constraints
    
    async def _check_compatibility(
        self,
        dep_name: str,
        dep_info: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> List[str]:
        """Check for compatibility issues."""
        issues = []
        
        # Use pattern integration to identify compatibility problems
        pattern_result = await self.magi_planning.technique_registry.execute(
            {
                'dependency': dep_info,
                'plan': plan
            },
            tags=['code', 'patterns']
        )
        
        return [
            issue['description']
            for issue in pattern_result.output.get('compatibility_issues', [])
        ]
