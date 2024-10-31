"""Task research component for MAGI agent."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.utils.logging import setup_logger
from ..research.integration import ResearchIntegration

logger = setup_logger(__name__)

class TaskResearch:
    """Handles task analysis and research for MAGI agent."""
    
    def __init__(self):
        self.research_integration = ResearchIntegration()
        self.analysis_history = []
    
    async def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a task to understand requirements and context.
        
        Args:
            task: Task to analyze
            
        Returns:
            Analysis results
        """
        try:
            logger.info(f"Analyzing task: {task.get('type', 'unknown')}")
            
            # Extract task components
            task_type = task.get('type', 'unknown')
            content = task.get('content', {})
            requirements = content.get('requirements', {})
            
            # Analyze requirements
            analysis = {
                'task_type': task_type,
                'complexity': self._assess_complexity(content),
                'requirements': self._analyze_requirements(requirements),
                'risks': self._identify_risks(content),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing task: {str(e)}")
            return {
                'error': str(e),
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_complexity(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Assess task complexity."""
        # Count requirements and constraints
        req_count = len(content.get('requirements', {}))
        constraint_count = len(content.get('constraints', []))
        
        # Calculate base complexity
        base_complexity = min(1.0, (req_count + constraint_count) / 10)
        
        # Adjust for specific factors
        adjustments = []
        
        if 'mathematical' in str(content).lower():
            adjustments.append(0.2)
        if 'optimization' in str(content).lower():
            adjustments.append(0.3)
        if 'compression' in str(content).lower():
            adjustments.append(0.4)
        
        # Apply adjustments
        final_complexity = base_complexity + sum(adjustments)
        
        return {
            'score': min(1.0, final_complexity),
            'base_complexity': base_complexity,
            'adjustments': adjustments,
            'factors': {
                'requirements_count': req_count,
                'constraints_count': constraint_count
            }
        }
    
    def _analyze_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements."""
        analysis = {
            'categories': {},
            'dependencies': [],
            'critical_requirements': []
        }
        
        # Categorize requirements
        for key, value in requirements.items():
            category = self._categorize_requirement(key, value)
            if category not in analysis['categories']:
                analysis['categories'][category] = []
            analysis['categories'][category].append(key)
            
            # Check for dependencies
            deps = self._find_dependencies(key, value)
            if deps:
                analysis['dependencies'].extend(deps)
            
            # Check if critical
            if self._is_critical_requirement(key, value):
                analysis['critical_requirements'].append(key)
        
        return analysis
    
    def _identify_risks(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential risks in the task."""
        risks = []
        
        # Check for common risk patterns
        if 'accuracy' in str(content).lower():
            risks.append({
                'type': 'accuracy',
                'description': 'Task requires high accuracy',
                'severity': 'high'
            })
        
        if 'memory' in str(content).lower():
            risks.append({
                'type': 'resource',
                'description': 'Task may have high memory requirements',
                'severity': 'medium'
            })
        
        if 'performance' in str(content).lower():
            risks.append({
                'type': 'performance',
                'description': 'Task has performance requirements',
                'severity': 'medium'
            })
        
        return risks
    
    def _categorize_requirement(self, key: str, value: Any) -> str:
        """Categorize a requirement."""
        key_lower = key.lower()
        
        if 'accuracy' in key_lower or 'precision' in key_lower:
            return 'quality'
        elif 'time' in key_lower or 'speed' in key_lower:
            return 'performance'
        elif 'memory' in key_lower or 'storage' in key_lower:
            return 'resource'
        elif 'maintain' in key_lower or 'preserve' in key_lower:
            return 'preservation'
        else:
            return 'general'
    
    def _find_dependencies(self, key: str, value: Any) -> List[str]:
        """Find dependencies in a requirement."""
        deps = []
        
        # Check for explicit dependencies
        if isinstance(value, dict) and 'depends_on' in value:
            deps.extend(value['depends_on'])
        
        # Check for implicit dependencies in text
        if isinstance(value, str):
            if 'after' in value.lower():
                deps.append(f"implicit: {value}")
            if 'requires' in value.lower():
                deps.append(f"implicit: {value}")
        
        return deps
    
    def _is_critical_requirement(self, key: str, value: Any) -> bool:
        """Determine if a requirement is critical."""
        # Check key indicators
        key_lower = key.lower()
        if 'critical' in key_lower or 'essential' in key_lower:
            return True
        
        # Check value indicators
        if isinstance(value, dict):
            return value.get('critical', False)
        
        # Check text content
        if isinstance(value, str):
            indicators = ['must', 'required', 'essential', 'critical']
            return any(ind in value.lower() for ind in indicators)
        
        return False
