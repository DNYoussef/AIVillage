"""Tool management functionality for MAGI."""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum
from .tool_persistence import ToolPersistence
from .tool_creator import ToolCreator
from .tool_optimization import ToolOptimizer
from ..core.project_planner import ToolRequirement, ToolStatus

logger = logging.getLogger(__name__)

@dataclass
class Tool:
    """Represents a tool in the system."""
    name: str
    implementation: str
    metadata: Dict[str, Any]
    version: str
    created_at: str
    last_modified: str
    usage_count: int
    performance_metrics: Dict[str, float]

class ToolManager:
    """
    Manages tool creation, modification, and usage tracking.
    """
    
    def __init__(
        self,
        llm=None,
        tool_persistence=None,
        tool_creator=None,
        tool_optimizer=None,
        continuous_learner=None
    ):
        self.llm = llm
        self.tool_persistence = tool_persistence or ToolPersistence("tools_storage")
        self.tool_creator = tool_creator or ToolCreator(llm=llm, continuous_learner=continuous_learner)
        self.tool_optimizer = tool_optimizer
        self.continuous_learner = continuous_learner
        self.tools: Dict[str, Any] = {}
        self.default_tool_parameters = {}
        self.creation_strategies = {}
        self.tool_dependencies = {}
        self.performance_history = {}

    async def create_tool(
        self,
        name: str,
        implementation: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Create a new tool.

        :param name: Tool name
        :param implementation: Tool implementation code
        :param metadata: Tool metadata
        :return: Tool ID
        """
        # Validate implementation
        validation_result = await self._validate_implementation(
            implementation, metadata)
        
        if not validation_result['is_valid']:
            raise ValueError(
                f"Invalid tool implementation: {validation_result['issues']}")
            
        # Create tool
        tool = await self.tool_creator.create_tool(
            name=name,
            implementation=implementation,
            metadata=metadata
        )
        
        # Store tool
        self.tools[name] = tool
        
        # Track dependencies
        self.tool_dependencies[name] = self._extract_dependencies(implementation)
        
        return name

    async def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    async def update_tool(
        self,
        name: str,
        implementation: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Update an existing tool."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' does not exist")
            
        # Validate new implementation
        validation_result = await self._validate_implementation(
            implementation, metadata)
        
        if not validation_result['is_valid']:
            raise ValueError(
                f"Invalid tool implementation: {validation_result['issues']}")
            
        # Update tool
        tool = await self.tool_creator.update_tool(
            name=name,
            implementation=implementation,
            metadata=metadata
        )
        
        # Update storage
        self.tools[name] = tool
        
        # Update dependencies
        self.tool_dependencies[name] = self._extract_dependencies(implementation)
        
        return name

    async def consider_removing_tool(self, tool: str) -> bool:
        """Consider removing a tool based on usage and performance."""
        removal_prompt = f"""
        Consider the implications of removing the '{tool}' tool due to low usage and performance.
        Provide a recommendation (remove/keep) with a brief explanation.
        """
        recommendation = await self.llm.generate(removal_prompt)
        if "remove" in recommendation.lower():
            del self.tools[tool]
            self.tool_persistence.delete_tool(tool)
            logger.info(f"Removed underperforming tool: {tool}")
            return True
        else:
            logger.info(f"Kept underperforming tool: {tool}. Reason: {recommendation}")
            return False

    async def consider_new_tools(self, tool_usage: Dict[str, int], recent_tasks: List[str]):
        """Consider creating new tools based on usage patterns and recent tasks."""
        new_tool_prompt = f"""
        Based on these recent tasks and current tool usage, suggest a new tool that could improve capabilities:
        
        Recent Tasks: {recent_tasks}
        Current Tools: {list(tool_usage.keys())}
        """
        new_tool_suggestion = await self.llm.generate(new_tool_prompt)
        await self.create_new_tool(new_tool_suggestion)

    async def create_new_tool(self, tool_suggestion: str):
        """Create a new tool based on a suggestion."""
        tool_creation_prompt = f"""
        Create a new tool based on this suggestion: {tool_suggestion}
        
        Provide the following:
        1. Tool Name
        2. Tool Description
        3. Python Code Implementation
        4. Required Parameters
        """
        tool_creation_result = await self.llm.generate(tool_creation_prompt)
        
        # Parse the result and create the tool
        lines = tool_creation_result.split('\n')
        tool_name = lines[0].split(': ')[1]
        tool_description = lines[1].split(': ')[1]
        tool_code = '\n'.join(lines[3:])
        
        await self.tool_creator.create_tool(
            name=tool_name,
            description=tool_description,
            parameters={},
            code=tool_code
        )
        logger.info(f"Created new tool: {tool_name}")

    async def optimize_tool(self, tool_name: str):
        """Optimize a frequently used tool."""
        tool_data = self.tool_persistence.load_tool(tool_name)
        if tool_data:
            optimization_prompt = f"""
            Optimize this tool code for better performance:
            
            {tool_data['code']}
            """
            optimized_code = await self.llm.generate(optimization_prompt)
            
            await self.tool_creator.create_tool(
                name=f"{tool_name}_optimized",
                code=optimized_code,
                description=f"Optimized version of {tool_name}",
                parameters=tool_data['parameters']
            )

    async def assess_tool_creation_feasibility(
        self,
        name: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """
        Assess if a tool can be feasibly created.

        :param name: Tool name
        :param requirements: Tool requirements
        :return: Feasibility assessment
        """
        # Check if similar tool exists
        similar_tool = await self._find_similar_tool(name, requirements)
        if similar_tool:
            return {
                'is_feasible': True,
                'strategy': 'modify',
                'similar_tool': similar_tool
            }
            
        # Check if requirements can be met
        requirements_analysis = await self._analyze_requirements(requirements)
        if not requirements_analysis['can_meet_requirements']:
            return {
                'is_feasible': False,
                'reason': requirements_analysis['blockers']
            }
            
        # Estimate complexity
        complexity = await self._estimate_complexity(requirements)
        if complexity > 0.8:  # Threshold for "too complex"
            return {
                'is_feasible': False,
                'reason': f"Tool would be too complex (score: {complexity})"
            }
            
        return {
            'is_feasible': True,
            'strategy': 'create',
            'estimated_complexity': complexity
        }

    async def check_dependency_availability(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        # Check if dependency is a tool
        if dependency in self.tools:
            return True
            
        # Check if dependency is a system package
        try:
            __import__(dependency)
            return True
        except ImportError:
            return False

    def update_default_tool_parameters(self, common_parameters: List[str]):
        """Update the default parameters used when creating new tools."""
        self.default_tool_parameters = {param: None for param in common_parameters}

    async def simplify_tool_creation_process(self):
        """Simplify the tool creation process."""
        # Create templates for common tool types
        templates = await self.generate_tool_templates()
        self.tool_creator.update_templates(templates)

    async def enhance_tool_creation_process(self):
        """Enhance the tool creation process."""
        # Add more advanced features
        enhancements = await self.generate_tool_enhancements()
        self.tool_creator.apply_enhancements(enhancements)

    async def prune_unused_tools(self, usage_count: Dict[str, int]):
        """Remove or archive tools that haven't been used."""
        unused_tools = [tool for tool, usage in usage_count.items() if usage == 0]
        for tool in unused_tools:
            await self.archive_tool(tool)

    async def archive_tool(self, tool_name: str):
        """Archive a tool for potential future use."""
        tool_data = self.tool_persistence.load_tool(tool_name)
        if tool_data:
            self.tool_persistence.archive_tool(tool_name, tool_data)
            del self.tools[tool_name]
            logger.info(f"Archived unused tool: {tool_name}")

    async def improve_tool_creation(self):
        """Improve the tool creation process."""
        template = await self.generate_improved_tool_template()
        self.update_tool_creation_process(template)

    async def generate_improved_tool_template(self) -> str:
        """Generate an improved template for tool creation."""
        template_prompt = """
        Generate an improved template for tool creation that includes:
        1. Better error handling
        2. Performance optimization
        3. Input validation
        4. Documentation standards
        5. Testing requirements
        """
        return await self.llm.generate(template_prompt)

    def update_tool_creation_process(self, template: str):
        """Update the tool creation process with a new template."""
        self.tool_creator.update_template(template)

    async def _validate_implementation(
        self,
        implementation: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate tool implementation."""
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check syntax
        try:
            compile(implementation, '<string>', 'exec')
        except SyntaxError as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Syntax error: {str(e)}")
            
        # Check required metadata
        required_fields = ['purpose', 'requirements', 'dependencies']
        for field in required_fields:
            if field not in metadata:
                validation_result['warnings'].append(
                    f"Missing metadata field: {field}")
                
        # Check for potential security issues
        security_issues = self._check_security_issues(implementation)
        if security_issues:
            validation_result['is_valid'] = False
            validation_result['issues'].extend(security_issues)
            
        return validation_result

    def _check_security_issues(self, implementation: str) -> List[str]:
        """Check for potential security issues in implementation."""
        issues = []
        
        # Check for dangerous imports
        dangerous_imports = ['os.system', 'subprocess', 'eval', 'exec']
        for imp in dangerous_imports:
            if imp in implementation:
                issues.append(f"Potentially dangerous import/usage: {imp}")
                
        # Check for file operations
        file_ops = ['open(', 'write(', 'read(']
        for op in file_ops:
            if op in implementation:
                issues.append(
                    f"File operation detected: {op}. "
                    "Ensure proper security measures are in place.")
                
        return issues

    def _extract_dependencies(self, implementation: str) -> List[str]:
        """Extract dependencies from implementation."""
        dependencies = []
        
        # Extract import statements
        lines = implementation.split('\n')
        for line in lines:
            if line.strip().startswith('import '):
                dependencies.append(line.split()[1])
            elif line.strip().startswith('from '):
                dependencies.append(line.split()[1])
                
        return dependencies

    async def _find_similar_tool(
        self,
        name: str,
        requirements: List[str]
    ) -> Optional[str]:
        """Find a similar existing tool."""
        for tool_name, tool in self.tools.items():
            # Check name similarity
            if self._calculate_name_similarity(name, tool_name) > 0.8:
                return tool_name
                
            # Check requirements overlap
            tool_reqs = tool.metadata.get('requirements', [])
            overlap = len(set(requirements) & set(tool_reqs)) / len(requirements)
            if overlap > 0.7:
                return tool_name
                
        return None

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two tool names."""
        # Simple Levenshtein distance implementation
        if len(name1) < len(name2):
            return self._calculate_name_similarity(name2, name1)
            
        if len(name2) == 0:
            return 0.0
            
        previous_row = range(len(name2) + 1)
        for i, c1 in enumerate(name1):
            current_row = [i + 1]
            for j, c2 in enumerate(name2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        # Convert distance to similarity score
        max_len = max(len(name1), len(name2))
        return 1 - (previous_row[-1] / max_len)

    async def _analyze_requirements(
        self,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Analyze if requirements can be met."""
        analysis = {
            'can_meet_requirements': True,
            'blockers': [],
            'warnings': []
        }
        
        for req in requirements:
            # Check if requirement needs external dependencies
            if 'requires' in req.lower():
                dep = req.split('requires')[-1].strip()
                if not await self.check_dependency_availability(dep):
                    analysis['blockers'].append(
                        f"Missing required dependency: {dep}")
                    analysis['can_meet_requirements'] = False
                    
            # Check for complexity indicators
            if any(word in req.lower() for word in ['complex', 'advanced', 'sophisticated']):
                analysis['warnings'].append(
                    f"Potentially complex requirement: {req}")
                    
        return analysis

    async def _estimate_complexity(self, requirements: List[str]) -> float:
        """Estimate implementation complexity."""
        complexity_score = 0.0
        
        # Analyze each requirement
        for req in requirements:
            # Check for complexity indicators
            if 'complex' in req.lower():
                complexity_score += 0.3
            if 'advanced' in req.lower():
                complexity_score += 0.2
            if 'simple' in req.lower():
                complexity_score -= 0.1
                
            # Check for dependency complexity
            if 'requires' in req.lower():
                complexity_score += 0.15
                
            # Check for integration requirements
            if 'integrate' in req.lower():
                complexity_score += 0.25
                
        return min(max(complexity_score, 0.0), 1.0)  # Ensure score is between 0 and 1
