from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class RequirementType(Enum):
    FUNCTIONAL = "functional"
    TECHNICAL = "technical"
    INFRASTRUCTURE = "infrastructure"
    INTEGRATION = "integration"

class ToolStatus(Enum):
    AVAILABLE = "available"
    NEEDS_CREATION = "needs_creation"
    NEEDS_MODIFICATION = "needs_modification"

@dataclass
class Requirement:
    id: str
    type: RequirementType
    description: str
    priority: int
    dependencies: List[str]
    tools_needed: List[str]
    estimated_complexity: float

@dataclass
class ToolRequirement:
    name: str
    purpose: str
    status: ToolStatus
    requirements: List[str]
    dependencies: List[str]
    estimated_complexity: float

@dataclass
class ProjectPlan:
    requirements: List[Requirement]
    tool_requirements: List[ToolRequirement]
    build_steps: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    estimated_timeline: Dict[str, float]

class ProjectPlanner:
    """
    Handles project planning and requirement analysis for the MagiAgent.
    """
    
    def __init__(self, tool_manager, knowledge_manager, llm):
        self.tool_manager = tool_manager
        self.knowledge_manager = knowledge_manager
        self.llm = llm
        self.requirement_cache = {}
        self.tool_analysis_cache = {}

    async def analyze_project_request(self, request: str) -> ProjectPlan:
        """
        Analyze a project request and create a comprehensive project plan.

        :param request: The project request description
        :return: A ProjectPlan object containing requirements and build steps
        """
        # Extract requirements
        requirements = await self._extract_requirements(request)
        
        # Analyze tool requirements
        tool_requirements = await self._analyze_tool_requirements(requirements)
        
        # Generate build steps
        build_steps = await self._generate_build_steps(requirements, tool_requirements)
        
        # Analyze dependencies
        dependencies = self._analyze_dependencies(requirements, tool_requirements)
        
        # Estimate timeline
        timeline = await self._estimate_timeline(requirements, tool_requirements, build_steps)
        
        return ProjectPlan(
            requirements=requirements,
            tool_requirements=tool_requirements,
            build_steps=build_steps,
            dependencies=dependencies,
            estimated_timeline=timeline
        )

    async def _extract_requirements(self, request: str) -> List[Requirement]:
        """Extract and categorize requirements from the project request."""
        # Generate requirement analysis prompt
        analysis_prompt = f"""
        Analyze this project request and break it down into specific requirements.
        For each requirement, provide:
        1. Type (functional/technical/infrastructure/integration)
        2. Description
        3. Priority (1-5)
        4. Dependencies
        5. Tools needed
        6. Estimated complexity (0-1)

        Project Request:
        {request}
        """
        
        # Get analysis from LLM
        analysis = await self.llm.generate(analysis_prompt)
        
        # Parse requirements from analysis
        requirements = []
        current_req = None
        
        for line in analysis.split('\n'):
            if line.startswith('Requirement'):
                if current_req:
                    requirements.append(self._create_requirement(current_req))
                current_req = {'id': line.split(':')[1].strip()}
            elif current_req and ':' in line:
                key, value = line.split(':', 1)
                current_req[key.strip().lower()] = value.strip()
                
        if current_req:
            requirements.append(self._create_requirement(current_req))
            
        return requirements

    def _create_requirement(self, req_dict: Dict[str, Any]) -> Requirement:
        """Create a Requirement object from a dictionary."""
        return Requirement(
            id=req_dict['id'],
            type=RequirementType(req_dict['type'].lower()),
            description=req_dict['description'],
            priority=int(req_dict['priority']),
            dependencies=req_dict.get('dependencies', '').split(','),
            tools_needed=req_dict.get('tools needed', '').split(','),
            estimated_complexity=float(req_dict.get('estimated complexity', 0.5))
        )

    async def _analyze_tool_requirements(self, requirements: List[Requirement]) -> List[ToolRequirement]:
        """Analyze tool requirements and determine which tools need to be created or modified."""
        tool_requirements = []
        needed_tools = set()
        
        # Collect all needed tools
        for req in requirements:
            needed_tools.update(req.tools_needed)
            
        # Analyze each tool
        for tool_name in needed_tools:
            if tool_name in self.tool_analysis_cache:
                tool_requirements.append(self.tool_analysis_cache[tool_name])
                continue
                
            # Check if tool exists
            existing_tool = await self.tool_manager.get_tool(tool_name)
            
            if existing_tool:
                # Analyze if tool needs modification
                needs_modification = await self._check_tool_modification_needed(
                    existing_tool, requirements)
                
                status = (ToolStatus.NEEDS_MODIFICATION if needs_modification 
                         else ToolStatus.AVAILABLE)
            else:
                status = ToolStatus.NEEDS_CREATION
                
            # Create tool requirement
            tool_req = await self._create_tool_requirement(
                tool_name, requirements, status)
            
            tool_requirements.append(tool_req)
            self.tool_analysis_cache[tool_name] = tool_req
            
        return tool_requirements

    async def _check_tool_modification_needed(
        self, 
        existing_tool: Any, 
        requirements: List[Requirement]
    ) -> bool:
        """Check if an existing tool needs modification to meet requirements."""
        tool_analysis_prompt = f"""
        Analyze if this existing tool needs modifications to meet the requirements:

        Tool:
        {existing_tool}

        Requirements:
        {requirements}

        Determine if modifications are needed and explain why/why not.
        """
        
        analysis = await self.llm.generate(tool_analysis_prompt)
        return "needs modification" in analysis.lower()

    async def _create_tool_requirement(
        self,
        tool_name: str,
        requirements: List[Requirement],
        status: ToolStatus
    ) -> ToolRequirement:
        """Create a tool requirement specification."""
        tool_spec_prompt = f"""
        Create a detailed specification for the tool '{tool_name}' based on these requirements:
        {requirements}

        Provide:
        1. Purpose
        2. Required functionality
        3. Dependencies
        4. Estimated complexity (0-1)
        """
        
        spec = await self.llm.generate(tool_spec_prompt)
        spec_lines = spec.split('\n')
        
        return ToolRequirement(
            name=tool_name,
            purpose=spec_lines[0].split(':', 1)[1].strip(),
            status=status,
            requirements=[r.strip() for r in spec_lines[1].split(':', 1)[1].strip().split(',')],
            dependencies=[d.strip() for d in spec_lines[2].split(':', 1)[1].strip().split(',')],
            estimated_complexity=float(spec_lines[3].split(':', 1)[1].strip())
        )

    async def _generate_build_steps(
        self,
        requirements: List[Requirement],
        tool_requirements: List[ToolRequirement]
    ) -> List[Dict[str, Any]]:
        """Generate ordered build steps for the project."""
        # Create build steps prompt
        build_steps_prompt = f"""
        Generate ordered build steps for this project:

        Requirements:
        {requirements}

        Tool Requirements:
        {tool_requirements}

        For each step provide:
        1. Step number and name
        2. Description
        3. Dependencies (previous steps needed)
        4. Tools needed
        5. Estimated time
        6. Success criteria
        """
        
        steps_analysis = await self.llm.generate(build_steps_prompt)
        
        # Parse build steps
        build_steps = []
        current_step = None
        
        for line in steps_analysis.split('\n'):
            if line.startswith('Step'):
                if current_step:
                    build_steps.append(current_step)
                current_step = {'name': line.split(':', 1)[1].strip()}
            elif current_step and ':' in line:
                key, value = line.split(':', 1)
                current_step[key.strip().lower()] = value.strip()
                
        if current_step:
            build_steps.append(current_step)
            
        return build_steps

    def _analyze_dependencies(
        self,
        requirements: List[Requirement],
        tool_requirements: List[ToolRequirement]
    ) -> Dict[str, List[str]]:
        """Analyze and create a dependency graph for requirements and tools."""
        dependencies = {}
        
        # Add requirement dependencies
        for req in requirements:
            dependencies[req.id] = req.dependencies
            
        # Add tool dependencies
        for tool in tool_requirements:
            dependencies[tool.name] = tool.dependencies
            
        return dependencies

    async def _estimate_timeline(
        self,
        requirements: List[Requirement],
        tool_requirements: List[ToolRequirement],
        build_steps: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Estimate timeline for project completion."""
        timeline = {}
        
        # Calculate base estimates
        for step in build_steps:
            # Get complexity factors
            step_reqs = [r for r in requirements if r.id in step.get('dependencies', [])]
            step_tools = [t for t in tool_requirements if t.name in step.get('tools needed', [])]
            
            # Calculate time estimate based on complexities
            req_complexity = sum(r.estimated_complexity for r in step_reqs)
            tool_complexity = sum(t.estimated_complexity for t in step_tools)
            
            # Base estimate (in hours)
            base_estimate = (req_complexity + tool_complexity) * 8
            
            # Add buffer for uncertainties (20%)
            timeline[step['name']] = base_estimate * 1.2
            
        return timeline

    async def validate_plan(self, plan: ProjectPlan) -> Dict[str, Any]:
        """
        Validate the project plan for completeness and consistency.
        
        :param plan: The project plan to validate
        :return: Validation results and any identified issues
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for circular dependencies
        circular_deps = self._check_circular_dependencies(plan.dependencies)
        if circular_deps:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                f"Circular dependencies detected: {circular_deps}")
            
        # Validate tool requirements
        tool_issues = await self._validate_tool_requirements(plan.tool_requirements)
        if tool_issues:
            validation_results['warnings'].extend(tool_issues)
            
        # Validate build steps
        step_issues = self._validate_build_steps(plan.build_steps)
        if step_issues:
            validation_results['issues'].extend(step_issues)
            
        # Check timeline feasibility
        timeline_issues = self._check_timeline_feasibility(plan.estimated_timeline)
        if timeline_issues:
            validation_results['warnings'].extend(timeline_issues)
            
        return validation_results

    def _check_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Check for circular dependencies in the dependency graph."""
        visited = set()
        path = []
        circular = []

        def visit(node):
            if node in path:
                circular.append(path[path.index(node):] + [node])
                return
            if node in visited:
                return
                
            visited.add(node)
            path.append(node)
            
            for dep in dependencies.get(node, []):
                visit(dep)
                
            path.pop()

        for node in dependencies:
            visit(node)
            
        return circular

    async def _validate_tool_requirements(
        self,
        tool_requirements: List[ToolRequirement]
    ) -> List[str]:
        """Validate tool requirements for feasibility and completeness."""
        issues = []
        
        for tool in tool_requirements:
            # Check if tool creation is feasible
            if tool.status == ToolStatus.NEEDS_CREATION:
                feasibility = await self.tool_manager.assess_tool_creation_feasibility(
                    tool.name, tool.requirements)
                if not feasibility['is_feasible']:
                    issues.append(
                        f"Tool '{tool.name}' may not be feasible to create: "
                        f"{feasibility['reason']}")
                    
            # Check if required dependencies are available
            for dep in tool.dependencies:
                if not await self.tool_manager.check_dependency_availability(dep):
                    issues.append(
                        f"Tool '{tool.name}' has unavailable dependency: {dep}")
                    
        return issues

    def _validate_build_steps(self, build_steps: List[Dict[str, Any]]) -> List[str]:
        """Validate build steps for completeness and correct ordering."""
        issues = []
        
        # Check for missing dependencies
        all_steps = {step['name'] for step in build_steps}
        for step in build_steps:
            for dep in step.get('dependencies', []):
                if dep not in all_steps:
                    issues.append(
                        f"Build step '{step['name']}' depends on missing step: {dep}")
                    
        # Check for missing success criteria
        for step in build_steps:
            if 'success criteria' not in step or not step['success criteria']:
                issues.append(
                    f"Build step '{step['name']}' is missing success criteria")
                
        return issues

    def _check_timeline_feasibility(self, timeline: Dict[str, float]) -> List[str]:
        """Check if the estimated timeline is feasible."""
        issues = []
        
        total_time = sum(timeline.values())
        if total_time > 160:  # More than 4 weeks (40 hours/week)
            issues.append(
                f"Total estimated time ({total_time} hours) may be too long for "
                "a single project")
            
        # Check for unrealistic estimates
        for step, time in timeline.items():
            if time < 1:
                issues.append(
                    f"Step '{step}' has unrealistically short estimate: {time} hours")
            elif time > 40:
                issues.append(
                    f"Step '{step}' has very long estimate: {time} hours, "
                    "consider breaking it down")
                
        return issues
