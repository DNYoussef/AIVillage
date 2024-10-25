from typing import Dict, Any, List, Optional, Tuple
import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from .project_planner import ProjectPlan, ToolRequirement, ToolStatus

logger = logging.getLogger(__name__)

class BuildStage(Enum):
    TOOL_PREPARATION = "tool_preparation"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    TESTING = "testing"
    DEPLOYMENT = "deployment"

class BuildStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BuildContext:
    """Context for the current build process."""
    project_plan: ProjectPlan
    current_stage: BuildStage
    status: BuildStatus
    tools_created: List[str]
    tools_modified: List[str]
    completed_steps: List[str]
    build_artifacts: Dict[str, Any]
    start_time: datetime
    metrics: Dict[str, Any]

class BuildOrchestrator:
    """
    Coordinates the build process, including tool creation and project implementation.
    """
    
    def __init__(self, tool_manager, knowledge_manager, llm):
        self.tool_manager = tool_manager
        self.knowledge_manager = knowledge_manager
        self.llm = llm
        self.current_builds: Dict[str, BuildContext] = {}
        self.build_history: List[BuildContext] = []

    async def orchestrate_build(self, project_id: str, project_plan: ProjectPlan) -> BuildContext:
        """
        Orchestrate the complete build process for a project.

        :param project_id: Unique identifier for the project
        :param project_plan: The project plan to execute
        :return: Build context with results
        """
        # Initialize build context
        context = BuildContext(
            project_plan=project_plan,
            current_stage=BuildStage.TOOL_PREPARATION,
            status=BuildStatus.NOT_STARTED,
            tools_created=[],
            tools_modified=[],
            completed_steps=[],
            build_artifacts={},
            start_time=datetime.now(),
            metrics={"stage_times": {}}
        )
        
        self.current_builds[project_id] = context
        
        try:
            # Prepare tools
            await self._prepare_tools(context)
            
            # Execute build stages
            for stage in BuildStage:
                context.current_stage = stage
                context.status = BuildStatus.IN_PROGRESS
                
                stage_start = datetime.now()
                
                if stage == BuildStage.TOOL_PREPARATION:
                    continue  # Already done
                elif stage == BuildStage.IMPLEMENTATION:
                    await self._execute_implementation(context)
                elif stage == BuildStage.INTEGRATION:
                    await self._execute_integration(context)
                elif stage == BuildStage.TESTING:
                    await self._execute_testing(context)
                elif stage == BuildStage.DEPLOYMENT:
                    await self._execute_deployment(context)
                    
                # Record stage metrics
                stage_time = (datetime.now() - stage_start).total_seconds()
                context.metrics["stage_times"][stage.value] = stage_time
                
            context.status = BuildStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Build failed: {str(e)}")
            context.status = BuildStatus.FAILED
            context.metrics["failure_reason"] = str(e)
            raise
            
        finally:
            # Record build history
            self.build_history.append(context)
            
        return context

    async def _prepare_tools(self, context: BuildContext):
        """Prepare all necessary tools for the build."""
        tool_requirements = context.project_plan.tool_requirements
        
        # Group tools by status
        tools_by_status = {
            ToolStatus.NEEDS_CREATION: [],
            ToolStatus.NEEDS_MODIFICATION: [],
            ToolStatus.AVAILABLE: []
        }
        
        for tool in tool_requirements:
            tools_by_status[tool.status].append(tool)
            
        # Create new tools
        for tool in tools_by_status[ToolStatus.NEEDS_CREATION]:
            created_tool = await self._create_tool(tool)
            context.tools_created.append(created_tool)
            
        # Modify existing tools
        for tool in tools_by_status[ToolStatus.NEEDS_MODIFICATION]:
            modified_tool = await self._modify_tool(tool)
            context.tools_modified.append(modified_tool)

    async def _create_tool(self, tool_requirement: ToolRequirement) -> str:
        """Create a new tool based on requirements."""
        # Generate tool creation prompt
        creation_prompt = f"""
        Create a new tool with the following specifications:
        Name: {tool_requirement.name}
        Purpose: {tool_requirement.purpose}
        Requirements: {tool_requirement.requirements}
        Dependencies: {tool_requirement.dependencies}

        Provide the complete implementation including:
        1. Function signature and parameters
        2. Input validation
        3. Error handling
        4. Documentation
        5. Example usage
        """
        
        # Get implementation from LLM
        implementation = await self.llm.generate(creation_prompt)
        
        # Create tool using tool manager
        tool_id = await self.tool_manager.create_tool(
            name=tool_requirement.name,
            implementation=implementation,
            metadata={
                "purpose": tool_requirement.purpose,
                "requirements": tool_requirement.requirements,
                "dependencies": tool_requirement.dependencies
            }
        )
        
        return tool_id

    async def _modify_tool(self, tool_requirement: ToolRequirement) -> str:
        """Modify an existing tool to meet new requirements."""
        # Get existing tool
        existing_tool = await self.tool_manager.get_tool(tool_requirement.name)
        
        # Generate modification prompt
        modification_prompt = f"""
        Modify this existing tool to meet new requirements:

        Existing Tool:
        {existing_tool}

        New Requirements:
        {tool_requirement.requirements}

        Provide the modified implementation maintaining:
        1. Backward compatibility where possible
        2. Existing functionality that's still needed
        3. Updated documentation
        """
        
        # Get modified implementation from LLM
        modified_implementation = await self.llm.generate(modification_prompt)
        
        # Update tool using tool manager
        tool_id = await self.tool_manager.update_tool(
            name=tool_requirement.name,
            implementation=modified_implementation,
            metadata={
                "purpose": tool_requirement.purpose,
                "requirements": tool_requirement.requirements,
                "dependencies": tool_requirement.dependencies
            }
        )
        
        return tool_id

    async def _execute_implementation(self, context: BuildContext):
        """Execute the implementation stage of the build."""
        build_steps = context.project_plan.build_steps
        
        for step in build_steps:
            if step['name'] in context.completed_steps:
                continue
                
            # Check if dependencies are met
            dependencies = step.get('dependencies', [])
            if not all(dep in context.completed_steps for dep in dependencies):
                continue
                
            # Execute build step
            try:
                result = await self._execute_build_step(step, context)
                context.build_artifacts[step['name']] = result
                context.completed_steps.append(step['name'])
            except Exception as e:
                logger.error(f"Failed to execute build step '{step['name']}': {str(e)}")
                raise

    async def _execute_build_step(self, step: Dict[str, Any], context: BuildContext) -> Any:
        """Execute a single build step."""
        # Generate step execution prompt
        execution_prompt = f"""
        Execute this build step:
        Name: {step['name']}
        Description: {step['description']}
        Tools needed: {step['tools needed']}
        Success criteria: {step['success criteria']}

        Previous steps completed: {context.completed_steps}
        Available artifacts: {list(context.build_artifacts.keys())}

        Provide the implementation ensuring:
        1. All success criteria are met
        2. Proper error handling
        3. Integration with previous steps
        4. Documentation of any assumptions or decisions
        """
        
        # Get implementation from LLM
        implementation = await self.llm.generate(execution_prompt)
        
        # Execute implementation using required tools
        tools = step['tools needed'].split(',')
        result = await self._execute_implementation_with_tools(
            implementation, tools, context)
        
        return result

    async def _execute_implementation_with_tools(
        self,
        implementation: str,
        tools: List[str],
        context: BuildContext
    ) -> Any:
        """Execute implementation using specified tools."""
        # Prepare tool instances
        tool_instances = {}
        for tool_name in tools:
            tool = await self.tool_manager.get_tool(tool_name)
            tool_instances[tool_name] = tool
            
        # Create execution context
        execution_context = {
            'tools': tool_instances,
            'artifacts': context.build_artifacts,
            'knowledge_base': self.knowledge_manager
        }
        
        # Execute implementation
        # This is a simplified version - in reality, you'd need a proper
        # execution environment
        result = await self._safe_execute(implementation, execution_context)
        
        return result

    async def _safe_execute(self, code: str, context: Dict[str, Any]) -> Any:
        """Safely execute code in a controlled environment."""
        # This is a placeholder - you'd need proper sandboxing
        # and security measures in a real implementation
        try:
            # Create a safe execution environment
            globals_dict = {
                'tools': context['tools'],
                'artifacts': context['artifacts'],
                'knowledge_base': context['knowledge_base'],
                'asyncio': asyncio
            }
            
            # Execute the code
            exec(code, globals_dict)
            
            # Return the result if available
            return globals_dict.get('result')
            
        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
            raise

    async def _execute_integration(self, context: BuildContext):
        """Execute the integration stage of the build."""
        # Generate integration plan
        integration_plan = await self._generate_integration_plan(context)
        
        # Execute integration steps
        for step in integration_plan:
            await self._execute_integration_step(step, context)
            
        # Verify integration
        await self._verify_integration(context)

    async def _execute_testing(self, context: BuildContext):
        """Execute the testing stage of the build."""
        # Generate test plan
        test_plan = await self._generate_test_plan(context)
        
        # Execute tests
        test_results = await self._execute_tests(test_plan, context)
        
        # Analyze results
        await self._analyze_test_results(test_results, context)

    async def _execute_deployment(self, context: BuildContext):
        """Execute the deployment stage of the build."""
        # Generate deployment plan
        deployment_plan = await self._generate_deployment_plan(context)
        
        # Execute deployment steps
        for step in deployment_plan:
            await self._execute_deployment_step(step, context)
            
        # Verify deployment
        await self._verify_deployment(context)

    async def get_build_status(self, project_id: str) -> Optional[BuildContext]:
        """Get the current status of a build."""
        return self.current_builds.get(project_id)

    async def pause_build(self, project_id: str):
        """Pause a running build."""
        if project_id in self.current_builds:
            context = self.current_builds[project_id]
            context.status = BuildStatus.PAUSED
            # Implement pause logic

    async def resume_build(self, project_id: str):
        """Resume a paused build."""
        if project_id in self.current_builds:
            context = self.current_builds[project_id]
            if context.status == BuildStatus.PAUSED:
                context.status = BuildStatus.IN_PROGRESS
                # Implement resume logic
