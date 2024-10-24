"""Tool management functionality for MAGI."""

from typing import Dict, Any, List
import logging
from .tool_persistence import ToolPersistence
from .tool_creator import ToolCreator

logger = logging.getLogger(__name__)

class ToolManager:
    def __init__(self, llm=None, tool_persistence=None, tool_creator=None, continuous_learner=None):
        self.llm = llm
        self.tool_persistence = tool_persistence or ToolPersistence("tools_storage")
        self.tool_creator = tool_creator or ToolCreator(llm=llm, continuous_learner=continuous_learner)
        self.continuous_learner = continuous_learner
        self.tools: Dict[str, Any] = {}
        self.default_tool_parameters = {}

    async def consider_removing_tool(self, tool: str) -> bool:
        """Consider removing a tool based on usage and performance."""
        removal_prompt = f"Consider the implications of removing the '{tool}' tool due to low usage and performance. Provide a recommendation (remove/keep) with a brief explanation."
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
