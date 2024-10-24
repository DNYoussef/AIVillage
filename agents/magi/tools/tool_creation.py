"""Dynamic tool creation system for MAGI."""

from typing import Dict, Any, List, Optional, Callable, Union
import ast
import inspect
import textwrap
import logging
from dataclasses import dataclass
from datetime import datetime

from ..core.exceptions import ToolError, ToolCreationError
from ..utils.logging import setup_logger
from .tool_management import ToolManager

logger = setup_logger(__name__)

@dataclass
class ToolTemplate:
    """Template for tool creation."""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    returns: Dict[str, Any]
    code_template: str
    validation_rules: List[Dict[str, Any]]
    required_imports: List[str]

class ToolCreator:
    """
    Dynamic tool creation system.
    
    Responsibilities:
    - Tool code generation
    - Code validation and safety checks
    - Tool testing and verification
    - Tool registration with ToolManager
    """
    
    def __init__(self, tool_manager: ToolManager):
        """
        Initialize tool creator.
        
        Args:
            tool_manager: Tool management system
        """
        self.tool_manager = tool_manager
        self.templates: Dict[str, ToolTemplate] = self._load_templates()
    
    async def create_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        returns: Dict[str, Any],
        code: str,
        template_name: Optional[str] = None,
        auto_register: bool = True
    ) -> Callable:
        """
        Create a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters specification
            returns: Tool return type specification
            code: Tool implementation code
            template_name: Name of template to use (optional)
            auto_register: Whether to automatically register the tool
            
        Returns:
            Created tool function
        """
        # Apply template if specified
        if template_name:
            if template_name not in self.templates:
                raise ToolCreationError(f"Template '{template_name}' not found")
            code = self._apply_template(self.templates[template_name], code)
        
        # Validate code
        self._validate_code(code)
        
        # Create function
        tool_fn = self._create_function(name, code)
        
        # Validate function signature matches parameters
        self._validate_signature(tool_fn, parameters)
        
        # Test the function
        await self._test_tool(tool_fn, parameters)
        
        # Register if requested
        if auto_register:
            self.tool_manager.register_tool(
                name=name,
                tool_fn=tool_fn,
                description=description,
                parameters=parameters,
                returns=returns
            )
        
        return tool_fn
    
    def add_template(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        returns: Dict[str, Any],
        code_template: str,
        validation_rules: List[Dict[str, Any]],
        required_imports: List[str]
    ) -> None:
        """Add a new tool template."""
        if name in self.templates:
            raise ToolError(f"Template '{name}' already exists")
        
        template = ToolTemplate(
            name=name,
            description=description,
            parameters=parameters,
            returns=returns,
            code_template=code_template,
            validation_rules=validation_rules,
            required_imports=required_imports
        )
        
        self.templates[name] = template
        logger.info(f"Added template '{name}'")
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates."""
        return [
            {
                'name': template.name,
                'description': template.description,
                'parameters': template.parameters,
                'returns': template.returns
            }
            for template in self.templates.values()
        ]
    
    def _load_templates(self) -> Dict[str, ToolTemplate]:
        """Load built-in templates."""
        templates = {}
        
        # Basic async function template
        templates['async_function'] = ToolTemplate(
            name='async_function',
            description='Basic asynchronous function template',
            parameters={
                'input_params': {
                    'type': 'Dict[str, Any]',
                    'description': 'Input parameters'
                }
            },
            returns={
                'type': 'Any',
                'description': 'Function result'
            },
            code_template=textwrap.dedent("""
                async def {function_name}({param_list}):
                    \"\"\"
                    {docstring}
                    \"\"\"
                    {code}
            """).strip(),
            validation_rules=[
                {'type': 'syntax'},
                {'type': 'async_compatible'},
                {'type': 'no_system_calls'}
            ],
            required_imports=['asyncio']
        )
        
        # Data processing template
        templates['data_processor'] = ToolTemplate(
            name='data_processor',
            description='Template for data processing functions',
            parameters={
                'data': {
                    'type': 'Union[List, Dict, pd.DataFrame]',
                    'description': 'Input data'
                },
                'options': {
                    'type': 'Dict[str, Any]',
                    'description': 'Processing options'
                }
            },
            returns={
                'type': 'Dict[str, Any]',
                'description': 'Processed data and metadata'
            },
            code_template=textwrap.dedent("""
                async def {function_name}(data: Union[List, Dict, pd.DataFrame], options: Dict[str, Any]) -> Dict[str, Any]:
                    \"\"\"
                    {docstring}
                    \"\"\"
                    # Validate input
                    if not isinstance(data, (list, dict, pd.DataFrame)):
                        raise ValueError("Invalid data type")
                    
                    # Process data
                    {code}
                    
                    # Return results
                    return {
                        'result': processed_data,
                        'metadata': {
                            'input_type': type(data).__name__,
                            'output_type': type(processed_data).__name__,
                            'options_used': options
                        }
                    }
            """).strip(),
            validation_rules=[
                {'type': 'syntax'},
                {'type': 'async_compatible'},
                {'type': 'no_system_calls'},
                {'type': 'data_validation'}
            ],
            required_imports=['pandas as pd', 'typing']
        )
        
        return templates
    
    def _apply_template(self, template: ToolTemplate, code: str) -> str:
        """Apply template to code."""
        # Add required imports
        imports = '\n'.join(f"import {imp}" for imp in template.required_imports)
        
        # Format template
        formatted_code = template.code_template.format(
            code=textwrap.indent(code, ' ' * 4)
        )
        
        return f"{imports}\n\n{formatted_code}"
    
    def _validate_code(self, code: str) -> None:
        """
        Validate tool code.
        
        Checks:
        1. Syntax validity
        2. No dangerous operations
        3. Proper async/await usage
        4. Resource usage constraints
        """
        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ToolCreationError(f"Invalid syntax: {str(e)}")
        
        # Check for dangerous operations
        dangerous_calls = self._find_dangerous_calls(tree)
        if dangerous_calls:
            raise ToolCreationError(
                f"Code contains dangerous operations: {', '.join(dangerous_calls)}"
            )
        
        # Check async/await usage
        if 'async def' in code:
            self._validate_async_code(tree)
        
        # Additional checks could be added here
    
    def _find_dangerous_calls(self, tree: ast.AST) -> List[str]:
        """Find potentially dangerous function calls."""
        dangerous_calls = []
        
        class DangerousFunctionVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for system/os operations
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'system']:
                        dangerous_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'spawn']:
                        dangerous_calls.append(f"{node.func.value.id}.{node.func.attr}")
                self.generic_visit(node)
        
        DangerousFunctionVisitor().visit(tree)
        return dangerous_calls
    
    def _validate_async_code(self, tree: ast.AST) -> None:
        """Validate async/await usage."""
        # Check for proper async/await usage
        async_functions = []
        await_calls = []
        
        class AsyncValidator(ast.NodeVisitor):
            def visit_AsyncFunctionDef(self, node):
                async_functions.append(node.name)
                self.generic_visit(node)
            
            def visit_Await(self, node):
                await_calls.append(node)
                self.generic_visit(node)
        
        validator = AsyncValidator()
        validator.visit(tree)
        
        if not async_functions:
            raise ToolCreationError("No async functions found in async code")
        
        if not await_calls:
            logger.warning("Async function contains no await expressions")
    
    def _create_function(self, name: str, code: str) -> Callable:
        """Create function from code."""
        # Create namespace for function
        namespace = {}
        
        # Execute code in namespace
        try:
            exec(code, namespace)
        except Exception as e:
            raise ToolCreationError(f"Error creating function: {str(e)}")
        
        # Get function from namespace
        if name not in namespace:
            raise ToolCreationError(f"Function '{name}' not found in code")
        
        return namespace[name]
    
    def _validate_signature(
        self,
        func: Callable,
        parameters: Dict[str, Dict[str, Any]]
    ) -> None:
        """Validate function signature matches parameters."""
        sig = inspect.signature(func)
        
        # Check parameter names
        param_names = set(parameters.keys())
        sig_names = set(sig.parameters.keys())
        if param_names != sig_names:
            raise ToolCreationError(
                f"Function signature {sig_names} doesn't match "
                f"parameters specification {param_names}"
            )
        
        # Check parameter types (if type hints are present)
        for name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                expected_type = parameters[name]['type']
                if str(param.annotation) != expected_type:
                    raise ToolCreationError(
                        f"Parameter '{name}' type hint {param.annotation} "
                        f"doesn't match specified type {expected_type}"
                    )
    
    async def _test_tool(
        self,
        func: Callable,
        parameters: Dict[str, Dict[str, Any]]
    ) -> None:
        """Test tool with sample inputs."""
        # Generate test inputs
        test_inputs = self._generate_test_inputs(parameters)
        
        # Run tests
        for inputs in test_inputs:
            try:
                if inspect.iscoroutinefunction(func):
                    await func(**inputs)
                else:
                    func(**inputs)
            except Exception as e:
                raise ToolCreationError(
                    f"Tool failed testing with inputs {inputs}: {str(e)}"
                )
    
    def _generate_test_inputs(
        self,
        parameters: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate test inputs for parameters."""
        # Generate basic test cases
        test_inputs = [{}]  # Start with empty input if no required params
        
        for name, info in parameters.items():
            param_type = info['type']
            required = info.get('required', True)
            
            # Generate sample value based on type
            if param_type == 'int':
                value = 0
            elif param_type == 'float':
                value = 0.0
            elif param_type == 'str':
                value = "test"
            elif param_type == 'bool':
                value = True
            elif param_type.startswith('List'):
                value = []
            elif param_type.startswith('Dict'):
                value = {}
            else:
                value = None
            
            if required:
                # Add value to all existing test cases
                for inputs in test_inputs:
                    inputs[name] = value
            else:
                # Create new test cases with and without optional param
                new_inputs = []
                for inputs in test_inputs:
                    with_param = inputs.copy()
                    with_param[name] = value
                    new_inputs.append(with_param)
                test_inputs.extend(new_inputs)
        
        return test_inputs

# Example usage
if __name__ == "__main__":
    async def main():
        # Create tool manager and creator
        manager = ToolManager("tools")
        creator = ToolCreator(manager)
        
        # Create a tool using the async_function template
        code = """
        result = a + b
        return result
        """
        
        tool = await creator.create_tool(
            name="add_numbers",
            description="Add two numbers together",
            parameters={
                'a': {'type': 'int', 'description': 'First number'},
                'b': {'type': 'int', 'description': 'Second number'}
            },
            returns={'type': 'int', 'description': 'Sum of the numbers'},
            code=code,
            template_name='async_function'
        )
        
        # Test the created tool
        result = await manager.execute_tool("add_numbers", a=5, b=3)
        print(f"Result: {result.result}")
    
    import asyncio
    asyncio.run(main())
