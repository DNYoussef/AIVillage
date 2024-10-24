"""Dynamic tool creation functionality for MAGI."""

from typing import Dict, Any, List, Optional
import ast
import inspect
import logging
import json
from datetime import datetime
from ..core.exceptions import ToolError, ToolCreationError
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ToolCreator:
    """
    Handles dynamic creation and validation of tools.
    
    Capabilities:
    - Tool code generation and validation
    - Hypothesis testing and falsification
    - Self-reflection and improvement
    - Continuous learning integration
    - Safety and security checks
    """
    
    def __init__(self, llm=None, continuous_learner=None):
        """
        Initialize tool creator.
        
        Args:
            llm: Language model for code generation
            continuous_learner: Continuous learning component
        """
        self.llm = llm
        self.continuous_learner = continuous_learner
        self.restricted_imports = {
            'os', 'sys', 'subprocess', 'eval', 'exec'
        }
        self.creation_history: List[Dict[str, Any]] = []
        self.hypotheses: List[Dict[str, Any]] = []
    
    async def create_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: Tool parameters specification
            code: Tool implementation code (optional)
            
        Returns:
            Created tool data
        """
        # Generate hypotheses about tool behavior
        hypotheses = await self._generate_hypotheses(name, description, parameters)
        
        # Generate or validate code
        if code is None:
            code = await self._generate_code(name, description, parameters, hypotheses)
        
        # Validate code
        self._validate_code(code)
        
        # Create function
        tool_fn = self._create_function(name, code)
        
        # Validate function signature matches parameters
        self._validate_signature(tool_fn, parameters)
        
        # Test hypotheses
        test_results = await self._test_hypotheses(tool_fn, parameters, hypotheses)
        
        # Perform self-reflection
        reflection = await self._reflect_on_creation(
            name, description, parameters, code, test_results
        )
        
        # Update continuous learning
        if self.continuous_learner:
            await self.continuous_learner.learn_from_tool_creation(
                name, code, description, parameters
            )
        
        # Record creation
        self.creation_history.append({
            'name': name,
            'description': description,
            'parameters': parameters,
            'code': code,
            'hypotheses': hypotheses,
            'test_results': test_results,
            'reflection': reflection,
            'timestamp': datetime.now()
        })
        
        return {
            'name': name,
            'code': code,
            'description': description,
            'parameters': parameters,
            'function': tool_fn,
            'hypotheses': hypotheses,
            'test_results': test_results,
            'reflection': reflection
        }
    
    async def _generate_hypotheses(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate testable hypotheses about tool behavior."""
        if not self.llm:
            return []
        
        prompt = f"""
        Generate testable hypotheses about the behavior of this tool:
        
        Name: {name}
        Description: {description}
        Parameters: {json.dumps(parameters, indent=2)}
        
        For each hypothesis, provide:
        1. The hypothesis statement
        2. Expected behavior
        3. Test cases to verify
        4. Potential edge cases
        5. Success criteria
        
        Format as JSON list of dictionaries.
        """
        
        response = await self.llm.complete(prompt)
        return json.loads(response.text)
    
    async def _generate_code(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        hypotheses: List[Dict[str, Any]]
    ) -> str:
        """Generate code for tool implementation."""
        if not self.llm:
            raise ToolCreationError("LLM required for code generation")
        
        prompt = f"""
        Generate Python code for a tool with:
        
        Name: {name}
        Description: {description}
        Parameters: {json.dumps(parameters, indent=2)}
        
        The code should:
        1. Handle all specified parameters
        2. Include input validation
        3. Handle errors gracefully
        4. Be well-documented
        5. Follow Python best practices
        
        Consider these hypotheses about the tool's behavior:
        {json.dumps(hypotheses, indent=2)}
        
        Return only the code, no explanations.
        """
        
        response = await self.llm.complete(prompt)
        return response.text.strip()
    
    async def _test_hypotheses(
        self,
        func: Any,
        parameters: Dict[str, Any],
        hypotheses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Test hypotheses about tool behavior."""
        results = []
        
        for hypothesis in hypotheses:
            test_cases = hypothesis['test_cases']
            success_criteria = hypothesis['success_criteria']
            
            test_results = []
            for test_case in test_cases:
                try:
                    if inspect.iscoroutinefunction(func):
                        result = await func(**test_case['inputs'])
                    else:
                        result = func(**test_case['inputs'])
                    
                    success = self._evaluate_success(
                        result,
                        test_case['expected'],
                        success_criteria
                    )
                    
                    test_results.append({
                        'inputs': test_case['inputs'],
                        'expected': test_case['expected'],
                        'actual': result,
                        'success': success
                    })
                    
                except Exception as e:
                    test_results.append({
                        'inputs': test_case['inputs'],
                        'expected': test_case['expected'],
                        'error': str(e),
                        'success': False
                    })
            
            results.append({
                'hypothesis': hypothesis['statement'],
                'test_results': test_results,
                'success_rate': sum(r['success'] for r in test_results) / len(test_results)
            })
        
        return results
    
    def _evaluate_success(
        self,
        actual: Any,
        expected: Any,
        criteria: Dict[str, Any]
    ) -> bool:
        """Evaluate if a result meets success criteria."""
        # Implement success criteria evaluation
        # This is a placeholder - implement actual evaluation logic
        return actual == expected
    
    async def _reflect_on_creation(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        code: str,
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform self-reflection on tool creation."""
        if not self.llm:
            return {}
        
        prompt = f"""
        Reflect on the creation of this tool:
        
        Name: {name}
        Description: {description}
        Parameters: {json.dumps(parameters, indent=2)}
        Code: {code}
        Test Results: {json.dumps(test_results, indent=2)}
        
        Provide:
        1. Strengths and weaknesses of the implementation
        2. Potential improvements
        3. Lessons learned
        4. Recommendations for future tool creation
        
        Format as JSON.
        """
        
        response = await self.llm.complete(prompt)
        return json.loads(response.text)
    
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
    
    def _find_dangerous_calls(self, tree: ast.AST) -> List[str]:
        """Find potentially dangerous function calls."""
        dangerous_calls = []
        
        class DangerousFunctionVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for name in node.names:
                    if name.name in self.restricted_imports:
                        dangerous_calls.append(f"import {name.name}")
            
            def visit_ImportFrom(self, node):
                if node.module in self.restricted_imports:
                    dangerous_calls.append(f"from {node.module}")
            
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        dangerous_calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'spawn']:
                        dangerous_calls.append(f"{node.func.value.id}.{node.func.attr}")
        
        DangerousFunctionVisitor().visit(tree)
        return dangerous_calls
    
    def _validate_async_code(self, tree: ast.AST) -> None:
        """Validate async/await usage."""
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
    
    def _create_function(self, name: str, code: str) -> Any:
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
        func: Any,
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
    
    async def get_creation_insights(self) -> Dict[str, Any]:
        """Get insights from tool creation history."""
        if not self.llm or not self.creation_history:
            return {}
        
        prompt = f"""
        Analyze this tool creation history:
        {json.dumps(self.creation_history, indent=2)}
        
        Provide insights on:
        1. Common patterns in successful tools
        2. Frequent issues and their solutions
        3. Areas for improvement in the creation process
        4. Recommendations for future tool creation
        
        Format as JSON.
        """
        
        response = await self.llm.complete(prompt)
        return json.loads(response.text)
