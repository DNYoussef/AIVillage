"""Enhanced Magi agent with improved code generation and experimentation capabilities."""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import asyncio
import json
import re
from difflib import SequenceMatcher

from config.unified_config import UnifiedConfig, AgentConfig
from ..openrouter_agent import OpenRouterAgent, AgentInteraction
from ..local_agent import LocalAgent
from ...data.complexity_evaluator import ComplexityEvaluator

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Manages code generation and validation."""
    
    def __init__(self):
        self.generation_history: List[Dict[str, Any]] = []
        self.metrics = {
            "success_rate": 1.0,
            "average_quality": 0.0,
            "test_coverage": 0.0,
            "optimization_score": 0.0
        }
    
    async def generate_code(self,
                          task: str,
                          language: str,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate code with validation and documentation."""
        # Parse requirements
        requirements = self._parse_requirements(task)
        
        # Generate code structure
        structure = self._plan_code_structure(requirements, language)
        
        # Prepare generation context
        generation_context = {
            "task": task,
            "language": language,
            "requirements": requirements,
            "structure": structure,
            **(context or {})
        }
        
        return {
            "context": generation_context,
            "structure": structure,
            "requirements": requirements
        }
    
    def _parse_requirements(self, task: str) -> Dict[str, List[str]]:
        """Parse requirements from task description."""
        requirements = {
            "functional": [],
            "non_functional": [],
            "constraints": []
        }
        
        # Extract functional requirements
        func_patterns = [
            r"must\s+([^.]+)",
            r"should\s+([^.]+)",
            r"needs? to\s+([^.]+)"
        ]
        for pattern in func_patterns:
            matches = re.finditer(pattern, task, re.IGNORECASE)
            requirements["functional"].extend(match.group(1).strip() for match in matches)
        
        # Extract non-functional requirements
        nonfunc_patterns = [
            r"performance[:\s]+([^.]+)",
            r"security[:\s]+([^.]+)",
            r"scalability[:\s]+([^.]+)"
        ]
        for pattern in nonfunc_patterns:
            matches = re.finditer(pattern, task, re.IGNORECASE)
            requirements["non_functional"].extend(match.group(1).strip() for match in matches)
        
        # Extract constraints
        constraint_patterns = [
            r"must not\s+([^.]+)",
            r"should not\s+([^.]+)",
            r"cannot\s+([^.]+)"
        ]
        for pattern in constraint_patterns:
            matches = re.finditer(pattern, task, re.IGNORECASE)
            requirements["constraints"].extend(match.group(1).strip() for match in matches)
        
        return requirements
    
    def _plan_code_structure(self, 
                           requirements: Dict[str, List[str]],
                           language: str) -> Dict[str, Any]:
        """Plan code structure based on requirements."""
        structure = {
            "components": [],
            "interfaces": [],
            "dependencies": [],
            "tests": []
        }
        
        # Extract components from functional requirements
        for req in requirements["functional"]:
            component = self._extract_component(req)
            if component:
                structure["components"].append(component)
        
        # Plan interfaces
        structure["interfaces"] = self._plan_interfaces(structure["components"])
        
        # Identify dependencies
        structure["dependencies"] = self._identify_dependencies(
            structure["components"],
            structure["interfaces"]
        )
        
        # Plan tests
        structure["tests"] = self._plan_tests(requirements["functional"])
        
        return structure
    
    def _extract_component(self, requirement: str) -> Optional[Dict[str, Any]]:
        """Extract component information from requirement."""
        # This would be more sophisticated in practice
        words = requirement.lower().split()
        
        # Look for common component indicators
        component_types = {
            "class": ["class", "object", "type"],
            "function": ["function", "method", "routine"],
            "interface": ["interface", "api", "contract"],
            "module": ["module", "package", "library"]
        }
        
        for word in words:
            for comp_type, indicators in component_types.items():
                if word in indicators:
                    return {
                        "type": comp_type,
                        "name": self._generate_name(requirement),
                        "requirement": requirement
                    }
        
        return None
    
    def _generate_name(self, requirement: str) -> str:
        """Generate appropriate name from requirement."""
        # This would be more sophisticated in practice
        words = re.findall(r'\w+', requirement.lower())
        return "".join(w.capitalize() for w in words[:3])
    
    def _plan_interfaces(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Plan interfaces between components."""
        interfaces = []
        
        for i, comp1 in enumerate(components):
            for comp2 in components[i+1:]:
                if self._should_interface(comp1, comp2):
                    interfaces.append({
                        "name": f"{comp1['name']}To{comp2['name']}Interface",
                        "source": comp1["name"],
                        "target": comp2["name"],
                        "methods": self._plan_interface_methods(comp1, comp2)
                    })
        
        return interfaces
    
    def _should_interface(self, comp1: Dict[str, Any], comp2: Dict[str, Any]) -> bool:
        """Determine if two components should have an interface."""
        # This would be more sophisticated in practice
        return comp1["type"] != comp2["type"]
    
    def _plan_interface_methods(self,
                              comp1: Dict[str, Any],
                              comp2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan methods for an interface between components."""
        # This would be more sophisticated in practice
        return [
            {
                "name": f"connect{comp1['name']}To{comp2['name']}",
                "parameters": ["context"],
                "return_type": "void"
            }
        ]
    
    def _identify_dependencies(self,
                             components: List[Dict[str, Any]],
                             interfaces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify dependencies between components."""
        dependencies = []
        
        for interface in interfaces:
            dependencies.append({
                "source": interface["source"],
                "target": interface["target"],
                "type": "uses",
                "through": interface["name"]
            })
        
        return dependencies
    
    def _plan_tests(self, requirements: List[str]) -> List[Dict[str, Any]]:
        """Plan tests based on requirements."""
        tests = []
        
        for req in requirements:
            tests.append({
                "name": f"test_{self._generate_test_name(req)}",
                "requirement": req,
                "type": "unit",
                "assertions": self._plan_test_assertions(req)
            })
        
        return tests
    
    def _generate_test_name(self, requirement: str) -> str:
        """Generate appropriate test name from requirement."""
        # This would be more sophisticated in practice
        words = re.findall(r'\w+', requirement.lower())
        return "_".join(words[:3])
    
    def _plan_test_assertions(self, requirement: str) -> List[Dict[str, Any]]:
        """Plan test assertions for a requirement."""
        # This would be more sophisticated in practice
        return [
            {
                "type": "equality",
                "description": f"Verify {requirement}",
                "expected": "success"
            }
        ]

class ExperimentManager:
    """Manages code experimentation and validation."""
    
    def __init__(self):
        self.experiments: List[Dict[str, Any]] = []
        self.metrics = {
            "success_rate": 1.0,
            "average_performance": 0.0,
            "coverage": 0.0
        }
    
    async def validate_code(self,
                          code: str,
                          language: str,
                          requirements: Dict[str, List[str]]) -> Dict[str, Any]:
        """Validate generated code against requirements."""
        validation_results = {
            "passes_syntax": self._check_syntax(code, language),
            "meets_requirements": self._check_requirements(code, requirements),
            "test_results": await self._run_tests(code, language),
            "metrics": self._calculate_metrics(code, language)
        }
        
        return validation_results
    
    def _check_syntax(self, code: str, language: str) -> bool:
        """Check code syntax."""
        # This would use actual syntax checking in practice
        try:
            if language == "python":
                compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False
    
    def _check_requirements(self,
                          code: str,
                          requirements: Dict[str, List[str]]) -> Dict[str, bool]:
        """Check if code meets requirements."""
        results = {}
        
        for req_type, reqs in requirements.items():
            for req in reqs:
                # This would be more sophisticated in practice
                results[req] = self._requirement_implemented(code, req)
        
        return results
    
    def _requirement_implemented(self, code: str, requirement: str) -> bool:
        """Check if a specific requirement is implemented."""
        # This would be more sophisticated in practice
        keywords = re.findall(r'\w+', requirement.lower())
        return all(keyword in code.lower() for keyword in keywords)
    
    async def _run_tests(self, code: str, language: str) -> Dict[str, Any]:
        """Run tests on the code."""
        # This would run actual tests in practice
        return {
            "total_tests": 1,
            "passed_tests": 1,
            "coverage": 100.0
        }
    
    def _calculate_metrics(self, code: str, language: str) -> Dict[str, float]:
        """Calculate code quality metrics."""
        return {
            "complexity": len(code.split('\n')) / 100,  # Simplified
            "maintainability": 0.8,
            "performance": 0.9
        }

class MagiAgent:
    """
    Enhanced Magi agent specializing in code generation and experimentation.
    Uses openai/o1-mini-2024-09-12 as frontier model and
    ibm-granite/granite-3b-code-instruct-128k as local model.
    """
    
    def __init__(self,
                 openrouter_agent: OpenRouterAgent,
                 config: UnifiedConfig):
        """
        Initialize enhanced MagiAgent.
        
        Args:
            openrouter_agent: OpenRouterAgent instance
            config: UnifiedConfig instance
        """
        self.config = config
        self.agent_config = config.get_agent_config("magi")
        self.frontier_agent = openrouter_agent
        self.local_agent = LocalAgent(
            model_config=self.agent_config.local_model,
            config=config
        )
        
        # Initialize support systems
        self.code_generator = CodeGenerator()
        self.experiment_manager = ExperimentManager()
        self.complexity_evaluator = ComplexityEvaluator(config)
        
        # Code history tracking (preserved from original)
        self.code_history: List[Dict[str, Any]] = []
        
        # Performance tracking (preserved from original with enhancements)
        self.performance_metrics: Dict[str, float] = {
            "code_quality": 0.0,
            "solution_efficiency": 0.0,
            "test_coverage": 0.0,
            "documentation_quality": 0.0,
            "local_model_performance": 0.0
        }
        
        logger.info(f"Initialized MagiAgent with:")
        logger.info(f"  Frontier model: {openrouter_agent.model}")
        logger.info(f"  Local model: {openrouter_agent.local_model}")
    
    async def generate_code(self,
                          task: str,
                          context: Optional[str] = None,
                          language: Optional[str] = None,
                          requirements: Optional[List[str]] = None) -> AgentInteraction:
        """
        Generate code with validation and optimization.
        
        Args:
            task: Code generation task
            context: Optional codebase context
            language: Target programming language (defaults to python)
            requirements: Optional specific requirements
            
        Returns:
            AgentInteraction containing the generated code
        """
        start_time = time.time()
        language = language or "python"  # Default to python if not specified
        
        try:
            # Evaluate task complexity (preserved from original)
            complexity_evaluation = await self.complexity_evaluator.evaluate_complexity(
                agent_type="magi",
                task=f"Code generation: {task}",
                context={"language": language, "requirements": requirements}
            )
            
            # Plan code generation (enhanced)
            generation_plan = await self.code_generator.generate_code(
                task=task,
                language=language,
                context={"codebase_context": context} if context else None
            )
            
            # Try local model first if task isn't complex (preserved from original)
            local_response = None
            if not complexity_evaluation["is_complex"]:
                try:
                    local_response = await self.local_agent.generate_response(
                        prompt=self._construct_coding_prompt(task, context, language, requirements),
                        system_prompt=self._get_coding_system_prompt(language),
                        temperature=0.2,
                        max_tokens=1000
                    )
                except Exception as e:
                    logger.warning(f"Local model code generation failed: {str(e)}")
            
            # Use frontier model if task is complex or local model failed
            if complexity_evaluation["is_complex"] or not local_response:
                interaction = await self.frontier_agent.generate_response(
                    prompt=self._construct_coding_prompt(task, context, language, requirements),
                    system_prompt=self._get_coding_system_prompt(language),
                    temperature=0.2,
                    max_tokens=1500
                )
                
                # If we tried local model, record performance comparison (preserved from original)
                if local_response:
                    self._record_model_comparison(local_response, interaction)
            else:
                # Convert local response to AgentInteraction format (preserved from original)
                interaction = AgentInteraction(
                    prompt=task,
                    response=local_response["response"],
                    model=local_response["model"],
                    timestamp=time.time(),
                    metadata={
                        **local_response["metadata"],
                        "generation_plan": generation_plan
                    }
                )
            
            # Extract code from response
            code = self._extract_code(interaction.response)
            
            # Validate code (enhanced)
            validation_results = await self.experiment_manager.validate_code(
                code=code,
                language=language,
                requirements=generation_plan["requirements"]
            )
            
            # Add validation results to interaction metadata
            interaction.metadata["validation_results"] = validation_results
            
            # Update code history (preserved from original)
            self._update_code_history(
                task=task,
                interaction=interaction,
                was_complex=complexity_evaluation["is_complex"],
                language=language
            )
            
            # Update performance metrics
            duration = time.time() - start_time
            self._update_metrics(interaction, {
                "duration": duration,
                "complexity_score": complexity_evaluation["complexity_score"],
                "validation_results": validation_results
            })
            
            return interaction
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            raise
    
    def _construct_coding_prompt(self, 
                               task: str, 
                               context: Optional[str],
                               language: Optional[str],
                               requirements: Optional[List[str]]) -> str:
        """Construct detailed coding prompt (preserved from original)."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Code Context:\n{context}\n")
            
        prompt_parts.append(f"Task Description:\n{task}\n")
        
        if language:
            prompt_parts.append(f"Programming Language: {language}\n")
            
        if requirements:
            prompt_parts.append("Requirements:")
            for req in requirements:
                prompt_parts.append(f"- {req}")
        
        prompt_parts.append("""
        Please provide:
        1. Code implementation
        2. Brief explanation of the approach
        3. Any important considerations or assumptions
        4. Example usage (if applicable)
        """)
        
        return "\n\n".join(prompt_parts)
    
    def _get_coding_system_prompt(self, language: Optional[str]) -> str:
        """Get appropriate system prompt (preserved from original)."""
        base_prompt = """You are Magi, an expert coding AI specializing in software development and technical problem-solving.
        Your approach:
        1. Write clean, efficient, and maintainable code
        2. Follow best practices and design patterns
        3. Consider edge cases and error handling
        4. Provide clear documentation and explanations
        5. Optimize for performance where appropriate"""
        
        if language:
            language_specific = f"\nYou are writing code in {language}. Follow {language}-specific best practices and conventions."
            return base_prompt + language_specific
            
        return base_prompt
    
    def _extract_code(self, response: str) -> str:
        """Extract code blocks from response."""
        # Look for code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        
        if code_blocks:
            # Return the largest code block
            return max(code_blocks, key=len).strip()
        
        # If no code blocks found, try to extract indented code
        lines = response.split('\n')
        code_lines = [line for line in lines if line.strip().startswith('    ')]
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # If no code found, return original response
        return response
    
    def _update_code_history(self, 
                           task: str, 
                           interaction: AgentInteraction,
                           was_complex: bool,
                           language: Optional[str]):
        """Update code history (preserved from original)."""
        code_record = {
            "task": task,
            "timestamp": interaction.timestamp,
            "was_complex": was_complex,
            "language": language,
            "model_used": interaction.model,
            "tokens_used": interaction.metadata["usage"]["total_tokens"] if "usage" in interaction.metadata else 0
        }
        self.code_history.append(code_record)
        
        # Keep last 100 coding tasks
        if len(self.code_history) > 100:
            self.code_history = self.code_history[-100:]
    
    def _record_model_comparison(self, local_response: Dict[str, Any], frontier_interaction: AgentInteraction):
        """Record model comparison (preserved from original)."""
        similarity = SequenceMatcher(
            None, 
            local_response["response"], 
            frontier_interaction.response
        ).ratio()
        
        self.local_agent.record_performance({
            "code_similarity": similarity,
            "was_used": similarity > 0.8
        })
    
    def _update_metrics(self,
                       interaction: AgentInteraction,
                       performance: Dict[str, Any]):
        """Update comprehensive performance metrics."""
        validation_results = performance.get("validation_results", {})
        
        # Update code quality (preserved from original)
        if "metrics" in validation_results:
            metrics = validation_results["metrics"]
            self.performance_metrics["code_quality"] = metrics.get("maintainability", 0)
        
        # Update solution efficiency (preserved from original)
        if "metrics" in validation_results:
            self.performance_metrics["solution_efficiency"] = validation_results["metrics"].get("performance", 0)
        
        # Update test coverage (preserved from original)
        if "test_results" in validation_results:
            self.performance_metrics["test_coverage"] = validation_results["test_results"].get("coverage", 0)
        
        # Update documentation quality (preserved from original)
        self.performance_metrics["documentation_quality"] = self._evaluate_documentation(interaction.response)
        
        # Update local model performance (preserved from original)
        local_metrics = self.local_agent.get_performance_metrics()
        if "code_similarity" in local_metrics:
            self.performance_metrics["local_model_performance"] = local_metrics["code_similarity"]
    
    def _evaluate_documentation(self, response: str) -> float:
        """Evaluate documentation quality."""
        # This would be more sophisticated in practice
        has_comments = '"""' in response or "'''" in response
        has_inline_comments = '#' in response
        has_usage_example = 'Example' in response or 'Usage:' in response
        
        score = 0.0
        if has_comments: score += 0.4
        if has_inline_comments: score += 0.3
        if has_usage_example: score += 0.3
        
        return score
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics (preserved from original)."""
        metrics = self.performance_metrics.copy()
        
        # Add code generator metrics
        metrics.update(self.code_generator.metrics)
        
        # Add experiment manager metrics
        metrics.update({
            f"experiment_{k}": v 
            for k, v in self.experiment_manager.metrics.items()
        })
        
        # Add local model metrics
        local_metrics = self.local_agent.get_performance_metrics()
        metrics.update({
            f"local_{k}": v 
            for k, v in local_metrics.items()
        })
        
        return metrics
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get training data (preserved from original)."""
        return self.frontier_agent.get_training_data()
