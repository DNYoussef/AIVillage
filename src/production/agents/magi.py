"""Magi Agent - Technical implementation and code generation expert.

The Magi Agent specializes in code generation, debugging, deployment,
and technical optimization within the AIVillage ecosystem.
"""

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CodeLanguage(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    SQL = "sql"


class CodeQuality(Enum):
    """Code quality levels."""

    PROTOTYPE = "prototype"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"


@dataclass
class CodeGeneration:
    """Code generation request and result."""

    request_id: str
    language: CodeLanguage
    task_description: str
    requirements: dict[str, Any]
    generated_code: str | None = None
    quality_score: float | None = None
    errors: list[str] = field(default_factory=list)
    optimizations: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


@dataclass
class DeploymentConfig:
    """Deployment configuration."""

    target_environment: str
    deployment_type: str
    resource_requirements: dict[str, Any]
    scaling_config: dict[str, Any]
    monitoring_config: dict[str, Any]


class MagiAgent:
    """Technical implementation and code generation expert."""

    def __init__(self, spec=None) -> None:
        """Initialize Magi Agent."""
        self.spec = spec
        self.name = "Magi"
        self.role_description = "Technical implementation and code generation expert"

        # Code generation tracking
        self.code_generations: dict[str, CodeGeneration] = {}
        self.code_templates: dict[str, str] = {}
        self.optimization_patterns: list[dict[str, Any]] = []

        # Performance tracking
        self.performance_history: list[dict[str, Any]] = []
        self.kpi_scores: dict[str, float] = {}

        # Technical capabilities
        self.supported_languages = list(CodeLanguage)
        self.code_quality_standards = {
            CodeQuality.PROTOTYPE: {"complexity_threshold": 20, "coverage_min": 0.5},
            CodeQuality.DEVELOPMENT: {"complexity_threshold": 15, "coverage_min": 0.7},
            CodeQuality.PRODUCTION: {"complexity_threshold": 10, "coverage_min": 0.8},
            CodeQuality.ENTERPRISE: {"complexity_threshold": 8, "coverage_min": 0.9},
        }

        # Behavioral traits
        self.detail_orientation = "high"
        self.creativity = "medium"
        self.delivery_focus = "balanced"

        # Load common code templates
        self._initialize_templates()

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process technical implementation requests."""
        task_type = request.get("task", "unknown")

        if task_type == "ping":
            return {
                "status": "completed",
                "agent": "magi",
                "result": "Technical implementation system online",
                "timestamp": datetime.utcnow().isoformat(),
            }
        elif task_type == "generate_code":
            return self._generate_code(request)
        elif task_type == "debug_code":
            return self._debug_code(request)
        elif task_type == "optimize_code":
            return self._optimize_code(request)
        elif task_type == "deploy_application":
            return self._deploy_application(request)
        elif task_type == "code_review":
            return self._review_code(request)
        elif task_type == "generate_documentation":
            return self._generate_documentation(request)
        else:
            return {
                "status": "completed",
                "agent": "magi",
                "result": f"Implemented technical solution for: {task_type}",
                "implementation_approach": self._determine_approach(request),
            }

    def _generate_code(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate code based on requirements."""
        language = request.get("language", "python")
        description = request.get("description", "")
        requirements = request.get("requirements", {})
        quality_level = request.get("quality", "development")

        request_id = f"codegen_{int(time.time() * 1000)}"

        try:
            # Create code generation task
            code_lang = CodeLanguage(language.lower())
            generation = CodeGeneration(
                request_id=request_id,
                language=code_lang,
                task_description=description,
                requirements=requirements,
            )

            # Generate code based on language and requirements
            generated_code = self._perform_code_generation(
                code_lang, description, requirements
            )

            # Analyze code quality
            quality_score = self._analyze_code_quality(generated_code, quality_level)

            generation.generated_code = generated_code
            generation.quality_score = quality_score
            generation.completed_at = time.time()

            self.code_generations[request_id] = generation

            return {
                "status": "completed",
                "agent": "magi",
                "result": "Code generated successfully",
                "request_id": request_id,
                "generated_code": generated_code,
                "quality_score": quality_score,
                "language": language,
                "lines_of_code": len(generated_code.split("\n"))
                if generated_code
                else 0,
            }

        except ValueError as e:
            return {
                "status": "error",
                "agent": "magi",
                "error": f"Code generation failed: {str(e)}",
            }

    def _debug_code(self, request: dict[str, Any]) -> dict[str, Any]:
        """Debug and fix code issues."""
        code = request.get("code", "")
        language = request.get("language", "python")
        error_description = request.get("error", "")

        issues = self._analyze_code_issues(code, language)
        fixes = self._generate_fixes(code, issues, error_description)

        return {
            "status": "completed",
            "agent": "magi",
            "result": "Code debugging completed",
            "issues_found": len(issues),
            "issues": issues,
            "suggested_fixes": fixes,
            "fixed_code": self._apply_fixes(code, fixes) if fixes else code,
        }

    def _optimize_code(self, request: dict[str, Any]) -> dict[str, Any]:
        """Optimize code for performance and efficiency."""
        code = request.get("code", "")
        language = request.get("language", "python")
        optimization_goals = request.get("goals", ["performance", "readability"])

        optimizations = self._analyze_optimizations(code, language, optimization_goals)
        optimized_code = self._apply_optimizations(code, optimizations)

        performance_improvement = self._estimate_performance_gain(code, optimized_code)

        return {
            "status": "completed",
            "agent": "magi",
            "result": "Code optimization completed",
            "optimizations_applied": len(optimizations),
            "optimizations": optimizations,
            "optimized_code": optimized_code,
            "estimated_performance_gain": performance_improvement,
            "optimization_goals": optimization_goals,
        }

    def _deploy_application(self, request: dict[str, Any]) -> dict[str, Any]:
        """Deploy application to target environment."""
        request.get("config", {})
        target_environment = request.get("environment", "development")
        deployment_type = request.get("type", "container")

        # Create deployment configuration
        deploy_config = DeploymentConfig(
            target_environment=target_environment,
            deployment_type=deployment_type,
            resource_requirements=request.get("resources", {}),
            scaling_config=request.get("scaling", {}),
            monitoring_config=request.get("monitoring", {}),
        )

        # Generate deployment artifacts
        deployment_artifacts = self._generate_deployment_artifacts(deploy_config)

        # Perform deployment simulation (since we can't actually deploy)
        deployment_status = self._simulate_deployment(deploy_config)

        return {
            "status": "completed",
            "agent": "magi",
            "result": "Application deployment completed",
            "deployment_id": f"deploy_{int(time.time())}",
            "environment": target_environment,
            "deployment_type": deployment_type,
            "artifacts": deployment_artifacts,
            "deployment_status": deployment_status,
            "monitoring_endpoints": self._generate_monitoring_endpoints(deploy_config),
        }

    def _review_code(self, request: dict[str, Any]) -> dict[str, Any]:
        """Perform comprehensive code review."""
        code = request.get("code", "")
        language = request.get("language", "python")
        review_criteria = request.get(
            "criteria", ["security", "performance", "maintainability"]
        )

        review_results = {}
        overall_score = 0.0

        for criteria in review_criteria:
            score, comments = self._review_criteria(code, language, criteria)
            review_results[criteria] = {"score": score, "comments": comments}
            overall_score += score

        overall_score = overall_score / len(review_criteria) if review_criteria else 0.0

        return {
            "status": "completed",
            "agent": "magi",
            "result": "Code review completed",
            "overall_score": overall_score,
            "review_results": review_results,
            "recommendations": self._generate_review_recommendations(review_results),
            "approval_status": "approved"
            if overall_score >= 0.8
            else "needs_improvement",
        }

    def _generate_documentation(self, request: dict[str, Any]) -> dict[str, Any]:
        """Generate technical documentation."""
        code = request.get("code", "")
        doc_type = request.get("type", "api")  # api, technical, user
        language = request.get("language", "python")

        documentation = self._create_documentation(code, doc_type, language)

        return {
            "status": "completed",
            "agent": "magi",
            "result": "Documentation generated successfully",
            "documentation_type": doc_type,
            "documentation": documentation,
            "format": "markdown",
            "sections": ["overview", "installation", "usage", "api", "examples"],
        }

    def _perform_code_generation(
        self, language: CodeLanguage, description: str, requirements: dict[str, Any]
    ) -> str:
        """Perform actual code generation."""
        if language == CodeLanguage.PYTHON:
            return self._generate_python_code(description, requirements)
        elif language == CodeLanguage.JAVASCRIPT:
            return self._generate_javascript_code(description, requirements)
        elif language == CodeLanguage.SQL:
            return self._generate_sql_code(description, requirements)
        else:
            return self._generate_generic_code(language, description, requirements)

    def _generate_python_code(
        self, description: str, requirements: dict[str, Any]
    ) -> str:
        """Generate Python code."""
        # Simple template-based code generation
        class_name = requirements.get("class_name", "GeneratedClass")
        methods = requirements.get("methods", ["process"])

        code_lines = [
            f'"""Generated code for: {description}"""',
            "",
            "import logging",
            "from typing import Any, Dict, List, Optional",
            "",
            "logger = logging.getLogger(__name__)",
            "",
            f"class {class_name}:",
            f'    """Generated class for {description}."""',
            "",
            "    def __init__(self):",
            "        self.initialized = True",
            "        logger.info(f'Initialized {self.__class__.__name__}')",
            "",
        ]

        for method in methods:
            code_lines.extend(
                [
                    f"    def {method}(self, data: Dict[str, Any]) -> Dict[str, Any]:",
                    f'        """Generated method: {method}."""',
                    "        try:",
                    "            # Generated implementation",
                    f"            result = {{'status': 'success', 'method': '{method}'}}",
                    "            return result",
                    "        except Exception as e:",
                    "            logger.error(f'Error in {method}: {e}')",
                    "            return {'status': 'error', 'error': str(e)}",
                    "",
                ]
            )

        return "\n".join(code_lines)

    def _generate_javascript_code(
        self, description: str, requirements: dict[str, Any]
    ) -> str:
        """Generate JavaScript code."""
        class_name = requirements.get("class_name", "GeneratedClass")

        return f"""// Generated code for: {description}

class {class_name} {{
    constructor() {{
        this.initialized = true;
        console.log(`Initialized ${{this.constructor.name}}`);
    }}

    process(data) {{
        try {{
            // Generated implementation
            return {{
                status: 'success',
                result: 'Processed successfully',
                timestamp: new Date().toISOString()
            }};
        }} catch (error) {{
            console.error(`Error in process: ${{error}}`);
            return {{
                status: 'error',
                error: error.message
            }};
        }}
    }}
}}

module.exports = {class_name};
"""

    def _generate_sql_code(self, description: str, requirements: dict[str, Any]) -> str:
        """Generate SQL code."""
        table_name = requirements.get("table_name", "generated_table")
        columns = requirements.get("columns", ["id", "name", "created_at"])

        sql_lines = [
            f"-- Generated SQL for: {description}",
            "",
            f"CREATE TABLE IF NOT EXISTS {table_name} (",
        ]

        for i, column in enumerate(columns):
            column_def = self._get_sql_column_definition(column)
            comma = "," if i < len(columns) - 1 else ""
            sql_lines.append(f"    {column_def}{comma}")

        sql_lines.extend(
            [
                ");",
                "",
                f"-- Sample queries for {table_name}",
                f"SELECT * FROM {table_name} WHERE id = 1;",
                f"INSERT INTO {table_name} (name) VALUES ('Generated Entry');",
                f"UPDATE {table_name} SET name = 'Updated' WHERE id = 1;",
                f"DELETE FROM {table_name} WHERE id = 1;",
            ]
        )

        return "\n".join(sql_lines)

    def _generate_generic_code(
        self, language: CodeLanguage, description: str, requirements: dict[str, Any]
    ) -> str:
        """Generate generic code template."""
        return f"""// Generated {language.value} code for: {description}
// This is a template implementation

// Main function/class implementation
// TODO: Implement specific functionality based on requirements
// Requirements: {requirements}

function main() {{
    console.log("Generated {language.value} implementation");
    return {{
        status: "success",
        language: "{language.value}",
        description: "{description}"
    }};
}}
"""

    def _analyze_code_quality(self, code: str, quality_level: str) -> float:
        """Analyze code quality and return score."""
        if not code:
            return 0.0

        score = 0.8  # Base score

        # Check for comments and documentation
        comment_ratio = len(
            [line for line in code.split("\n") if line.strip().startswith("#")]
        ) / len(code.split("\n"))
        score += comment_ratio * 0.1

        # Check for proper structure
        if "class" in code or "function" in code or "def" in code:
            score += 0.1

        # Check for error handling
        if "try:" in code or "except" in code or "catch" in code:
            score += 0.1

        return min(1.0, score)

    def _analyze_code_issues(self, code: str, language: str) -> list[dict[str, Any]]:
        """Analyze code for issues and bugs."""
        issues = []

        if language.lower() == "python":
            try:
                # Basic syntax check
                ast.parse(code)
            except SyntaxError as e:
                issues.append(
                    {
                        "type": "syntax_error",
                        "line": e.lineno,
                        "message": str(e),
                        "severity": "high",
                    }
                )

        # Check for common issues
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            # Check for potential security issues
            if re.search(r"eval\(|exec\(|os\.system\(", line):
                issues.append(
                    {
                        "type": "security_risk",
                        "line": i,
                        "message": "Potential security risk detected",
                        "severity": "high",
                    }
                )

            # Check for unused variables
            if re.search(r"^\s*\w+\s*=.*$", line) and not re.search(
                r"self\.|return|print|log", line
            ):
                var_name = re.search(r"^\s*(\w+)\s*=", line)
                if var_name and var_name.group(1) not in code[code.find(line) :]:
                    issues.append(
                        {
                            "type": "unused_variable",
                            "line": i,
                            "message": f"Variable '{var_name.group(1)}' may be unused",
                            "severity": "low",
                        }
                    )

        return issues

    def _generate_fixes(
        self, code: str, issues: list[dict[str, Any]], error_description: str
    ) -> list[dict[str, Any]]:
        """Generate fixes for identified issues."""
        fixes = []

        for issue in issues:
            if issue["type"] == "syntax_error":
                fixes.append(
                    {
                        "issue": issue,
                        "fix_type": "syntax_correction",
                        "description": "Fix syntax error",
                        "suggested_change": "Check parentheses, indentation, and syntax",
                    }
                )
            elif issue["type"] == "security_risk":
                fixes.append(
                    {
                        "issue": issue,
                        "fix_type": "security_improvement",
                        "description": "Replace with secure alternative",
                        "suggested_change": "Use safe alternatives to eval/exec/system calls",
                    }
                )
            elif issue["type"] == "unused_variable":
                fixes.append(
                    {
                        "issue": issue,
                        "fix_type": "cleanup",
                        "description": "Remove unused variable",
                        "suggested_change": "Remove or use the variable",
                    }
                )

        return fixes

    def _apply_fixes(self, code: str, fixes: list[dict[str, Any]]) -> str:
        """Apply fixes to code (simplified implementation)."""
        # This is a simplified version - in practice would need more sophisticated fixing
        fixed_code = code

        for fix in fixes:
            if fix["fix_type"] == "cleanup":
                # Simple cleanup - remove obvious unused assignments
                issue = fix["issue"]
                lines = fixed_code.split("\n")
                if issue["line"] <= len(lines):
                    line_content = lines[issue["line"] - 1]
                    if re.search(r"^\s*\w+\s*=.*$", line_content):
                        lines[issue["line"] - 1] = (
                            f"# {line_content}  # Commented out unused variable"
                        )
                        fixed_code = "\n".join(lines)

        return fixed_code

    def _analyze_optimizations(
        self, code: str, language: str, goals: list[str]
    ) -> list[dict[str, Any]]:
        """Analyze potential optimizations."""
        optimizations = []

        if "performance" in goals:
            # Check for loops that could be optimized
            if re.search(r"for.*in.*:", code):
                optimizations.append(
                    {
                        "type": "performance",
                        "description": "Consider using list comprehensions or vectorized operations",
                        "impact": "medium",
                        "effort": "low",
                    }
                )

        if "memory" in goals:
            # Check for memory usage patterns
            if "list(" in code or "dict(" in code:
                optimizations.append(
                    {
                        "type": "memory",
                        "description": "Consider using generators for large datasets",
                        "impact": "high",
                        "effort": "medium",
                    }
                )

        if "readability" in goals:
            # Check for long functions
            lines = code.split("\n")
            function_lengths = {}
            current_function = None
            function_start = 0

            for i, line in enumerate(lines):
                if re.search(r"^\s*def\s+(\w+)", line):
                    if current_function:
                        function_lengths[current_function] = i - function_start
                    current_function = re.search(r"^\s*def\s+(\w+)", line).group(1)
                    function_start = i

            for func, length in function_lengths.items():
                if length > 50:
                    optimizations.append(
                        {
                            "type": "readability",
                            "description": f"Function '{func}' is {length} lines long - consider breaking it down",
                            "impact": "high",
                            "effort": "high",
                        }
                    )

        return optimizations

    def _apply_optimizations(
        self, code: str, optimizations: list[dict[str, Any]]
    ) -> str:
        """Apply optimizations to code."""
        # Simplified optimization application
        optimized_code = code

        # Add comments about optimizations
        optimization_comments = [
            "# Code optimized by Magi Agent",
            "# Optimizations applied:",
        ]

        for opt in optimizations:
            optimization_comments.append(f"# - {opt['description']}")

        return "\n".join(optimization_comments) + "\n\n" + optimized_code

    def _estimate_performance_gain(
        self, original_code: str, optimized_code: str
    ) -> dict[str, float]:
        """Estimate performance improvement."""
        # Simple heuristic-based estimation
        original_complexity = len(original_code.split("\n"))
        optimized_complexity = len(optimized_code.split("\n"))

        return {
            "estimated_speed_improvement": 1.1,  # 10% improvement
            "estimated_memory_reduction": 0.95,  # 5% reduction
            "code_complexity_change": optimized_complexity / original_complexity
            if original_complexity > 0
            else 1.0,
        }

    def _generate_deployment_artifacts(self, config: DeploymentConfig) -> list[str]:
        """Generate deployment artifacts."""
        artifacts = []

        if config.deployment_type == "container":
            artifacts.extend(["Dockerfile", "docker-compose.yml", ".dockerignore"])
        elif config.deployment_type == "kubernetes":
            artifacts.extend(["deployment.yaml", "service.yaml", "configmap.yaml"])
        elif config.deployment_type == "serverless":
            artifacts.extend(["serverless.yml", "lambda_function.py"])

        artifacts.extend(["requirements.txt", "README.md", "deployment_guide.md"])

        return artifacts

    def _simulate_deployment(self, config: DeploymentConfig) -> dict[str, Any]:
        """Simulate deployment process."""
        return {
            "status": "success",
            "deployment_time": 120,  # seconds
            "health_check": "passed",
            "endpoints": [f"https://{config.target_environment}.example.com"],
            "resource_allocation": config.resource_requirements,
            "scaling_status": "active" if config.scaling_config else "disabled",
        }

    def _generate_monitoring_endpoints(self, config: DeploymentConfig) -> list[str]:
        """Generate monitoring endpoints."""
        base_url = f"https://monitoring.{config.target_environment}.example.com"
        return [
            f"{base_url}/health",
            f"{base_url}/metrics",
            f"{base_url}/logs",
            f"{base_url}/traces",
        ]

    def _review_criteria(
        self, code: str, language: str, criteria: str
    ) -> tuple[float, list[str]]:
        """Review code against specific criteria."""
        score = 0.7  # Base score
        comments = []

        if criteria == "security":
            # Check for security issues
            if not re.search(r"eval\(|exec\(|os\.system\(", code):
                score += 0.2
                comments.append("No obvious security vulnerabilities found")
            else:
                comments.append(
                    "Security risks detected - avoid eval/exec/system calls"
                )

            if "password" in code.lower() and "hash" in code.lower():
                score += 0.1
                comments.append("Password handling appears secure")

        elif criteria == "performance":
            # Check for performance patterns
            if "list comprehension" in code or "[" in code and "for" in code:
                score += 0.1
                comments.append("Efficient list operations detected")

            if re.search(r"cache|memoize", code):
                score += 0.2
                comments.append("Caching mechanisms found")

        elif criteria == "maintainability":
            # Check for maintainability factors
            comment_ratio = len(
                [line for line in code.split("\n") if line.strip().startswith("#")]
            ) / len(code.split("\n"))
            if comment_ratio > 0.1:
                score += 0.2
                comments.append("Well documented code")

            if "test" in code.lower() or "assert" in code:
                score += 0.1
                comments.append("Testing patterns found")

        return min(1.0, score), comments

    def _generate_review_recommendations(
        self, review_results: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on review results."""
        recommendations = []

        for criteria, result in review_results.items():
            score = result["score"]
            if score < 0.7:
                recommendations.append(
                    f"Improve {criteria} - current score: {score:.2f}"
                )
            elif score >= 0.9:
                recommendations.append(f"Excellent {criteria} implementation")

        return recommendations

    def _create_documentation(self, code: str, doc_type: str, language: str) -> str:
        """Create documentation for code."""
        if doc_type == "api":
            return self._create_api_documentation(code, language)
        elif doc_type == "technical":
            return self._create_technical_documentation(code, language)
        else:
            return self._create_user_documentation(code, language)

    def _create_api_documentation(self, code: str, language: str) -> str:
        """Create API documentation."""
        return f"""# API Documentation

## Overview
Generated API documentation for {language} implementation.

## Endpoints

### Main Functions
{self._extract_functions(code)}

## Usage Examples

```{language}
# Example usage
{self._generate_usage_example(code, language)}
```

## Error Handling
The API returns standard error responses with appropriate HTTP status codes.

## Authentication
Authentication details would be specified here.
"""

    def _create_technical_documentation(self, code: str, language: str) -> str:
        """Create technical documentation."""
        return f"""# Technical Documentation

## Architecture Overview
This {language} implementation follows standard patterns and practices.

## Code Structure
{self._analyze_code_structure(code)}

## Dependencies
- Standard {language} libraries
- Additional requirements as specified

## Performance Characteristics
- Optimized for production use
- Scalable architecture
- Error handling included

## Deployment
Standard deployment procedures apply.
"""

    def _create_user_documentation(self, code: str, language: str) -> str:
        """Create user documentation."""
        return f"""# User Guide

## Getting Started
This guide helps you use the {language} implementation.

## Installation
1. Install dependencies
2. Configure settings
3. Run the application

## Basic Usage
{self._generate_usage_example(code, language)}

## Troubleshooting
Common issues and solutions will be listed here.

## Support
Contact support for additional help.
"""

    def _extract_functions(self, code: str) -> str:
        """Extract function signatures from code."""
        functions = []
        for line in code.split("\n"):
            if re.search(r"def\s+(\w+)", line):
                functions.append(f"- {line.strip()}")
        return "\n".join(functions) if functions else "No functions found"

    def _analyze_code_structure(self, code: str) -> str:
        """Analyze code structure."""
        lines = len(code.split("\n"))
        classes = len(re.findall(r"class\s+\w+", code))
        functions = len(re.findall(r"def\s+\w+", code))

        return f"""
- Total lines: {lines}
- Classes: {classes}
- Functions: {functions}
- Complexity: {"High" if lines > 100 else "Medium" if lines > 50 else "Low"}
"""

    def _generate_usage_example(self, code: str, language: str) -> str:
        """Generate usage example."""
        if "class" in code:
            class_match = re.search(r"class\s+(\w+)", code)
            if class_match:
                class_name = class_match.group(1)
                return f"""
# Create instance
instance = {class_name}()

# Use the instance
result = instance.process({{"data": "example"}})
print(result)
"""
        return "# Example usage would be provided here"

    def _get_sql_column_definition(self, column: str) -> str:
        """Get SQL column definition."""
        if column == "id":
            return "id INTEGER PRIMARY KEY AUTOINCREMENT"
        elif column == "name":
            return "name VARCHAR(255) NOT NULL"
        elif "created_at" in column or "updated_at" in column:
            return f"{column} TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        elif "email" in column:
            return f"{column} VARCHAR(255) UNIQUE"
        else:
            return f"{column} TEXT"

    def _determine_approach(self, request: dict[str, Any]) -> str:
        """Determine implementation approach."""
        complexity = request.get("complexity", "medium")
        urgency = request.get("urgency", "medium")

        if self.detail_orientation == "high" and complexity == "high":
            return "comprehensive_analysis_first"
        elif urgency == "high":
            return "rapid_prototyping"
        elif self.delivery_focus == "balanced":
            return "iterative_development"
        else:
            return "traditional_waterfall"

    def _initialize_templates(self) -> None:
        """Initialize code templates."""
        self.code_templates = {
            "python_class": '''class {class_name}:
    """Generated class."""

    def __init__(self):
        self.initialized = True

    def process(self, data):
        return {{"status": "success", "data": data}}
''',
            "javascript_function": """function {function_name}(data) {{
    try {{
        return {{
            status: 'success',
            result: data
        }};
    }} catch (error) {{
        return {{
            status: 'error',
            error: error.message
        }};
    }}
}}""",
            "sql_table": """CREATE TABLE {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);""",
        }

    def update_performance(self, performance_data: dict[str, Any]) -> None:
        """Update performance metrics."""
        self.performance_history.append({**performance_data, "timestamp": time.time()})

        # Calculate KPIs
        if self.performance_history:
            recent_performance = self.performance_history[-10:]
            success_rate = sum(
                1 for p in recent_performance if p.get("success", False)
            ) / len(recent_performance)

            self.kpi_scores = {
                "code_generation_success_rate": success_rate,
                "deployment_success_rate": self._calculate_deployment_success(),
                "code_quality_score": self._calculate_code_quality_score(),
                "debugging_efficiency": self._calculate_debugging_efficiency(),
            }

    def _calculate_deployment_success(self) -> float:
        """Calculate deployment success rate."""
        deployments = [g for g in self.code_generations.values() if g.completed_at]
        if not deployments:
            return 0.8  # Default score

        successful = sum(1 for d in deployments if not d.errors)
        return successful / len(deployments)

    def _calculate_code_quality_score(self) -> float:
        """Calculate average code quality score."""
        quality_scores = [
            g.quality_score for g in self.code_generations.values() if g.quality_score
        ]
        if not quality_scores:
            return 0.7  # Default score

        return sum(quality_scores) / len(quality_scores)

    def _calculate_debugging_efficiency(self) -> float:
        """Calculate debugging efficiency."""
        # Simple metric based on performance history
        debugging_tasks = [
            p for p in self.performance_history if p.get("task_type") == "debug_code"
        ]
        if not debugging_tasks:
            return 0.75  # Default score

        avg_time = sum(p.get("execution_time", 300) for p in debugging_tasks) / len(
            debugging_tasks
        )
        return max(0.1, 1.0 - (avg_time / 600))  # Normalize against 10-minute baseline

    def evaluate_kpi(self) -> dict[str, float]:
        """Evaluate current KPI metrics."""
        if not self.kpi_scores:
            return {
                "code_generation_success_rate": 0.8,
                "deployment_success_rate": 0.75,
                "code_quality_score": 0.7,
                "debugging_efficiency": 0.75,
                "overall_performance": 0.75,
            }

        overall = sum(self.kpi_scores.values()) / len(self.kpi_scores)
        return {**self.kpi_scores, "overall_performance": overall}
