"""
AIVillage Unified Linting Manager with MCP Integration
Coordinates multi-agent code review with GitHub automation
"""

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import tempfile
import shutil

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import error handling system with fallbacks
try:
    from .error_handler import error_handler, ErrorContext, LintingError, ToolNotFoundError, ConfigurationError
    from .linting_cache_system import cache_manager, linting_cache, CacheKey
    logger.info("Loaded advanced error handling and caching systems")
except ImportError:
    try:
        from error_handler import error_handler, ErrorContext, LintingError, ToolNotFoundError, ConfigurationError
        from linting_cache_system import cache_manager, linting_cache, CacheKey
        logger.info("Loaded error handling and caching from local modules")
    except ImportError:
        # Create minimal fallback error handling
        logger.warning("Advanced error handling not available, using basic error handling")
        
        @dataclass
        class ErrorContext:
            operation: str
            tool: str
            target_paths: List[str]
            config: Dict[str, Any]
            timestamp: str = None
            session_id: str = None
            
            def __post_init__(self):
                if self.timestamp is None:
                    self.timestamp = datetime.now().isoformat()
                if self.session_id is None:
                    self.session_id = f"{self.operation}_{id(self)}"
        
        class LintingError(Exception):
            pass
        
        class ToolNotFoundError(LintingError):
            pass
        
        class ConfigurationError(LintingError):
            pass
        
        def error_handler(context):
            """Simple error handler decorator"""
            def decorator(func):
                def wrapper(*args, **kwargs):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        logger.error(f"Error in {context.operation}: {e}")
                        raise
                return wrapper
            return decorator
        
        # Minimal cache implementation
        class MinimalCache:
            def __init__(self):
                self._cache = {}
            
            def get(self, key):
                return self._cache.get(str(key))
            
            def set(self, key, value, ttl=3600):
                self._cache[str(key)] = value
            
            def clear(self):
                self._cache.clear()
        
        linting_cache = MinimalCache()
        
        @dataclass
        class CacheKey:
            tool: str
            files_hash: str
            config_hash: str
            
            def __str__(self):
                return f"{self.tool}:{self.files_hash}:{self.config_hash}"


@dataclass
class LintingResult:
    """Container for linting results from a single tool"""
    tool_name: str
    status: str  # 'passed', 'failed', 'error', 'skipped'
    exit_code: int
    stdout: str
    stderr: str
    issues_found: int
    critical_issues: int
    security_issues: int
    performance_issues: int
    style_issues: int
    execution_time: float
    file_count: int
    cached: bool = False
    config_used: Dict[str, Any] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.config_used is None:
            self.config_used = {}
        if self.recommendations is None:
            self.recommendations = []


@dataclass 
class QualityMetrics:
    """Quality metrics calculated from linting results"""
    overall_score: float
    security_score: float
    maintainability_score: float
    performance_score: float
    reliability_score: float
    quality_gate_status: str
    technical_debt_ratio: float = 0.0
    complexity_score: float = 0.0
    test_coverage: float = 0.0


class UnifiedLintingPipeline:
    """Main unified linting pipeline orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = {}
        self.start_time = None
        self.session_id = f"linting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with fallbacks"""
        default_config = {
            "python": {
                "tools": {
                    "ruff": {
                        "enabled": True,
                        "args": ["check", ".", "--output-format=json"],
                        "timeout": 120,
                        "fail_on_error": False
                    },
                    "black": {
                        "enabled": True, 
                        "args": ["--check", "--diff", "."],
                        "timeout": 60,
                        "fail_on_error": False
                    },
                    "mypy": {
                        "enabled": True,
                        "args": [".", "--ignore-missing-imports", "--json-report=/tmp/mypy-report"],
                        "timeout": 180,
                        "fail_on_error": False
                    },
                    "bandit": {
                        "enabled": True,
                        "args": ["-r", ".", "-f", "json", "-o", "/tmp/bandit-report.json"],
                        "timeout": 120,
                        "fail_on_error": False
                    }
                },
                "quality_gates": {
                    "min_overall_score": 70,
                    "max_critical_issues": 0,
                    "max_security_issues": 5
                }
            },
            "frontend": {
                "tools": {
                    "eslint": {
                        "enabled": True,
                        "args": [".", "--ext", ".ts,.tsx,.js,.jsx", "--format=json"],
                        "timeout": 120,
                        "fail_on_error": False
                    },
                    "prettier": {
                        "enabled": True,
                        "args": ["--check", "."],
                        "timeout": 60, 
                        "fail_on_error": False
                    }
                },
                "quality_gates": {
                    "min_overall_score": 75,
                    "max_critical_issues": 0
                }
            },
            "security": {
                "tools": {
                    "bandit": {
                        "enabled": True,
                        "args": ["-r", ".", "-ll", "-f", "json"],
                        "timeout": 180,
                        "fail_on_error": True
                    },
                    "safety": {
                        "enabled": True,
                        "args": ["check", "--json"],
                        "timeout": 120,
                        "fail_on_error": True
                    },
                    "semgrep": {
                        "enabled": True,
                        "args": ["--config=auto", ".", "--json"],
                        "timeout": 300,
                        "fail_on_error": True
                    },
                    "detect-secrets": {
                        "enabled": True,
                        "args": ["scan", "--baseline", ".secrets.baseline"],
                        "timeout": 60,
                        "fail_on_error": True
                    }
                },
                "quality_gates": {
                    "min_security_score": 80,
                    "max_critical_vulnerabilities": 0,
                    "max_high_vulnerabilities": 3
                }
            },
            "execution": {
                "parallel": True,
                "max_workers": 4,
                "cache_enabled": True,
                "cache_ttl": 3600,
                "output_formats": ["json"],
                "github_integration": True
            }
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                import yaml
                with open(self.config_path) as f:
                    user_config = yaml.safe_load(f)
                # Merge user config with defaults
                return self._deep_merge(default_config, user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    async def run_linting(self, language: str, paths: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Main entry point for running linting"""
        self.start_time = datetime.now()
        logger.info(f"Starting {language} linting pipeline (session: {self.session_id})")
        
        if language not in self.config:
            raise ConfigurationError(f"No configuration found for language: {language}")
        
        target_paths = paths or ["."]
        language_config = self.config[language]
        
        # Run tools
        if self.config.get("execution", {}).get("parallel", True):
            results = await self._run_tools_parallel(language_config, target_paths)
        else:
            results = await self._run_tools_sequential(language_config, target_paths)
        
        # Calculate metrics
        quality_metrics = self._calculate_quality_metrics(results, language_config)
        
        # Generate report
        return self._generate_report(language, results, quality_metrics, target_paths)
    
    async def _run_tools_parallel(self, config: Dict[str, Any], paths: List[str]) -> Dict[str, LintingResult]:
        """Run linting tools in parallel"""
        tools = config.get("tools", {})
        enabled_tools = [(name, tool_config) for name, tool_config in tools.items() 
                        if tool_config.get("enabled", True)]
        
        if not enabled_tools:
            logger.warning("No enabled tools found")
            return {}
        
        # Create semaphore to limit concurrent executions
        max_workers = self.config.get("execution", {}).get("max_workers", 4)
        semaphore = asyncio.Semaphore(max_workers)
        
        tasks = []
        for tool_name, tool_config in enabled_tools:
            task = self._run_single_tool_with_semaphore(semaphore, tool_name, tool_config, paths)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and exceptions
        tool_results = {}
        for i, (tool_name, _) in enumerate(enabled_tools):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Tool {tool_name} failed: {result}")
                tool_results[tool_name] = LintingResult(
                    tool_name=tool_name,
                    status="error",
                    exit_code=1,
                    stdout="",
                    stderr=str(result),
                    issues_found=0,
                    critical_issues=0,
                    security_issues=0,
                    performance_issues=0,
                    style_issues=0,
                    execution_time=0.0,
                    file_count=0
                )
            else:
                tool_results[tool_name] = result
        
        return tool_results
    
    async def _run_tools_sequential(self, config: Dict[str, Any], paths: List[str]) -> Dict[str, LintingResult]:
        """Run linting tools sequentially"""
        tools = config.get("tools", {})
        results = {}
        
        for tool_name, tool_config in tools.items():
            if not tool_config.get("enabled", True):
                continue
                
            try:
                result = await self._run_single_tool(tool_name, tool_config, paths)
                results[tool_name] = result
            except Exception as e:
                logger.error(f"Tool {tool_name} failed: {e}")
                results[tool_name] = LintingResult(
                    tool_name=tool_name,
                    status="error",
                    exit_code=1,
                    stdout="",
                    stderr=str(e),
                    issues_found=0,
                    critical_issues=0,
                    security_issues=0,
                    performance_issues=0,
                    style_issues=0,
                    execution_time=0.0,
                    file_count=0
                )
        
        return results
    
    async def _run_single_tool_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                            tool_name: str, tool_config: Dict[str, Any], 
                                            paths: List[str]) -> LintingResult:
        """Run a single tool with semaphore protection"""
        async with semaphore:
            return await self._run_single_tool(tool_name, tool_config, paths)
    
    async def _run_single_tool(self, tool_name: str, tool_config: Dict[str, Any], 
                              paths: List[str]) -> LintingResult:
        """Run a single linting tool"""
        start_time = datetime.now()
        
        # Check cache first
        if self.config.get("execution", {}).get("cache_enabled", True):
            cache_key = self._generate_cache_key(tool_name, tool_config, paths)
            cached_result = linting_cache.get(cache_key)
            if cached_result:
                logger.info(f"{tool_name}: Using cached result")
                cached_result.cached = True
                return cached_result
        
        # Check if tool exists
        if not self._tool_exists(tool_name):
            raise ToolNotFoundError(f"Tool '{tool_name}' not found in PATH")
        
        # Build command
        command = [tool_name] + tool_config.get("args", [])
        timeout = tool_config.get("timeout", 120)
        
        logger.info(f"Running {tool_name}: {' '.join(command)}")
        
        try:
            # Run the tool
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ToolNotFoundError(f"Tool '{tool_name}' timed out after {timeout}s")
            
            exit_code = process.returncode
            stdout_str = stdout.decode('utf-8', errors='ignore')
            stderr_str = stderr.decode('utf-8', errors='ignore')
            
        except FileNotFoundError:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found")
        
        # Parse results
        result = self._parse_tool_output(tool_name, exit_code, stdout_str, stderr_str, tool_config)
        result.execution_time = (datetime.now() - start_time).total_seconds()
        
        # Cache the result
        if self.config.get("execution", {}).get("cache_enabled", True):
            cache_ttl = self.config.get("execution", {}).get("cache_ttl", 3600)
            linting_cache.set(cache_key, result, ttl=cache_ttl)
        
        return result
    
    def _tool_exists(self, tool_name: str) -> bool:
        """Check if a tool exists in PATH"""
        return shutil.which(tool_name) is not None
    
    def _generate_cache_key(self, tool_name: str, tool_config: Dict[str, Any], paths: List[str]) -> CacheKey:
        """Generate cache key for a tool execution"""
        # Hash file contents
        files_hash = hashlib.md5()
        for path in paths:
            path_obj = Path(path)
            if path_obj.is_file():
                with open(path_obj, 'rb') as f:
                    files_hash.update(f.read())
            elif path_obj.is_dir():
                for file_path in path_obj.rglob('*.py'):
                    try:
                        with open(file_path, 'rb') as f:
                            files_hash.update(f.read())
                    except (OSError, IOError):
                        continue
        
        # Hash config
        config_hash = hashlib.md5(
            json.dumps(tool_config, sort_keys=True).encode()
        ).hexdigest()
        
        return CacheKey(
            tool=tool_name,
            files_hash=files_hash.hexdigest(),
            config_hash=config_hash
        )
    
    def _parse_tool_output(self, tool_name: str, exit_code: int, stdout: str, 
                          stderr: str, config: Dict[str, Any]) -> LintingResult:
        """Parse tool output and extract metrics"""
        issues_found = 0
        critical_issues = 0
        security_issues = 0
        performance_issues = 0
        style_issues = 0
        file_count = 0
        
        # Tool-specific parsing
        if tool_name == "ruff":
            issues_found, critical_issues, style_issues = self._parse_ruff_output(stdout)
        elif tool_name == "black":
            # Black only reports formatting issues
            if exit_code != 0:
                style_issues = stdout.count("would reformat")
                issues_found = style_issues
        elif tool_name == "mypy":
            issues_found = stdout.count("error:")
            critical_issues = stdout.count("error:") - stdout.count("note:")
        elif tool_name == "bandit":
            issues_found, critical_issues, security_issues = self._parse_bandit_output(stdout)
        elif tool_name == "eslint":
            issues_found, critical_issues, style_issues = self._parse_eslint_output(stdout)
        elif tool_name == "semgrep":
            issues_found, critical_issues, security_issues = self._parse_semgrep_output(stdout)
        else:
            # Generic parsing - count lines with common issue patterns
            lines = stdout.split('\n') + stderr.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['error', 'critical', 'high']):
                    issues_found += 1
                    if any(keyword in line.lower() for keyword in ['critical', 'high']):
                        critical_issues += 1
        
        # Determine status
        fail_on_error = config.get("fail_on_error", False)
        if exit_code == 0:
            status = "passed"
        elif issues_found == 0:
            status = "passed" 
        elif fail_on_error and (critical_issues > 0 or security_issues > 0):
            status = "failed"
        else:
            status = "warning"
        
        return LintingResult(
            tool_name=tool_name,
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            issues_found=issues_found,
            critical_issues=critical_issues,
            security_issues=security_issues,
            performance_issues=performance_issues,
            style_issues=style_issues,
            execution_time=0.0,  # Will be set by caller
            file_count=file_count,
            config_used=config
        )
    
    def _parse_ruff_output(self, output: str) -> tuple[int, int, int]:
        """Parse ruff JSON output"""
        try:
            if output.strip():
                data = json.loads(output)
                if isinstance(data, list):
                    issues = len(data)
                    critical = sum(1 for item in data if item.get("level") == "error")
                    style = sum(1 for item in data if item.get("level") == "warning")
                    return issues, critical, style
        except (json.JSONDecodeError, KeyError):
            # Fall back to text parsing
            lines = output.split('\n')
            issues = len([line for line in lines if ' error ' in line or ' warning ' in line])
            critical = len([line for line in lines if ' error ' in line])
            style = len([line for line in lines if ' warning ' in line])
            return issues, critical, style
        return 0, 0, 0
    
    def _parse_bandit_output(self, output: str) -> tuple[int, int, int]:
        """Parse bandit JSON output"""
        try:
            if output.strip():
                data = json.loads(output)
                results = data.get("results", [])
                issues = len(results)
                critical = len([r for r in results if r.get("issue_severity") in ["HIGH", "CRITICAL"]])
                security = issues  # All bandit issues are security-related
                return issues, critical, security
        except (json.JSONDecodeError, KeyError):
            pass
        return 0, 0, 0
    
    def _parse_eslint_output(self, output: str) -> tuple[int, int, int]:
        """Parse ESLint JSON output"""
        try:
            if output.strip():
                data = json.loads(output)
                if isinstance(data, list):
                    issues = sum(file_result.get("errorCount", 0) + file_result.get("warningCount", 0) 
                               for file_result in data)
                    critical = sum(file_result.get("errorCount", 0) for file_result in data)
                    style = sum(file_result.get("warningCount", 0) for file_result in data)
                    return issues, critical, style
        except (json.JSONDecodeError, KeyError):
            pass
        return 0, 0, 0
    
    def _parse_semgrep_output(self, output: str) -> tuple[int, int, int]:
        """Parse semgrep JSON output"""
        try:
            if output.strip():
                data = json.loads(output)
                results = data.get("results", [])
                issues = len(results)
                critical = len([r for r in results if r.get("extra", {}).get("severity") in ["ERROR", "WARNING"]])
                security = issues  # Semgrep is primarily security-focused
                return issues, critical, security
        except (json.JSONDecodeError, KeyError):
            pass
        return 0, 0, 0
    
    def _calculate_quality_metrics(self, results: Dict[str, LintingResult], 
                                  config: Dict[str, Any]) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        if not results:
            return QualityMetrics(
                overall_score=100.0,
                security_score=100.0,
                maintainability_score=100.0,
                performance_score=100.0,
                reliability_score=100.0,
                quality_gate_status="passed"
            )
        
        # Aggregate metrics
        total_issues = sum(r.issues_found for r in results.values())
        total_critical = sum(r.critical_issues for r in results.values()) 
        total_security = sum(r.security_issues for r in results.values())
        total_performance = sum(r.performance_issues for r in results.values())
        total_style = sum(r.style_issues for r in results.values())
        
        # Calculate scores (0-100)
        # Base score starts at 100, deduct points for issues
        security_score = max(0, 100 - (total_security * 10) - (total_critical * 5))
        maintainability_score = max(0, 100 - (total_style * 2) - (total_issues * 1))
        performance_score = max(0, 100 - (total_performance * 8))
        reliability_score = max(0, 100 - (total_critical * 8) - (total_issues * 1))
        
        # Overall weighted score
        overall_score = (
            security_score * 0.3 +
            maintainability_score * 0.25 +
            performance_score * 0.2 +
            reliability_score * 0.25
        )
        
        # Determine quality gate status
        quality_gates = config.get("quality_gates", {})
        min_score = quality_gates.get("min_overall_score", 70)
        max_critical = quality_gates.get("max_critical_issues", 0)
        max_security = quality_gates.get("max_security_issues", 5)
        
        if total_critical > max_critical:
            gate_status = "failed_critical"
        elif total_security > max_security:
            gate_status = "failed_security"
        elif overall_score < min_score:
            gate_status = "failed_overall"
        elif total_issues > 50:  # Warning threshold
            gate_status = "warning_quality"
        else:
            gate_status = "passed"
        
        return QualityMetrics(
            overall_score=round(overall_score, 2),
            security_score=round(security_score, 2),
            maintainability_score=round(maintainability_score, 2),
            performance_score=round(performance_score, 2),
            reliability_score=round(reliability_score, 2),
            quality_gate_status=gate_status,
            technical_debt_ratio=min(100, total_issues * 0.5),
            complexity_score=round((total_critical + total_security) * 2.5, 2)
        )
    
    def _generate_report(self, language: str, results: Dict[str, LintingResult], 
                        metrics: QualityMetrics, paths: List[str]) -> Dict[str, Any]:
        """Generate comprehensive linting report"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Generate recommendations
        recommendations = []
        total_critical = sum(r.critical_issues for r in results.values())
        total_security = sum(r.security_issues for r in results.values())
        total_style = sum(r.style_issues for r in results.values())
        
        if total_critical > 0:
            recommendations.append(f"Fix {total_critical} critical issues immediately")
        if total_security > 0:
            recommendations.append(f"Address {total_security} security vulnerabilities") 
        if total_style > 10:
            recommendations.append(f"Clean up {total_style} style issues for better maintainability")
        if metrics.overall_score < 80:
            recommendations.append("Overall code quality needs improvement")
        
        # Summary statistics
        pipeline_summary = {
            "total_tools_run": len(results),
            "successful_tools": len([r for r in results.values() if r.status == "passed"]),
            "failed_tools": len([r for r in results.values() if r.status == "failed"]),
            "total_issues_found": sum(r.issues_found for r in results.values()),
            "critical_issues": sum(r.critical_issues for r in results.values()),
            "security_issues": sum(r.security_issues for r in results.values()),
            "performance_issues": sum(r.performance_issues for r in results.values()),
            "style_issues": sum(r.style_issues for r in results.values()),
            "total_execution_time": round(total_time, 2),
            "cache_hit_rate": len([r for r in results.values() if r.cached]) / len(results) * 100 if results else 0
        }
        
        return {
            "language": language,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "target_paths": paths,
            "pipeline_summary": pipeline_summary,
            "quality_metrics": asdict(metrics),
            "tool_results": {name: asdict(result) for name, result in results.items()},
            "recommendations": recommendations,
            "status": "completed",
            "github_integration": {
                "workflow_run": os.getenv("GITHUB_RUN_ID"),
                "commit_sha": os.getenv("GITHUB_SHA"),
                "branch": os.getenv("GITHUB_REF_NAME"),
                "actor": os.getenv("GITHUB_ACTOR")
            }
        }


# Create global instance
unified_linting_manager = UnifiedLintingPipeline()

async def main():
    """CLI entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVillage Unified Linting Manager")
    parser.add_argument("--language", default="python", choices=["python", "frontend", "security"])
    parser.add_argument("--paths", nargs="+", default=["."])
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    manager = UnifiedLintingPipeline(config_path=args.config)
    result = await manager.run_linting(args.language, args.paths)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())