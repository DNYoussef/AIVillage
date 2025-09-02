#!/usr/bin/env python3
"""
AIVillage Unified Linting CLI Runner
MCP-Enhanced Multi-Agent Code Review Orchestrator
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Set up basic logging first
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the unified linting manager with proper error handling
try:
    from .unified_linting_manager import UnifiedLintingPipeline, unified_linting_manager
    logger.info("Successfully imported unified linting manager from package")
except ImportError:
    try:
        from unified_linting_manager import UnifiedLintingPipeline, unified_linting_manager
        logger.info("Successfully imported unified linting manager from local directory")
    except ImportError as e:
        logger.error(f"Failed to import unified linting manager: {e}")
        # Create a minimal fallback
        import asyncio
        from pathlib import Path
        
        class MinimalLintingResult:
            def __init__(self, tool, status="error", message="Tool unavailable"):
                self.tool = tool
                self.status = status
                self.issues_found = 0
                self.critical_issues = 0
                self.security_issues = 0
                self.performance_issues = 0
                self.style_issues = 0
                self.execution_time = 0.0
                self.suggestions = [message]
                self.details = {"error": message}
                
        class UnifiedLintingPipeline:
            def __init__(self, config_path=None):
                self.config_path = config_path
                logger.error("Using minimal fallback - full linting unavailable")
            
            async def run_full_pipeline(self, target_paths=None):
                return {
                    "error": "Unified linting manager not available",
                    "fallback": True,
                    "pipeline_summary": {
                        "timestamp": "unknown",
                        "total_tools_run": 0,
                        "total_issues_found": 0,
                        "critical_issues": 0,
                        "security_issues": 0,
                        "performance_issues": 0,
                        "style_issues": 0,
                        "total_execution_time": 0.0
                    },
                    "recommendations": ["Install required linting tools and dependencies"],
                    "next_steps": ["Check installation and configuration"]
                }
            
            async def run_python_linting(self, target_paths):
                return {"fallback": MinimalLintingResult("python", "error", "Python linting unavailable")}
            
            async def run_frontend_linting(self, target_paths):
                return {"fallback": MinimalLintingResult("frontend", "error", "Frontend linting unavailable")}
            
            async def run_security_linting(self, target_paths):
                return {"fallback": MinimalLintingResult("security", "error", "Security linting unavailable")}
            
            async def initialize_mcp_connections(self):
                logger.warning("MCP connections not available in fallback mode")
                return False
        
        unified_linting_manager = UnifiedLintingPipeline()


def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('unified_linting.log')
        ]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def print_summary(results: Dict, language: str):
    """Print a formatted summary of linting results"""
    
    if "pipeline_summary" in results:
        summary = results["pipeline_summary"]
        print(f"\n🔍 {language.upper()} Linting Summary")
        print("=" * 50)
        print(f"Tools Run: {summary.get('total_tools_run', 0)}")
        print(f"Total Issues: {summary.get('total_issues_found', 0)}")
        print(f"Critical Issues: {summary.get('critical_issues', 0)}")
        print(f"Security Issues: {summary.get('security_issues', 0)}")
        print(f"Performance Issues: {summary.get('performance_issues', 0)}")
        print(f"Style Issues: {summary.get('style_issues', 0)}")
        print(f"Execution Time: {summary.get('total_execution_time', 0):.2f}s")
        
        if "quality_metrics" in results:
            metrics = results["quality_metrics"]
            print(f"\n📊 Quality Metrics")
            print(f"Overall Score: {metrics.get('overall_score', 0)}/100")
            print(f"Security Score: {metrics.get('security_score', 0)}/100")
            print(f"Quality Gate: {metrics.get('quality_gate_status', 'unknown')}")
            
        if "recommendations" in results and results["recommendations"]:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(results["recommendations"][:5], 1):
                print(f"  {i}. {rec}")
        
    else:
        # Individual tool results
        total_issues = sum(result.get('issues_found', 0) for result in results.values())
        critical_issues = sum(result.get('critical_issues', 0) for result in results.values())
        
        print(f"\n🔍 {language.upper()} Linting Results")
        print("=" * 40)
        print(f"Tools: {', '.join(results.keys())}")
        print(f"Total Issues: {total_issues}")
        print(f"Critical Issues: {critical_issues}")
        
        for tool, result in results.items():
            status_emoji = "✅" if result.get('status') == 'passed' else "❌" if result.get('status') == 'failed' else "⚠️"
            print(f"  {status_emoji} {tool}: {result.get('issues_found', 0)} issues")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="AIVillage Unified Linting Pipeline with MCP Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all linting
  python run_unified_linting.py --language=all
  
  # Run Python linting only  
  python run_unified_linting.py --language=python --paths src/ core/
  
  # Run security scan with output
  python run_unified_linting.py --language=security --output=security-results.json
  
  # Full pipeline with custom config
  python run_unified_linting.py --config=custom_config.yml --debug
        """
    )
    
    parser.add_argument(
        "--language", 
        choices=["python", "frontend", "security", "documentation", "all"], 
        default="all",
        help="Language/type to lint"
    )
    
    parser.add_argument(
        "--paths", 
        nargs="+", 
        default=None,
        help="Specific paths to lint (default: auto-detect)"
    )
    
    parser.add_argument(
        "--output", 
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--format", 
        choices=["json", "sarif", "text", "summary"],
        default="summary",
        help="Output format"
    )
    
    parser.add_argument(
        "--fail-on-critical", 
        action="store_true",
        help="Exit with non-zero code if critical issues found"
    )
    
    parser.add_argument(
        "--skip-cache", 
        action="store_true",
        help="Skip performance cache and run fresh scans"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be run without executing"
    )
    
    parser.add_argument(
        "--tools",
        nargs="+",
        help="Specific tools to run (overrides language selection)"
    )
    
    parser.add_argument(
        "--github-integration",
        action="store_true", 
        help="Enable GitHub workflow integration features"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AIVillage Unified Linting Pipeline")
    
    try:
        # Initialize pipeline
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error(f"Configuration file not found: {args.config}")
                return 1
            manager = UnifiedLintingPipeline(config_path)
        else:
            manager = unified_linting_manager
            
        # Handle dry run
        if args.dry_run:
            logger.info("DRY RUN MODE - Showing execution plan")
            print("\n🔍 Execution Plan:")
            if args.language == "all":
                print("- Python linting (ruff, black, mypy, bandit)")
                print("- Frontend linting (eslint, prettier, typescript)")
                print("- Security scanning (detect-secrets, semgrep, pip-audit)")
            elif args.language == "python":
                print("- Python tools: ruff, black, mypy, bandit")
            elif args.language == "frontend":
                print("- Frontend tools: eslint, prettier, typescript")
            elif args.language == "security":
                print("- Security tools: detect-secrets, semgrep, pip-audit")
            
            if args.paths:
                print(f"- Target paths: {', '.join(args.paths)}")
            else:
                print("- Target paths: auto-detected")
                
            return 0
        
        # Determine target paths
        if args.paths:
            target_paths = args.paths
        else:
            # Auto-detect based on common project structures
            potential_paths = [".", "src/", "core/", "infrastructure/", "apps/", "lib/"]
            target_paths = [p for p in potential_paths if Path(p).exists()]
            if not target_paths:
                target_paths = ["."]
        
        logger.info(f"Target paths: {target_paths}")
        logger.info(f"Language selection: {args.language}")
        
        # Initialize MCP connections if GitHub integration enabled
        if args.github_integration:
            logger.info("Initializing GitHub MCP integration...")
            await manager.initialize_mcp_connections()
        
        # Run linting based on selection
        results = {}
        
        if args.tools:
            # Custom tool selection
            logger.info(f"Running custom tools: {args.tools}")
            # This would require extending the manager to support custom tool selection
            logger.warning("Custom tool selection not yet implemented")
            return 1
            
        elif args.language == "python":
            logger.info("Running Python linting pipeline...")
            results = await manager.run_python_linting(target_paths)
            
        elif args.language == "frontend":
            logger.info("Running frontend linting pipeline...")
            results = await manager.run_frontend_linting(target_paths)
            
        elif args.language == "security":
            logger.info("Running security linting pipeline...")
            results = await manager.run_security_linting(target_paths)
            
        elif args.language == "documentation":
            logger.info("Documentation linting not yet implemented")
            return 1
            
        else:  # "all"
            logger.info("Running full unified linting pipeline...")
            results = await manager.run_full_pipeline(target_paths)
        
        # Process and output results
        exit_code = 0
        
        # Check for critical issues
        if isinstance(results, dict):
            if "pipeline_summary" in results:
                critical_issues = results["pipeline_summary"].get("critical_issues", 0)
            else:
                critical_issues = sum(
                    result.get("critical_issues", 0) 
                    for result in results.values() 
                    if isinstance(result, dict)
                )
            
            if critical_issues > 0 and args.fail_on_critical:
                logger.error(f"Found {critical_issues} critical issues - failing build")
                exit_code = 1
        
        # Output results based on format
        if args.format == "json":
            output_data = json.dumps(results, indent=2, default=str)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_data)
                logger.info(f"Results saved to {args.output}")
            else:
                print(output_data)
                
        elif args.format == "sarif":
            # Convert to SARIF format (simplified)
            sarif_data = convert_to_sarif(results)
            output_data = json.dumps(sarif_data, indent=2)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_data)
            else:
                print(output_data)
                
        elif args.format == "text":
            # Detailed text output
            output_data = format_text_results(results)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output_data)
            else:
                print(output_data)
                
        else:  # "summary"
            print_summary(results, args.language)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Full results saved to {args.output}")
        
        logger.info("Linting pipeline completed successfully")
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Linting interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Linting pipeline failed: {e}", exc_info=args.debug)
        return 1


def convert_to_sarif(results: Dict) -> Dict:
    """Convert linting results to SARIF format"""
    sarif = {
        "$schema": "https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-schema.json",
        "version": "2.1.0",
        "runs": []
    }
    
    # Extract tool results
    if "tool_results" in results:
        tool_results = results["tool_results"]
    else:
        tool_results = results
    
    for tool_name, tool_result in tool_results.items():
        if not isinstance(tool_result, dict):
            continue
            
        run = {
            "tool": {
                "driver": {
                    "name": tool_name,
                    "version": "1.0.0"
                }
            },
            "results": []
        }
        
        # Convert issues to SARIF results
        if tool_result.get("details", {}).get("issues"):
            for issue in tool_result["details"]["issues"]:
                result = {
                    "ruleId": issue.get("code", "unknown"),
                    "message": {"text": issue.get("message", "Issue found")},
                    "level": "error" if tool_result.get("critical_issues", 0) > 0 else "warning",
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": issue.get("filename", "unknown")},
                            "region": {
                                "startLine": issue.get("line", 1),
                                "startColumn": issue.get("column", 1)
                            }
                        }
                    }]
                }
                run["results"].append(result)
        
        sarif["runs"].append(run)
    
    return sarif


def format_text_results(results: Dict) -> str:
    """Format results as detailed text output"""
    output_lines = []
    
    if "pipeline_summary" in results:
        summary = results["pipeline_summary"]
        output_lines.extend([
            "AIVillage Unified Linting Report",
            "=" * 50,
            f"Timestamp: {summary.get('timestamp', 'unknown')}",
            f"Tools Run: {summary.get('total_tools_run', 0)}",
            f"Total Issues: {summary.get('total_issues_found', 0)}",
            f"Critical Issues: {summary.get('critical_issues', 0)}",
            f"Security Issues: {summary.get('security_issues', 0)}",
            f"Performance Issues: {summary.get('performance_issues', 0)}",
            f"Style Issues: {summary.get('style_issues', 0)}",
            f"Execution Time: {summary.get('total_execution_time', 0):.2f}s",
            ""
        ])
        
        if "quality_metrics" in results:
            metrics = results["quality_metrics"]
            output_lines.extend([
                "Quality Metrics:",
                "-" * 20,
                f"Overall Score: {metrics.get('overall_score', 0)}/100",
                f"Security Score: {metrics.get('security_score', 0)}/100",
                f"Performance Score: {metrics.get('performance_score', 0)}/100",
                f"Style Score: {metrics.get('style_score', 0)}/100",
                f"Quality Gate: {metrics.get('quality_gate_status', 'unknown')}",
                ""
            ])
        
        if "tool_results" in results:
            output_lines.append("Tool Results:")
            output_lines.append("-" * 20)
            
            for tool, result in results["tool_results"].items():
                status_symbol = "✅" if result.get('status') == 'passed' else "❌" if result.get('status') == 'failed' else "⚠️"
                output_lines.extend([
                    f"{status_symbol} {tool}:",
                    f"  Status: {result.get('status', 'unknown')}",
                    f"  Issues: {result.get('issues_found', 0)}",
                    f"  Critical: {result.get('critical_issues', 0)}",
                    f"  Time: {result.get('execution_time', 0):.2f}s"
                ])
                
                if result.get('suggestions'):
                    output_lines.append("  Suggestions:")
                    for suggestion in result['suggestions'][:3]:
                        output_lines.append(f"    - {suggestion}")
                
                output_lines.append("")
        
        if "recommendations" in results and results["recommendations"]:
            output_lines.extend([
                "Recommendations:",
                "-" * 20
            ])
            for i, rec in enumerate(results["recommendations"], 1):
                output_lines.append(f"{i}. {rec}")
            output_lines.append("")
        
        if "next_steps" in results and results["next_steps"]:
            output_lines.extend([
                "Next Steps:",
                "-" * 20
            ])
            for i, step in enumerate(results["next_steps"], 1):
                output_lines.append(f"{i}. {step}")
    
    return "\n".join(output_lines)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)