#!/usr/bin/env python3
"""MCP Server Diagnostics and Repair Tool.

This tool diagnoses and fixes common MCP server issues including:
1. Package availability and installation
2. Configuration validation
3. Network connectivity
4. Timeout issues
5. Path and environment problems
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MCPServerDiagnostics:
    """Comprehensive MCP server diagnostics and repair."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.config_path = self.project_root / "config" / "mcp_config.json"
        self.mcp_dir = self.project_root / ".mcp"
        self.results = {}
        
    def run_full_diagnostics(self) -> Dict[str, any]:
        """Run comprehensive diagnostics on all MCP servers."""
        
        logger.info("Starting MCP server diagnostics...")
        
        # Ensure MCP directory exists
        self.mcp_dir.mkdir(exist_ok=True)
        
        # Load configuration
        if not self.config_path.exists():
            self.results["config_error"] = "MCP config file not found"
            return self.results
            
        try:
            with open(self.config_path) as f:
                config = json.load(f)
        except Exception as e:
            self.results["config_error"] = f"Failed to load config: {e}"
            return self.results
        
        servers = config.get("mcpServers", {})
        
        # Test each server
        for server_name, server_config in servers.items():
            if server_config.get("disabled", False):
                logger.info(f"Skipping disabled server: {server_name}")
                continue
                
            logger.info(f"Testing server: {server_name}")
            self.results[server_name] = self.test_server(server_name, server_config)
        
        # Generate repair recommendations
        self.results["repair_recommendations"] = self.generate_repair_recommendations()
        
        return self.results
    
    def test_server(self, server_name: str, config: Dict) -> Dict[str, any]:
        """Test individual MCP server configuration and availability."""
        
        result = {
            "server_name": server_name,
            "config": config,
            "tests": {}
        }
        
        # Test 1: Command availability
        command = config.get("command", "")
        result["tests"]["command_available"] = self.test_command_available(command)
        
        # Test 2: Package availability (for NPX packages)
        if "npx" in config.get("args", []):
            package_name = self.extract_package_name(config.get("args", []))
            result["tests"]["package_available"] = self.test_package_available(package_name)
        
        # Test 3: Environment variables
        result["tests"]["environment"] = self.test_environment(config.get("env", {}))
        
        # Test 4: File paths
        result["tests"]["file_paths"] = self.test_file_paths(config)
        
        # Test 5: Quick execution test (with timeout)
        result["tests"]["execution"] = self.test_server_execution(server_name, config)
        
        # Overall status
        result["status"] = self.calculate_server_status(result["tests"])
        
        return result
    
    def test_command_available(self, command: str) -> Dict[str, any]:
        """Test if the base command is available."""
        try:
            if command == "cmd":
                # Windows cmd is always available
                return {"available": True, "details": "Windows cmd shell"}
            elif command == "python":
                result = subprocess.run([command, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                return {
                    "available": result.returncode == 0,
                    "details": result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
                }
            else:
                result = subprocess.run([command, "--version"], 
                                      capture_output=True, text=True, timeout=10)
                return {
                    "available": result.returncode == 0,
                    "details": result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
                }
        except Exception as e:
            return {"available": False, "details": str(e)}
    
    def extract_package_name(self, args: List[str]) -> Optional[str]:
        """Extract NPM package name from arguments."""
        for i, arg in enumerate(args):
            if arg == "-y" and i + 1 < len(args):
                return args[i + 1]
        return None
    
    def test_package_available(self, package_name: Optional[str]) -> Dict[str, any]:
        """Test if NPM package is available."""
        if not package_name:
            return {"available": False, "details": "No package name found"}
        
        try:
            # Test global installation
            result = subprocess.run(["npm", "list", "-g", package_name], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return {"available": True, "details": f"Globally installed: {package_name}"}
            
            # Test if package exists in registry
            result = subprocess.run(["npm", "view", package_name, "version"], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                return {
                    "available": True, 
                    "details": f"Available in registry: {package_name}@{version}",
                    "needs_install": True
                }
            else:
                return {
                    "available": False, 
                    "details": f"Package not found: {package_name}"
                }
                
        except Exception as e:
            return {"available": False, "details": f"Error checking package: {e}"}
    
    def test_environment(self, env_vars: Dict[str, str]) -> Dict[str, any]:
        """Test environment variable configuration."""
        results = {}
        
        for key, value in env_vars.items():
            if key == "PATH":
                # Test PATH components
                path_components = value.split(";")
                valid_paths = []
                for path_comp in path_components:
                    if path_comp != "%PATH%" and Path(path_comp).exists():
                        valid_paths.append(path_comp)
                
                results[key] = {
                    "configured": True,
                    "valid_paths": valid_paths,
                    "total_paths": len(path_components)
                }
            else:
                # Test specific path or file
                if Path(value).exists():
                    results[key] = {"configured": True, "exists": True, "value": value}
                else:
                    results[key] = {"configured": True, "exists": False, "value": value}
        
        return results
    
    def test_file_paths(self, config: Dict) -> Dict[str, any]:
        """Test file paths referenced in configuration."""
        results = {}
        
        # Check Python file paths in args
        args = config.get("args", [])
        for arg in args:
            if arg.endswith(".py"):
                path = self.project_root / arg
                results[arg] = {
                    "exists": path.exists(),
                    "absolute_path": str(path.absolute()),
                    "is_file": path.is_file() if path.exists() else False
                }
        
        return results
    
    def test_server_execution(self, server_name: str, config: Dict) -> Dict[str, any]:
        """Test server execution with short timeout."""
        try:
            command = config.get("command", "")
            args = config.get("args", [])
            env_vars = config.get("env", {})
            
            # Prepare command
            if command == "cmd":
                full_command = [command] + args + ["--help"]
            elif command == "python":
                full_command = [command] + args + ["--help"]
            else:
                full_command = [command, "--help"]
            
            # Quick execution test with 15 second timeout
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, **env_vars} if env_vars else None
            )
            
            return {
                "executable": True,
                "return_code": result.returncode,
                "stdout_preview": result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout,
                "stderr_preview": result.stderr[:200] + "..." if len(result.stderr) > 200 else result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "executable": False,
                "error": "Execution timeout (15s)",
                "details": "Server may be hanging or require interactive input"
            }
        except Exception as e:
            return {
                "executable": False,
                "error": str(e),
                "details": "Failed to execute server command"
            }
    
    def calculate_server_status(self, tests: Dict) -> str:
        """Calculate overall server status based on test results."""
        
        # Check critical tests
        command_ok = tests.get("command_available", {}).get("available", False)
        package_ok = tests.get("package_available", {}).get("available", False)
        execution_ok = tests.get("execution", {}).get("executable", False)
        
        if command_ok and package_ok and execution_ok:
            return "healthy"
        elif command_ok and package_ok:
            return "configured_but_failing"
        elif command_ok:
            return "missing_dependencies"
        else:
            return "broken"
    
    def generate_repair_recommendations(self) -> List[Dict[str, str]]:
        """Generate repair recommendations based on diagnostic results."""
        
        recommendations = []
        
        for server_name, result in self.results.items():
            if server_name == "repair_recommendations":
                continue
                
            status = result.get("status", "unknown")
            tests = result.get("tests", {})
            
            if status == "broken":
                recommendations.append({
                    "server": server_name,
                    "priority": "critical",
                    "issue": "Command not available",
                    "fix": f"Install or fix base command: {result['config']['command']}"
                })
            
            elif status == "missing_dependencies":
                package_test = tests.get("package_available", {})
                if package_test.get("needs_install"):
                    package_name = self.extract_package_name(result["config"].get("args", []))
                    recommendations.append({
                        "server": server_name,
                        "priority": "high",
                        "issue": "Package not installed",
                        "fix": f"Run: npm install -g {package_name}"
                    })
            
            elif status == "configured_but_failing":
                execution_test = tests.get("execution", {})
                if "timeout" in execution_test.get("error", "").lower():
                    recommendations.append({
                        "server": server_name,
                        "priority": "medium",
                        "issue": "Server execution timeout",
                        "fix": f"Increase timeout in config or check server implementation"
                    })
        
        return recommendations
    
    def apply_automatic_repairs(self) -> Dict[str, any]:
        """Apply automatic repairs for common issues."""
        
        repair_results = {}
        recommendations = self.results.get("repair_recommendations", [])
        
        for rec in recommendations:
            server = rec["server"]
            fix = rec["fix"]
            
            logger.info(f"Attempting repair for {server}: {fix}")
            
            try:
                if "npm install -g" in fix:
                    # Extract package name and install
                    package_name = fix.split("npm install -g ")[1]
                    result = subprocess.run(
                        ["npm", "install", "-g", package_name],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    
                    repair_results[server] = {
                        "attempted": True,
                        "success": result.returncode == 0,
                        "details": result.stdout if result.returncode == 0 else result.stderr
                    }
                
                else:
                    repair_results[server] = {
                        "attempted": False,
                        "reason": "Manual intervention required",
                        "recommendation": fix
                    }
                    
            except Exception as e:
                repair_results[server] = {
                    "attempted": True,
                    "success": False,
                    "error": str(e)
                }
        
        return repair_results


import os

def main():
    """Main diagnostics execution."""
    
    diagnostics = MCPServerDiagnostics()
    
    print("=" * 80)
    print("MCP SERVER DIAGNOSTICS AND REPAIR")
    print("=" * 80)
    
    # Run diagnostics
    results = diagnostics.run_full_diagnostics()
    
    # Print results
    print("\nDIAGNOSTIC RESULTS:")
    print("-" * 40)
    
    for server_name, result in results.items():
        if server_name == "repair_recommendations":
            continue
            
        status = result.get("status", "unknown")
        print(f"\n{server_name.upper()}: {status}")
        
        tests = result.get("tests", {})
        for test_name, test_result in tests.items():
            if isinstance(test_result, dict):
                available = test_result.get("available", test_result.get("executable", "unknown"))
                print(f"  - {test_name}: {available}")
            else:
                print(f"  - {test_name}: {test_result}")
    
    # Print recommendations
    recommendations = results.get("repair_recommendations", [])
    if recommendations:
        print("\nREPAIR RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['server']} ({rec['priority']}): {rec['issue']}")
            print(f"   Fix: {rec['fix']}")
    
    # Apply automatic repairs
    print("\nAPPLYING AUTOMATIC REPAIRS:")
    print("-" * 40)
    repair_results = diagnostics.apply_automatic_repairs()
    
    for server, result in repair_results.items():
        if result.get("attempted"):
            success = result.get("success", False)
            print(f"{server}: {'SUCCESS' if success else 'FAILED'}")
            if not success and "error" in result:
                print(f"  Error: {result['error']}")
        else:
            print(f"{server}: MANUAL INTERVENTION REQUIRED")
            print(f"  Action: {result.get('recommendation', 'See diagnostics')}")
    
    print("\n" + "=" * 80)
    print("Diagnostics complete. Re-run to verify repairs.")
    

if __name__ == "__main__":
    main()