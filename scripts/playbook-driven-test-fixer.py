#!/usr/bin/env python3
"""
Playbook-Driven Test Failure Analysis and Fixing System
Uses existing .claude/playbooks as decision trees for intelligent test failure resolution
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PlaybookDrivenTestFixer:
    """Uses existing playbooks as decision trees for test failure resolution"""

    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.claude_dir = self.project_root / ".claude"
        self.playbooks_dir = self.claude_dir / "playbooks"
        self.test_failures_dir = self.claude_dir / "test-failures"
        self.failures_db = self.test_failures_dir / "failures.db"

        # Ensure directories exist
        self.test_failures_dir.mkdir(parents=True, exist_ok=True)

        # Load available playbooks
        self.playbooks = self._load_playbooks()

        print(f"🎯 Loaded {len(self.playbooks)} playbooks as decision trees")

    def _load_playbooks(self) -> Dict[str, Dict[str, Any]]:
        """Load all available playbooks"""
        playbooks = {}

        if not self.playbooks_dir.exists():
            print("⚠️ No playbooks directory found")
            return playbooks

        # Find all YAML playbooks
        for playbook_file in self.playbooks_dir.rglob("*.yml"):
            try:
                with open(playbook_file, "r") as f:
                    playbook_data = yaml.safe_load(f)
                    if playbook_data and isinstance(playbook_data, dict):
                        playbook_name = playbook_file.stem
                        playbooks[playbook_name] = {
                            "data": playbook_data,
                            "file_path": playbook_file,
                            "category": self._categorize_playbook(playbook_file),
                        }
                        print(f"✓ Loaded playbook: {playbook_name}")

            except Exception as e:
                print(f"⚠️ Failed to load playbook {playbook_file}: {e}")

        return playbooks

    def _categorize_playbook(self, playbook_file: Path) -> str:
        """Categorize playbook based on its purpose"""
        file_parts = playbook_file.parts

        if "slo" in str(playbook_file).lower():
            return "performance"
        elif "flake" in str(playbook_file).lower():
            return "stability"
        elif "cve" in str(playbook_file).lower():
            return "security"
        elif "loops" in file_parts:
            return "operational"
        else:
            return "general"

    def analyze_and_fix_with_playbooks(self, failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point: analyze failure and apply appropriate playbook"""
        print("🔍 Analyzing test failures with playbook decision trees...")

        try:
            # Step 1: Classify the failure type
            failure_classification = self._classify_failure_type(failure_data)
            print(f"📊 Failure classification: {failure_classification}")

            # Step 2: Select appropriate playbook
            selected_playbook = self._select_playbook(failure_classification, failure_data)
            print(f"📋 Selected playbook: {selected_playbook['name']}")

            # Step 3: Execute playbook phases
            execution_result = self._execute_playbook_phases(selected_playbook, failure_data)

            # Step 4: Validate results
            validation_result = self._validate_playbook_execution(execution_result)

            return {
                "status": "success",
                "classification": failure_classification,
                "playbook_used": selected_playbook["name"],
                "execution_result": execution_result,
                "validation": validation_result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            error_msg = f"Playbook-driven analysis failed: {e}"
            print(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}

    def _classify_failure_type(self, failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify failure type to determine appropriate playbook"""
        failures = self._extract_failures_list(failure_data)

        classification = {
            "primary_type": "unknown",
            "secondary_types": [],
            "severity": "medium",
            "patterns": [],
            "characteristics": {},
        }

        # Analyze failure patterns
        timing_issues = 0
        resource_issues = 0
        security_issues = 0
        performance_issues = 0
        stability_issues = 0

        for failure in failures:
            message = failure.get("failure_message", "").lower()
            stack_trace = failure.get("stack_trace", "").lower()
            combined_text = f"{message} {stack_trace}"

            # Pattern detection
            if any(keyword in combined_text for keyword in ["timeout", "slow", "hang", "delay"]):
                timing_issues += 1
                classification["patterns"].append("timing_dependency")

            if any(keyword in combined_text for keyword in ["memory", "resource", "connection", "file", "lock"]):
                resource_issues += 1
                classification["patterns"].append("resource_dependency")

            if any(keyword in combined_text for keyword in ["security", "auth", "permission", "certificate", "ssl"]):
                security_issues += 1
                classification["patterns"].append("security_issue")

            if any(keyword in combined_text for keyword in ["performance", "latency", "throughput", "load"]):
                performance_issues += 1
                classification["patterns"].append("performance_issue")

            if any(keyword in combined_text for keyword in ["flaky", "intermittent", "random", "sometimes"]):
                stability_issues += 1
                classification["patterns"].append("stability_issue")

        # Determine primary type based on pattern frequency
        pattern_counts = {
            "timing": timing_issues,
            "resource": resource_issues,
            "security": security_issues,
            "performance": performance_issues,
            "stability": stability_issues,
        }

        primary_pattern = max(pattern_counts.items(), key=lambda x: x[1])
        classification["primary_type"] = primary_pattern[0]
        classification["characteristics"] = pattern_counts

        # Determine severity
        if security_issues > 0:
            classification["severity"] = "critical"
        elif performance_issues > len(failures) * 0.5:
            classification["severity"] = "high"
        elif stability_issues > len(failures) * 0.3:
            classification["severity"] = "high"
        else:
            classification["severity"] = "medium"

        return classification

    def _select_playbook(self, classification: Dict[str, Any], failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select the most appropriate playbook based on classification"""
        primary_type = classification["primary_type"]
        severity = classification["severity"]
        patterns = classification["patterns"]

        # Playbook selection logic
        selected_playbook = None
        selection_score = 0

        for playbook_name, playbook_info in self.playbooks.items():
            score = self._calculate_playbook_relevance_score(playbook_info, primary_type, severity, patterns)

            if score > selection_score:
                selection_score = score
                selected_playbook = {"name": playbook_name, "info": playbook_info, "relevance_score": score}

        # Default to flakes playbook if no better match (most general)
        if not selected_playbook and "flakes" in self.playbooks:
            selected_playbook = {"name": "flakes", "info": self.playbooks["flakes"], "relevance_score": 0.5}

        return selected_playbook or {"name": "generic", "info": {"data": {"stages": []}}, "relevance_score": 0.1}

    def _calculate_playbook_relevance_score(
        self, playbook_info: Dict[str, Any], primary_type: str, severity: str, patterns: List[str]
    ) -> float:
        """Calculate how relevant a playbook is for the given failure characteristics"""
        score = 0.0
        playbook_data = playbook_info["data"]
        category = playbook_info["category"]

        # Category matching
        category_scores = {
            "stability": {"stability": 1.0, "timing": 0.8, "resource": 0.6},
            "performance": {"performance": 1.0, "timing": 0.7, "resource": 0.5},
            "security": {"security": 1.0},
            "operational": {"timing": 0.5, "resource": 0.5, "performance": 0.5},
        }

        if category in category_scores and primary_type in category_scores[category]:
            score += category_scores[category][primary_type]

        # Pattern matching in playbook content
        playbook_text = str(playbook_data).lower()
        for pattern in patterns:
            if pattern.replace("_", " ") in playbook_text:
                score += 0.2

        # Specific playbook characteristics
        if "flake" in playbook_text and "stability" in patterns:
            score += 0.5
        if "slo" in playbook_text and "performance" in patterns:
            score += 0.5
        if "cve" in playbook_text and "security" in patterns:
            score += 0.5

        return min(score, 1.0)  # Cap at 1.0

    def _execute_playbook_phases(
        self, selected_playbook: Dict[str, Any], failure_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the selected playbook phases step by step"""
        playbook_data = selected_playbook["info"]["data"]
        playbook_name = selected_playbook["name"]

        print(f"🎭 Executing playbook: {playbook_name}")

        execution_result = {
            "playbook_name": playbook_name,
            "phases_executed": [],
            "outputs": {},
            "success": False,
            "errors": [],
        }

        # Get phases/stages from playbook
        phases = self._extract_phases_from_playbook(playbook_data)

        for phase in phases:
            try:
                print(f"🔄 Executing phase: {phase['name']}")
                phase_result = self._execute_playbook_phase(phase, failure_data)

                execution_result["phases_executed"].append(
                    {"name": phase["name"], "result": phase_result, "timestamp": datetime.now().isoformat()}
                )

                # Store phase outputs
                if phase_result.get("outputs"):
                    execution_result["outputs"].update(phase_result["outputs"])

                # Check if phase failed critically
                if phase_result.get("critical_failure"):
                    execution_result["errors"].append(f"Critical failure in phase {phase['name']}")
                    break

            except Exception as e:
                error_msg = f"Phase {phase['name']} failed: {e}"
                print(f"❌ {error_msg}")
                execution_result["errors"].append(error_msg)

        execution_result["success"] = len(execution_result["errors"]) == 0

        return execution_result

    def _extract_phases_from_playbook(self, playbook_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract executable phases from playbook structure"""
        phases = []

        # Try different playbook structures
        if "stages" in playbook_data:
            # CVE/SLO style playbook
            for stage in playbook_data["stages"]:
                if isinstance(stage, dict) and "name" in stage:
                    phases.append(stage)

        elif "phases" in playbook_data:
            # Flakes style playbook
            for phase_name, phase_data in playbook_data["phases"].items():
                if isinstance(phase_data, dict):
                    phase = phase_data.copy()
                    phase["name"] = phase_name
                    phases.append(phase)

        # Default phases if none found
        if not phases:
            phases = [
                {"name": "analyze", "description": "Analyze the failure"},
                {"name": "fix", "description": "Apply automated fix"},
                {"name": "validate", "description": "Validate the fix"},
            ]

        return phases

    def _execute_playbook_phase(self, phase: Dict[str, Any], failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single playbook phase"""
        phase_name = phase.get("name", "unnamed")
        phase_description = phase.get("description", "No description")

        print(f"  📋 {phase_name}: {phase_description}")

        # Map phase to execution strategy
        execution_strategies = {
            "harvest": self._execute_harvest_phase,
            "cluster": self._execute_cluster_phase,
            "stabilize": self._execute_stabilize_phase,
            "classify": self._execute_classify_phase,
            "analyze": self._execute_analyze_phase,
            "fix": self._execute_fix_phase,
            "validate": self._execute_validate_phase,
        }

        # Find matching strategy
        strategy_func = None
        for strategy_name, func in execution_strategies.items():
            if strategy_name in phase_name.lower():
                strategy_func = func
                break

        # Default to generic execution
        if not strategy_func:
            strategy_func = self._execute_generic_phase

        # Execute the phase
        return strategy_func(phase, failure_data)

    def _execute_harvest_phase(self, phase: Dict[str, Any], failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute harvest/detection phase using Claude Flow"""
        print("    🔍 Harvesting failure patterns...")

        try:
            # Use Claude Flow to analyze failures
            harvest_prompt = self._create_harvest_prompt(failure_data, phase)

            # Execute with Claude Flow researcher agent
            result = subprocess.run(
                ["npx", "claude-flow", "sparc", "run", "researcher", harvest_prompt],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,
            )

            if result.returncode == 0:
                analysis_output = result.stdout

                # Save analysis results
                output_file = self.test_failures_dir / "harvest_analysis.json"
                analysis_data = {
                    "phase": "harvest",
                    "analysis": analysis_output,
                    "patterns_detected": self._extract_patterns_from_analysis(analysis_output),
                    "timestamp": datetime.now().isoformat(),
                }

                with open(output_file, "w") as f:
                    json.dump(analysis_data, f, indent=2)

                return {
                    "success": True,
                    "outputs": {"harvest_analysis": str(output_file)},
                    "patterns_found": len(analysis_data["patterns_detected"]),
                }
            else:
                return {"success": False, "error": f"Claude Flow analysis failed: {result.stderr}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_stabilize_phase(self, phase: Dict[str, Any], failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stabilization phase using TDD London School approach"""
        print("    🔧 Stabilizing tests with TDD London School approach...")

        try:
            # Create stabilization prompt based on playbook guidance
            stabilize_prompt = self._create_stabilization_prompt(failure_data, phase)

            # Execute with Claude Flow tester agent (TDD specialist)
            result = subprocess.run(
                ["npx", "claude-flow", "sparc", "run", "tester", stabilize_prompt],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=600,
            )

            if result.returncode == 0:
                fix_output = result.stdout

                # Apply the fixes
                fixes_applied = self._apply_stabilization_fixes(fix_output)

                # Save stabilization results
                output_file = self.test_failures_dir / "stabilization_results.json"
                stabilization_data = {
                    "phase": "stabilize",
                    "fixes_applied": fixes_applied,
                    "stabilization_output": fix_output,
                    "timestamp": datetime.now().isoformat(),
                }

                with open(output_file, "w") as f:
                    json.dump(stabilization_data, f, indent=2)

                return {
                    "success": True,
                    "outputs": {"stabilization_results": str(output_file)},
                    "fixes_applied": len(fixes_applied),
                }
            else:
                return {"success": False, "error": f"Stabilization failed: {result.stderr}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_validate_phase(self, phase: Dict[str, Any], failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation phase with comprehensive testing"""
        print("    ✅ Validating fixes with comprehensive testing...")

        try:
            # Run tests to validate fixes
            validation_result = self._run_validation_tests(failure_data)

            # Save validation results
            output_file = self.test_failures_dir / "validation_results.json"
            with open(output_file, "w") as f:
                json.dump(validation_result, f, indent=2)

            return {
                "success": validation_result.get("all_tests_passed", False),
                "outputs": {"validation_results": str(output_file)},
                "test_results": validation_result,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_generic_phase(self, phase: Dict[str, Any], failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic phase using appropriate Claude Flow agent"""
        phase_name = phase.get("name", "generic")
        print(f"    🔄 Executing generic phase: {phase_name}")

        # Select appropriate agent based on phase characteristics
        agent_type = self._select_agent_for_phase(phase)

        # Create generic prompt
        generic_prompt = self._create_generic_phase_prompt(phase, failure_data)

        try:
            # Execute with selected agent
            result = subprocess.run(
                ["npx", "claude-flow", "sparc", "run", agent_type, generic_prompt],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,
            )

            if result.returncode == 0:
                return {"success": True, "output": result.stdout, "agent_used": agent_type}
            else:
                return {"success": False, "error": f"Agent {agent_type} failed: {result.stderr}"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_harvest_prompt(self, failure_data: Dict[str, Any], phase: Dict[str, Any]) -> str:
        """Create prompt for harvest phase based on playbook guidance"""
        failures = self._extract_failures_list(failure_data)

        prompt = f"""
Analyze the following test failures using the harvest phase methodology:

**Phase**: {phase.get('name', 'harvest')}
**Description**: {phase.get('description', 'Analyze and harvest failure patterns')}

**Test Failures to Analyze**:
{json.dumps(failures, indent=2)}

**Analysis Tasks** (based on playbook guidance):
1. Detect flake patterns and root causes
2. Identify timing dependencies (sleep, timeout, race conditions)
3. Find resource dependencies (file system, network, database)
4. Discover concurrency issues (thread safety, async/await)
5. Spot environment dependencies (OS, browser, device specific)
6. Detect data dependencies (test order, shared state)

**Required Output**:
- Categorized failure patterns
- Root cause analysis for each pattern
- Recommendations for stabilization approach
- Priority scoring based on impact

Please provide a comprehensive analysis following the harvest phase methodology.
"""
        return prompt

    def _create_stabilization_prompt(self, failure_data: Dict[str, Any], phase: Dict[str, Any]) -> str:
        """Create stabilization prompt using TDD London School approach"""
        failures = self._extract_failures_list(failure_data)

        prompt = f"""
Stabilize the following test failures using TDD London School methodology:

**Phase**: {phase.get('name', 'stabilize')}
**Approach**: London School TDD with mock-based isolation

**Test Failures to Stabilize**:
{json.dumps(failures, indent=2)}

**TDD London School Stabilization Techniques**:

1. **Deterministic Mocking**: Replace time.sleep() with mock controls
2. **Event-Driven Waiting**: Use explicit waits instead of sleeps  
3. **Clock Injection**: Inject controllable time sources
4. **Resource Mocking**: Mock file system, network, database
5. **Cleanup Automation**: Ensure resource cleanup in finally blocks
6. **Thread Synchronization**: Use proper locking mechanisms
7. **Async Isolation**: Isolate async operations with mocks

**Validation Strategy**:
1. Mock all external dependencies
2. Verify behavior contracts, not implementation
3. Use property-based testing for timing ranges
4. Validate with stress testing (100+ runs)

**Required Output**:
- Stabilized test code for each failure
- Mock-based isolation implementations
- Behavioral test contracts
- Cleanup and resource management code

Please provide complete, working stabilized test code following TDD London School principles.
"""
        return prompt

    def _run_validation_tests(self, failure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation tests to verify fixes"""
        failures = self._extract_failures_list(failure_data)

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": [],
            "all_tests_passed": True,
            "pass_rate": 0.0,
            "failures": [],
        }

        for failure in failures:
            test_name = failure.get("test_name", "")
            file_path = failure.get("file_path", "")

            if not test_name and not file_path:
                continue

            # Run the specific test
            test_result = self._run_single_test(test_name, file_path)
            validation_results["tests_run"].append(test_result)

            if not test_result.get("passed", False):
                validation_results["all_tests_passed"] = False
                validation_results["failures"].append(test_result)

        # Calculate pass rate
        if validation_results["tests_run"]:
            passed_tests = sum(1 for t in validation_results["tests_run"] if t.get("passed"))
            validation_results["pass_rate"] = passed_tests / len(validation_results["tests_run"])

        return validation_results

    def _run_single_test(self, test_name: str, file_path: str) -> Dict[str, Any]:
        """Run a single test and return results"""
        try:
            # Determine test command
            if file_path.endswith(".py"):
                cmd = ["python", "-m", "pytest", file_path, "-v", "--tb=short"]
                if test_name:
                    cmd.extend(["-k", test_name])
            elif file_path.endswith((".js", ".ts")):
                cmd = ["npm", "test", "--", "--testNamePattern=" + test_name]
            else:
                cmd = ["python", "-m", "pytest", "-k", test_name] if test_name else ["python", "-m", "pytest"]

            # Run test
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root, timeout=60)

            return {
                "test_name": test_name,
                "file_path": file_path,
                "passed": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "command": " ".join(cmd),
            }

        except Exception as e:
            return {
                "test_name": test_name,
                "file_path": file_path,
                "passed": False,
                "error": str(e),
                "command": "failed_to_run",
            }

    # Additional helper methods for the main execution flow
    def _extract_failures_list(self, failure_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract list of failures from various data formats"""
        if isinstance(failure_data, dict):
            if "categorized" in failure_data:
                failures = []
                for category, category_failures in failure_data["categorized"].items():
                    failures.extend(category_failures)
                return failures
            elif "raw_failures" in failure_data:
                return failure_data["raw_failures"]
            elif "failures" in failure_data:
                return failure_data["failures"]
        elif isinstance(failure_data, list):
            return failure_data

        return []

    def _extract_patterns_from_analysis(self, analysis_output: str) -> List[str]:
        """Extract patterns from Claude's analysis output"""
        patterns = []

        # Look for common pattern indicators
        pattern_keywords = ["timing", "resource", "concurrency", "environment", "data", "flake"]

        for line in analysis_output.split("\n"):
            line_lower = line.lower()
            for keyword in pattern_keywords:
                if keyword in line_lower and "pattern" in line_lower:
                    patterns.append(keyword)

        return list(set(patterns))  # Remove duplicates

    def _apply_stabilization_fixes(self, fix_output: str) -> List[Dict[str, Any]]:
        """Apply stabilization fixes from Claude's output"""
        fixes_applied = []

        # Extract code blocks from output
        import re

        code_blocks = re.findall(r"```(?:python|js|javascript)?\n(.*?)```", fix_output, re.DOTALL)

        for i, code_block in enumerate(code_blocks):
            try:
                # For now, save the fix code for manual review
                fix_file = self.test_failures_dir / f"generated_fix_{i}.py"
                with open(fix_file, "w") as f:
                    f.write(code_block)

                fixes_applied.append(
                    {"fix_number": i, "file_path": str(fix_file), "code_length": len(code_block), "applied": True}
                )

            except Exception as e:
                fixes_applied.append({"fix_number": i, "error": str(e), "applied": False})

        return fixes_applied

    def _select_agent_for_phase(self, phase: Dict[str, Any]) -> str:
        """Select appropriate Claude Flow agent for a phase"""
        phase_name = phase.get("name", "").lower()

        if "analyze" in phase_name or "research" in phase_name:
            return "researcher"
        elif "fix" in phase_name or "patch" in phase_name:
            return "coder"
        elif "test" in phase_name or "validate" in phase_name:
            return "tester"
        elif "review" in phase_name:
            return "reviewer"
        elif "plan" in phase_name:
            return "planner"
        else:
            return "coder"  # Default to coder

    def _create_generic_phase_prompt(self, phase: Dict[str, Any], failure_data: Dict[str, Any]) -> str:
        """Create generic prompt for any phase"""
        return f"""
Execute the following phase from the test stabilization playbook:

**Phase**: {phase.get('name', 'Unknown')}
**Description**: {phase.get('description', 'No description provided')}

**Context**: Test failure analysis and fixing

**Failure Data**:
{json.dumps(self._extract_failures_list(failure_data), indent=2)}

**Instructions**: 
{phase.get('instructions', 'Apply appropriate analysis and fixes for the given failures')}

Please execute this phase according to the playbook methodology and provide actionable results.
"""

    def _validate_playbook_execution(self, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the overall playbook execution"""
        validation = {
            "overall_success": execution_result.get("success", False),
            "phases_completed": len(execution_result.get("phases_executed", [])),
            "critical_errors": len(execution_result.get("errors", [])),
            "outputs_generated": len(execution_result.get("outputs", {})),
            "recommendations": [],
        }

        # Add recommendations based on results
        if not validation["overall_success"]:
            validation["recommendations"].append("Review error logs and retry failed phases")

        if validation["phases_completed"] < 3:
            validation["recommendations"].append("Consider running additional stabilization phases")

        if validation["outputs_generated"] == 0:
            validation["recommendations"].append("No outputs generated - check agent configuration")

        return validation


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Playbook-driven test failure analysis and fixing")
    parser.add_argument("--failure-data", required=True, help="JSON file containing test failure data")
    parser.add_argument("--playbook", help="Specific playbook to use (optional)")
    parser.add_argument("--output-dir", default=".claude/test-failures", help="Output directory for results")

    args = parser.parse_args()

    # Load failure data
    if os.path.isfile(args.failure_data):
        with open(args.failure_data, "r") as f:
            failure_data = json.load(f)
    else:
        try:
            failure_data = json.loads(args.failure_data)
        except json.JSONDecodeError:
            print(f"❌ Invalid failure data: {args.failure_data}")
            sys.exit(1)

    # Initialize playbook-driven fixer
    fixer = PlaybookDrivenTestFixer()

    # Execute playbook-driven analysis and fixing
    result = fixer.analyze_and_fix_with_playbooks(failure_data)

    # Save results
    result_file = Path(args.output_dir) / "playbook_execution_results.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    if result.get("status") == "success":
        print("\n✅ Playbook execution completed successfully!")
        print(f"📋 Playbook used: {result['playbook_used']}")
        print(f"🔄 Phases executed: {len(result['execution_result']['phases_executed'])}")
        print(f"📊 Results saved to: {result_file}")
    else:
        print(f"\n❌ Playbook execution failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
