#!/usr/bin/env python3
"""
Pre-commit Configuration Validation Script

Validates the optimized pre-commit configuration meets performance and reliability targets.
Provides detailed analysis and recommendations for further optimization.
"""

from pathlib import Path
import subprocess  # nosec B404
import sys

import yaml


class PreCommitValidator:
    """Validate pre-commit configuration optimization."""

    def __init__(self):
        """Initialize validator."""
        self.repo_root = Path(__file__).parent.parent.parent
        self.config_file = self.repo_root / ".pre-commit-config.yaml"
        self.target_time = 120.0  # 2 minutes
        self.warning_time = 90.0  # 1.5 minutes

    def validate_configuration(self) -> dict:
        """Validate the pre-commit configuration."""
        print("🔍 Validating Pre-commit Configuration Optimization")
        print("=" * 60)

        results = {
            "config_valid": False,
            "performance_optimized": False,
            "file_filtering_effective": False,
            "parallel_execution_enabled": False,
            "timeouts_configured": False,
            "caching_implemented": False,
            "recommendations": [],
            "estimated_performance": "unknown",
        }

        # Load configuration
        config = self._load_config()
        if not config:
            results["recommendations"].append("❌ Could not load .pre-commit-config.yaml")
            return results

        results["config_valid"] = True
        print("✅ Configuration file loaded successfully")

        # Validate optimizations
        results.update(self._validate_file_filtering(config))
        results.update(self._validate_parallel_execution(config))
        results.update(self._validate_timeouts(config))
        results.update(self._validate_caching(config))
        results.update(self._validate_performance_settings(config))

        # Analyze file scope
        scope_analysis = self._analyze_file_scope()
        print(f"📊 File Scope Analysis: {scope_analysis}")

        # Estimate performance
        results["estimated_performance"] = self._estimate_performance(config, scope_analysis)

        return results

    def _load_config(self) -> dict:
        """Load pre-commit configuration."""
        try:
            with open(self.config_file) as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"❌ Error loading configuration: {e}")
            return {}

    def _validate_file_filtering(self, config: dict) -> dict:
        """Validate file filtering optimization."""
        results = {"file_filtering_effective": False}

        # Check global exclude pattern
        global_exclude = config.get("exclude", "")
        if "deprecated/" in global_exclude and "__pycache__/" in global_exclude:
            print("✅ Comprehensive global exclusion pattern configured")
            results["file_filtering_effective"] = True
        else:
            results["recommendations"] = results.get("recommendations", [])
            results["recommendations"].append("⚠️  Consider adding comprehensive global exclusion pattern")

        # Check per-hook file filtering
        hooks_with_files = 0
        total_hooks = 0

        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                total_hooks += 1
                if "files" in hook:
                    hooks_with_files += 1

        if hooks_with_files > 0:
            print(f"✅ {hooks_with_files}/{total_hooks} hooks have file type filtering")
        else:
            results["recommendations"] = results.get("recommendations", [])
            results["recommendations"].append("⚠️  Consider adding file type filtering to hooks (files: \\.py$)")

        return results

    def _validate_parallel_execution(self, config: dict) -> dict:
        """Validate parallel execution optimization."""
        results = {"parallel_execution_enabled": False}

        parallel_configs = []

        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                hook_id = hook.get("id", "")
                args = hook.get("args", [])

                # Check for parallel execution flags
                if hook_id == "black" and "--workers" in " ".join(args):
                    parallel_configs.append("Black")
                elif hook_id == "isort" and "--jobs" in " ".join(args):
                    parallel_configs.append("isort")

        if parallel_configs:
            print(f"✅ Parallel execution enabled for: {', '.join(parallel_configs)}")
            results["parallel_execution_enabled"] = True
        else:
            results["recommendations"] = results.get("recommendations", [])
            results["recommendations"].append(
                "⚠️  Enable parallel execution for Black (--workers=4) and isort (--jobs=4)"
            )

        return results

    def _validate_timeouts(self, config: dict) -> dict:
        """Validate timeout configuration."""
        results = {"timeouts_configured": False}

        timeout_hooks = []

        for repo in config.get("repos", []):
            if repo.get("repo") == "local":
                for hook in repo.get("hooks", []):
                    entry = hook.get("entry", "")
                    if "timeout" in entry:
                        timeout_hooks.append(hook.get("id", "unknown"))

        if timeout_hooks:
            print(f"✅ Timeouts configured for: {', '.join(timeout_hooks)}")
            results["timeouts_configured"] = True
        else:
            results["recommendations"] = results.get("recommendations", [])
            results["recommendations"].append("⚠️  Consider adding timeouts to slow local hooks (timeout 30s)")

        return results

    def _validate_caching(self, config: dict) -> dict:
        """Validate caching implementation."""
        results = {"caching_implemented": False}

        cache_indicators = []

        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                args = hook.get("args", [])
                if "--cache" in " ".join(args) or "--fast-mode" in " ".join(args):
                    cache_indicators.append(hook.get("id", "unknown"))

        # Check for cache files
        cache_files = [".god-object-cache.json", ".magic-literal-cache.json", ".pre-commit-metrics.json"]

        existing_caches = [f for f in cache_files if (self.repo_root / f).exists()]

        if cache_indicators or existing_caches:
            print(f"✅ Caching implemented: {len(cache_indicators)} hooks, {len(existing_caches)} cache files")
            results["caching_implemented"] = True
        else:
            results["recommendations"] = results.get("recommendations", [])
            results["recommendations"].append("⚠️  Implement caching for expensive analysis hooks")

        return results

    def _validate_performance_settings(self, config: dict) -> dict:
        """Validate performance-oriented settings."""
        results = {"performance_optimized": False}

        optimizations = []

        # Check fail_fast setting
        if config.get("fail_fast", False):
            optimizations.append("fail_fast enabled")

        # Check default_stages
        if config.get("default_stages") == ["commit"]:
            optimizations.append("default_stages optimized")

        # Check stage usage
        stage_hooks = 0
        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                if "stages" in hook:
                    stage_hooks += 1

        if stage_hooks > 0:
            optimizations.append(f"{stage_hooks} hooks use stage optimization")

        if len(optimizations) >= 2:
            print(f"✅ Performance settings optimized: {', '.join(optimizations)}")
            results["performance_optimized"] = True
        else:
            results["recommendations"] = results.get("recommendations", [])
            results["recommendations"].append(
                "⚠️  Enable performance settings: fail_fast, default_stages, stage separation"
            )

        return results

    def _analyze_file_scope(self) -> dict:
        """Analyze effective file scope after filtering."""
        analysis = {"total_py_files": 0, "excluded_files": 0, "effective_files": 0, "reduction_percentage": 0.0}

        try:
            # Count total Python files
            result = subprocess.run(  # nosec B603 B607
                ["find", ".", "-name", "*.py", "-type", "f"], capture_output=True, text=True, cwd=self.repo_root
            )

            if result.returncode == 0:
                analysis["total_py_files"] = len(result.stdout.strip().split("\n"))

            # Count effective files (after exclusions)
            result = subprocess.run(  # nosec B603 B607
                [
                    "find",
                    ".",
                    "-name",
                    "*.py",
                    "-type",
                    "f",
                    "!",
                    "-path",
                    "*/deprecated/*",
                    "!",
                    "-path",
                    "*/archive/*",
                    "!",
                    "-path",
                    "*/backups/*",
                    "!",
                    "-path",
                    "*/__pycache__/*",
                    "!",
                    "-path",
                    "*/.git/*",
                    "!",
                    "-path",
                    "*/.pytest_cache/*",
                ],
                capture_output=True,
                text=True,
                cwd=self.repo_root,
            )

            if result.returncode == 0:
                effective_files = len([line for line in result.stdout.strip().split("\n") if line])
                analysis["effective_files"] = effective_files
                analysis["excluded_files"] = analysis["total_py_files"] - effective_files

                if analysis["total_py_files"] > 0:
                    analysis["reduction_percentage"] = analysis["excluded_files"] / analysis["total_py_files"] * 100

        except subprocess.SubprocessError:
            print("⚠️  Could not analyze file scope")

        return analysis

    def _estimate_performance(self, config: dict, scope: dict) -> str:
        """Estimate performance category based on configuration and scope."""
        score = 0
        max_score = 10

        # File filtering (0-3 points)
        if scope.get("reduction_percentage", 0) > 50:
            score += 3
        elif scope.get("reduction_percentage", 0) > 25:
            score += 2
        elif scope.get("reduction_percentage", 0) > 10:
            score += 1

        # Parallel execution (0-2 points)
        if self._has_parallel_execution(config):
            score += 2

        # Timeouts (0-2 points)
        if self._has_timeouts(config):
            score += 2

        # Caching (0-2 points)
        if self._has_caching(config):
            score += 2

        # Performance settings (0-1 point)
        if config.get("fail_fast", False):
            score += 1

        performance_percentage = (score / max_score) * 100

        if performance_percentage >= 80:
            return f"excellent ({performance_percentage:.0f}%)"
        elif performance_percentage >= 60:
            return f"good ({performance_percentage:.0f}%)"
        elif performance_percentage >= 40:
            return f"moderate ({performance_percentage:.0f}%)"
        else:
            return f"poor ({performance_percentage:.0f}%)"

    def _has_parallel_execution(self, config: dict) -> bool:
        """Check if parallel execution is configured."""
        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                args = " ".join(hook.get("args", []))
                if "--workers" in args or "--jobs" in args:
                    return True
        return False

    def _has_timeouts(self, config: dict) -> bool:
        """Check if timeouts are configured."""
        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                if "timeout" in hook.get("entry", ""):
                    return True
        return False

    def _has_caching(self, config: dict) -> bool:
        """Check if caching is implemented."""
        for repo in config.get("repos", []):
            for hook in repo.get("hooks", []):
                args = " ".join(hook.get("args", []))
                if "--cache" in args or "--fast-mode" in args:
                    return True
        return False

    def print_summary(self, results: dict) -> None:
        """Print validation summary."""
        print("\n🎯 Optimization Validation Summary")
        print("=" * 40)

        checks = [
            ("Configuration Valid", results["config_valid"]),
            ("File Filtering", results["file_filtering_effective"]),
            ("Parallel Execution", results["parallel_execution_enabled"]),
            ("Timeouts Configured", results["timeouts_configured"]),
            ("Caching Implemented", results["caching_implemented"]),
            ("Performance Optimized", results["performance_optimized"]),
        ]

        passed = sum(1 for _, status in checks if status)
        total = len(checks)

        for check, status in checks:
            icon = "✅" if status else "❌"
            print(f"{icon} {check}")

        print(f"\n📊 Overall Score: {passed}/{total} ({passed/total*100:.0f}%)")
        print(f"🚀 Estimated Performance: {results['estimated_performance']}")

        if results["recommendations"]:
            print(f"\n💡 Recommendations ({len(results['recommendations'])}):")
            for rec in results["recommendations"]:
                print(f"  {rec}")

        # Performance prediction
        if passed >= 5:
            print(f"\n🎉 Configuration is well-optimized for <{self.target_time}s target!")
        elif passed >= 3:
            print(f"\n⚠️  Configuration needs improvement to achieve <{self.target_time}s target")
        else:
            print("\n🔴 Configuration requires significant optimization for performance target")


def main():
    """Main entry point."""
    validator = PreCommitValidator()
    results = validator.validate_configuration()
    validator.print_summary(results)

    # Return appropriate exit code
    if results["config_valid"] and results["performance_optimized"]:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
