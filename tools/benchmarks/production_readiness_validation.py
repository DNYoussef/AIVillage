#!/usr/bin/env python3
"""
Production Readiness Validation

Final validation of AIVillage system performance for production deployment.
Uses real system measurements to determine deployment readiness.
"""

import json
import sys
from datetime import datetime
from pathlib import Path


class ProductionReadinessValidator:
    """Validates system readiness for production deployment based on real performance data."""

    def __init__(self):
        self.criteria = {
            "minimum_success_rate": 0.90,
            "maximum_average_latency_ms": 2000,
            "minimum_throughput_ops_sec": 5.0,
            "minimum_component_coverage": 0.60,  # 60% of components tested
            "maximum_critical_issues": 2,
        }

        self.production_requirements = {
            "high_availability": 0.95,  # 95% uptime requirement
            "performance_sla": 1000,  # 1s max response time SLA
            "scalability_target": 100,  # Handle 100 concurrent users
            "reliability_target": 0.99,  # 99% success rate target
        }

    def validate_performance_data(self, report_file: str) -> dict:
        """Validate production readiness based on performance report."""
        try:
            with open(report_file) as f:
                data = json.load(f)

            return self._assess_production_readiness(data)

        except Exception as e:
            return {
                "ready": False,
                "error": f"Failed to load performance data: {e}",
                "recommendation": "Fix performance data collection before deployment",
            }

    def _assess_production_readiness(self, data: dict) -> dict:
        """Assess production readiness based on performance metrics."""
        summary = data.get("performance_test_summary", {})
        components = data.get("component_results", [])

        # Calculate key metrics
        components_tested = summary.get("components_tested", 0)
        total_operations = summary.get("total_operations", 0)
        average_success_rate = summary.get("average_success_rate", 0)

        if not components:
            return {
                "ready": False,
                "score": 0.0,
                "critical_issues": ["No component performance data available"],
                "recommendation": "Complete performance testing before deployment",
            }

        # Calculate component coverage (assuming 4 core components)
        component_coverage = components_tested / 4.0

        # Calculate weighted average latency
        total_weight = sum(c["items_processed"] for c in components)
        if total_weight > 0:
            weighted_latency = sum(c["latency_ms"] * c["items_processed"] for c in components) / total_weight
        else:
            weighted_latency = float("inf")

        # Calculate combined throughput
        combined_throughput = sum(c["throughput"] for c in components)

        # Validate against criteria
        validation_results = self._validate_criteria(
            average_success_rate, weighted_latency, combined_throughput, component_coverage, components
        )

        # Calculate readiness score
        readiness_score = self._calculate_readiness_score(validation_results)

        return {
            "ready": readiness_score >= 0.8,  # 80% threshold for production
            "score": readiness_score,
            "validation_results": validation_results,
            "key_metrics": {
                "success_rate": average_success_rate,
                "average_latency_ms": weighted_latency,
                "combined_throughput": combined_throughput,
                "component_coverage": component_coverage,
                "total_operations": total_operations,
            },
            "production_assessment": self._generate_production_assessment(readiness_score, validation_results),
            "deployment_recommendation": self._generate_deployment_recommendation(readiness_score, validation_results),
        }

    def _validate_criteria(self, success_rate, latency, throughput, coverage, components):
        """Validate individual performance criteria."""
        results = {}

        # Success rate validation
        results["success_rate"] = {
            "value": success_rate,
            "threshold": self.criteria["minimum_success_rate"],
            "passed": success_rate >= self.criteria["minimum_success_rate"],
            "impact": "critical" if success_rate < 0.8 else "minor",
        }

        # Latency validation
        results["latency"] = {
            "value": latency,
            "threshold": self.criteria["maximum_average_latency_ms"],
            "passed": latency <= self.criteria["maximum_average_latency_ms"],
            "impact": "major" if latency > 5000 else "minor",
        }

        # Throughput validation
        results["throughput"] = {
            "value": throughput,
            "threshold": self.criteria["minimum_throughput_ops_sec"],
            "passed": throughput >= self.criteria["minimum_throughput_ops_sec"],
            "impact": "major" if throughput < 1.0 else "minor",
        }

        # Coverage validation
        results["coverage"] = {
            "value": coverage,
            "threshold": self.criteria["minimum_component_coverage"],
            "passed": coverage >= self.criteria["minimum_component_coverage"],
            "impact": "major" if coverage < 0.5 else "minor",
        }

        # Component-specific validation
        results["components"] = {}
        critical_issues = []

        for component in components:
            comp_name = component["component"]
            comp_success = component["success_rate"]
            comp_latency = component["latency_ms"]
            comp_throughput = component["throughput"]

            results["components"][comp_name] = {
                "success_rate_ok": comp_success >= 0.90,
                "latency_ok": comp_latency <= 2000,
                "throughput_ok": comp_throughput >= 0.5,
                "overall_ok": comp_success >= 0.90 and comp_latency <= 2000 and comp_throughput >= 0.5,
            }

            if comp_success < 0.80:
                critical_issues.append(f"{comp_name}: Low success rate ({comp_success:.1%})")
            if comp_latency > 5000:
                critical_issues.append(f"{comp_name}: High latency ({comp_latency:.0f}ms)")

        results["critical_issues"] = critical_issues
        results["critical_issues_count"] = len(critical_issues)

        return results

    def _calculate_readiness_score(self, validation_results):
        """Calculate overall readiness score (0-1)."""
        weights = {
            "success_rate": 0.30,  # 30% weight - most important
            "latency": 0.25,  # 25% weight
            "throughput": 0.20,  # 20% weight
            "coverage": 0.25,  # 25% weight
        }

        score = 0.0

        for criterion, weight in weights.items():
            if criterion in validation_results and validation_results[criterion]["passed"]:
                score += weight
            elif criterion in validation_results:
                # Partial credit based on how close to threshold
                value = validation_results[criterion]["value"]
                threshold = validation_results[criterion]["threshold"]

                if criterion == "latency":  # Lower is better
                    partial = max(0, 1 - (value - threshold) / threshold) if value > threshold else 1
                else:  # Higher is better
                    partial = value / threshold if threshold > 0 else 0
                    partial = min(1.0, partial)

                score += weight * partial

        # Penalize critical issues
        critical_penalty = min(0.3, validation_results["critical_issues_count"] * 0.1)
        score = max(0.0, score - critical_penalty)

        return score

    def _generate_production_assessment(self, score, validation_results):
        """Generate production readiness assessment."""

        if score >= 0.90:
            return {
                "level": "PRODUCTION_READY",
                "confidence": "HIGH",
                "description": "System meets all production criteria with high confidence",
                "deployment_window": "Immediate",
            }
        elif score >= 0.80:
            return {
                "level": "CONDITIONALLY_READY",
                "confidence": "MEDIUM",
                "description": "System meets core criteria with minor issues",
                "deployment_window": "Within 1 week after addressing minor issues",
            }
        elif score >= 0.60:
            return {
                "level": "NEEDS_IMPROVEMENT",
                "confidence": "LOW",
                "description": "System has significant performance issues requiring attention",
                "deployment_window": "2-4 weeks after major improvements",
            }
        else:
            return {
                "level": "NOT_READY",
                "confidence": "VERY_LOW",
                "description": "System has critical issues preventing production deployment",
                "deployment_window": "1-3 months after major fixes",
            }

    def _generate_deployment_recommendation(self, score, validation_results):
        """Generate specific deployment recommendations."""

        recommendations = []

        # Success rate recommendations
        success_rate_info = validation_results.get("success_rate", {})
        if not success_rate_info.get("passed", False):
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Reliability",
                    "issue": f"Success rate {success_rate_info['value']:.1%} below {success_rate_info['threshold']:.1%} minimum",
                    "action": "Investigate and fix error conditions causing failures",
                }
            )

        # Latency recommendations
        latency_info = validation_results.get("latency", {})
        if not latency_info.get("passed", False):
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Performance",
                    "issue": f"Average latency {latency_info['value']:.0f}ms exceeds {latency_info['threshold']}ms threshold",
                    "action": "Optimize slow components and add caching where appropriate",
                }
            )

        # Throughput recommendations
        throughput_info = validation_results.get("throughput", {})
        if not throughput_info.get("passed", False):
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Scalability",
                    "issue": f"Combined throughput {throughput_info['value']:.1f} ops/sec below {throughput_info['threshold']} minimum",
                    "action": "Identify bottlenecks and implement performance optimizations",
                }
            )

        # Coverage recommendations
        coverage_info = validation_results.get("coverage", {})
        if not coverage_info.get("passed", False):
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Testing",
                    "issue": f"Component coverage {coverage_info['value']:.1%} below {coverage_info['threshold']:.1%} minimum",
                    "action": "Complete performance testing for remaining system components",
                }
            )

        # Critical issues
        critical_issues = validation_results.get("critical_issues", [])
        for issue in critical_issues:
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "category": "System Health",
                    "issue": issue,
                    "action": "Immediate investigation and resolution required",
                }
            )

        # Component-specific recommendations
        components_info = validation_results.get("components", {})
        for comp_name, comp_results in components_info.items():
            if not comp_results.get("overall_ok", True):
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": f"{comp_name} Component",
                        "issue": "Performance issues detected",
                        "action": f"Optimize {comp_name} component performance and reliability",
                    }
                )

        return recommendations

    def generate_validation_report(self, validation_results: dict) -> str:
        """Generate human-readable validation report."""

        report = []
        report.append("AIVillage Production Readiness Validation Report")
        report.append("=" * 55)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall assessment
        assessment = validation_results["production_assessment"]
        report.append(f"Production Readiness: {assessment['level']}")
        report.append(f"Confidence Level: {assessment['confidence']}")
        report.append(f"Readiness Score: {validation_results['score']:.1%}")
        report.append(f"Deployment Window: {assessment['deployment_window']}")
        report.append("")
        report.append(f"Assessment: {assessment['description']}")
        report.append("")

        # Key metrics
        metrics = validation_results["key_metrics"]
        report.append("Key Performance Metrics:")
        report.append("-" * 30)
        report.append(f"Success Rate: {metrics['success_rate']:.1%}")
        report.append(f"Average Latency: {metrics['average_latency_ms']:.0f}ms")
        report.append(f"Combined Throughput: {metrics['combined_throughput']:.1f} ops/sec")
        report.append(f"Component Coverage: {metrics['component_coverage']:.1%}")
        report.append(f"Total Operations: {metrics['total_operations']}")
        report.append("")

        # Recommendations
        recommendations = validation_results["deployment_recommendation"]
        if recommendations:
            report.append(f"Deployment Recommendations ({len(recommendations)}):")
            report.append("-" * 40)

            # Group by priority
            by_priority = {}
            for rec in recommendations:
                priority = rec["priority"]
                if priority not in by_priority:
                    by_priority[priority] = []
                by_priority[priority].append(rec)

            for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                if priority in by_priority:
                    report.append(f"\n{priority} Priority:")
                    for rec in by_priority[priority]:
                        report.append(f"  [{rec['category']}] {rec['issue']}")
                        report.append(f"    Action: {rec['action']}")
        else:
            report.append("No critical recommendations - system ready for deployment!")

        # Final decision
        report.append("")
        report.append("=" * 55)
        if validation_results["ready"]:
            report.append("DECISION: APPROVE FOR PRODUCTION DEPLOYMENT")
            report.append("System meets production readiness criteria")
        else:
            report.append("DECISION: DEPLOYMENT BLOCKED")
            report.append("Address critical issues before production deployment")

        return "\n".join(report)


def main():
    """Main validation process."""
    print("AIVillage Production Readiness Validation")
    print("=" * 45)
    print("Validating system performance for production deployment")
    print()

    validator = ProductionReadinessValidator()

    # Look for performance report
    report_file = "tools/benchmarks/quick_demo_report.json"

    if not Path(report_file).exists():
        print(f"Error: Performance report not found: {report_file}")
        print("Run performance tests first to generate validation data")
        return 1

    print(f"Loading performance data from: {report_file}")

    try:
        # Validate production readiness
        validation_results = validator.validate_performance_data(report_file)

        if "error" in validation_results:
            print(f"Validation Error: {validation_results['error']}")
            print(f"Recommendation: {validation_results['recommendation']}")
            return 1

        # Generate and display report
        report_text = validator.generate_validation_report(validation_results)
        print(report_text)

        # Save detailed report
        detailed_report_file = "tools/benchmarks/production_readiness_validation.json"
        with open(detailed_report_file, "w") as f:
            json.dump(validation_results, f, indent=2)

        # Save readable report
        readable_report_file = "tools/benchmarks/production_readiness_report.txt"
        with open(readable_report_file, "w") as f:
            f.write(report_text)

        print(f"\nDetailed results saved to: {detailed_report_file}")
        print(f"Readable report saved to: {readable_report_file}")

        # Return exit code based on readiness
        return 0 if validation_results["ready"] else 1

    except Exception as e:
        print(f"Validation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
