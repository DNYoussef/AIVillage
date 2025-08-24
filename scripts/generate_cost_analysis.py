#!/usr/bin/env python3
"""
Generate comprehensive cloud cost analysis for AIVillage deployments.

This script analyzes costs across different cloud providers and deployment
scenarios, providing detailed reports and optimization recommendations.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "packages"))

from core.deployment.cloud_cost_analyzer import CloudProvider, DeploymentType, create_cost_analyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate cloud cost analysis for AIVillage deployments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze production deployment on AWS
  python generate_cost_analysis.py --deployment production --provider aws

  # Compare costs across multiple providers
  python generate_cost_analysis.py --deployment production --provider aws,azure,gcp

  # Analyze all deployment types on AWS
  python generate_cost_analysis.py --deployment all --provider aws

  # Generate detailed report with optimization recommendations
  python generate_cost_analysis.py --deployment production --provider aws --detailed --output cost_analysis.json
        """,
    )

    parser.add_argument(
        "--deployment",
        choices=["development", "staging", "production", "high_availability", "edge_distributed", "all"],
        default="production",
        help="Deployment type to analyze (default: production)",
    )

    parser.add_argument(
        "--provider",
        default="aws",
        help="Cloud provider(s) to analyze (comma-separated). Options: aws,azure,gcp,digitalocean (default: aws)",
    )

    parser.add_argument("--region", default="us-east-1", help="Cloud region for analysis (default: us-east-1)")

    parser.add_argument("--output", type=Path, help="Output file for detailed JSON report")

    parser.add_argument("--detailed", action="store_true", help="Include detailed resource breakdown in output")

    parser.add_argument("--compare", action="store_true", help="Generate provider comparison table")

    parser.add_argument(
        "--format", choices=["table", "json", "summary"], default="table", help="Output format (default: table)"
    )

    return parser.parse_args()


def parse_providers(providers_str: str) -> list[CloudProvider]:
    """Parse provider string into CloudProvider enums."""
    provider_map = {
        "aws": CloudProvider.AWS,
        "azure": CloudProvider.AZURE,
        "gcp": CloudProvider.GCP,
        "digitalocean": CloudProvider.DIGITAL_OCEAN,
        "linode": CloudProvider.LINODE,
        "hetzner": CloudProvider.HETZNER,
        "vultr": CloudProvider.VULTR,
    }

    providers = []
    for provider_name in providers_str.split(","):
        provider_name = provider_name.strip().lower()
        if provider_name in provider_map:
            providers.append(provider_map[provider_name])
        else:
            print(f"Warning: Unknown provider '{provider_name}', skipping")

    return providers


def parse_deployments(deployment_str: str) -> list[DeploymentType]:
    """Parse deployment string into DeploymentType enums."""
    deployment_map = {
        "development": DeploymentType.DEVELOPMENT,
        "staging": DeploymentType.STAGING,
        "production": DeploymentType.PRODUCTION,
        "high_availability": DeploymentType.HIGH_AVAILABILITY,
        "edge_distributed": DeploymentType.EDGE_DISTRIBUTED,
    }

    if deployment_str == "all":
        return list(deployment_map.values())
    else:
        return [deployment_map.get(deployment_str, DeploymentType.PRODUCTION)]


def format_currency(amount: float) -> str:
    """Format currency amount."""
    return f"${amount:,.2f}"


def print_summary_table(analysis, provider: CloudProvider):
    """Print summary cost table."""
    print(f"\n{'='*60}")
    print(f"COST ANALYSIS: {analysis.deployment_type.value.upper()} ON {provider.value.upper()}")
    print(f"{'='*60}")

    print(f"Region: {analysis.region}")
    print(f"Analysis Date: {analysis.analysis_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Confidence Level: {analysis.confidence_level:.0%}")

    print("\nCOST SUMMARY:")
    print(f"  Monthly Cost: {format_currency(float(analysis.total_monthly_cost))}")
    print(f"  Annual Cost:  {format_currency(float(analysis.total_annual_cost))}")

    if analysis.potential_savings:
        total_savings = sum(analysis.potential_savings.values())
        optimized_monthly = float(analysis.total_monthly_cost) - float(total_savings)
        print(f"  Potential Monthly Savings: {format_currency(float(total_savings))}")
        print(f"  Optimized Monthly Cost:    {format_currency(optimized_monthly)}")


def print_resource_breakdown(analysis):
    """Print detailed resource cost breakdown."""
    print("\nRESOURCE BREAKDOWN:")
    print(f"{'Resource':<25} {'Type':<12} {'Monthly Cost':<15} {'Instance':<15}")
    print("-" * 70)

    for cost in analysis.resource_costs:
        resource_name = cost.resource_spec.name[:24]
        resource_type = cost.resource_spec.resource_type.value[:11]
        monthly_cost = format_currency(float(cost.monthly_cost))
        instance_info = cost.instance_type[:14] if cost.instance_type else "N/A"

        print(f"{resource_name:<25} {resource_type:<12} {monthly_cost:<15} {instance_info:<15}")

    # Cost by category
    categories = {}
    for cost in analysis.resource_costs:
        category = cost.resource_spec.resource_type.value
        if category not in categories:
            categories[category] = 0
        categories[category] += float(cost.monthly_cost)

    print("\nCOST BY CATEGORY:")
    for category, cost in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (cost / float(analysis.total_monthly_cost)) * 100
        print(f"  {category.title():<15}: {format_currency(cost):<12} ({percentage:.1f}%)")


def print_optimization_recommendations(analysis):
    """Print optimization recommendations."""
    if not analysis.recommendations:
        return

    print("\nOPTIMIZATION RECOMMENDATIONS:")
    for i, recommendation in enumerate(analysis.recommendations, 1):
        print(f"  {i}. {recommendation}")

    if analysis.potential_savings:
        print("\nPOTENTIAL SAVINGS BREAKDOWN:")
        for category, savings in analysis.potential_savings.items():
            print(f"  {category}: {format_currency(float(savings))}/month")


def print_comparison_table(comparisons):
    """Print provider comparison table."""
    print("\nPROVIDER COMPARISON:")
    print(f"{'Provider':<15} {'Monthly Cost':<15} {'Annual Cost':<15} {'Savings Potential':<18}")
    print("-" * 65)

    for provider, analysis in comparisons.items():
        monthly_cost = format_currency(float(analysis.total_monthly_cost))
        annual_cost = format_currency(float(analysis.total_annual_cost))
        savings = format_currency(float(sum(analysis.potential_savings.values())))

        print(f"{provider.value.upper():<15} {monthly_cost:<15} {annual_cost:<15} {savings:<18}")

    # Find cheapest option
    cheapest = min(comparisons.items(), key=lambda x: x[1].total_monthly_cost)
    print(
        f"\nMost Cost-Effective: {cheapest[0].value.upper()} at {format_currency(float(cheapest[1].total_monthly_cost))}/month"
    )


async def main():
    """Main execution function."""
    args = parse_arguments()

    # Initialize cost analyzer
    print("Initializing cloud cost analyzer...")
    analyzer = await create_cost_analyzer()

    # Parse providers and deployments
    providers = parse_providers(args.provider)
    deployments = parse_deployments(args.deployment)

    if not providers:
        print("ERROR: No valid providers specified")
        return 1

    print(f"Analyzing {len(deployments)} deployment(s) across {len(providers)} provider(s)")

    all_analyses = []
    all_comparisons = {}

    # Analyze each deployment type
    for deployment_type in deployments:
        print(f"\nAnalyzing {deployment_type.value} deployment...")

        if len(providers) > 1 or args.compare:
            # Multi-provider comparison
            comparisons = analyzer.compare_providers(deployment_type, providers, args.region)
            all_comparisons[deployment_type] = comparisons

            if args.format == "table":
                print_comparison_table(comparisons)

            # Add all analyses for potential detailed output
            for provider, analysis in comparisons.items():
                all_analyses.append((deployment_type, provider, analysis))
        else:
            # Single provider analysis
            provider = providers[0]
            analysis = analyzer.analyze_deployment(deployment_type, provider, args.region)
            all_analyses.append((deployment_type, provider, analysis))

            if args.format == "table":
                print_summary_table(analysis, provider)
                if args.detailed:
                    print_resource_breakdown(analysis)
                    print_optimization_recommendations(analysis)

    # Generate JSON output if requested
    if args.output or args.format == "json":
        output_data = {
            "analysis_metadata": {
                "generated_at": analysis.analysis_date.isoformat(),
                "region": args.region,
                "deployments_analyzed": [d.value for d in deployments],
                "providers_analyzed": [p.value for p in providers],
            },
            "analyses": [],
            "comparisons": {},
        }

        # Add individual analyses
        for deployment_type, provider, analysis in all_analyses:
            report = analyzer.generate_cost_report(analysis, "detailed" if args.detailed else "summary")
            output_data["analyses"].append(
                {"deployment_type": deployment_type.value, "provider": provider.value, **report}
            )

        # Add comparisons
        for deployment_type, comparisons in all_comparisons.items():
            comparison_data = {}
            for provider, analysis in comparisons.items():
                comparison_data[provider.value] = {
                    "monthly_cost": float(analysis.total_monthly_cost),
                    "annual_cost": float(analysis.total_annual_cost),
                    "potential_savings": float(sum(analysis.potential_savings.values())),
                }
            output_data["comparisons"][deployment_type.value] = comparison_data

        if args.output:
            # Save to file
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nDetailed report saved to {args.output}")

        if args.format == "json":
            # Print to stdout
            print(json.dumps(output_data, indent=2, default=str))

    # Summary output
    if args.format == "summary":
        total_monthly = sum(float(analysis.total_monthly_cost) for _, _, analysis in all_analyses)
        total_annual = total_monthly * 12

        print("\nSUMMARY:")
        print(f"  Total Monthly Cost: {format_currency(total_monthly)}")
        print(f"  Total Annual Cost:  {format_currency(total_annual)}")
        print(f"  Deployments: {len(deployments)}")
        print(f"  Providers: {len(providers)}")

    print("\nCost analysis complete!")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)
