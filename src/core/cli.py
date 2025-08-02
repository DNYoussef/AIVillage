import argparse
import json
from typing import List


def _compute_reliability(sent: int, received: int, dropped: int = 0) -> float:
    """Calculate simple reliability metric."""
    total = sent + dropped
    if total == 0:
        return 0.0
    return received / total


def reliability_metrics() -> float:
    """CLI entry point for reliability metrics."""
    parser = argparse.ArgumentParser(description="Generate reliability metrics")
    parser.add_argument("--sent", type=int, required=True, help="Messages sent")
    parser.add_argument("--received", type=int, required=True, help="Messages received")
    parser.add_argument("--dropped", type=int, default=0, help="Messages dropped")
    args = parser.parse_args()
    reliability = _compute_reliability(args.sent, args.received, args.dropped)
    result = {
        "sent": args.sent,
        "received": args.received,
        "dropped": args.dropped,
        "reliability": reliability,
    }
    print(json.dumps(result))
    return reliability


def latency_metrics() -> dict:
    """CLI entry point for latency metrics."""
    parser = argparse.ArgumentParser(description="Generate latency metrics")
    parser.add_argument("latencies", nargs="+", type=float, help="Latency samples in ms")
    args = parser.parse_args()
    latencies: List[float] = args.latencies
    avg = sum(latencies) / len(latencies)
    metrics = {
        "average_latency": avg,
        "min_latency": min(latencies),
        "max_latency": max(latencies),
    }
    print(json.dumps(metrics))
    return metrics


def main() -> None:
    """Core CLI with reliability and latency subcommands."""
    parser = argparse.ArgumentParser(description="AIVillage core CLI")
    subparsers = parser.add_subparsers(dest="command")

    rel = subparsers.add_parser("reliability", help="Generate reliability metrics")
    rel.add_argument("--sent", type=int, required=True)
    rel.add_argument("--received", type=int, required=True)
    rel.add_argument("--dropped", type=int, default=0)

    lat = subparsers.add_parser("latency", help="Generate latency metrics")
    lat.add_argument("latencies", nargs="+", type=float)

    args = parser.parse_args()
    if args.command == "reliability":
        reliability = _compute_reliability(args.sent, args.received, args.dropped)
        result = {
            "sent": args.sent,
            "received": args.received,
            "dropped": args.dropped,
            "reliability": reliability,
        }
        print(json.dumps(result))
    elif args.command == "latency":
        latencies = args.latencies
        avg = sum(latencies) / len(latencies)
        metrics = {
            "average_latency": avg,
            "min_latency": min(latencies),
            "max_latency": max(latencies),
        }
        print(json.dumps(metrics))
    else:
        parser.print_help()
