#!/usr/bin/env python3
"""Validate Magi Agent Success and Capabilities

This script validates the successful creation of the Magi agent
and confirms all achievements from the Agent Forge pipeline.
"""

from datetime import datetime
import json
from pathlib import Path
import sys

# Force UTF-8 encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def validate_magi_creation():
    """Validate the Magi agent creation success."""
    print("=" * 70)
    print("MAGI AGENT VALIDATION REPORT")
    print("=" * 70)

    # Check for results file
    results_path = Path("D:/AgentForge/memory_efficient_magi_20250726_033506/memory_efficient_scaled_results.json")

    if not results_path.exists():
        print("❌ ERROR: Magi results file not found")
        return False

    # Load and validate results
    with open(results_path) as f:
        data = json.load(f)

    print("\n✅ MAGI CREATION CONFIRMED")
    print(f"Run ID: {data['run_id']}")
    if "timestamp" in data:
        print(f"Timestamp: {data['timestamp']}")

    # Validate training scale
    results = data["results"]
    questions = results["questions_processed"]
    duration = data["duration_seconds"]

    print("\n📊 TRAINING SCALE VALIDATION:")
    print(f"  Questions Processed: {questions:,} {'✅' if questions >= 9960 else '❌'}")
    print(f"  Training Duration: {duration:.1f} seconds")
    print(f"  Processing Rate: {questions / duration:.1f} questions/second")

    # Validate specialization achievement
    spec_score = results["final_specialization_score"]
    baseline = results["level_results"][0]["overall_capability"]
    improvement = (spec_score - baseline) / baseline * 100

    print("\n🎯 SPECIALIZATION ACHIEVEMENT:")
    print(f"  Final Score: {spec_score:.4f} {'✅' if spec_score >= 0.80 else '❌'}")
    print(f"  Baseline Score: {baseline:.4f}")
    print(f"  Improvement: {improvement:.1f}%")

    # Validate capabilities
    capabilities = results["final_capabilities"]
    print("\n🔥 CAPABILITY VALIDATION:")

    mastery_count = 0
    for cap, score in capabilities.items():
        level = "MASTERY" if score >= 0.90 else "EXPERT" if score >= 0.75 else "ADVANCED"
        if score >= 0.90:
            mastery_count += 1
        status = "✅" if score >= 0.75 else "⚠️" if score >= 0.60 else "❌"
        print(f"  {cap:<25} {score:.3f} [{level}] {status}")

    print(f"\n  Mastery Level Skills: {mastery_count}/6")

    # Validate geometric progression
    print("\n📈 GEOMETRIC PROGRESSION VALIDATION:")
    print(f"  Levels Completed: {len(results['level_results'])}")
    print(f"  Snapshots Recorded: {len(results['geometric_snapshots'])}")

    # Show progression
    print("\n  Level Progression:")
    for i, level in enumerate(results["level_results"]):
        print(f"    Level {i}: Score={level['overall_capability']:.4f}, " f"Questions={level['questions_completed']}")

    # Validate memory efficiency
    print("\n💾 MEMORY EFFICIENCY:")
    if "memory_stats" in data:
        mem = data["memory_stats"]
        print(f"  Peak Memory: {mem.get('peak_memory_mb', 'N/A')} MB")
        print(f"  Average Memory: {mem.get('avg_memory_mb', 'N/A')} MB")
    else:
        print("  Memory stats not recorded")

    # Final validation summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY:")

    validations = [
        ("Scale Achievement", questions >= 9960),
        ("Specialization Score", spec_score >= 0.80),
        ("Mastery Capabilities", mastery_count >= 3),
        ("Geometric Progression", len(results["geometric_snapshots"]) > 0),
        ("Completion Time", duration < 300),  # Under 5 minutes
    ]

    all_passed = True
    for check, passed in validations:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:<25} {status}")
        if not passed:
            all_passed = False

    print(f"\n{'=' * 70}")
    if all_passed:
        print("🎉 MAGI AGENT CREATION: FULLY VALIDATED AND OPERATIONAL")
        print("🧙 The first AI Village agent is ready for deployment!")
    else:
        print("⚠️  Some validations failed - review needed")

    print(f"{'=' * 70}")

    # Save validation report
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "magi_run_id": data["run_id"],
        "training_duration": duration,
        "questions_processed": questions,
        "specialization_score": spec_score,
        "improvement_percentage": improvement,
        "capabilities": capabilities,
        "mastery_count": mastery_count,
        "all_validations_passed": all_passed,
    }

    report_path = Path("magi_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Validation report saved to: {report_path}")

    return all_passed


if __name__ == "__main__":
    success = validate_magi_creation()
    exit(0 if success else 1)
