#!/usr/bin/env python3
"""
Monitor Recursive Training Progress
===================================

Monitors the Titans recursive training and provides real-time analysis.
"""

import time
import re
from pathlib import Path

def monitor_recursive_training():
    """Monitor and analyze recursive training progress"""

    print("=== RECURSIVE TRAINING MONITOR ===")
    print("Monitoring Titans memory updates and recursive learning...\n")

    log_file = Path("logs/recursive_training.log")

    # Track metrics
    losses = []
    act_steps = []
    memory_updates = []
    improvements = []

    last_pos = 0

    while True:
        try:
            # Read new log entries
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_pos)
                    new_lines = f.readlines()
                    last_pos = f.tell()

                for line in new_lines:
                    # Parse step info
                    if "Step" in line and "/1000" in line:
                        step_match = re.search(r"Step (\d+)/(\d+)", line)
                        if step_match:
                            current_step = int(step_match.group(1))
                            total_steps = int(step_match.group(2))
                            print(f"\n--- Step {current_step}/{total_steps} ---")

                    # Parse loss
                    if "Loss:" in line:
                        loss_match = re.search(r"Loss: ([\d.]+) \(avg: ([\d.]+)\)", line)
                        if loss_match:
                            current_loss = float(loss_match.group(1))
                            avg_loss = float(loss_match.group(2))
                            losses.append(current_loss)
                            print(f"  Loss: {current_loss:.4f} (avg: {avg_loss:.4f})")

                    # Parse ACT steps
                    if "ACT steps:" in line:
                        act_match = re.search(r"ACT steps: ([\d.]+) \(avg: ([\d.]+)\)", line)
                        if act_match:
                            current_act = float(act_match.group(1))
                            avg_act = float(act_match.group(2))
                            act_steps.append(current_act)
                            print(f"  ACT steps: {current_act:.2f} (avg: {avg_act:.2f})")

                    # Parse memory updates
                    if "Memory updates:" in line:
                        mem_match = re.search(r"Memory updates: ([\d.]+)", line)
                        if mem_match:
                            total_updates = float(mem_match.group(1))
                            memory_updates.append(total_updates)
                            print(f"  Memory updates: {total_updates:.0f}")

                    # Parse recursive improvement
                    if "Recursive improvement:" in line:
                        imp_match = re.search(r"Recursive improvement: ([-\d.]+)%", line)
                        if imp_match:
                            improvement = float(imp_match.group(1))
                            improvements.append(improvement)
                            print(f"  Recursive improvement: {improvement:.2f}%")

                            # Analyze trend
                            if improvement > 1.0:
                                print("  ✓ POSITIVE RECURSIVE EFFECT - Memory is helping!")
                            elif improvement > 0:
                                print("  → Slight improvement - recursive loop working")
                            else:
                                print("  ⚠ No improvement yet - may need more steps")

                    # Check for recommendations
                    if "RECOMMENDATION:" in line:
                        print(f"\n{line.strip()}")

                    # Final results
                    if "TRAINING COMPLETE" in line:
                        print("\n=== TRAINING COMPLETE ===")

                    if "Total improvement:" in line:
                        total_imp_match = re.search(r"Total improvement: ([-\d.]+)%", line)
                        if total_imp_match:
                            total_imp = float(total_imp_match.group(1))
                            print(f"\nFINAL RESULTS:")
                            print(f"  Total improvement: {total_imp:.2f}%")

                            if total_imp > 5:
                                print("  ✅ RECURSIVE LEARNING SUCCESSFUL!")
                                print("  Memory significantly improved predictions")
                            elif total_imp > 2:
                                print("  ✓ Moderate recursive effect achieved")
                            else:
                                print("  ⚠ Limited recursive effect")

                    if "RECURSIVE LEARNING SUCCESSFUL" in line:
                        print("\n✅ SUCCESS: Recursive self-improvement validated!")

            # Periodic analysis
            if len(losses) > 0 and len(losses) % 50 == 0:
                print(f"\n=== ANALYSIS at {len(losses)} steps ===")

                # Loss trend
                if len(losses) >= 100:
                    early_avg = sum(losses[-100:-50]) / 50
                    recent_avg = sum(losses[-50:]) / 50
                    trend = (early_avg - recent_avg) / early_avg * 100
                    print(f"  Loss trend: {trend:.2f}% improvement")

                # ACT adaptation
                if len(act_steps) >= 100:
                    early_act = sum(act_steps[-100:-50]) / 50
                    recent_act = sum(act_steps[-50:]) / 50
                    print(f"  ACT evolution: {early_act:.2f} → {recent_act:.2f} steps")

                # Memory activity
                if len(memory_updates) >= 2:
                    update_rate = memory_updates[-1] - memory_updates[-2] if len(memory_updates) >= 2 else 0
                    print(f"  Memory update rate: {update_rate:.0f} updates/100 steps")

                # Recursive effect
                if improvements:
                    avg_improvement = sum(improvements[-5:]) / min(5, len(improvements))
                    if avg_improvement > 1.0:
                        print(f"  ✓ Strong recursive effect: {avg_improvement:.2f}% avg improvement")
                        print("  → Recommend scaling to 10,000 steps")
                    elif avg_improvement > 0:
                        print(f"  Moderate recursive effect: {avg_improvement:.2f}%")
                    else:
                        print(f"  Limited recursive effect: {avg_improvement:.2f}%")

            time.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_recursive_training()