#!/usr/bin/env python3
"""
Monitor REAL Cognate Training Progress
======================================

Monitors the training progress and provides real-time updates.
"""

import time
import json
from pathlib import Path

def monitor_training_progress():
    """Monitor the training progress"""
    log_file = Path("logs/real_cognate_evomerge.log")
    models_dir = Path("models/cognate_real")

    print("=== REAL Cognate Training Monitor ===")
    print("Monitoring training progress...")

    last_size = 0

    while True:
        try:
            # Check log file
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size > last_size:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_size)
                        new_lines = f.read()
                        if new_lines.strip():
                            print("--- Latest Log Updates ---")
                            for line in new_lines.strip().split('\n')[-5:]:
                                if 'Step' in line or 'Loss' in line or 'Model' in line or 'SUCCESS' in line or 'ERROR' in line:
                                    print(line)
                    last_size = current_size

            # Check model directories
            if models_dir.exists():
                model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
                print(f"\nModels created: {len(model_dirs)}")
                for model_dir in model_dirs:
                    metadata_file = model_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                            training_stats = metadata.get('training_stats', {})
                            print(f"  {metadata['model_name']}: {training_stats.get('mode', 'unknown')} - {metadata['parameter_count']:,} params")

            time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training_progress()