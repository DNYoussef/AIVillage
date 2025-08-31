#!/usr/bin/env python3
"""
Performance monitoring script for CI/CD pipeline
"""

import json
import time
import psutil
import os
from datetime import datetime

def collect_performance_metrics():
    """Collect system performance metrics"""
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free
            }
        },
        "process_count": len(psutil.pids()),
        "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
    }
    
    return metrics

def main():
    print("Collecting performance metrics...")
    
    try:
        metrics = collect_performance_metrics()
        
        # Save metrics to file
        with open('test_performance_summary.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Performance metrics collected:")
        print(f"  CPU Usage: {metrics['system']['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {metrics['system']['memory']['percent']:.1f}%")
        print(f"  Process Count: {metrics['process_count']}")
        
    except Exception as e:
        print(f"Error collecting metrics: {e}")
        # Create minimal fallback metrics
        fallback_metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "error",
            "message": str(e),
            "system": {
                "cpu_percent": 0,
                "memory": {"percent": 0},
                "disk": {"free": 0}
            }
        }
        
        with open('test_performance_summary.json', 'w') as f:
            json.dump(fallback_metrics, f, indent=2)

if __name__ == "__main__":
    main()
