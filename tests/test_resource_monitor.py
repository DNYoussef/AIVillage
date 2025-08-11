"""Test Resource Monitor functionality"""

import time

from src.core.resources.resource_monitor import (
    get_all_metrics,
    get_cpu_usage,
    get_memory_usage,
    get_monitor_instance,
)

print("=== Testing Individual Metrics ===")

# Test CPU usage
print("\n1. CPU Usage:")
cpu_values = []
for i in range(3):
    cpu = get_cpu_usage()
    cpu_values.append(cpu)
    print(f"   Reading {i + 1}: {cpu:.1f}%")
    time.sleep(0.5)

print(f"   CPU values are changing: {len(set(cpu_values)) > 1}")

# Test memory usage
print("\n2. Memory Usage:")
mem = get_memory_usage()
print(f"   Total: {mem['total_gb']:.2f} GB")
print(f"   Available: {mem['available_gb']:.2f} GB")
print(f"   Used: {mem['used_gb']:.2f} GB")
print(f"   Percent: {mem['percent']:.1f}%")
print(f"   All values are numeric: {all(isinstance(v, int | float) for v in mem.values())}")

# Test disk usage
print("\n3. Disk Usage:")
monitor = get_monitor_instance()
disk = monitor.get_disk_usage()
print(f"   Total: {disk['total_gb']:.2f} GB")
print(f"   Free: {disk['free_gb']:.2f} GB")
print(f"   Used: {disk['used_gb']:.2f} GB")
print(f"   Percent: {disk['percent']:.1f}%")

# Test network usage
print("\n4. Network Usage:")
net = monitor.get_network_usage()
print(f"   Bytes sent: {net['bytes_sent_mb']:.2f} MB")
print(f"   Bytes received: {net['bytes_recv_mb']:.2f} MB")
print(f"   Packets sent: {net['packets_sent']}")
print(f"   Packets received: {net['packets_recv']}")
if net["latency_ms"]:
    print(f"   Network latency: {net['latency_ms']:.1f} ms")

# Test battery (if available)
print("\n5. Battery Status:")
battery = monitor.get_battery()
if battery:
    print(f"   Level: {battery['percent']:.1f}%")
    print(f"   Plugged in: {battery['power_plugged']}")
else:
    print("   No battery detected (desktop system)")

# Test continuous monitoring
print("\n=== Testing Continuous Monitoring ===")
print("Getting metrics 5 times with 1-second delays...")
metrics_list = []
for i in range(5):
    metrics = get_all_metrics()
    metrics_list.append(metrics)
    print(f"   Run {i + 1}: CPU={metrics['cpu_percent']:.1f}%, Memory={metrics['memory']['percent']:.1f}%")
    time.sleep(1)

# Check if values are changing
cpu_changing = len({m["cpu_percent"] for m in metrics_list}) > 1
print(f"\nCPU values changing: {cpu_changing}")
print(
    f"Timestamps incrementing: {all(metrics_list[i]['timestamp'] < metrics_list[i + 1]['timestamp'] for i in range(4))}"
)

# Test resource allocation checks
print("\n=== Testing Resource Allocation Checks ===")
can_allocate_1gb = monitor.can_allocate(memory_gb=1.0)
can_allocate_huge = monitor.can_allocate(memory_gb=1000.0)
print(f"Can allocate 1 GB: {can_allocate_1gb}")
print(f"Can allocate 1000 GB: {can_allocate_huge}")

# Test model size checks
can_run_small = monitor.can_run_model(size_mb=100)
can_run_huge = monitor.can_run_model(size_mb=1000000)
print(f"Can run 100 MB model: {can_run_small}")
print(f"Can run 1 TB model: {can_run_huge}")

# Check history tracking
print("\n=== History Tracking ===")
print(f"CPU history length: {len(monitor.cpu_history)}")
print(f"Memory history length: {len(monitor.memory_history)}")
print(f"Full metrics history length: {len(monitor.history)}")
if monitor.cpu_history:
    print(f"Average CPU over time: {sum(monitor.cpu_history) / len(monitor.cpu_history):.1f}%")

print("\n✅ Resource Monitor Test Complete!")
