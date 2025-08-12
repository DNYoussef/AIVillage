"""Mobile device monitoring and resource management."""

# Safe imports with fallbacks for cross-platform compatibility
try:
    from .device_profiler import DeviceProfile, DeviceProfiler, ResourceSnapshot
except ImportError:
    DeviceProfile = DeviceProfiler = ResourceSnapshot = None

try:
    from .mobile_metrics import MetricsCollector, MobileMetrics
except ImportError:
    MetricsCollector = MobileMetrics = None

try:
    from .resource_allocator import ResourceAllocation, ResourceAllocator

    AllocationStrategy = ResourcePool = None  # These may not exist in current impl
except ImportError:
    ResourceAllocator = ResourceAllocation = AllocationStrategy = ResourcePool = None

# Only export what's actually available
__all__ = [
    name
    for name in [
        "AllocationStrategy",
        "DeviceProfile",
        "DeviceProfiler",
        "MetricsCollector",
        "MobileMetrics",
        "ResourceAllocator",
        "ResourceAllocation",
        "ResourcePool",
        "ResourceSnapshot",
    ]
    if globals().get(name) is not None
]
