"""DEPRECATED: Enhanced Device Profiler for Sprint 6 Resource Management.

This module has been consolidated into src/production/monitoring/mobile/device_profiler.py

All features from this module have been merged into the production version, including:
- Comprehensive resource monitoring
- Evolution suitability scoring
- Real-time profiling capabilities
- Cross-platform compatibility

Please update your imports to use:
  from src.production.monitoring.mobile.device_profiler import DeviceProfiler, DeviceProfile, ResourceSnapshot

This shim will be removed in a future version.
"""

import warnings

from src.production.monitoring.mobile.device_profiler import (
    DeviceProfile as _DeviceProfile,
)
from src.production.monitoring.mobile.device_profiler import (
    DeviceProfiler as _DeviceProfiler,
)
from src.production.monitoring.mobile.device_profiler import DeviceType as _DeviceType
from src.production.monitoring.mobile.device_profiler import PowerState as _PowerState
from src.production.monitoring.mobile.device_profiler import (
    ResourceSnapshot as _ResourceSnapshot,
)
from src.production.monitoring.mobile.device_profiler import (
    ThermalState as _ThermalState,
)
from src.production.monitoring.mobile.device_profiler import (
    get_device_profile as _get_device_profile,
)
from src.production.monitoring.mobile.device_profiler import (
    get_device_profiler as _get_device_profiler,
)

warnings.warn(
    "src.core.resources.device_profiler is deprecated. "
    "Use src.production.monitoring.mobile.device_profiler instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Export the production implementations with deprecation warning
DeviceProfiler = _DeviceProfiler
DeviceProfile = _DeviceProfile
ResourceSnapshot = _ResourceSnapshot
DeviceType = _DeviceType
PowerState = _PowerState
ThermalState = _ThermalState
get_device_profiler = _get_device_profiler
get_device_profile = _get_device_profile
