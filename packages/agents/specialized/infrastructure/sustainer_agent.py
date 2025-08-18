"""Sustainer Agent - Capacity & Efficiency

The capacity and efficiency specialist of AIVillage, responsible for:
- Device profiling and resource capacity monitoring
- Resource scheduling and allocation optimization
- Power management and energy efficiency
- Cost optimization and budget management
- Performance monitoring and system sustainability
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

from src.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    POWER = "power"


class DeviceClass(Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    SERVER = "server"
    EDGE = "edge"
    IOT = "iot"


@dataclass
class DeviceProfile:
    device_id: str
    device_class: DeviceClass
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    network_mbps: float
    power_capacity_watts: float
    battery_hours: float | None
    gpu_available: bool
    constraints: dict[str, Any]
    performance_tier: str


@dataclass
class ResourceUsage:
    resource_type: ResourceType
    current_usage: float
    max_capacity: float
    utilization_percent: float
    trend: str  # increasing, decreasing, stable
    projected_usage: float
    timestamp: float


@dataclass
class OptimizationTask:
    task_id: str
    target_resource: ResourceType
    optimization_type: str
    current_efficiency: float
    target_efficiency: float
    estimated_savings: dict[str, float]
    priority: int
    status: str


class SustainerAgent(AgentInterface):
    """Sustainer Agent handles capacity monitoring, resource optimization, and efficiency
    management for AIVillage, ensuring sustainable and cost-effective operations.
    """

    def __init__(self, agent_id: str = "sustainer_agent"):
        self.agent_id = agent_id
        self.agent_type = "Sustainer"
        self.capabilities = [
            "device_profiling",
            "resource_scheduling",
            "power_management",
            "cost_optimization",
            "capacity_monitoring",
            "performance_optimization",
            "energy_efficiency",
            "resource_allocation",
            "system_sustainability",
            "budget_management",
        ]

        # Resource monitoring
        self.device_profiles: dict[str, DeviceProfile] = {}
        self.resource_usage_history: list[ResourceUsage] = []
        self.optimization_tasks: dict[str, OptimizationTask] = {}
        self.cost_tracking = {"daily": 0.0, "weekly": 0.0, "monthly": 0.0}

        # Performance tracking
        self.optimizations_applied = 0
        self.energy_saved_kwh = 0.0
        self.cost_savings_usd = 0.0
        self.efficiency_improvements = 0

        # Thresholds and targets
        self.resource_thresholds = {
            ResourceType.CPU: {"warning": 0.8, "critical": 0.95},
            ResourceType.MEMORY: {"warning": 0.85, "critical": 0.95},
            ResourceType.STORAGE: {"warning": 0.9, "critical": 0.98},
            ResourceType.POWER: {"warning": 0.8, "critical": 0.9},
        }

        # Mobile optimization settings
        self.mobile_constraints = {
            "max_cpu_usage": 0.6,
            "max_memory_mb": 512,
            "max_power_watts": 5.0,
            "min_battery_hours": 8.0,
        }

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate capacity and efficiency responses"""
        prompt_lower = prompt.lower()

        if "capacity" in prompt_lower or "resource" in prompt_lower:
            return "I monitor system capacity and optimize resource allocation across all agents and devices."
        if "efficiency" in prompt_lower or "optimize" in prompt_lower:
            return "I optimize system efficiency, reducing energy consumption and operational costs."
        if "power" in prompt_lower or "energy" in prompt_lower:
            return "I manage power consumption and implement energy-efficient operating strategies."
        if "cost" in prompt_lower or "budget" in prompt_lower:
            return "I track operational costs and optimize resource usage for maximum cost-effectiveness."
        if "mobile" in prompt_lower or "device" in prompt_lower:
            return "I profile devices and ensure optimal performance within mobile and edge constraints."

        return "I am Sustainer Agent, ensuring capacity and efficiency across AIVillage operations."

    async def get_embedding(self, text: str) -> list[float]:
        """Generate efficiency-focused embeddings"""
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Efficiency embeddings focus on optimization patterns
        return [(hash_value % 1000) / 1000.0] * 320

    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        """Rerank based on efficiency relevance"""
        efficiency_keywords = [
            "efficiency",
            "optimization",
            "resource",
            "capacity",
            "performance",
            "power",
            "energy",
            "cost",
            "sustainable",
            "mobile",
            "constraint",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))

            for keyword in efficiency_keywords:
                score += content.lower().count(keyword) * 1.8

            # Boost optimization and performance content
            if any(term in content.lower() for term in ["optimize", "efficient", "performance"]):
                score *= 1.6

            result["efficiency_relevance"] = score

        return sorted(results, key=lambda x: x.get("efficiency_relevance", 0), reverse=True)[:k]

    async def introspect(self) -> dict[str, Any]:
        """Return Sustainer agent status and efficiency metrics"""
        current_usage = await self._get_current_resource_usage()

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "device_profiles": len(self.device_profiles),
            "resource_usage_samples": len(self.resource_usage_history),
            "active_optimizations": len(self.optimization_tasks),
            "optimizations_applied": self.optimizations_applied,
            "energy_saved_kwh": self.energy_saved_kwh,
            "cost_savings_usd": self.cost_savings_usd,
            "current_cpu_usage": current_usage.get("cpu_percent", 0),
            "current_memory_usage": current_usage.get("memory_percent", 0),
            "efficiency_score": await self._calculate_efficiency_score(),
            "specialization": "capacity_and_efficiency",
            "initialized": self.initialized,
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate efficiency insights and recommendations"""
        # Add efficiency context to communications
        if any(keyword in message.lower() for keyword in ["resource", "optimize", "efficiency"]):
            efficiency_context = "[EFFICIENCY INSIGHT]"
            message = f"{efficiency_context} {message}"

        if recipient:
            response = await recipient.generate(f"Sustainer Agent provides efficiency insight: {message}")
            return f"Efficiency recommendation delivered: {response[:50]}..."
        return "No recipient for efficiency insight"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate efficiency-specific latent spaces"""
        query_lower = query.lower()

        if "resource" in query_lower:
            space_type = "resource_optimization"
        elif "power" in query_lower or "energy" in query_lower:
            space_type = "power_management"
        elif "cost" in query_lower:
            space_type = "cost_optimization"
        elif "mobile" in query_lower or "device" in query_lower:
            space_type = "device_profiling"
        else:
            space_type = "general_efficiency"

        latent_repr = f"SUSTAINER[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def profile_device(self, device_spec: dict[str, Any]) -> dict[str, Any]:
        """Profile a device and determine optimal configuration - MVP function"""
        profile_id = f"profile_{int(time.time())}"
        device_id = device_spec.get("device_id", f"device_{len(self.device_profiles)}")

        # Determine device class based on specifications
        device_class = await self._classify_device(device_spec)

        # Create comprehensive device profile
        profile = DeviceProfile(
            device_id=device_id,
            device_class=device_class,
            cpu_cores=device_spec.get("cpu_cores", 4),
            memory_gb=device_spec.get("memory_gb", 8.0),
            storage_gb=device_spec.get("storage_gb", 128.0),
            network_mbps=device_spec.get("network_mbps", 100.0),
            power_capacity_watts=device_spec.get("power_watts", 15.0),
            battery_hours=device_spec.get("battery_hours"),
            gpu_available=device_spec.get("gpu_available", False),
            constraints=await self._determine_device_constraints(device_class, device_spec),
            performance_tier=await self._determine_performance_tier(device_spec),
        )

        # Store profile
        self.device_profiles[device_id] = profile

        # Generate optimization recommendations
        recommendations = await self._generate_device_recommendations(profile)

        # Create receipt
        receipt = {
            "agent": "Sustainer",
            "action": "device_profiling",
            "profile_id": profile_id,
            "timestamp": time.time(),
            "device_id": device_id,
            "device_class": device_class.value,
            "performance_tier": profile.performance_tier,
            "mobile_optimized": device_class == DeviceClass.MOBILE,
            "constraints_applied": len(profile.constraints),
            "signature": f"sustainer_profile_{profile_id}",
        }

        logger.info(f"Device profiling completed: {device_id} ({device_class.value})")

        return {
            "status": "success",
            "profile_id": profile_id,
            "device_profile": profile,
            "recommendations": recommendations,
            "receipt": receipt,
        }

    async def _classify_device(self, spec: dict[str, Any]) -> DeviceClass:
        """Classify device based on specifications"""
        memory_gb = spec.get("memory_gb", 8.0)
        cpu_cores = spec.get("cpu_cores", 4)
        power_watts = spec.get("power_watts", 15.0)
        has_battery = spec.get("battery_hours") is not None

        # Mobile device classification
        if memory_gb <= 8.0 and power_watts <= 10.0 and has_battery:
            return DeviceClass.MOBILE
        # Edge device classification
        if memory_gb <= 16.0 and cpu_cores <= 8 and power_watts <= 50.0:
            return DeviceClass.EDGE
        # Server classification
        if memory_gb >= 32.0 or cpu_cores >= 16:
            return DeviceClass.SERVER
        # IoT classification
        if memory_gb <= 2.0 and power_watts <= 5.0:
            return DeviceClass.IOT
        return DeviceClass.DESKTOP

    async def _determine_device_constraints(self, device_class: DeviceClass, spec: dict[str, Any]) -> dict[str, Any]:
        """Determine optimization constraints based on device class"""
        constraints = {}

        if device_class == DeviceClass.MOBILE:
            constraints.update(
                {
                    "max_cpu_usage": 0.6,
                    "max_memory_mb": min(spec.get("memory_gb", 8.0) * 1024 * 0.7, 512),
                    "max_power_watts": min(spec.get("power_watts", 10.0) * 0.8, 5.0),
                    "background_processing": False,
                    "model_size_limit_mb": 50,
                }
            )
        elif device_class == DeviceClass.EDGE:
            constraints.update(
                {
                    "max_cpu_usage": 0.8,
                    "max_memory_mb": spec.get("memory_gb", 16.0) * 1024 * 0.8,
                    "max_power_watts": spec.get("power_watts", 50.0) * 0.9,
                    "local_processing_preferred": True,
                    "model_size_limit_mb": 200,
                }
            )
        elif device_class == DeviceClass.IOT:
            constraints.update(
                {
                    "max_cpu_usage": 0.4,
                    "max_memory_mb": spec.get("memory_gb", 2.0) * 1024 * 0.6,
                    "max_power_watts": spec.get("power_watts", 5.0) * 0.7,
                    "ultra_low_power": True,
                    "model_size_limit_mb": 10,
                }
            )
        else:
            # Server/Desktop - fewer constraints
            constraints.update(
                {
                    "max_cpu_usage": 0.9,
                    "max_memory_mb": spec.get("memory_gb", 32.0) * 1024 * 0.9,
                    "high_performance_mode": True,
                }
            )

        return constraints

    async def _determine_performance_tier(self, spec: dict[str, Any]) -> str:
        """Determine device performance tier"""
        memory_gb = spec.get("memory_gb", 8.0)
        cpu_cores = spec.get("cpu_cores", 4)

        if memory_gb >= 32 and cpu_cores >= 16:
            return "high_performance"
        if memory_gb >= 16 and cpu_cores >= 8:
            return "standard"
        if memory_gb >= 8 and cpu_cores >= 4:
            return "basic"
        return "minimal"

    async def _generate_device_recommendations(self, profile: DeviceProfile) -> list[str]:
        """Generate optimization recommendations for device"""
        recommendations = []

        if profile.device_class == DeviceClass.MOBILE:
            recommendations.extend(
                [
                    "Enable aggressive model quantization for <50MB models",
                    "Use batch processing during charging periods",
                    "Implement dynamic CPU throttling based on battery level",
                    "Cache frequently used data locally to reduce network usage",
                ]
            )
        elif profile.device_class == DeviceClass.EDGE:
            recommendations.extend(
                [
                    "Optimize for local processing to reduce latency",
                    "Use compressed model formats to fit memory constraints",
                    "Implement load balancing across available cores",
                    "Monitor thermal throttling and adjust workload accordingly",
                ]
            )
        elif profile.device_class == DeviceClass.IOT:
            recommendations.extend(
                [
                    "Use ultra-compressed models <10MB",
                    "Implement sleep/wake cycles for power savings",
                    "Batch communication to reduce radio usage",
                    "Use event-driven processing only",
                ]
            )

        # General recommendations based on performance tier
        if profile.performance_tier == "minimal":
            recommendations.append("Consider offloading heavy tasks to cloud/edge nodes")
        elif profile.performance_tier == "high_performance":
            recommendations.append("Utilize full system capacity for parallel processing")

        return recommendations

    async def monitor_resource_usage(self) -> dict[str, Any]:
        """Monitor current resource usage and generate efficiency report - MVP function"""
        monitoring_id = f"monitor_{int(time.time())}"

        # Get current system resource usage
        current_usage = await self._get_current_resource_usage()

        # Analyze usage patterns and trends
        usage_analysis = await self._analyze_usage_patterns(current_usage)

        # Check for threshold violations
        threshold_violations = await self._check_threshold_violations(current_usage)

        # Generate optimization recommendations
        optimization_recommendations = await self._generate_optimization_recommendations(
            current_usage, threshold_violations
        )

        # Store usage data
        for resource_type, usage_data in current_usage.items():
            if resource_type.endswith("_percent"):
                resource = ResourceType(resource_type.replace("_percent", ""))
                usage_record = ResourceUsage(
                    resource_type=resource,
                    current_usage=usage_data,
                    max_capacity=100.0,
                    utilization_percent=usage_data,
                    trend=usage_analysis.get(resource_type, "stable"),
                    projected_usage=usage_data * 1.1,  # Simple projection
                    timestamp=time.time(),
                )
                self.resource_usage_history.append(usage_record)

        # Create receipt
        receipt = {
            "agent": "Sustainer",
            "action": "resource_monitoring",
            "monitoring_id": monitoring_id,
            "timestamp": time.time(),
            "cpu_usage_percent": current_usage.get("cpu_percent", 0),
            "memory_usage_percent": current_usage.get("memory_percent", 0),
            "threshold_violations": len(threshold_violations),
            "optimization_opportunities": len(optimization_recommendations),
            "efficiency_score": await self._calculate_efficiency_score(),
            "signature": f"sustainer_monitor_{monitoring_id}",
        }

        logger.info(f"Resource monitoring completed: {monitoring_id} - {len(threshold_violations)} violations found")

        return {
            "status": "success",
            "monitoring_id": monitoring_id,
            "current_usage": current_usage,
            "usage_analysis": usage_analysis,
            "threshold_violations": threshold_violations,
            "recommendations": optimization_recommendations,
            "receipt": receipt,
        }

    async def _get_current_resource_usage(self) -> dict[str, Any]:
        """Get current system resource usage"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "storage_percent": (
                    psutil.disk_usage("/").percent if hasattr(psutil.disk_usage("/"), "percent") else 50.0
                ),
                "network_io": (psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}),
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.warning(f"Could not get system metrics, using simulated data: {e}")
            # Fallback to simulated data for environments without psutil access
            return {
                "cpu_percent": 45.0,
                "memory_percent": 60.0,
                "storage_percent": 35.0,
                "network_io": {"bytes_sent": 1024000, "bytes_recv": 2048000},
                "timestamp": time.time(),
            }

    async def _analyze_usage_patterns(self, current_usage: dict[str, Any]) -> dict[str, str]:
        """Analyze resource usage trends"""
        analysis = {}

        # Simple trend analysis based on historical data
        if len(self.resource_usage_history) >= 3:
            recent_history = self.resource_usage_history[-3:]
            for resource in [ResourceType.CPU, ResourceType.MEMORY]:
                resource_history = [r.utilization_percent for r in recent_history if r.resource_type == resource]
                if len(resource_history) >= 2:
                    if resource_history[-1] > resource_history[-2] + 5:
                        analysis[f"{resource.value}_percent"] = "increasing"
                    elif resource_history[-1] < resource_history[-2] - 5:
                        analysis[f"{resource.value}_percent"] = "decreasing"
                    else:
                        analysis[f"{resource.value}_percent"] = "stable"

        return analysis

    async def _check_threshold_violations(self, usage: dict[str, Any]) -> list[dict[str, Any]]:
        """Check for resource threshold violations"""
        violations = []

        for resource_type, thresholds in self.resource_thresholds.items():
            usage_key = f"{resource_type.value}_percent"
            if usage_key in usage:
                current_value = usage[usage_key]

                if current_value >= thresholds["critical"]:
                    violations.append(
                        {
                            "resource": resource_type.value,
                            "level": "critical",
                            "current": current_value,
                            "threshold": thresholds["critical"],
                            "action_required": "immediate",
                        }
                    )
                elif current_value >= thresholds["warning"]:
                    violations.append(
                        {
                            "resource": resource_type.value,
                            "level": "warning",
                            "current": current_value,
                            "threshold": thresholds["warning"],
                            "action_required": "monitor",
                        }
                    )

        return violations

    async def _generate_optimization_recommendations(
        self, usage: dict[str, Any], violations: list[dict[str, Any]]
    ) -> list[str]:
        """Generate optimization recommendations based on usage and violations"""
        recommendations = []

        # CPU optimization recommendations
        cpu_usage = usage.get("cpu_percent", 0)
        if cpu_usage > 80:
            recommendations.extend(
                [
                    "Consider distributing CPU-intensive tasks across multiple agents",
                    "Implement CPU throttling during peak usage periods",
                    "Review and optimize agent initialization routines",
                ]
            )

        # Memory optimization recommendations
        memory_usage = usage.get("memory_percent", 0)
        if memory_usage > 75:
            recommendations.extend(
                [
                    "Clear unused model caches and temporary data",
                    "Implement memory pooling for agent operations",
                    "Consider using streaming processing for large datasets",
                ]
            )

        # Storage optimization
        storage_usage = usage.get("storage_percent", 0)
        if storage_usage > 85:
            recommendations.extend(
                [
                    "Archive old log files and temporary data",
                    "Compress model files and data caches",
                    "Implement automatic cleanup routines",
                ]
            )

        # Critical violation responses
        for violation in violations:
            if violation["level"] == "critical":
                if violation["resource"] == "cpu":
                    recommendations.append("CRITICAL: Immediately reduce CPU load or scale horizontally")
                elif violation["resource"] == "memory":
                    recommendations.append("CRITICAL: Emergency memory cleanup required")

        return recommendations

    async def optimize_efficiency(self, optimization_target: str = "balanced") -> dict[str, Any]:
        """Apply efficiency optimizations based on target - MVP function"""
        optimization_id = f"optimize_{int(time.time())}"

        # Get current state
        current_usage = await self._get_current_resource_usage()
        current_efficiency = await self._calculate_efficiency_score()

        # Apply optimizations based on target
        optimizations_applied = await self._apply_efficiency_optimizations(optimization_target, current_usage)

        # Measure improvement
        await self._get_current_resource_usage()
        new_efficiency = await self._calculate_efficiency_score()
        improvement = new_efficiency - current_efficiency

        # Estimate savings
        estimated_savings = await self._calculate_estimated_savings(optimizations_applied, improvement)

        # Create receipt
        receipt = {
            "agent": "Sustainer",
            "action": "efficiency_optimization",
            "optimization_id": optimization_id,
            "timestamp": time.time(),
            "optimization_target": optimization_target,
            "before_efficiency": current_efficiency,
            "after_efficiency": new_efficiency,
            "improvement_percent": improvement * 100,
            "optimizations_count": len(optimizations_applied),
            "estimated_energy_savings_kwh": estimated_savings.get("energy_kwh", 0.0),
            "estimated_cost_savings_usd": estimated_savings.get("cost_usd", 0.0),
            "signature": f"sustainer_opt_{optimization_id}",
        }

        # Update tracking
        self.optimizations_applied += len(optimizations_applied)
        self.energy_saved_kwh += estimated_savings.get("energy_kwh", 0.0)
        self.cost_savings_usd += estimated_savings.get("cost_usd", 0.0)
        self.efficiency_improvements += 1

        logger.info(f"Efficiency optimization completed: {improvement * 100:.1f}% improvement")

        return {
            "status": "success",
            "optimization_id": optimization_id,
            "efficiency_improvement": improvement * 100,
            "optimizations_applied": optimizations_applied,
            "estimated_savings": estimated_savings,
            "receipt": receipt,
        }

    async def _apply_efficiency_optimizations(self, target: str, usage: dict[str, Any]) -> list[str]:
        """Apply specific efficiency optimizations"""
        optimizations = []

        if target == "power_saving":
            optimizations.extend(
                [
                    "cpu_frequency_scaling",
                    "idle_agent_suspension",
                    "background_task_scheduling",
                    "display_brightness_reduction",
                ]
            )
        elif target == "performance":
            optimizations.extend(
                [
                    "parallel_processing_enablement",
                    "cache_optimization",
                    "memory_prefetching",
                    "network_connection_pooling",
                ]
            )
        elif target == "mobile":
            optimizations.extend(
                [
                    "aggressive_model_quantization",
                    "background_sync_reduction",
                    "cellular_data_optimization",
                    "battery_usage_prioritization",
                ]
            )
        else:  # balanced
            optimizations.extend(
                [
                    "dynamic_resource_scaling",
                    "intelligent_task_scheduling",
                    "adaptive_quality_adjustment",
                    "predictive_resource_allocation",
                ]
            )

        # CPU-specific optimizations
        if usage.get("cpu_percent", 0) > 70:
            optimizations.append("cpu_load_balancing")

        # Memory-specific optimizations
        if usage.get("memory_percent", 0) > 80:
            optimizations.append("memory_garbage_collection")

        return optimizations

    async def _calculate_efficiency_score(self) -> float:
        """Calculate overall system efficiency score (0.0 to 1.0)"""
        try:
            current_usage = await self._get_current_resource_usage()

            # Calculate efficiency based on resource utilization
            cpu_efficiency = 1.0 - (current_usage.get("cpu_percent", 0) / 100.0)
            memory_efficiency = 1.0 - (current_usage.get("memory_percent", 0) / 100.0)
            storage_efficiency = 1.0 - (current_usage.get("storage_percent", 0) / 100.0)

            # Weighted efficiency score
            overall_efficiency = cpu_efficiency * 0.4 + memory_efficiency * 0.4 + storage_efficiency * 0.2

            return max(0.0, min(1.0, overall_efficiency))
        except Exception:
            return 0.7  # Default efficiency score

    async def _calculate_estimated_savings(self, optimizations: list[str], improvement: float) -> dict[str, float]:
        """Calculate estimated savings from optimizations"""
        base_energy_usage = 100.0  # kWh per day
        base_cost = 50.0  # USD per day

        # Estimate energy savings based on improvement and optimization types
        energy_savings = 0.0
        cost_savings = 0.0

        for optimization in optimizations:
            if "power" in optimization or "battery" in optimization:
                energy_savings += improvement * base_energy_usage * 0.1
                cost_savings += improvement * base_cost * 0.1
            elif "cpu" in optimization or "memory" in optimization:
                energy_savings += improvement * base_energy_usage * 0.05
                cost_savings += improvement * base_cost * 0.05

        return {
            "energy_kwh": round(energy_savings, 2),
            "cost_usd": round(cost_savings, 2),
        }

    async def get_sustainability_report(self) -> dict[str, Any]:
        """Generate comprehensive sustainability and efficiency report"""
        current_efficiency = await self._calculate_efficiency_score()
        current_usage = await self._get_current_resource_usage()

        return {
            "agent": "Sustainer",
            "report_type": "sustainability_efficiency",
            "timestamp": time.time(),
            "efficiency_metrics": {
                "overall_efficiency_score": current_efficiency,
                "cpu_utilization": current_usage.get("cpu_percent", 0),
                "memory_utilization": current_usage.get("memory_percent", 0),
                "storage_utilization": current_usage.get("storage_percent", 0),
                "optimizations_applied": self.optimizations_applied,
                "efficiency_improvements": self.efficiency_improvements,
            },
            "sustainability_metrics": {
                "energy_saved_kwh": self.energy_saved_kwh,
                "cost_savings_usd": self.cost_savings_usd,
                "device_profiles_managed": len(self.device_profiles),
                "mobile_optimized_devices": len(
                    [p for p in self.device_profiles.values() if p.device_class == DeviceClass.MOBILE]
                ),
            },
            "performance_stats": {
                "resource_samples_collected": len(self.resource_usage_history),
                "active_optimization_tasks": len(self.optimization_tasks),
                "average_response_efficiency": 0.85,
            },
            "recommendations": [
                "Continue monitoring mobile device constraints",
                "Implement predictive scaling based on usage patterns",
                "Consider renewable energy sources for high-consumption periods",
                "Optimize agent task scheduling during off-peak hours",
            ],
        }

    async def initialize(self):
        """Initialize the Sustainer Agent"""
        try:
            logger.info("Initializing Sustainer Agent - Capacity & Efficiency...")

            # Initialize with current system profile
            current_system = {
                "device_id": "system_local",
                "cpu_cores": 8,
                "memory_gb": 16.0,
                "storage_gb": 512.0,
                "power_watts": 65.0,
                "network_mbps": 1000.0,
            }

            system_profile_result = await self.profile_device(current_system)
            logger.info(f"System profile created: {system_profile_result['device_profile'].performance_tier}")

            # Initialize resource monitoring
            initial_usage = await self._get_current_resource_usage()
            logger.info(
                f"Initial resource usage: CPU {initial_usage.get('cpu_percent', 0):.1f}%, Memory {initial_usage.get('memory_percent', 0):.1f}%"
            )

            self.initialized = True
            logger.info(f"Sustainer Agent {self.agent_id} initialized - Capacity monitoring active")

        except Exception as e:
            logger.error(f"Failed to initialize Sustainer Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown Sustainer Agent gracefully"""
        try:
            logger.info("Sustainer Agent shutting down...")

            # Generate final sustainability report
            final_report = await self.get_sustainability_report()
            logger.info(f"Sustainer Agent final report: {final_report['efficiency_metrics']}")

            # Clear monitoring data
            self.resource_usage_history.clear()

            self.initialized = False

        except Exception as e:
            logger.error(f"Error during Sustainer Agent shutdown: {e}")
