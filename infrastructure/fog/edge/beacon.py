"""
Edge Capability Beacon

Implements the "peripheral nervous system" for fog computing by:
- Advertising edge device capabilities to the fog network
- Discovering nearby fog nodes via mDNS and BitChat mesh
- Maintaining real-time capability and resource information
- Integrating with BetaNet transport for secure communication

The beacon acts as the sensory system, constantly monitoring and
advertising what resources are available on the edge device.
"""

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
import socket
import time
from typing import Any
from uuid import uuid4

import psutil

logger = logging.getLogger(__name__)


class PowerProfile(str, Enum):
    """Device power profile for scheduling decisions"""

    BATTERY_SAVER = "battery_saver"  # Minimal workload, optimize for battery
    BALANCED = "balanced"  # Standard workload capacity
    PERFORMANCE = "performance"  # Maximum workload capacity
    CHARGING = "charging"  # Device is charging, high capacity


class DeviceType(str, Enum):
    """Type of edge device"""

    MOBILE_PHONE = "mobile_phone"  # iOS/Android phone
    TABLET = "tablet"  # Tablet device
    LAPTOP = "laptop"  # Laptop/notebook
    DESKTOP = "desktop"  # Desktop computer
    EMBEDDED = "embedded"  # Embedded/IoT device
    SERVER = "server"  # Server hardware


class RuntimeCapability(str, Enum):
    """Supported execution runtimes"""

    WASI = "wasi"  # WebAssembly System Interface
    MICROVM = "microvm"  # Lightweight VM (Firecracker)
    OCI = "oci"  # OCI containers (Docker)
    NATIVE = "native"  # Native binary execution


@dataclass
class EdgeCapability:
    """Edge device capability advertisement"""

    # Device identity
    device_id: str = field(default_factory=lambda: str(uuid4()))
    device_name: str = ""
    device_type: DeviceType = DeviceType.DESKTOP
    operator_namespace: str = ""

    # Network connectivity
    endpoint: str = ""  # BetaNet endpoint
    public_ip: str = ""  # Public IP (if available)
    private_ip: str = ""  # Private/local IP
    port: int = 0  # Listening port

    # Hardware capabilities
    cpu_cores: float = 0.0  # Available CPU cores
    memory_mb: int = 0  # Available memory MB
    disk_mb: int = 0  # Available disk MB
    bandwidth_mbps: float = 0.0  # Network bandwidth Mbps

    # Runtime support
    supported_runtimes: set[RuntimeCapability] = field(default_factory=set)
    max_concurrent_jobs: int = 1  # Maximum parallel executions

    # Current utilization
    cpu_usage_percent: float = 0.0  # Current CPU usage
    memory_usage_percent: float = 0.0  # Current memory usage
    active_jobs: int = 0  # Currently running jobs

    # Power and thermal
    power_profile: PowerProfile = PowerProfile.BALANCED
    battery_percent: float | None = None  # Battery level (mobile)
    thermal_state: str = "normal"  # thermal state

    # Security features
    has_tpm: bool = False  # TPM chip available
    has_secure_boot: bool = False  # Secure boot enabled
    attestation_available: bool = False  # Can provide attestation

    # Geographic and network
    region: str = "unknown"  # Geographic region
    timezone: str = "UTC"  # Device timezone
    last_seen: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Quality metrics
    reliability_score: float = 1.0  # Historical reliability 0.0-1.0
    latency_ms: float = 0.0  # Network latency estimate
    trust_score: float = 0.5  # Trust score 0.0-1.0

    # Marketplace pricing
    spot_price_per_cpu_hour: float = 0.10  # Spot pricing rate
    on_demand_price_per_cpu_hour: float = 0.15  # On-demand pricing rate
    accepts_marketplace_bids: bool = True  # Participates in marketplace
    min_job_duration_minutes: int = 5  # Minimum job duration
    max_job_duration_hours: int = 24  # Maximum job duration
    pricing_tier: str = "basic"  # "basic", "standard", "premium"
    min_trust_required: float = 0.0  # Minimum bidder trust required
    cost_per_gb_hour_disk: float = 0.001  # Disk pricing
    cost_per_gb_network: float = 0.01  # Network pricing

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert sets to lists for JSON serialization
        data["supported_runtimes"] = list(self.supported_runtimes)
        # Convert datetime to ISO string
        data["last_seen"] = self.last_seen.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeCapability":
        """Create from dictionary"""
        # Convert lists back to sets
        if "supported_runtimes" in data:
            data["supported_runtimes"] = set(data["supported_runtimes"])
        # Convert ISO string back to datetime
        if "last_seen" in data:
            data["last_seen"] = datetime.fromisoformat(data["last_seen"])

        return cls(**data)


class CapabilityBeacon:
    """
    Edge capability beacon - the peripheral nervous system

    Continuously monitors and advertises device capabilities to the fog network.
    Acts as the sensory layer that feeds information to the autonomic control layer.
    """

    def __init__(
        self,
        device_name: str,
        operator_namespace: str,
        device_type: DeviceType = DeviceType.DESKTOP,
        betanet_endpoint: str = "",
        advertisement_interval: float = 30.0,
        discovery_port: int = 5353,
    ):
        """
        Initialize capability beacon

        Args:
            device_name: Human-readable device name
            operator_namespace: Operator namespace (org/team)
            device_type: Type of device
            betanet_endpoint: BetaNet endpoint for communication
            advertisement_interval: How often to advertise capabilities (seconds)
            discovery_port: mDNS discovery port
        """

        self.device_name = device_name
        self.operator_namespace = operator_namespace
        self.device_type = device_type
        self.betanet_endpoint = betanet_endpoint
        self.advertisement_interval = advertisement_interval
        self.discovery_port = discovery_port

        # Current capability state
        self.capability = EdgeCapability(
            device_name=device_name,
            device_type=device_type,
            operator_namespace=operator_namespace,
            endpoint=betanet_endpoint,
        )

        # Network discovery
        self._mdns_socket: socket.socket | None = None
        self._discovery_task: asyncio.Task | None = None
        self._advertisement_task: asyncio.Task | None = None

        # Discovered peers
        self.discovered_peers: dict[str, EdgeCapability] = {}
        self.fog_gateways: set[str] = set()

        # Event callbacks
        self.on_peer_discovered = None
        self.on_gateway_discovered = None
        self.on_capability_changed = None

        # State
        self._running = False
        self._last_capability_hash = None

    async def start(self):
        """Start the capability beacon"""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting capability beacon for {self.device_name}")

        # Initialize capability detection
        await self._detect_capabilities()

        # Start discovery and advertisement tasks
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        self._advertisement_task = asyncio.create_task(self._advertisement_loop())

        logger.info(
            f"Capability beacon started: {self.capability.cpu_cores} cores, "
            f"{self.capability.memory_mb}MB RAM, "
            f"runtimes: {self.capability.supported_runtimes}"
        )

    async def stop(self):
        """Stop the capability beacon"""
        if not self._running:
            return

        self._running = False
        logger.info("Stopping capability beacon")

        # Cancel tasks
        if self._discovery_task:
            self._discovery_task.cancel()
        if self._advertisement_task:
            self._advertisement_task.cancel()

        # Close network resources
        if self._mdns_socket:
            self._mdns_socket.close()
            self._mdns_socket = None

    async def _detect_capabilities(self):
        """Detect device capabilities and resources"""

        # CPU capabilities
        self.capability.cpu_cores = float(psutil.cpu_count(logical=True))

        # Memory capabilities
        memory = psutil.virtual_memory()
        self.capability.memory_mb = int(memory.total / (1024 * 1024))

        # Disk capabilities
        disk = psutil.disk_usage("/")
        self.capability.disk_mb = int(disk.free / (1024 * 1024))

        # Network capabilities
        self.capability.private_ip = self._get_local_ip()
        self.capability.bandwidth_mbps = await self._detect_bandwidth()

        # Runtime support detection
        runtimes = set()

        # Check for WASI runtime
        if await self._check_wasi_support():
            runtimes.add(RuntimeCapability.WASI)

        # Check for MicroVM support (Firecracker)
        if await self._check_microvm_support():
            runtimes.add(RuntimeCapability.MICROVM)

        # Check for OCI/Docker support
        if await self._check_oci_support():
            runtimes.add(RuntimeCapability.OCI)

        # Native execution always supported
        runtimes.add(RuntimeCapability.NATIVE)

        self.capability.supported_runtimes = runtimes

        # Power profile detection
        battery = psutil.sensors_battery()
        if battery:
            self.capability.battery_percent = battery.percent
            if battery.power_plugged:
                self.capability.power_profile = PowerProfile.CHARGING
            elif battery.percent < 20:
                self.capability.power_profile = PowerProfile.BATTERY_SAVER
            else:
                self.capability.power_profile = PowerProfile.BALANCED
        else:
            self.capability.power_profile = PowerProfile.PERFORMANCE

        # Security features
        self.capability.has_tpm = self._detect_tpm()
        self.capability.has_secure_boot = self._detect_secure_boot()

        # Geographic location (rough estimate)
        self.capability.region = self._detect_region()

        logger.info(
            f"Detected capabilities: {self.capability.cpu_cores} cores, "
            f"{self.capability.memory_mb}MB, power: {self.capability.power_profile}"
        )

    async def _update_utilization(self):
        """Update current resource utilization"""

        # CPU utilization
        self.capability.cpu_usage_percent = psutil.cpu_percent(interval=1.0)

        # Memory utilization
        memory = psutil.virtual_memory()
        self.capability.memory_usage_percent = memory.percent

        # Battery status
        battery = psutil.sensors_battery()
        if battery:
            self.capability.battery_percent = battery.percent

            # Update power profile based on battery
            if battery.power_plugged:
                self.capability.power_profile = PowerProfile.CHARGING
            elif battery.percent < 15:
                self.capability.power_profile = PowerProfile.BATTERY_SAVER
            elif battery.percent < 50:
                self.capability.power_profile = PowerProfile.BALANCED
            else:
                self.capability.power_profile = PowerProfile.PERFORMANCE

        # Thermal state (simplified)
        if self.capability.cpu_usage_percent > 90:
            self.capability.thermal_state = "hot"
        elif self.capability.cpu_usage_percent > 70:
            self.capability.thermal_state = "warm"
        else:
            self.capability.thermal_state = "normal"

        # Update marketplace pricing based on current conditions
        await self._update_marketplace_pricing()

        # Update timestamp
        self.capability.last_seen = datetime.now(UTC)

    async def _update_marketplace_pricing(self):
        """Update marketplace pricing based on current device conditions"""

        # Base pricing rates
        base_spot_rate = 0.10  # Base spot rate per CPU-hour
        base_on_demand_rate = 0.15  # Base on-demand rate per CPU-hour

        # Utilization-based pricing adjustments
        cpu_utilization = self.capability.cpu_usage_percent / 100.0
        memory_utilization = self.capability.memory_usage_percent / 100.0
        avg_utilization = (cpu_utilization + memory_utilization) / 2.0

        # Utilization multiplier: higher utilization = higher prices
        utilization_multiplier = 1.0 + (avg_utilization * 0.5)  # Up to 50% premium

        # Power profile adjustments
        power_multipliers = {
            PowerProfile.CHARGING: 0.8,  # 20% discount when charging
            PowerProfile.PERFORMANCE: 1.0,  # Normal pricing
            PowerProfile.BALANCED: 1.1,  # 10% premium for balanced mode
            PowerProfile.BATTERY_SAVER: 1.5,  # 50% premium for battery saver
        }
        power_multiplier = power_multipliers.get(self.capability.power_profile, 1.0)

        # Thermal state adjustments
        thermal_multipliers = {
            "normal": 1.0,  # Normal pricing
            "warm": 1.2,  # 20% premium when warm
            "hot": 1.5,  # 50% premium when hot
        }
        thermal_multiplier = thermal_multipliers.get(self.capability.thermal_state, 1.0)

        # Trust score bonus: higher trust = higher prices
        trust_multiplier = 1.0 + (self.capability.trust_score * 0.3)  # Up to 30% premium

        # Device type adjustments
        device_multipliers = {
            DeviceType.SERVER: 0.8,  # 20% discount for servers
            DeviceType.DESKTOP: 1.0,  # Normal pricing
            DeviceType.LAPTOP: 1.1,  # 10% premium for laptops
            DeviceType.TABLET: 1.3,  # 30% premium for tablets
            DeviceType.MOBILE_PHONE: 1.5,  # 50% premium for phones
            DeviceType.EMBEDDED: 0.9,  # 10% discount for embedded
        }
        device_multiplier = device_multipliers.get(self.capability.device_type, 1.0)

        # Calculate final pricing
        spot_price = (
            base_spot_rate
            * utilization_multiplier
            * power_multiplier
            * thermal_multiplier
            * trust_multiplier
            * device_multiplier
        )

        on_demand_price = (
            base_on_demand_rate
            * utilization_multiplier
            * power_multiplier
            * thermal_multiplier
            * trust_multiplier
            * device_multiplier
        )

        # Update capability with new pricing
        self.capability.spot_price_per_cpu_hour = round(spot_price, 4)
        self.capability.on_demand_price_per_cpu_hour = round(on_demand_price, 4)

        # Determine pricing tier based on device capabilities
        if self.capability.has_tpm and self.capability.has_secure_boot and self.capability.attestation_available:
            self.capability.pricing_tier = "premium"
        elif self.capability.cpu_cores >= 4 and self.capability.memory_mb >= 8192:
            self.capability.pricing_tier = "standard"
        else:
            self.capability.pricing_tier = "basic"

        # Adjust marketplace participation based on power state
        if self.capability.power_profile == PowerProfile.BATTERY_SAVER:
            # Disable marketplace participation when in battery saver mode
            self.capability.accepts_marketplace_bids = False
        elif self.capability.battery_percent is not None and self.capability.battery_percent < 20:
            # Disable when battery is very low
            self.capability.accepts_marketplace_bids = False
        else:
            self.capability.accepts_marketplace_bids = True

        # Adjust minimum trust requirements based on pricing tier
        tier_trust_requirements = {
            "basic": 0.0,  # No minimum trust
            "standard": 0.3,  # Some trust required
            "premium": 0.7,  # High trust required
        }
        self.capability.min_trust_required = tier_trust_requirements.get(self.capability.pricing_tier, 0.0)

        # Log pricing updates periodically
        if hasattr(self, "_last_pricing_log"):
            if time.time() - self._last_pricing_log > 300:  # Log every 5 minutes
                self._log_pricing_update()
                self._last_pricing_log = time.time()
        else:
            self._last_pricing_log = time.time()
            self._log_pricing_update()

    def _log_pricing_update(self):
        """Log current pricing information"""
        logger.info(
            f"Marketplace pricing update: "
            f"spot=${self.capability.spot_price_per_cpu_hour:.4f}/cpu-hour, "
            f"on-demand=${self.capability.on_demand_price_per_cpu_hour:.4f}/cpu-hour, "
            f"tier={self.capability.pricing_tier}, "
            f"accepting_bids={self.capability.accepts_marketplace_bids}, "
            f"trust_required={self.capability.min_trust_required:.2f}"
        )

    def get_marketplace_listing(self) -> dict[str, Any]:
        """Get marketplace listing information for this device"""

        available_cpu = self.capability.cpu_cores * (1.0 - self.capability.cpu_usage_percent / 100.0)
        available_memory = self.capability.memory_mb * (1.0 - self.capability.memory_usage_percent / 100.0)

        return {
            "listing_id": f"edge_{self.capability.device_id}",
            "node_id": self.capability.device_id,
            "cpu_cores": available_cpu,
            "memory_gb": available_memory / 1024.0,  # Convert MB to GB
            "disk_gb": self.capability.disk_mb / 1024.0,  # Convert MB to GB
            "spot_price_per_cpu_hour": self.capability.spot_price_per_cpu_hour,
            "on_demand_price_per_cpu_hour": self.capability.on_demand_price_per_cpu_hour,
            "pricing_tier": self.capability.pricing_tier,
            "trust_score": self.capability.trust_score,
            "latency_ms": self.capability.latency_ms,
            "accepts_spot_bids": self.capability.accepts_marketplace_bids,
            "accepts_on_demand": self.capability.accepts_marketplace_bids,
            "min_trust_required": self.capability.min_trust_required,
            "min_duration_minutes": self.capability.min_job_duration_minutes,
            "max_duration_hours": self.capability.max_job_duration_hours,
            "power_profile": self.capability.power_profile.value,
            "thermal_state": self.capability.thermal_state,
            "device_type": self.capability.device_type.value,
            "region": self.capability.region,
            "supported_runtimes": list(self.capability.supported_runtimes),
            "available_until": None,  # Always available (subject to power constraints)
            "endpoint": self.capability.endpoint,
        }

    def estimate_job_cost(
        self, cpu_cores: float, memory_gb: float, disk_gb: float, duration_hours: float, bid_type: str = "spot"
    ) -> float:
        """Estimate cost for job execution on this device"""

        # Base pricing
        if bid_type == "spot":
            cpu_rate = self.capability.spot_price_per_cpu_hour
        else:
            cpu_rate = self.capability.on_demand_price_per_cpu_hour

        # Calculate costs
        cpu_cost = cpu_cores * duration_hours * cpu_rate
        memory_cost = memory_gb * duration_hours * (cpu_rate * 0.1)  # Memory is 10% of CPU rate
        disk_cost = disk_gb * duration_hours * self.capability.cost_per_gb_hour_disk

        total_cost = cpu_cost + memory_cost + disk_cost

        return round(total_cost, 4)

    async def _advertisement_loop(self):
        """Continuously advertise capabilities"""

        while self._running:
            try:
                # Update current utilization
                await self._update_utilization()

                # Check if capabilities changed significantly
                current_hash = hash(str(self.capability.to_dict()))
                if current_hash != self._last_capability_hash:
                    self._last_capability_hash = current_hash

                    # Notify gateway of capability changes
                    await self._advertise_to_gateways()

                    # Trigger callback
                    if self.on_capability_changed:
                        await self.on_capability_changed(self.capability)

                # Advertise via mDNS
                await self._mdns_advertise()

                # Advertise via BetaNet
                await self._betanet_advertise()

                await asyncio.sleep(self.advertisement_interval)

            except Exception as e:
                logger.error(f"Advertisement loop error: {e}")
                await asyncio.sleep(5.0)

    async def _discovery_loop(self):
        """Continuously discover peer devices and gateways"""

        while self._running:
            try:
                # Discover via mDNS
                await self._mdns_discover()

                # Discover via BetaNet mesh
                await self._betanet_discover()

                await asyncio.sleep(10.0)  # Discovery every 10 seconds

            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(5.0)

    async def _mdns_advertise(self):
        """Advertise capabilities via mDNS"""

        # Implement proper mDNS advertisement\n        try:\n            from zeroconf import Zeroconf, ServiceInfo\n            import socket\n            \n            zeroconf = Zeroconf()\n            info = ServiceInfo(\n                "_fog-edge._tcp.local.",\n                f"{self.capability.device_id}._fog-edge._tcp.local.",\n                addresses=[socket.inet_aton(self.capability.private_ip)],\n                port=8080,\n                properties={\n                    "version": "1.0",\n                    "cpu_cores": str(self.capability.cpu_cores),\n                    "memory_mb": str(self.capability.memory_mb)\n                }\n            )\n            zeroconf.register_service(info)\n            logger.info(f"Advertised fog edge service via mDNS")\n        except ImportError:\n            logger.warning("zeroconf library not available for mDNS")"
        # For now, log the advertisement
        logger.debug(
            f"mDNS advertisement: {self.device_name} with "
            f"{self.capability.cpu_cores} cores, "
            f"{len(self.capability.supported_runtimes)} runtimes"
        )

    async def _mdns_discover(self):
        """Discover peers via mDNS"""

        # Implement proper mDNS discovery
        try:
            from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

            class FogServiceListener(ServiceListener):
                def __init__(self, beacon):
                    self.beacon = beacon

                def add_service(self, zeroconf, type, name):
                    info = zeroconf.get_service_info(type, name)
                    if info and "_fog-edge._tcp.local." in type:
                        # Discovered a fog edge device
                        device_capability = self.beacon._parse_service_info(info)
                        if device_capability:
                            self.beacon.discovered_peers[device_capability.device_id] = device_capability
                            logger.info(f"Discovered fog peer: {device_capability.device_id}")

            zeroconf = Zeroconf()
            listener = FogServiceListener(self)
            ServiceBrowser(zeroconf, "_fog-edge._tcp.local.", listener)

            logger.info("Started mDNS discovery for fog peers")

        except ImportError:
            logger.warning("zeroconf library not available for mDNS discovery")

    async def _betanet_advertise(self):
        """Advertise capabilities via BetaNet mesh"""

        if not self.betanet_endpoint:
            return

        # TODO: Integrate with actual BetaNet transport
        # This would send capability advertisements through BitChat mesh
        message = {
            "type": "capability_advertisement",
            "device_id": self.capability.device_id,
            "capability": self.capability.to_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.debug(f"BetaNet advertisement: {message['type']}")

    async def _betanet_discover(self):
        """Discover peers via BetaNet mesh"""

        # TODO: Integrate with actual BetaNet transport
        # This would listen for capability advertisements from other devices
        pass

    async def _advertise_to_gateways(self):
        """Send capability update to known fog gateways"""

        for gateway_endpoint in self.fog_gateways:
            try:
                # TODO: Send HTTPS POST to gateway endpoint
                # POST /v1/fog/admin/nodes/{device_id}/heartbeat
                logger.debug(f"Sending heartbeat to gateway: {gateway_endpoint}")

            except Exception as e:
                logger.warning(f"Failed to advertise to gateway {gateway_endpoint}: {e}")

    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    async def _check_wasi_support(self) -> bool:
        """Check if WASI runtime is available"""

        # TODO: Check for wasmtime, wasmer, or other WASI runtime
        try:
            proc = await asyncio.create_subprocess_exec(
                "wasmtime", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    async def _check_microvm_support(self) -> bool:
        """Check if MicroVM (Firecracker) support is available"""

        # TODO: Check for Firecracker binary and KVM support
        try:
            proc = await asyncio.create_subprocess_exec(
                "firecracker", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    async def _check_oci_support(self) -> bool:
        """Check if OCI container support is available"""

        # Check for Docker
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            pass

        # Check for Podman
        try:
            proc = await asyncio.create_subprocess_exec(
                "podman", "--version", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    def _detect_tpm(self) -> bool:
        """Detect TPM chip availability"""

        # TODO: Implement proper TPM detection
        # Check /dev/tpm0 on Linux, or use tpm2-tools
        import os

        return os.path.exists("/dev/tpm0")

    def _detect_secure_boot(self) -> bool:
        """Detect if secure boot is enabled"""

        # TODO: Implement proper secure boot detection
        try:
            with open("/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c", "rb") as f:
                data = f.read()
                return len(data) > 4 and data[4] == 1
        except (FileNotFoundError, PermissionError, OSError):
            return False

    def _detect_region(self) -> str:
        """Detect geographic region (rough estimate)"""

        # Implement proper region detection using system timezone
        import time

        try:
            # Use timezone-based detection first
            timezone = time.tzname[0] if time.tzname else None
            if timezone:
                # Map common timezones to regions
                timezone_map = {
                    "PST": "us-west-1",
                    "PDT": "us-west-1",
                    "MST": "us-west-2",
                    "MDT": "us-west-2",
                    "CST": "us-central-1",
                    "CDT": "us-central-1",
                    "EST": "us-east-1",
                    "EDT": "us-east-1",
                    "UTC": "eu-west-1",
                    "GMT": "eu-west-1",
                }
                if timezone in timezone_map:
                    return timezone_map[timezone]
        except Exception as e:
            import logging

            logging.exception("Exception in timezone-based region detection: %s", str(e))

        return "us-west-2"  # Default region

    def _parse_service_info(self, info) -> EdgeCapability | None:
        """Parse mDNS service info into EdgeCapability"""
        try:
            if info and info.properties:
                properties = {k.decode(): v.decode() for k, v in info.properties.items()}

                capability = EdgeCapability(
                    device_id=info.name.split(".")[0],
                    cpu_cores=int(properties.get("cpu_cores", 1)),
                    memory_mb=int(properties.get("memory_mb", 1024)),
                    disk_mb=int(properties.get("disk_mb", 10000)),
                    private_ip=str(info.addresses[0]) if info.addresses else "127.0.0.1",
                    bandwidth_mbps=float(properties.get("bandwidth_mbps", 100.0)),
                )

                return capability
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to parse mDNS service info: {e}")

        return None

    async def _detect_bandwidth(self) -> float:
        """Detect actual network bandwidth in Mbps"""
        try:
            import psutil

            # Get network interface speeds
            interfaces = psutil.net_if_stats()
            max_speed = 0

            for interface, stats in interfaces.items():
                if stats.isup and stats.speed > 0:
                    max_speed = max(max_speed, stats.speed)

            if max_speed > 0:
                return float(max_speed)

            # Fallback: estimate based on interface names
            interface_names = list(psutil.net_if_addrs().keys())
            for name in interface_names:
                name_lower = name.lower()
                if "eth" in name_lower or "lan" in name_lower:
                    return 1000.0  # Assume gigabit ethernet
                elif "wlan" in name_lower or "wifi" in name_lower:
                    return 100.0  # Assume 100 Mbps WiFi

            return 10.0  # Conservative default

        except Exception as e:
            logger.warning(f"Failed to detect bandwidth: {e}")
            return 100.0  # Default fallback

    def add_gateway(self, gateway_endpoint: str):
        """Add known fog gateway endpoint"""
        self.fog_gateways.add(gateway_endpoint)
        logger.info(f"Added fog gateway: {gateway_endpoint}")

    def remove_gateway(self, gateway_endpoint: str):
        """Remove fog gateway endpoint"""
        self.fog_gateways.discard(gateway_endpoint)

    def get_capability(self) -> EdgeCapability:
        """Get current device capability"""
        return self.capability

    def get_discovered_peers(self) -> list[EdgeCapability]:
        """Get list of discovered peer devices"""
        return list(self.discovered_peers.values())

    def update_job_count(self, active_jobs: int):
        """Update active job count"""
        self.capability.active_jobs = active_jobs

    def set_max_concurrent_jobs(self, max_jobs: int):
        """Set maximum concurrent jobs"""
        self.capability.max_concurrent_jobs = max_jobs
