#!/usr/bin/env python3
"""
Edge Computing Deployment Demo

Demonstrates the complete edge computing implementation with a realistic
multi-device deployment scenario showcasing all key features:

- Edge device registration and capability discovery
- Battery/thermal-aware deployment automation
- Fog computing orchestration
- Cross-device coordination protocols
- Mobile optimization and adaptive QoS
- Resource harvesting and tokenomics

Usage:
    python tools/deploy_edge_computing_demo.py
    python tools/deploy_edge_computing_demo.py --scenario mobile_fleet
    python tools/deploy_edge_computing_demo.py --scenario enterprise_edge
"""

import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import edge computing components
from infrastructure.fog.compute.harvest_manager import FogHarvestManager
from infrastructure.fog.edge.deployment.edge_deployer import (
    DeviceCapabilities,
    DeviceType,
    EdgeDeployer,
    NetworkQuality,
)
from infrastructure.fog.edge.fog_compute.fog_coordinator import FogCoordinator
from infrastructure.fog.edge.mobile.resource_manager import MobileResourceManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EdgeComputingDemo:
    """Comprehensive edge computing deployment demonstration"""

    def __init__(self, scenario: str = "default"):
        self.scenario = scenario
        self.edge_deployer = None
        self.fog_coordinator = None
        self.harvest_manager = None
        self.mobile_resource_manager = None
        self.demo_devices = []
        self.demo_results = {}

    async def initialize_system(self):
        """Initialize the complete edge computing system"""
        logger.info("🚀 Initializing AIVillage Edge Computing System...")

        # Initialize core components
        self.edge_deployer = EdgeDeployer(
            coordinator_id="demo_deployer", enable_fog_computing=True, enable_cross_device_coordination=True
        )

        self.fog_coordinator = FogCoordinator("demo_fog_coordinator")
        self.harvest_manager = FogHarvestManager("demo_harvest_manager", token_rate_per_hour=50)
        self.mobile_resource_manager = MobileResourceManager(harvest_enabled=True, token_rewards_enabled=True)

        logger.info("✅ Edge computing system initialized")

    def create_demo_devices(self) -> list[DeviceCapabilities]:
        """Create realistic device fleet based on scenario"""
        devices = []

        if self.scenario == "mobile_fleet":
            # Simulate a mobile device fleet (e.g., delivery company)
            devices = [
                # Driver smartphones
                DeviceCapabilities(
                    device_id="driver_phone_001",
                    device_type=DeviceType.SMARTPHONE,
                    device_name="Driver Phone - Route A",
                    cpu_cores=8,
                    cpu_freq_ghz=2.8,
                    ram_total_mb=6144,
                    ram_available_mb=3072,
                    has_gpu=True,
                    battery_powered=True,
                    battery_percent=75,
                    is_charging=False,
                    network_quality=NetworkQuality.GOOD,
                    network_type="4g",
                    supports_ml_frameworks=["tflite", "onnx"],
                ),
                DeviceCapabilities(
                    device_id="driver_phone_002",
                    device_type=DeviceType.SMARTPHONE,
                    device_name="Driver Phone - Route B",
                    cpu_cores=6,
                    cpu_freq_ghz=2.4,
                    ram_total_mb=4096,
                    ram_available_mb=2048,
                    battery_powered=True,
                    battery_percent=45,
                    is_charging=True,  # Charging in vehicle
                    network_quality=NetworkQuality.FAIR,
                    network_type="4g",
                ),
                # Vehicle tablets
                DeviceCapabilities(
                    device_id="vehicle_tablet_001",
                    device_type=DeviceType.TABLET,
                    device_name="Vehicle Dashboard Tablet",
                    cpu_cores=8,
                    cpu_freq_ghz=2.0,
                    ram_total_mb=8192,
                    ram_available_mb=4096,
                    has_gpu=True,
                    battery_powered=True,
                    battery_percent=90,
                    is_charging=True,  # Always charging in vehicle
                    network_quality=NetworkQuality.EXCELLENT,
                    network_type="5g",
                    supports_containers=True,
                ),
                # Warehouse IoT devices
                DeviceCapabilities(
                    device_id="warehouse_scanner_001",
                    device_type=DeviceType.IOT_DEVICE,
                    device_name="Warehouse Scanner",
                    cpu_cores=4,
                    cpu_freq_ghz=1.8,
                    ram_total_mb=2048,
                    ram_available_mb=1024,
                    battery_powered=True,
                    battery_percent=60,
                    is_charging=False,
                    network_quality=NetworkQuality.EXCELLENT,
                    network_type="wifi",
                ),
            ]

        elif self.scenario == "enterprise_edge":
            # Simulate enterprise edge deployment
            devices = [
                # Executive laptops
                DeviceCapabilities(
                    device_id="exec_laptop_001",
                    device_type=DeviceType.LAPTOP,
                    device_name="Executive MacBook Pro",
                    cpu_cores=10,
                    cpu_freq_ghz=3.2,
                    ram_total_mb=32768,
                    ram_available_mb=16384,
                    has_gpu=True,
                    gpu_model="M2 Max",
                    battery_powered=True,
                    battery_percent=85,
                    is_charging=False,
                    network_quality=NetworkQuality.EXCELLENT,
                    supports_containers=True,
                    supports_ml_frameworks=["pytorch", "tensorflow", "onnx"],
                ),
                # Developer workstations
                DeviceCapabilities(
                    device_id="dev_workstation_001",
                    device_type=DeviceType.DESKTOP,
                    device_name="Developer Workstation",
                    cpu_cores=16,
                    cpu_freq_ghz=3.8,
                    ram_total_mb=65536,
                    ram_available_mb=32768,
                    has_gpu=True,
                    gpu_model="RTX 4090",
                    gpu_memory_mb=24576,
                    battery_powered=False,  # Always plugged in
                    network_quality=NetworkQuality.EXCELLENT,
                    supports_containers=True,
                    supports_ml_frameworks=["pytorch", "tensorflow", "onnx", "cuda"],
                ),
                # Edge servers
                DeviceCapabilities(
                    device_id="edge_server_001",
                    device_type=DeviceType.EDGE_SERVER,
                    device_name="Edge Computing Server",
                    cpu_cores=32,
                    cpu_freq_ghz=2.9,
                    ram_total_mb=131072,  # 128GB
                    ram_available_mb=65536,
                    has_gpu=True,
                    gpu_model="A100",
                    gpu_memory_mb=81920,  # 80GB
                    battery_powered=False,
                    network_quality=NetworkQuality.EXCELLENT,
                    supports_containers=True,
                    supports_ml_frameworks=["pytorch", "tensorflow", "onnx", "tensorrt"],
                ),
            ]

        else:
            # Default mixed scenario
            devices = [
                DeviceCapabilities(
                    device_id="smartphone_demo",
                    device_type=DeviceType.SMARTPHONE,
                    device_name="Demo Smartphone",
                    cpu_cores=6,
                    cpu_freq_ghz=2.6,
                    ram_total_mb=6144,
                    ram_available_mb=3072,
                    has_gpu=True,
                    battery_powered=True,
                    battery_percent=80,
                    is_charging=True,
                    network_quality=NetworkQuality.GOOD,
                ),
                DeviceCapabilities(
                    device_id="laptop_demo",
                    device_type=DeviceType.LAPTOP,
                    device_name="Demo Laptop",
                    cpu_cores=8,
                    cpu_freq_ghz=2.8,
                    ram_total_mb=16384,
                    ram_available_mb=8192,
                    has_gpu=True,
                    battery_powered=True,
                    battery_percent=95,
                    is_charging=True,
                    network_quality=NetworkQuality.EXCELLENT,
                    supports_containers=True,
                ),
                DeviceCapabilities(
                    device_id="tablet_demo",
                    device_type=DeviceType.TABLET,
                    device_name="Demo Tablet",
                    cpu_cores=4,
                    cpu_freq_ghz=2.0,
                    ram_total_mb=4096,
                    ram_available_mb=2048,
                    battery_powered=True,
                    battery_percent=70,
                    is_charging=False,
                    network_quality=NetworkQuality.GOOD,
                ),
            ]

        self.demo_devices = devices
        return devices
